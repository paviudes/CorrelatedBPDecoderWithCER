# forward_gpu.jl
# ----------------------------------------------------------------------------
# GPU-accelerated forward pass for the Nachmani Neural BP decoder.
# Self-contained: does not modify, shadow, or depend on any function in the
# rest of the package. Wrapped in a module so loading this file cannot affect
# anything else.
#
# Apple Silicon (Metal.jl) is the default backend. Switching to NVIDIA (CUDA.jl)
# is a two-line change in the BACKEND section below.
#
# Designed for *testing only* — large batches, no autodiff, no training. Sparse
# matrices are densified (Metal has no usable sparse linalg); for the ~12-edge
# Tanner graphs of small codes this is essentially free.
#
# Usage:
#   include("src/forward_gpu.jl")
#   using .NeuralBPGPU
#   posteriors = NeuralBPGPU.forward_pass_gpu(bpnn, initial_llrs_batch, syndromes_batch)
#
# Inputs (CPU):
#   bpnn::NachmaniNeuralBP
#   initial_llrs_batch :: Matrix{Float32},       (n_bits × n_samples)
#   syndromes_batch    :: BitMatrix,             (n_checks × n_samples)
# Output (CPU):
#   posteriors :: Array{Float32, 3},             (n_bits × n_samples × n_layers)
# ----------------------------------------------------------------------------

#module NeuralBPGPU

# ============================================================================
# BACKEND — conditionally defined based on USE_GPU environment variable
#   Apple Silicon (M-series): ArrayT = Metal.MtlArray (when USE_GPU=1)
#   CPU fallback:             ArrayT = Array (when USE_GPU=0)
# ============================================================================
@static if CorrelatedBPDecoderWithCER.USE_GPU && CorrelatedBPDecoderWithCER.METAL_LOADED
    const ArrayT = CorrelatedBPDecoderWithCER.Metal.MtlArray
    const GPU_AVAILABLE = true
else
    const ArrayT = Array
    const GPU_AVAILABLE = false
end
# ============================================================================

# ----------------------------------------------------------------------------
# GPU-friendly activations. Pure broadcasts — work on MtlArray (or CuArray).
# Inlined here so this module has no dependency on utils.jl.
# ----------------------------------------------------------------------------

@inline function _safe_log_tanh_split(x)
    t = tanh.(abs.(x) .* 0.5f0)
    t_clipped = clamp.(t, eps(Float32), 1f0 - eps(Float32))
    magnitudes = log.(t_clipped)
    signs = x .< 0f0
    return magnitudes, signs
end

@inline function _safe_atanh_exp_signed(magnitudes, signs_bool_or_fp)
    e = exp.(magnitudes)
    e_clipped = clamp.(e, eps(Float32), 1f0 - eps(Float32))
    # Linear combination form, avoids relying on Bool-array branching support
    # (Metal handles Bool inconsistently across versions; Float32 is universal).
    sign_pm = 1f0 .- 2f0 .* signs_bool_or_fp           # true → -1, false → +1
    return 2f0 .* atanh.(sign_pm .* e_clipped)
end

# ----------------------------------------------------------------------------
# GPU-side state, built once per forward_pass call.
# ----------------------------------------------------------------------------

struct GPUState
    # Dense adjacency matrices on GPU
    adj_V2C_C2V        :: ArrayT{Float32, 2}
    adj_C2V_V2C        :: ArrayT{Float32, 2}
    adj_C2V_readout    :: ArrayT{Float32, 2}
    adj_initialize_V2C :: ArrayT{Float32, 2}

    # Per-layer dense message weights, masked by adj_C2V_V2C upfront so
    # we don't redo the multiply each layer.  Shape: (nb_neurons × nb_neurons × n_layers).
    W_msg_per_layer    :: ArrayT{Float32, 3}

    # Readout weight matrix (shared across layers in this Nachmani impl),
    # masked by adj_C2V_readout upfront.  Shape: (n_bits × nb_neurons).
    W_readout_masked   :: ArrayT{Float32, 2}

    # Per-layer LLR weights.  Shape: (n_bits × n_layers).
    weights_llrs       :: ArrayT{Float32, 2}

    # Pre-routed syndromes — neuron_to_checks indexing applied on CPU once,
    # then uploaded.  Shape: (nb_neurons × n_samples), Float32 in {0, 1}.
    syndromes_routed   :: ArrayT{Float32, 2}

    nb_neurons :: Int
    n_bits     :: Int
    n_layers   :: Int
end

# Convert a sparse matrix to a dense Float32 GPU array: Sparse → dense → Float32 → GPU .
_to_dense_gpu(M::AbstractMatrix) = ArrayT{Float32}(Float32.(Matrix(M)))

function _scatter_layer_weights(rows::Vector{Int}, cols::Vector{Int},
                                weights_flat::AbstractVector{<:Real},
                                n_layers::Int, dim_rows::Int, dim_cols::Int)
    """
    Given the weights for all layers in a flat vector, arrange them into a 3D tensor of shape (nb_neurons × nb_neurons × n_layers) according to the provided row and column indices, which specify where the non-zero weights should be placed for each layer.
    Arguments:
    - `rows::Vector{Int}`: Row indices for the non-zero weights in the adjacency matrix.
    - `cols::Vector{Int}`: Column indices for the non-zero weights in the adjacency matrix.
    - `weights_flat::AbstractVector{<:Real}`: A flat vector containing the weights for all layers, concatenated together. The length should be equal to `length(rows) * n_layers`.
    - `n_layers::Int`: The number of layers (iterations) in the NeuralBP model.
    - `dim_rows::Int`: The number of rows in the output tensor (should match nb_neurons).
    - `dim_cols::Int`: The number of columns in the output tensor (should match nb_neurons).
    """
    n_per_layer = length(rows)
    weights_tensor = zeros(Float32, dim_rows, dim_cols, n_layers)
    @inbounds for l in 1:n_layers
        offset = (l - 1) * n_per_layer
        for k in 1:n_per_layer
            weights_tensor[rows[k], cols[k], l] = weights_flat[offset + k]
        end
    end
    return weights_tensor
end

function build_gpu_state(bpnn, syndromes_batch::BitMatrix)
    """
    Build the GPUState struct by converting and uploading all necessary data from the CPU NeuralBP model and the input syndromes batch. This includes:
    - Adjacency matrices (densified and converted to Float32)
    - Per-layer message weight tensors, pre-masked by the adjacency matrices
    - Readout weight matrix, pre-masked by the readout adjacency
    - Per-layer LLR weights, reshaped and converted to Float32
    - Syndromes, routed according to neuron_to_checks and converted to Float32
    """
    base = bpnn.base
    
    # Adjacency
    adj_V2C_C2V = _to_dense_gpu(base.adj_V2C_C2V)
    adj_C2V_V2C_gpu = _to_dense_gpu(base.adj_C2V_V2C)
    adj_C2V_readout_gpu = _to_dense_gpu(base.adj_C2V_readout)
    adj_initialize_V2C = _to_dense_gpu(base.adj_initialize_V2C)

    # Per-layer message weight tensor, pre-masked by adj_C2V_V2C
    W_msg_cpu = _scatter_layer_weights(
        Vector{Int}(base.non_zero_rows_C2V_V2C),
        Vector{Int}(base.non_zero_cols_C2V_V2C),
        bpnn.weights_c2v_v2c,
        base.n_layers,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer,
    )
    # Multiply the weights by the adjacency mask so that the GPU doesn't have to do it every layer.
    @inbounds for l in 1:base.n_layers
        @views W_msg_cpu[:, :, l] .*= base.adj_C2V_V2C
    end
    W_msg_per_layer = ArrayT{Float32}(W_msg_cpu)

    # Readout weight matrix, pre-masked by adj_C2V_readout
    W_readout_cpu = zeros(Float32, base.code_n_bits, base.nb_neurons_per_layer)
    rows_r = Vector{Int}(base.non_zero_rows_C2V_readout)
    cols_r = Vector{Int}(base.non_zero_cols_C2V_readout)
    @inbounds for k in eachindex(rows_r)
        W_readout_cpu[rows_r[k], cols_r[k]] = bpnn.weights_c2v_readout[k]
    end
    # Multiply the readout weights by the adjacency mask so that the GPU doesn't have to do it every layer.
    W_readout_cpu .*= base.adj_C2V_readout
    W_readout_masked = _to_dense_gpu(W_readout_cpu)

    # Per-layer LLR weights as (n_bits × n_layers)
    weights_llrs_cpu = reshape(Vector{Float32}(bpnn.weights_llrs),
                               base.code_n_bits, base.n_layers)
    weights_llrs_gpu = _to_dense_gpu(weights_llrs_cpu)

    # Route syndromes once on CPU, upload as Float32
    syndromes_routed_gpu = _to_dense_gpu(syndromes_batch[Vector{Int}(base.neuron_to_checks), :])

    return GPUState(
        adj_V2C_C2V,
        adj_C2V_V2C_gpu,
        adj_C2V_readout_gpu,
        adj_initialize_V2C,
        W_msg_per_layer,
        W_readout_masked,
        weights_llrs_gpu,
        syndromes_routed_gpu,
        base.nb_neurons_per_layer,
        base.code_n_bits,
        base.n_layers,
    )
end

# ----------------------------------------------------------------------------
# Single-layer forward pass on GPU.
# Returns (new messages_c2v, posterior_llrs_at_this_layer).
# ----------------------------------------------------------------------------

function _compute_layer_gpu(messages_c2v, initial_llrs_gpu, gs::GPUState, layer::Int)
    # Slice this layer's weights.
    # Indexing the third dim of a 3D MtlArray gives a 2D MtlArray; this is fine
    # on Metal but allocates — for n_layers = 100 and n=12 the cost is trivial.
    W_msg = gs.W_msg_per_layer[:, :, layer]                  # (nb_neurons × nb_neurons)
    weights_llr_col = gs.weights_llrs[:, layer]              # (n_bits,)

    # ---- C → V (variable-to-check incoming form) ----
    m_v2c = W_msg * messages_c2v                              # (nb_neurons × batch)
    scaled_llrs = gs.adj_initialize_V2C * (weights_llr_col .* initial_llrs_gpu)
    m_v2c = m_v2c .+ scaled_llrs                              # avoid in-place .+= for safety on Metal

    m_v2c_mag, m_v2c_signs_bool = _safe_log_tanh_split(m_v2c)
    m_v2c_signs_fp = Float32.(m_v2c_signs_bool)               # for arithmetic parity

    # ---- V → C ----
    m_c2v_mag = gs.adj_V2C_C2V * m_v2c_mag                    # (nb_neurons × batch)

    # Parity: count negative-sign incoming messages, mod 2; XOR with syndrome.
    parity_fp = gs.adj_V2C_C2V * m_v2c_signs_fp               # integer-valued floats
    # (a + b) mod 2  ==  XOR for {0,1}-valued a, b
    combined = mod.(parity_fp, 2f0) .+ gs.syndromes_routed
    m_c2v_signs_fp = mod.(combined, 2f0)                      # 0 or 1 in Float32

    messages_c2v_new = _safe_atanh_exp_signed(m_c2v_mag, m_c2v_signs_fp)

    # ---- Readout ----
    posterior_llrs = (weights_llr_col .* initial_llrs_gpu) .+ gs.W_readout_masked * messages_c2v_new

    return messages_c2v_new, posterior_llrs
end

# ----------------------------------------------------------------------------
# Top-level entry point.
# ----------------------------------------------------------------------------

function forward_pass_gpu(
    bpnn,
    initial_llrs_batch::AbstractMatrix{Float32},
    syndromes_batch::BitMatrix;
    chunk_size::Int = 0,
)
    """
    Perform a forward pass of the NeuralBP model on GPU, returning the posterior LLRs at each layer.
    For codes / sample counts whose output tensor exceeds Metal's per-buffer
    size limit (`Metal.device().maxBufferLength`), the samples are processed
    in chunks. Each chunk's result is transferred to a CPU output tensor and
    discarded from GPU before the next chunk runs.
    Arguments:
    - `bpnn::NachmaniNeuralBP`: The trained NeuralBP model containing the base code structure and weights.
    - `initial_llrs_batch::AbstractMatrix{Float32}`: A matrix of shape (n_bits × n_samples) containing the initial LLRs for each bit and sample in the batch.
    - `syndromes_batch::BitMatrix`: A matrix of shape (n_checks × n_samples) where each column represents a syndrome corresponding to an error pattern.
    - `chunk_size::Int=0`: If positive, samples are processed in slices of this size. If 0 (default), an automatic estimate is computed from the device's `maxBufferLength`, leaving headroom for the per-layer intermediate allocations.
    """
    n_samples = size(initial_llrs_batch, 2)
    n_bits    = bpnn.base.code_n_bits
    n_layers  = bpnn.base.n_layers

    # ---- Auto-pick chunk size to stay under maxBufferLength ----
    # The output tensor (n_bits × chunk × n_layers × 4 bytes) is the largest
    # single buffer allocated. Use 1/4 of maxBufferLength as our budget so
    # the per-layer intermediates, masked weights, syndromes, etc., still fit.
    if chunk_size <= 0
        budget_bytes = try
            if GPU_AVAILABLE && CorrelatedBPDecoderWithCER.METAL_LOADED
                Int(CorrelatedBPDecoderWithCER.Metal.device().maxBufferLength) ÷ 4
            else
                Int(2^30)   # 1 GB fallback for CPU
            end
        catch
            Int(2^30)   # 1 GB fallback if the device query is unavailable
        end
        per_sample_bytes = n_bits * n_layers * 4
        chunk_size = max(1, budget_bytes ÷ per_sample_bytes)
        chunk_size = min(chunk_size, n_samples)
    end

    # ---- Single-shot path: everything fits in one chunk ----
    if chunk_size >= n_samples
        return _forward_pass_gpu_chunk(bpnn, initial_llrs_batch, syndromes_batch)
    end

    # ---- Chunked path: write each chunk's result into a CPU output tensor ----
    posterior_3d_cpu = zeros(Float32, n_bits, n_samples, n_layers)
    for start in 1:chunk_size:n_samples
        stop = min(start + chunk_size - 1, n_samples)
        chunk_llrs = initial_llrs_batch[:, start:stop]
        chunk_synd = syndromes_batch[:,    start:stop]
        chunk_out  = _forward_pass_gpu_chunk(bpnn, chunk_llrs, chunk_synd)
        @views posterior_3d_cpu[:, start:stop, :] .= chunk_out
    end
    return posterior_3d_cpu
end

function _forward_pass_gpu_chunk(
    bpnn,
    initial_llrs_batch::AbstractMatrix{Float32},
    syndromes_batch::BitMatrix,
)
    """
    Single-chunk forward pass on GPU. Same logic as the previous (non-chunked)
    forward_pass_gpu body — used either directly (when everything fits in one
    Metal buffer) or once per chunk by `forward_pass_gpu` for large sample
    counts.
    """
    gs = build_gpu_state(bpnn, syndromes_batch)
    n_samples = size(initial_llrs_batch, 2)

    initial_llrs_gpu = ArrayT{Float32}(Matrix{Float32}(initial_llrs_batch))
    messages_c2v = ArrayT{Float32}(zeros(Float32, gs.nb_neurons, n_samples))

    # Pre-allocate output 3D tensor on GPU and write per-layer slices into it.
    # Avoids GPU `cat` which has had reliability issues across Metal versions.
    posterior_3d_gpu = ArrayT{Float32}(zeros(Float32, gs.n_bits, n_samples, gs.n_layers))

    for layer in 1:gs.n_layers
        messages_c2v, post = _compute_layer_gpu(messages_c2v, initial_llrs_gpu, gs, layer)
        @views posterior_3d_gpu[:, :, layer] .= post
    end

    return Array(posterior_3d_gpu)   # transfer to CPU
end

# ----------------------------------------------------------------------------

#end # module NeuralBPGPU
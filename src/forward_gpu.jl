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
# ----------------------------------------------------------------------------

# ============================================================================
# BACKEND — the GPU array type is chosen by which backend package is compiled in
# for this PLATFORM (Metal on Apple, CUDA on Linux); see the backend block in
# CorrelatedBPDecoderWithCER.jl. `GPU_AVAILABLE` means "a GPU backend is present"
# — it does NOT mean the GPU is enabled. Whether the GPU is actually used is the
# RUNTIME decision `CorrelatedBPDecoderWithCER.gpu_active()` (checked by the
# caller in predict.jl), so this const is safe to bake at precompile time.
#   Apple Silicon (M-series): ArrayT = Metal.MtlArray
#   NVIDIA (HPC):             ArrayT = CUDA.CuArray
#   Neither backend present:  ArrayT = Array
# ============================================================================
@static if CorrelatedBPDecoderWithCER.METAL_LOADED
    const ArrayT = CorrelatedBPDecoderWithCER.Metal.MtlArray
    const GPU_AVAILABLE = true
elseif CorrelatedBPDecoderWithCER.CUDA_LOADED
    const ArrayT = CorrelatedBPDecoderWithCER.CUDA.CuArray
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

    # Per-layer dense message weights, masked by adj_C2V_V2C upfront so we don't
    # redo the multiply each layer. ONE (nb_neurons × nb_neurons) array PER
    # LAYER, rather than one (nb_neurons × nb_neurons × n_layers) tensor that
    # `_compute_layer_gpu` would have to slice: slicing the third dimension of a
    # device array allocates and copies the slice, which for 90 layers × 123
    # chunks was ~2 GB of pointless copies and 11k needless allocations.
    W_msg_layers       :: Vector{ArrayT{Float32, 2}}

    # Readout weight matrix (shared across layers in this Nachmani impl),
    # masked by adj_C2V_readout upfront.  Shape: (n_bits × nb_neurons).
    W_readout_masked   :: ArrayT{Float32, 2}

    # Per-layer LLR weights, again one array per layer.  Each: (n_bits,).
    weights_llr_layers :: Vector{ArrayT{Float32, 1}}

    # Pre-routed syndromes — neuron_to_checks indexing applied on CPU, then
    # uploaded.  Shape: (nb_neurons × n_samples), Float32 in {0, 1}.
    #
    # This is the ONLY field that depends on the chunk. Everything above is a
    # function of the model alone, so a state can be built once and reused for
    # every chunk of the same size via `update_gpu_state_syndromes!` — which is
    # what keeps the 16 MB weight tensor from being rebuilt and re-uploaded 123
    # times in a 10^6-sample run.
    syndromes_routed   :: ArrayT{Float32, 2}

    nb_neurons :: Int
    n_bits     :: Int
    n_layers   :: Int
    n_samples  :: Int

    # Enriched check node (soft_constraints.jl): the rule selector, the scalar
    # α, and the tables on the device. `soft_checks` is `nothing` for the
    # standard rule, so the tanh path allocates and uploads nothing extra.
    check_node_kind :: Int
    coupling_scale  :: Float32
    soft_checks     :: Union{GPUSoftCheckState, Nothing}
end

# Convert a sparse matrix to a dense Float32 GPU array: Sparse → dense → Float32 → GPU .
_to_dense_gpu(M::AbstractMatrix) = ArrayT{Float32}(Float32.(Matrix(M)))

# Same for an already-dense Float32 array of any rank (the enriched check
# node's tables are 3-D). No sparse round trip, no element-type change.
function _to_device(array::Array{Float32})::ArrayT{Float32}
    device_array::ArrayT{Float32} = ArrayT{Float32}(array)
    return device_array
end

"""
    _scatter_layer_weights(rows, cols, weights_flat, n_layers, dim_rows, dim_cols) -> Array{Float32, 3}

Arrange flat weight vector into 3D tensor of shape (nb_neurons × nb_neurons × n_layers).

Arguments:
- `rows`: Row indices for non-zero weights in adjacency matrix
- `cols`: Column indices for non-zero weights in adjacency matrix
- `weights_flat`: Flat vector with weights for all layers (length = length(rows) × n_layers)
- `n_layers`: Number of layers (iterations) in NeuralBP model
- `dim_rows`: Number of rows in output tensor (nb_neurons)
- `dim_cols`: Number of columns in output tensor (nb_neurons)
"""
function _scatter_layer_weights(rows::Vector{Int}, cols::Vector{Int},
                                weights_flat::AbstractVector{<:Real},
                                n_layers::Int, dim_rows::Int, dim_cols::Int)
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

"""
    build_gpu_state(bpnn, syndromes_batch) -> GPUState
    build_gpu_state(base, weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale, syndromes_batch) -> GPUState

Build GPUState by converting and uploading CPU data to GPU. Includes:
- Adjacency matrices (densified and converted to Float32)
- Per-layer message weight tensors, pre-masked by adjacency matrices
- Readout weight matrix, pre-masked by readout adjacency
- Per-layer LLR weights, reshaped and converted to Float32
- Syndromes, routed according to neuron_to_checks and converted to Float32
- For the enriched check node: the coupling tables and the enriched checks'
  syndromes (see soft_constraints.jl), plus the scalar α

Overload 1: Extract weights and base from NachmaniNeuralBP struct
Overload 2: Accept raw weights and base structure directly
"""
function build_gpu_state(bpnn, syndromes_batch::BitMatrix)
    base = bpnn.base
    gpustate = build_gpu_state(base, bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, bpnn.coupling_scale, syndromes_batch)
    return gpustate
end

function build_gpu_state(base, weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale, syndromes_batch::BitMatrix)

    # Adjacency
    adj_V2C_C2V = _to_dense_gpu(base.adj_V2C_C2V)
    adj_C2V_V2C_gpu = _to_dense_gpu(base.adj_C2V_V2C)
    adj_C2V_readout_gpu = _to_dense_gpu(base.adj_C2V_readout)
    adj_initialize_V2C = _to_dense_gpu(base.adj_initialize_V2C)

    # Per-layer message weight tensor, pre-masked by adj_C2V_V2C
    W_msg_cpu = _scatter_layer_weights(
        Vector{Int}(base.non_zero_rows_C2V_V2C),
        Vector{Int}(base.non_zero_cols_C2V_V2C),
        weights_c2v_v2c,
        base.n_layers,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer,
    )
    # Multiply the weights by the adjacency mask so that the GPU doesn't have to do it every layer.
    @inbounds for l in 1:base.n_layers
        @views W_msg_cpu[:, :, l] .*= base.adj_C2V_V2C
    end
    # Upload one 2-D array per layer, so the layer loop indexes a Julia Vector
    # instead of slicing (and thereby copying) a 3-D device array every layer.
    W_msg_layers = [ArrayT{Float32}(Matrix{Float32}(W_msg_cpu[:, :, l])) for l in 1:base.n_layers]

    # Readout weight matrix, pre-masked by adj_C2V_readout
    W_readout_cpu = zeros(Float32, base.code_n_bits, base.nb_neurons_per_layer)
    rows_r = Vector{Int}(base.non_zero_rows_C2V_readout)
    cols_r = Vector{Int}(base.non_zero_cols_C2V_readout)
    @inbounds for k in eachindex(rows_r)
        W_readout_cpu[rows_r[k], cols_r[k]] = weights_c2v_readout[k]
    end
    # Multiply the readout weights by the adjacency mask so that the GPU doesn't have to do it every layer.
    W_readout_cpu .*= base.adj_C2V_readout
    W_readout_masked = _to_dense_gpu(W_readout_cpu)

    # Per-layer LLR weights, again one array per layer.
    weights_llrs_cpu = reshape(Vector{Float32}(weights_llrs),
                               base.code_n_bits, base.n_layers)
    weights_llr_layers = [ArrayT{Float32}(Vector{Float32}(weights_llrs_cpu[:, l])) for l in 1:base.n_layers]

    # Route syndromes once on CPU, upload as Float32
    syndromes_routed_gpu = _to_dense_gpu(syndromes_batch[Vector{Int}(base.neuron_to_checks), :])

    # Enriched check node: tables and the enriched checks' syndromes, on device.
    soft_checks::Union{GPUSoftCheckState, Nothing} = nothing
    if base.check_node_kind == CHECK_NODE_ENRICHED
        soft_checks = build_gpu_soft_check_state(
            base.soft_check_tables, syndromes_batch, base.nb_neurons_per_layer, _to_device
        )
    end

    return GPUState(
        adj_V2C_C2V,
        adj_C2V_V2C_gpu,
        adj_C2V_readout_gpu,
        adj_initialize_V2C,
        W_msg_layers,
        W_readout_masked,
        weights_llr_layers,
        syndromes_routed_gpu,
        base.nb_neurons_per_layer,
        base.code_n_bits,
        base.n_layers,
        size(syndromes_batch, 2),
        base.check_node_kind,
        Float32(coupling_scale[1]),
        soft_checks,
    )
end

"""
    update_gpu_state_syndromes!(gs, base, syndromes_batch) -> Nothing

Point an existing `GPUState` at a new chunk of syndromes, in place.

Everything in a `GPUState` except `syndromes_routed` (and, for the enriched
check node, `syndromes_enriched`) is a function of the model, not of the chunk.
Rebuilding the whole state per chunk meant re-deriving and re-uploading the
16 MB per-layer weight tensor 123 times in a 10^6-sample run — ~2 GB of
transfers plus the CPU-side scatter each time — for weights that never change.

Requires the same sample count the state was built with, since the buffers are
sized for it. The caller rebuilds for a differently sized (final) chunk.
"""
function update_gpu_state_syndromes!(gs::GPUState, base, syndromes_batch::BitMatrix)::Nothing
    if size(syndromes_batch, 2) != gs.n_samples
        throw(DimensionMismatch(
            "update_gpu_state_syndromes!: state holds $(gs.n_samples) samples, " *
            "given $(size(syndromes_batch, 2)). Rebuild the state instead."))
    end
    copyto!(gs.syndromes_routed,
            Float32.(Matrix(syndromes_batch[Vector{Int}(base.neuron_to_checks), :])))
    if gs.soft_checks !== nothing
        copyto!(gs.soft_checks.syndromes_enriched,
                reshape(Float32.(Matrix(syndromes_batch[base.soft_check_tables.enriched_check_ids, :])),
                        1, gs.soft_checks.n_enriched_checks, gs.n_samples))
    end
    return nothing
end

"""
    release_gpu_state!(gs) -> Nothing

Free every device array a `GPUState` holds, including the enriched check node's
buffers. Call it when the chunk that built the state is done; the state must not
be used afterwards.

This is not tidiness, it is what keeps a long run alive. `build_gpu_state` runs
once per prediction chunk, and Julia's GC does not feel the device memory behind
these wrappers (see `release_device_array!`). The weight tensor alone is ~17 MB
per chunk; with the enriched check node the state is ~187 MB per chunk, which
reached ~12 GB by chunk 65 of a 10^6-sample run, paged the process out, and
filled the volume with swap.
"""
function release_gpu_state!(gs::GPUState)::Nothing
    release_device_array!(gs.adj_V2C_C2V)
    release_device_array!(gs.adj_C2V_V2C)
    release_device_array!(gs.adj_C2V_readout)
    release_device_array!(gs.adj_initialize_V2C)
    for layer_weights in gs.W_msg_layers
        release_device_array!(layer_weights)
    end
    release_device_array!(gs.W_readout_masked)
    for layer_llr_weights in gs.weights_llr_layers
        release_device_array!(layer_llr_weights)
    end
    release_device_array!(gs.syndromes_routed)
    if gs.soft_checks !== nothing
        release_gpu_soft_check_state!(gs.soft_checks)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Single-layer forward pass on GPU.
# Returns (new messages_c2v, posterior_llrs_at_this_layer).
# ----------------------------------------------------------------------------

function _compute_layer_gpu(messages_c2v, initial_llrs_gpu, gs::GPUState, layer::Int)
    # This layer's weights: a plain Julia Vector lookup, no device allocation
    # and no copy (they were uploaded per layer in `build_gpu_state`).
    W_msg = gs.W_msg_layers[layer]                           # (nb_neurons × nb_neurons)
    weights_llr_col = gs.weights_llr_layers[layer]           # (n_bits,)

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

    # ---- Enriched check node: overwrite the rows of coupled checks ----
    # Mirrors the CPU hook in compute_layer_with_weights!: the standard rule has
    # filled every row; the dense kernel replaces the enriched ones from the RAW
    # v2c messages `m_v2c` of this layer.
    #
    # The result is a buffer the kernel owns and rewrites on the next call (see
    # its docstring). Safe here: the next layer reads it at `W_msg * messages_c2v`
    # above, and this layer reads it in the readout below, both BEFORE the next
    # call happens. Do not hold a reference to it across layers.
    if gs.check_node_kind == CHECK_NODE_ENRICHED
        messages_c2v_new = apply_enriched_checks_gpu(messages_c2v_new, m_v2c, gs.coupling_scale, gs.soft_checks)
    end

    # ---- Readout ----
    posterior_llrs = (weights_llr_col .* initial_llrs_gpu) .+ gs.W_readout_masked * messages_c2v_new

    return messages_c2v_new, posterior_llrs
end

# ----------------------------------------------------------------------------
# Top-level entry point.
# ----------------------------------------------------------------------------

"""
    Perform forward pass of NeuralBP on GPU, returning posterior LLRs at each layer.

    For large datasets, samples are processed in chunks to avoid exceeding Metal's
    per-buffer size limit (`Metal.device().maxBufferLength`). Each chunk is transferred
    to CPU output tensor before the next chunk runs.

    forward_pass_gpu(bpnn, initial_llrs_batch, syndromes_batch; chunk_size=0) -> Array{Float32, 3}
    Arguments (overload 1):
    - `bpnn`: Trained NachmaniNeuralBP model with base code structure and weights
    - `initial_llrs_batch`: Matrix of shape (n_bits × n_samples) with initial LLRs
    - `syndromes_batch`: BitMatrix of shape (n_checks × n_samples) with syndromes
    - `chunk_size`: If >0, process samples in slices of this size. If 0 (default),
    auto-estimate from device's maxBufferLength with headroom for intermediates

    forward_pass_gpu(weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale, base, llrs_batch, syndromes_batch) -> Array{Float32, 3}
    Arguments (overload 2):
    - Raw weight vectors (including the length-1 `coupling_scale`) and base structure, bypassing NachmaniNeuralBP struct
    - Useful for testing or scenarios requiring direct weight manipulation
"""
function forward_pass_gpu(
    bpnn,
    initial_llrs_batch::AbstractMatrix{Float32},
    syndromes_batch::BitMatrix;
    chunk_size::Int = 0,
)
    n_samples = size(initial_llrs_batch::AbstractMatrix{Float32}, 2)
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
            elseif GPU_AVAILABLE && CorrelatedBPDecoderWithCER.CUDA_LOADED
                # CUDA.available_memory() returns free VRAM in bytes on the current device.
                # Use 1/4 of free memory so per-layer intermediates and weights still fit.
                Int(CorrelatedBPDecoderWithCER.CUDA.available_memory()) ÷ 4
            else
                Int(2^30)   # 1 GB fallback for CPU
            end
        catch
            Int(2^30)   # 1 GB fallback if the device query is unavailable
        end
        # The enriched check node keeps (2^d × n_enriched × chunk) tensors alive
        # per layer, which for the BB code is ~4x the output tensor's per-sample
        # footprint; zero for the standard rule.
        per_sample_bytes = n_bits * n_layers * 4 + enriched_kernel_bytes_per_sample(bpnn.base.soft_check_tables)
        chunk_size = max(1, budget_bytes ÷ per_sample_bytes)
        chunk_size = min(chunk_size, n_samples)
        # `maxBufferLength` is Metal's limit on a SINGLE buffer, not a statement
        # about free memory, and on unified memory every allocation competes
        # with the host — so a quarter of it is a far larger budget than the
        # enriched kernel should claim. CUDA's `available_memory()` IS a real
        # budget, so the cap would only slow that path down. No-op for the
        # standard check node either way.
        if CorrelatedBPDecoderWithCER.METAL_LOADED
            chunk_size = cap_batch_size_for_enriched_kernel(chunk_size, bpnn.base.soft_check_tables)
        end
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

function forward_pass_gpu(weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale, base, llrs_batch, syndromes_batch)
    gpustate = build_gpu_state(base, weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale, syndromes_batch)
    posterior_3d_gpu = _forward_pass_gpu_chunk(gpustate, llrs_batch)
    release_gpu_state!(gpustate)
    return posterior_3d_gpu
end

"""
    _forward_pass_gpu_chunk(bpnn, initial_llrs_batch, syndromes_batch) -> Array{Float32, 3}
    _forward_pass_gpu_chunk(gs::GPUState, initial_llrs_batch) -> Array{Float32, 3}

Single-chunk forward pass on GPU. Used either directly when everything fits in
one Metal buffer, or once per chunk by `forward_pass_gpu` for large sample counts.

Overload 1: Build GPUState from NachmaniNeuralBP struct, then compute
Overload 2: Accept pre-built GPUState directly (more efficient for repeated calls)
"""
function _forward_pass_gpu_chunk(
    bpnn::NeuralBP,
    initial_llrs_batch::AbstractMatrix{Float32},
    syndromes_batch::BitMatrix
)
    gs = build_gpu_state(bpnn, syndromes_batch)
    posterior_llrs = _forward_pass_gpu_chunk(gs, initial_llrs_batch)
    # This overload OWNS the state it just built, and `posterior_llrs` is already
    # a host array, so nothing returned refers to the device. Releasing here is
    # what stops per-chunk state from accumulating over a long run.
    release_gpu_state!(gs)
    return posterior_llrs
end

function _forward_pass_gpu_chunk(
    gs::GPUState,
    initial_llrs_batch::AbstractMatrix{Float32}
)
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

    posterior_llrs = Array(posterior_3d_gpu)   # transfer to CPU

    # The per-chunk working arrays are dead once the result is on the host. The
    # output tensor is the largest of them (n_bits x batch x n_layers): 202 MB at
    # 8192 samples and 90 layers, which is another ~25 GB over a 10^6-sample run
    # if left to the GC. `messages_c2v` is the enriched kernel's own buffer when
    # that node is active, so it is freed by `release_gpu_state!` instead.
    release_device_array!(posterior_3d_gpu)
    release_device_array!(initial_llrs_gpu)
    if gs.check_node_kind != CHECK_NODE_ENRICHED
        release_device_array!(messages_c2v)
    end
    return posterior_llrs
end

"""
    predict_recoveries_gpu(gs, initial_llrs_batch) -> Array{Bool, 3}

The decode of one chunk, returned as the hard-decision RECOVERIES at every
layer — `(n_bits × n_samples × n_layers)` of `Bool` — rather than the Float32
posterior LLRs.

This is what both prediction paths actually want: they immediately compute
`posterior .< 0` and never look at the magnitudes. Doing that threshold on the
device instead of the host is exact (same comparison, same values) and cuts the
per-sample cost of the layer-resolved output by 4.5x:

    Float32 on device 25920 B + Float32 on host 25920 B + Bool 6480 B
      ->  Bool on device 6480 B + Bool on host 6480 B

at 90 layers. That is 72% of the per-chunk memory in the old arrangement, and it
is what lets the batch size grow instead of the chunk count.

`forward_pass_gpu` is left alone for callers that genuinely want the LLRs (the
tests compare them against the CPU path).
"""
function predict_recoveries_gpu(
    gs::GPUState,
    initial_llrs_batch::AbstractMatrix{Float32}
)::Array{Bool, 3}
    n_samples = size(initial_llrs_batch, 2)

    initial_llrs_gpu = ArrayT{Float32}(Matrix{Float32}(initial_llrs_batch))
    messages_c2v = ArrayT{Float32}(zeros(Float32, gs.nb_neurons, n_samples))
    recoveries_gpu = ArrayT{Bool}(falses(gs.n_bits, n_samples, gs.n_layers))

    for layer in 1:gs.n_layers
        messages_c2v, post = _compute_layer_gpu(messages_c2v, initial_llrs_gpu, gs, layer)
        # Identical to the host-side `Array(posterior_llrs .< 0)` it replaces.
        @views recoveries_gpu[:, :, layer] .= post .< 0f0
    end

    recoveries::Array{Bool, 3} = Array(recoveries_gpu)

    release_device_array!(recoveries_gpu)
    release_device_array!(initial_llrs_gpu)
    if gs.check_node_kind != CHECK_NODE_ENRICHED
        release_device_array!(messages_c2v)
    end
    return recoveries
end
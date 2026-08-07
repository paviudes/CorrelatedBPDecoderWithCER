"""
    fmt_probs(prob1::Float64, prob2::Float64) -> String

Format the error-model parameters `per_qubit_error_prob` and
`neighbour_error_prob` into a string of the form `p_<prob1>_q_<prob2>`.
Both fields are zero-padded to the *maximum* decimal count of the pair — so
`fmt_probs(0.01, 0.001) == "p_0.010_q_0.001"`, not `"p_0.01_q_0.001"`.

This is the canonical filename tag for anything that identifies a
`(per_qubit_prob, neighbour_prob)` pair on disk: training/test datasets,
correlation-strengths files, trained-weight JSONs, and the
`error_model_parameters_description` field written into result CSVs.

Kept in the main package (not `PlotsForBPDecoder`) so that
`expts/submission/batch_commands.jl` — which runs on the cluster and must
not depend on `Plots.jl` — can format filenames without dragging in the
whole plotting stack.
"""
function fmt_probs(prob1::Float64, prob2::Float64)::String
    ndig = max(length(split(string(prob1), ".")[end]),
               length(split(string(prob2), ".")[end]))
    fmt = Printf.Format("%.$(ndig)f")
    return "p_$(Printf.format(fmt, prob1))_q_$(Printf.format(fmt, prob2))"
end

function fmt_prob(prob::Float64)::String
    """
    Single-parameter counterpart to `fmt_probs`, for the circuit-level error
    model that has only a per-element depolarizing probability `p` (no
    neighbour `q`). There is no second value to pad against, so `p` is rendered
    with its own decimal count: `fmt_prob(0.01) == "p_0.01"`,
    `fmt_prob(0.001) == "p_0.001"`. This is exactly what `fmt_probs` would
    produce for the `p_...` portion of a lone value. Mirrors the copy in
    `expts/submission/batch_commands.jl`.
    """
    ndig = length(split(string(prob), ".")[end])
    fmt = Printf.Format("%.$(ndig)f")
    return "p_$(Printf.format(fmt, prob))"
end

function random_values_around_one(size::Vector{Int}; scale::Float32=0.01f0)::Array{Float32}
    """
    Generate a matrix of random values around 1.0 with specified scale.
    Each element is sampled uniformly from the range [1 - scale, 1 + scale].
    """
    return ones(Float32, size...) .+ scale .* (2 .* rand(Float32, size...) .- 1)
end

function safe_log_tanh_split!(
    out_magnitudes::AbstractMatrix{Float32},
    out_signs::AbstractMatrix{Bool},
    x::AbstractMatrix{Float32}
)
    """
    Compute log(tanh(x/2)) safely, avoiding numerical issues, in-place.

    Instead of returning a complex number:
        log(tanh(x/2)) = log(tanh(|x|/2)) + i π * 1_{x < 0},

    we represent this as:
    - magnitude: log(tanh(|x|/2))
    - sign:      1_{x < 0}

    For large |x|, tanh(x/2) ≈ 1 ⇒ log(tanh(x/2)) ≈ 0.
    For small |x|, tanh(x/2) ≈ 0 ⇒ log(tanh(x/2)) → -∞.

    We clip tanh(|x|/2) to avoid numerical issues near 0 and 1.
    """

    @inbounds @simd for i in eachindex(x)
        xi = x[i]

        # magnitude part: tanh(|x|/2)
        t = tanh(abs(xi) * 0.5f0)

        # clip to avoid numerical issues
        t = ifelse(t < eps(Float32), eps(Float32),
            ifelse(t > 1.0f0 - eps(Float32), 1.0f0 - eps(Float32), t)
        )

        # log magnitude
        out_magnitudes[i] = log(t)

        # sign bit
        out_signs[i] = xi < 0f0
    end

    return nothing
end

function safe_log_tanh_split(x::AbstractMatrix{Float32})
    """
    Functional version (Zygote-friendly).

    Returns:
    - magnitudes
    - signs (Bool)
    """

    t = tanh.(abs.(x) .* 0.5f0)

    t_clipped = clamp.(t, eps(Float32), 1.0f0 - eps(Float32))

    magnitudes = log.(t_clipped)
    signs = x .< 0f0

    return magnitudes, signs
end

# Debugging functions
function debug_log_tanh_split(x::AbstractMatrix{Float32})
    """
    Debug version: uses direct tanh-domain representation.

    Instead of:
        log(tanh(x/2)) = magnitude + iπ sign

    we store:
        tanh(x/2) = sign * magnitude
    """

    t = tanh.(x .* 0.5f0)

    signs = t .< 0f0
    magnitudes = log.(abs.(t))

    return magnitudes, signs
end

function safe_atanh_exp_signed!(
    out::AbstractMatrix{Float32},
    magnitudes::AbstractMatrix{Float32},
    signs::AbstractMatrix{Bool}
)
    """
    Compute 2 * atanh(exp(x)) safely, avoiding numerical issues, in-place.

    Previously:
        x = a + i π b  ⇒ exp(x) = exp(a) * (-1)^b

    Now:
    - magnitudes = a
    - signs = b mod 2

    So:
        exp(x) = exp(magnitudes) * (sign ? -1 : +1)

    We:
    - compute exp(magnitudes), clip it
    - apply sign flip
    - compute 2 * atanh(...)
    """

    @inbounds @simd for i in eachindex(magnitudes)
        a = magnitudes[i]

        # exp + clip
        e = exp(a)
        e = ifelse(e < eps(Float32), eps(Float32),
            ifelse(e > 1.0f0 - eps(Float32), 1.0f0 - eps(Float32), e)
        )

        # apply sign
        e = signs[i] ? -e : e

        # final output
        out[i] = 2f0 * atanh(e)
    end

    return nothing
end

function safe_atanh_exp_signed(
    magnitudes::AbstractMatrix{Float32},
    signs::AbstractMatrix{Bool}
)
    """
    Functional version (Zygote-friendly).
    """

    e = exp.(magnitudes)
    e_clipped = clamp.(e, eps(Float32), 1.0f0 - eps(Float32))

    e_signed = ifelse.(signs, -e_clipped, e_clipped)

    return 2f0 .* atanh.(e_signed)
end

# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = 1.0f0 / (1.0f0 + exp(x))

function binary_entropy(p::T) where T <: Number
    """
    Define the binary entropy function: H(p) = -p log(p) - (1-p) log(1-p).
    """
    return -p * log(p) - (1 - p) * log(1 - p)
end

# Numerically stable version of the softplus function: log(1 + exp(x)).
# For x ≫ 0, naive log1p(exp(x)) overflows. We use the symmetry
#     softplus(x) = max(x, 0) + log1p(exp(-|x|))
# which is well-defined for any finite x in Float32.
@inline softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function binary_entropy_of_sigmoid(μ::Float32)::Float32
    """
    Numerically stable version of the binary entropy of σ(μ), using the
    convention σ(μ) = 1/(1 + exp(μ)) (positive μ ⇒ low error probability).

    Closed form (no log of saturated σ):
        h(σ(μ)) = softplus(μ) - (1 - σ(μ)) * μ.

    Derivation. With σ = 1/(1 + e^μ) we have:
        log σ       = -log(1 + e^μ) = -softplus(μ),
        1 - σ       = e^μ / (1 + e^μ),
        log(1 - σ)  = μ - log(1 + e^μ) = μ - softplus(μ).

    Substituting into h = -σ log σ - (1 - σ) log(1 - σ):

        -σ log σ            =  softplus(μ) / (1 + e^μ),
        -(1 - σ) log(1 - σ) = -(e^μ / (1 + e^μ)) * (μ - softplus(μ))
                            = -e^μ μ / (1 + e^μ)
                              + e^μ softplus(μ) / (1 + e^μ).

    Adding the two and factoring softplus(μ) out of the first and third
    pieces (whose coefficients sum to (1 + e^μ)/(1 + e^μ) = 1):

        h(σ(μ)) = softplus(μ) - e^μ μ / (1 + e^μ)
                = softplus(μ) - (1 - σ(μ)) * μ.

    Limits: at μ = 0, h = log 2 (max); as |μ| → ∞, h → 0. No log(0)
    anywhere, so no NaN even when σ(μ) saturates to 0 or 1 in Float32.
    """
    return softplus(μ) - (1f0 - sigmoid(μ)) * μ
end

function xor_affine!(
    out::AbstractMatrix{Bool},
    A,
    X::AbstractMatrix{Bool},
    Y::AbstractMatrix{Bool}
)
    """
    Compute: out = (A * X + Y) mod 2
    where
    - A is a sparse integer matrix
    - X, Y are boolean matrices
    """
    # Compute the integer product A * X (mod 2 will be applied later)
    integer_product = A * X
    # Apply XOR with Y and take mod 2 (which is just checking if the integer product is odd)
    @inbounds @simd for i in eachindex(integer_product)
        out[i] = isodd(integer_product[i]) ⊻ Y[i]
    end

    return nothing
end

function sparse_multiply!(
    out::Matrix{Float32},
    rows::Vector{Int},
    cols::Vector{Int},
    weights::AbstractVector{Float32},
    X::Matrix{Float32}
)
    """
    Compute Y = (A ⊙ W) * X in an efficient edge-wise manner.

    Here:
      - A is an implicit sparse binary matrix defined by (rows, cols),
        i.e., A[r, c] = 1 for each edge (r, c), and 0 otherwise.
      - W is a vector of learnable weights corresponding to each nonzero entry of A.
      - ⊙ denotes the Hadamard (element-wise) product.
      - X is a dense input matrix.
      - Y is the output matrix (`out`), overwritten in-place.

    This function avoids explicitly forming the sparse matrix A or (A ⊙ W),
    and instead performs the multiplication using edge-wise accumulation:
        Y[r, j] += W[e] * X[c, j]

    Arguments:
      - out      : Output matrix Y (preallocated, will be overwritten)
      - rows     : Row indices of nonzero entries of A
      - cols     : Column indices of nonzero entries of A
      - weights  : Values W[e] corresponding to each (rows[e], cols[e])
      - X        : Input matrix X

    Notes:
      - length(rows) == length(cols) == length(weights)
      - This implementation is compatible with Enzyme.jl (no sparse structures).
    """
    fill!(out, 0f0)

    n_edges = length(rows)
    n_cols_input = size(X, 2)

    @inbounds for e in 1:n_edges
        r = rows[e]
        c = cols[e]
        w = weights[e]

        @simd for j in 1:n_cols_input
            out[r, j] += w * X[c, j]
        end
    end

    return nothing
end

# ============================================================================
#                        GPU memory -> prediction batch size
# ============================================================================

"Accepted memory spellings: an optionally fractional number, an optional binary
unit, and an optional trailing `B`. No sign, so a parsed value is never negative."
const MEMORY_SPECIFICATION_PATTERN = r"^([0-9]*\.?[0-9]+)\s*([KMGT]?)B?$"

"Megabytes reserved for the GPU driver/context before any of our own allocations."
const DRIVER_CONTEXT_OVERHEAD_MB = 512

"Multiplier from each binary unit to megabytes. The empty unit is a bare number,
which SLURM reports in MB."
const MEGABYTES_PER_MEMORY_UNIT = Dict{String, Float64}(
    ""  => 1.0,
    "K" => 1.0 / 1024,
    "M" => 1.0,
    "G" => 1024.0,
    "T" => 1024.0 * 1024,
)

"""
    parse_memory(memory_description_string) -> Int   (megabytes)

Parse a SLURM-style memory string into MEGABYTES.

    parse_memory("16G")     == 16384
    parse_memory("1024M")   == 1024
    parse_memory("16GB")    == 16384
    parse_memory("2T")      == 2097152
    parse_memory("512")     == 512      # bare number: MB, as SLURM reports it

The bare-number case follows SLURM's own convention: `--mem-per-gpu=16G` is
exported to the job as `SLURM_MEM_PER_GPU=16384`, i.e. already in MB. Binary
units throughout (1G = 1024M), again matching SLURM.

Throws `ArgumentError` on anything unparseable rather than guessing — silently
defaulting a bad memory string would produce a batch size that OOMs deep into a
10^6-sample run.
"""
function parse_memory(memory_description_string::AbstractString)::Int
    normalised_memory::String = uppercase(strip(String(memory_description_string)))
    if isempty(normalised_memory)
        throw(ArgumentError("parse_memory: empty memory string."))
    end

    # `match` returns `nothing` when the string does not parse, so the honest
    # annotation is a Union. Declaring this `::String` would fail on every
    # SUCCESSFUL match (a RegexMatch does not convert to String) and would also
    # make the `.captures` accesses below invalid.
    memory_match::Union{RegexMatch, Nothing} = match(MEMORY_SPECIFICATION_PATTERN, normalised_memory)
    if memory_match === nothing
        throw(
            ArgumentError(
                "parse_memory: cannot parse \"$(memory_description_string)\". " *
                "Expected forms: \"16G\", \"16GB\", \"1024M\", \"2T\", or a bare number of MB."
            )
        )
    end

    # Both groups always participate when the pattern matches: group 1 requires at
    # least one digit, and group 2 can match the empty string.
    numeric_value::Float64 = parse(Float64, memory_match.captures[1])
    unit_symbol::String = String(memory_match.captures[2])

    megabytes::Int = max(1, round(Int, numeric_value * MEGABYTES_PER_MEMORY_UNIT[unit_symbol]))
    return megabytes
end

"""
    compute_optimal_batch_size_for(memory_in_mb; n_bits, n_layers, nb_neurons, ...) -> Int

Conservative prediction batch size that fits in `memory_in_mb` megabytes of GPU
memory.

THE MEMORY MODEL. Per sample, `_forward_pass_gpu_chunk` holds, in Float32:

  - the output tensor `posterior_3d_gpu`, `n_bits x batch x n_layers`  <- dominant
  - `initial_llrs_gpu` (`n_bits`), `messages_c2v` and `syndromes_routed`
    (`nb_neurons` each)
  - the per-layer temporaries inside `_compute_layer_gpu` and its two helpers
    (m_v2c, scaled_llrs, the tanh split's four arrays, m_c2v_mag, parity_fp,
    combined, the atanh helper's four, ...), each `nb_neurons x batch`.
    `layer_temporaries` counts them; the default is deliberately generous
    because GPU allocators pool rather than free eagerly.

so

    bytes/sample = 4 * (n_bits * n_layers + n_bits + layer_temporaries * nb_neurons)

Batch-INDEPENDENT allocations (the per-layer weight tensor
`nb_neurons^2 x n_layers`, the four adjacency matrices, and the driver context)
are subtracted off first as `fixed_overhead_mb`.

The result is then scaled by `headroom_fraction` and rounded DOWN to a power of
two, which alone gives up to a further 2x margin. With the defaults the estimate
is deliberately pessimistic; raise `headroom_fraction` if you have measured
headroom on your device.

Pass `fixed_overhead_mb` explicitly to override the derived overhead; a negative
value (the default) means derive it.

NOTE ON THE TWO BATCHING LEVELS. This sizes the OUTER loop in
`predict_and_check_neuralbp`, which also governs host-side memory (each chunk's
posterior tensor comes back to the CPU). `forward_pass_gpu` independently
sub-chunks on the device from `CUDA.available_memory()`, so this value is an
upper bound on device residency, not the only guard. Larger outer batches also
mean fewer `build_gpu_state` rebuilds (which re-upload the weight tensor every
call), so bigger is faster until host memory complains — hence `max_batch_size`.
"""
function compute_optimal_batch_size_for(
    memory_in_mb::Int;
    n_bits::Int = 72,
    n_layers::Int = 100,
    nb_neurons::Int = 216,
    layer_temporaries::Int = 24,
    headroom_fraction::Float64 = 0.25,
    fixed_overhead_mb::Int = -1,
    min_batch_size::Int = 256,
    max_batch_size::Int = 262144,
)::Int
    if memory_in_mb <= 0
        throw(ArgumentError("compute_optimal_batch_size_for: memory must be positive, got $(memory_in_mb) MB."))
    end

    # Batch-independent device allocations, in MB: the per-layer weight tensor
    # dominates, plus four adjacency matrices, plus ~512 MB of driver/context.
    # A negative `fixed_overhead_mb` means "derive it from the geometry".
    resolved_fixed_overhead_mb::Int = fixed_overhead_mb
    if fixed_overhead_mb < 0
        weight_tensor_mb::Int = (nb_neurons^2 * n_layers * 4) ÷ (1024 * 1024)
        adjacency_matrices_mb::Int = (4 * nb_neurons^2 * 4) ÷ (1024 * 1024)
        resolved_fixed_overhead_mb = DRIVER_CONTEXT_OVERHEAD_MB + weight_tensor_mb + adjacency_matrices_mb
    end

    usable_mb::Float64 = (memory_in_mb - resolved_fixed_overhead_mb) * headroom_fraction
    if usable_mb <= 0
        @warn "compute_optimal_batch_size_for: $(memory_in_mb) MB does not cover the " *
              "$(resolved_fixed_overhead_mb) MB of batch-independent overhead; falling back to " *
              "min_batch_size = $(min_batch_size)."
        return min_batch_size
    end

    bytes_per_sample::Int = 4 * (n_bits * n_layers + n_bits + layer_temporaries * nb_neurons)
    unrounded_batch_size::Int = floor(Int, usable_mb * 1024 * 1024 / bytes_per_sample)
    if unrounded_batch_size < min_batch_size
        return min_batch_size
    end

    # Round DOWN to a power of two: tidy in logs, and worth up to 2x of extra margin.
    rounded_batch_size::Int = 1 << floor(Int, log2(unrounded_batch_size))
    batch_size::Int = clamp(rounded_batch_size, min_batch_size, max_batch_size)
    return batch_size
end
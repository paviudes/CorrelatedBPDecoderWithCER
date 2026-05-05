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
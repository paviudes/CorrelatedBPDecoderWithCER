function random_values_around_one(size::Vector{Int}; scale::Float32=0.01f0)::Array{Float32}
    """
    Generate a matrix of random values around 1.0 with specified scale.
    Each element is sampled uniformly from the range [1 - scale, 1 + scale].
    """
    return ones(Float32, size...) .+ scale .* (2 .* rand(Float32, size...) .- 1)
end

#=
# Define safe versions of atanh(exp(x)) and log(tanh(x/2)) to avoid numerical issues.
function safe_atanh_exp(x::Matrix{ComplexF32})::Matrix{Float32}
    """
    Compute 2 * atanh(exp(x)) safely, avoiding numerical issues.
    While x is complex, we will implicitly assume that x is of the form x = a + i π b
    where a is a real number and b is an integer.
    Hence, exp(x) = exp(a) * exp(i π b) = exp(a) * (-1)^b.
    """
    real_part = real.(x)
    imaginary_part = Int.(imag.(x) ./ π)
    clipped_exp = clamp.(exp.(real_part), eps(Float32), 1.0f0 - eps(Float32)) .* (-1) .^ imaginary_part
    atanh_exp = 2 .* atanh.(clipped_exp)
    return atanh_exp
end

function safe_log_tanh(x::Matrix{Float32})::Matrix{ComplexF32}
    """
    Compute log(tanh(x/2)) safely, avoiding numerical issues.
    If x is negative, we will add i π to the log(tanh(|x|/2)).
    For large x, tanh(x/2) will be close to 1, and log(tanh(x/2)) will be close to 0.
    For small x, tanh(x/2) will be close to 0, and log(tanh(x/2)) will be close to -Inf.
    We will clip the values of tanh(x/2) to avoid numerical issues.
    """
    clipped_tanh = clamp.(tanh.(abs.(x) ./ 2), eps(Float32), 1.0f0 - eps(Float32))
    return log.(clipped_tanh) .+ im * Float32(π) * (x .< 0)
end
=#

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

# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = 1.0f0 / (1.0f0 + exp(x))
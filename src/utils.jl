function random_values_around_one(size::Vector{Int}; scale::Float32=0.01f0)::Array{Float32}
    """
    Generate a matrix of random values around 1.0 with specified scale.
    Each element is sampled uniformly from the range [1 - scale, 1 + scale].
    """
    return ones(Float32, size...) .+ scale .* (2 .* rand(Float32, size...) .- 1)
end

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

# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = 1.0f0 / (1.0f0 + exp(x))
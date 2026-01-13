function random_values_around_one(size::Vector{Int}; scale::Float32=0.01f0)::Array{Float32}
    """
    Generate a matrix of random values around 1.0 with specified scale.
    Each element is sampled uniformly from the range [1 - scale, 1 + scale].
    """
    return ones(Float32, size...) .+ scale .* (2 .* rand(Float32, size...) .- 1)
end

# Define safe versions of atanh(exp(x)) and log(tanh(x/2)) to avoid numerical issues.
function safe_atanh_exp(x::Matrix{Float32})::Matrix{Float32}
    """
    Compute atanh(exp(x)) safely, avoiding numerical issues.
    For large negative x, exp(x) will be very small, and atanh(exp(x)) will be close to 0.
    For large positive x, exp(x) will be very large, and atanh(exp(x)) will be close to Inf.
    We will clip the values of exp(x) to avoid numerical issues.
    """
    clipped_exp = clamp.(exp.(x), eps(Float32), 1.0f0 - eps(Float32))
    return atanh.(clipped_exp)
end

function safe_log_tanh(x::Matrix{Float32})::Matrix{Float32}
    """
    Compute log(tanh(x/2)) safely, avoiding numerical issues.
    We will assume that x is always positive.
    For large x, tanh(x/2) will be close to 1, and log(tanh(x/2)) will be close to 0.
    For small x, tanh(x/2) will be close to 0, and log(tanh(x/2)) will be close to -Inf.
    We will clip the values of tanh(x/2) to avoid numerical issues.
    """
    clipped_tanh = clamp.(tanh.(x ./ 2), eps(Float32), 1.0f0 - eps(Float32))
    return log.(clipped_tanh)
end

# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = 1.0f0 / (1.0f0 + exp(x))
import Base: +, *

struct ComplexPi32
    """
    Define a complex number type with Float32 real and imaginary parts, where the imaginary part is an integer multiple of π.
    This is useful for representing the contribution of the syndrome in the check-to-variable messages in the belief propagation algorithm, where the syndrome contributes a term of iπ for unsatisfied checks.
    Hence the complex structure represents a number of the form: re + im * π, where re is a Float32 value and im is an integer.
    """
    re::Float32
    im::Int
    function ComplexPi32(re::Float32, im::Int)::ComplexPi32
        return new(re, im)
    end
    
    # Additional constructor that accepts Bool and converts to Int
    function ComplexPi32(re::Float32, im::Bool)::ComplexPi32
        return new(re, convert(Int, im))
    end
end

#=
Define basic arithmetic operations for ComplexPi32, including addition, multiplication, and scalar multiplication.
=#
+(a::ComplexPi32, b::ComplexPi32) = ComplexPi32(a.re + b.re, a.im + b.im)
*(a::ComplexPi32, b::Int) = ComplexPi32(a.re * b, a.im * b)
*(a::Int, b::ComplexPi32) = ComplexPi32(a * b.re, a * b.im)

function safe_sinh(x::Float32; x_max::Float32=20.0f0, x_min::Float32=1.0f-6)::Float32
    """
    Compute sinh in a numerically stable way.
    We will clamp x to the range [-a_max, a_max] to prevent overflow in sinh(a).
    If x is very small, then sinh(x) ≈ x.
    """
    if abs(x) < x_min
        safe_real_sinh = x
    else
        a_clamped = clamp(x, -x_max, x_max)
        safe_real_sinh = sinh(a_clamped)
    end
    return safe_real_sinh
end

function safe_sinh(x::ComplexPi32; x_max::Float32=20.0f0, x_min::Float32=1.0f-6)::Float32
    """
    Compute sinh(x) for x = a + i π b where a is a Float32 value and b is an integer.
    We can use the relation:
        sinh(a + i π b) = sinh(a) * cos(π b) + i cosh(a) * sin(π b)
                        = sinh(a) * (-1)^b + i cosh(a) * 0 (Note: Since b is an integer, cos(π b) = (-1)^b and sin(π b) = 0.)
                        = sinh(a) * (-1)^b
    
    
    """
    safe_real_sinh = safe_sinh(x.re; x_max=x_max, x_min=x_min)
    result = safe_real_sinh * (-1.0f0) ^ (x.im)
    return result
end

safe_cosech(x::Union{ComplexPi32, Float32}; x_max::Float32=20.0f0, x_min::Float32=1.0f-6)::Float32 = 1.0f0 / safe_sinh(x; x_max=x_max, x_min=x_min)
safe_negative_cosech(x::Union{ComplexPi32, Float32}; x_max::Float32=20.0f0, x_min::Float32=1.0f-6)::Float32 = -safe_cosech(x; x_max=x_max, x_min=x_min)

function safe_atanh_exp(x::ComplexPi32)::Float32
    """
    Compute 2 * atanh(exp(x)) safely for x = a + i π b where a is a Float32 value and b is an integer.
    We can use the relation:
         2 * atanh(exp(a + i π b)) = 2 * atanh(exp(a) * exp(i π b))
                                   = 2 * atanh(exp(a) * (-1)^b)
                                   = (-1)^b * 2 * atanh(exp(a))
    Note that atanh(y) is not defined for |y| >= 1, so we need to ensure that exp(a) is less than 1 to avoid numerical issues. We can safely compute this by clamping exp(a) to a maximum value slightly less than 1.
    """
    ϵ = eps(Float32)
    exp_a = clamp(exp(x.re), 0.0f0, 1.0f0 - ϵ)
    result = (-1.0f0) ^ (x.im) * 2.0f0 * atanh(exp_a)
    return result
end

function safe_log_tanh(x::Float32)::ComplexPi32
    """
    Compute log(tanh(x/2)) safely, avoiding numerical issues.
    If x is positive:
        - we will clamp tanh(x/2) between [ϵ, 1 - ϵ] to avoid log(0) issues.
    If x is negative:
        - we will write log(tanh(x/2)) = log(-tanh(|x|/2)) = log(tanh(|x|/2)) + i π
    """
    ϵ = eps(Float32)
    if x >= 0.0f0
        tanh_value = tanh(x / 2.0f0)
        clipped_log_tanh = log(clamp(tanh_value, ϵ, 1.0f0 - ϵ))
        return ComplexPi32(clipped_log_tanh, 0)
    else
        # For negative x, compute log(tanh(|x|/2)) + i π
        abs_x = abs(x)
        tanh_value = tanh(abs_x / 2.0f0)
        clipped_tanh = clamp(tanh_value, ϵ, 1.0f0 - ϵ)
        clipped_log_tanh = log(clipped_tanh)
        return ComplexPi32(clipped_log_tanh, 1) # This represents log(tanh(|x|/2)) + i π
    end
end
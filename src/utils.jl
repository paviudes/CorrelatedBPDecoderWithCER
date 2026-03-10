# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = T(1) / (T(1) + exp(x))
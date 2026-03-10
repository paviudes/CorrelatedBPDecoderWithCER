function grad_message_c2v_wrt_weight(
    bpnn::NachmaniNeuralBP, 
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    time::Int, # t* in W^(t*)_(v*,c*;c*',v*)
    derivative_wrt_checks::Tuple{Int,Int}, # c* and c*' in W^(t*)_(v*,c*;c*',v*)
    derivative_wrt_vertex::Int # v* in W^(t*)_(v*,c*;c*',v*)
)
    """
    Compute the gradient of the (activated) messages from check nodes to variable nodes (C2V neurons) with respect to the learnable weights W_(c2v_v2c).

    We want to compute the derivative:
    ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*)
    
    We have the following derivatives based on the message passing rules for Nachmani et al. architecture:
    1. ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ W^(t)_(v*,c*;c*',v*)
    2. ∂ m^l_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(c' ∈ N(v) - c) m^t_(c'* -> v*) * δ(v = v*) * δ(c = c*) * δ(l-1, t) + W^(l-1)_(v,c ; c',v) * ∂ m^(l-1)_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*)

    We can compute ∂ a(m^l_(v' -> c)) / ∂ W^(t)_(v*,c*;c*',v*) using the chain rule.

    We will compute the derivatives for all layers l from t+1 to n_layers, since the messages at layer l only depend on the weights up to layer l-1.
    1. Set ∂ m^(t+1)_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) = m^t_(c'* -> v*) δ(v = v*) * δ(c = c*) for all (v,c) pairs.
    2. For l from t+2 to n_layers, do
        a. Compute ∂ a(m^l_(v -> c)) / ∂ W^(t)_(v*,c*;c*',v*) = a'(m^l_(v -> c)) ∂ m^l_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) : for all (v,c) pairs.
        b. Compute ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ W^(t)_(v*,c*;c*',v*) : for all (c,v) pairs.
        c. Compute ∂ m^l_(c -> v) / ∂ W^(t)_(v*,c*;c*',v*) = (a^-1)'(a(m^l_(c -> v))) * ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*) : for all (c,v) pairs.
        d. Compute ∂ m^(l+1)_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(c' ∈ N(v) - c) W^(l)_(v,c ; c',v) * ∂ m^l_(c' -> v) / ∂ W^(t)_(v*,c*;c*',v*) : for all (v,c) pairs.
    """
    n_layers = bpnn.base.n_layers
    # Initialize the gradients dictionaries: to zeros for all (c,v) pairs and (v,c) pairs for all layers.
    grad_c2v = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:time
        grad_c2v[l] = Dict{Tuple{Int,Int}, Float32}((check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_activated_c2v = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:time
        grad_activated_c2v[l] = Dict{Tuple{Int,Int}, Float32}((check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_v2c = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:(time+1)
        grad_v2c[l] = Dict{Tuple{Int,Int}, Float32}((vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_activated_v2c = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:time
        grad_activated_v2c[l] = Dict{Tuple{Int,Int}, Float32}((vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    
    for layer in (time + 1):n_layers
        
        # Base Case: ∂ m^(t+1)_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) = m^t_(c'* -> v*) δ(v = v*) * δ(c = c*)
        if (layer == time + 1)
            grad_v2c[time + 1][(derivative_wrt_vertex, derivative_wrt_checks[1])] = intermediate_c2v_messages[time][(derivative_wrt_checks[2], derivative_wrt_vertex)]
        end

        # a. Compute ∂ a(m^l_(v -> c)) / ∂ W^(t)_(v*,c*;c*',v*) = a'(m^l_(v -> c)) ∂ m^l_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) : for all (v,c) pairs.
        grad_activated_v2c[layer] = Dict{Tuple{Int,Int}, Float32}(
            (vertex, check) => bpnn.base.derivative_activation_function(intermediate_v2c_messages[layer][(vertex, check)]) * 
                               grad_v2c[layer][(vertex, check)] 
            for (check, vertex) in bpnn.base.edges
        )

        # b. Compute ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ W^(t)_(v*,c*;c*',v*) : for all (c,v) pairs.
        grad_activated_c2v[layer] = Dict{Tuple{Int,Int}, Float32}(
            (check, vertex) => sum(
                grad_activated_v2c[layer][(vertex_prime, check)] for vertex_prime in bpnn.base.neighbors_of_check[check] if vertex_prime != vertex
            ) for (check, vertex) in bpnn.base.edges
        )

        # c. Compute ∂ m^l_(c -> v) / ∂ W^(t)_(v*,c*;c*',v*) = (a^-1)'(a(m^l_(c -> v))) * ∂ a(m^l_(c -> v)) / ∂ W^(t)_(v*,c*;c*',v*) : for all (c,v) pairs.
        grad_c2v[layer] = Dict{Tuple{Int,Int}, Float32}(
            (check, vertex) => begin
                activated_message = bpnn.base.activation_function(intermediate_c2v_messages[layer][(check, vertex)])
                derivative_term = bpnn.base.derivative_inverse_activation_function(activated_message)
                gradient_term = grad_activated_c2v[layer][(check, vertex)]
                derivative_term * gradient_term
            end
            for (check, vertex) in bpnn.base.edges
        )

        # d. Compute ∂ m^(l+1)_(v -> c) / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(c' ∈ N(v) - c) W^(l)_(v,c ; c',v) * ∂ m^l_(c' -> v) / ∂ W^(t)_(v*,c*;c*',v*) : for all (v,c) pairs.
        if layer < n_layers # We only need to compute this for l < n_layers, since the messages at layer n_layers do not affect any messages at higher layers.
            grad_v2c[layer + 1] = Dict{Tuple{Int,Int}, Float32}(
                (vertex, check) => sum(
                    bpnn.weights_c2v_v2c[(layer, vertex, check, check_prime, vertex)] * grad_c2v[layer][(check_prime, vertex)] 
                    for check_prime in bpnn.base.neighbors_of_vertex[vertex] if check_prime != check
                , init=0.0f0) for (check, vertex) in bpnn.base.edges
            )
        end
    end
    return grad_c2v
end

function grad_llrs_wrt_weight(
    bpnn::NachmaniNeuralBP, 
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    time::Int, # t* in W^(t*)_(v*,c*;c*',v*)
    derivative_wrt_checks::Tuple{Int,Int}, # c* and c*' in W^(t*)_(v*,c*;c*',v*)
    derivative_wrt_vertex::Int # v* in W^(t*)_(v*,c*;c*',v*)
)
    """
    Compute the gradient of the LLRs at all layers with respect to the learnable weights W_(c2v_v2c).
    We want to compute the derivative ∂ μ^l_v / ∂ W^(t)_(v*,c*;c*',v*), for all layers l >= t. The derivative is given by:
    
    ∂ μ^l_v / ∂ W^(t)_(v*,c*;c*',v*) = ∑_(c ∈ N(v)) W^(l)_(v; c,v) * ∂ m^l_(c -> v) / ∂ W^(t)_(v*,c*;c*',v*)
    
    Note that we can compute ∂ m^l_(c -> v) / ∂ W^(t)_(v*,c*;c*',v*) using the `grad_message_c2v_wrt_weight` function defined above.
    """
    grad_c2v = grad_message_c2v_wrt_weight(
        bpnn, 
        intermediate_c2v_messages, 
        intermediate_v2c_messages, 
        time, 
        derivative_wrt_checks, 
        derivative_wrt_vertex
    )

    n_layers = bpnn.base.n_layers

    # For every layer l < time, the derivative ∂ μ^l_v / ∂ W^(t)_(v*,c*;c*',v*) is zero since the messages at layer l do not depend on the weights at layer t.
    grad_llrs = Dict{Tuple{Int,Int}, Float32}(
        (layer, vertex) => (layer < time) ? 0.0f0 : sum(
            bpnn.weights_c2v_readout[(layer, vertex, check, vertex)] * grad_c2v[layer][(check, vertex)] 
            for check in bpnn.base.neighbors_of_vertex[vertex]
        , init=0.0f0) for layer in time:n_layers for vertex in 1:bpnn.base.code_n_bits
    )
    return grad_llrs
end

function grad_message_c2v_wrt_bias(
    bpnn::NachmaniNeuralBP, 
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    time::Int, # t*
    derivative_wrt_vertex::Int # v*
)
    """
    Compute the gradient of the message m^(l)_(c -> v) with respect to the bias term b^(t)_(v*), for all layers l >= t.

    We have the following relations between the derivatives.
    1. ∂ a(m^l_(c -> v)) / ∂ b^(t)_(v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ b^(t)_(v*)
    2. ∂ m^l_(v -> c) / ∂ b^(t)_(v*) = l_(v*) δ(v = v*) * δ(l = t) + ∑_(c' ∈ N(v) - c) W^(l-1)_(v,c; c',v) * ∂ m^(l-1)_(c' -> v) / ∂ b^(t)_(v*)

    We can compute ∂ a(m^l_(v' -> c)) / ∂ b^(t)_(v*) using the chain rule.

    The iteration for computing the derivatives is as follows.
    1. Set ∂ m^t_(v -> c) / ∂ b^(t)_(v*) = l_(v*) δ(v = v*) for all (v,c) pairs.
    2. For l from t+1 to n_layers, do
        a. Compute ∂ a(m^l_(v -> c)) / ∂ b^(t)_(v*) = a'(m^l_(v -> c)) ∂ m^l_(v -> c) / ∂ b^(t)_(v*) : for all (v,c) pairs.
        b. Compute ∂ a(m^l_(c -> v)) / ∂ b^(t)_(v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ b^(t)_(v*) : for all (c,v) pairs.
        c. Compute ∂ m^l_(c -> v) / ∂ b^(t)_(v*) = (a^-1)'(a(m^l_(c -> v))) * ∂ a(m^l_(c -> v)) / ∂ b^(t)_(v*) : for all (c,v) pairs.
        d. Compute ∂ m^(l+1)_(v -> c) / ∂ b^(t)_(v*) = ∑_(c' ∈ N(v) - c) W^(l)_(v,c ; c',v) * ∂ m^l_(c' -> v) / ∂ b^(t)_(v*) : for all (v,c) pairs.
    """

    n_layers = bpnn.base.n_layers

    # Initialize the gradients dictionaries: to zeros for all (c,v) pairs and (v,c) pairs for all layers.
    grad_c2v = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:(time-1)
        grad_c2v[l] = Dict{Tuple{Int,Int}, Float32}((check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_activated_c2v = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:(time-1)
        grad_activated_c2v[l] = Dict{Tuple{Int,Int}, Float32}((check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_v2c = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:time
        grad_v2c[l] = Dict{Tuple{Int,Int}, Float32}((vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end
    grad_activated_v2c = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    for l in 1:(time-1)
        grad_activated_v2c[l] = Dict{Tuple{Int,Int}, Float32}((vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges)
    end

    for layer in time:n_layers
        
        # Base Case: ∂ m^t_(v -> c) / ∂ b^(t)_(v*) = l_(v*) δ(v = v*)
        if (layer == time)
            grad_v2c[time] = Dict{Tuple{Int,Int}, Float32}(
                (vertex, check) => (vertex == derivative_wrt_vertex) ? bpnn.initial_llrs[vertex] : 0.0f0 for (check, vertex) in bpnn.base.edges
            )
        end

        # a. Compute ∂ a(m^l_(v -> c)) / ∂ b^(t)_(v*) = a'(m^l_(v -> c)) ∂ m^l_(v -> c) / ∂ b^(t)_(v*) : for all (v,c) pairs.
        grad_activated_v2c[layer] = Dict{Tuple{Int,Int}, Float32}(
            (vertex, check) => bpnn.base.derivative_activation_function(intermediate_v2c_messages[layer][(vertex, check)]) * 
                               grad_v2c[layer][(vertex, check)] 
            for (check, vertex) in bpnn.base.edges
        )

        # b. Compute ∂ a(m^l_(c -> v)) / ∂ b^(t)_(v*) = ∑_(v' ∈ N(c) - v) ∂ a(m^l_(v' -> c)) / ∂ b^(t)_(v*) : for all (c,v) pairs.
        grad_activated_c2v[layer] = Dict{Tuple{Int,Int}, Float32}(
            (check, vertex) => sum(
                grad_activated_v2c[layer][(vertex_prime, check)] for vertex_prime in bpnn.base.neighbors_of_check[check] if vertex_prime != vertex
            , init=0.0f0) for (check, vertex) in bpnn.base.edges
        )

        # c. Compute ∂ m^l_(c -> v) / ∂ b^(t)_(v*) = (a^-1)'(a(m^l_(c -> v))) * ∂ a(m^l_(c -> v)) / ∂ b^(t)_(v*) : for all (c,v) pairs.
        grad_c2v[layer] = Dict{Tuple{Int,Int}, Float32}(
            (check, vertex) => begin
                activated_message = bpnn.base.activation_function(intermediate_c2v_messages[layer][(check, vertex)])
                derivative_term = bpnn.base.derivative_inverse_activation_function(activated_message)
                gradient_term = grad_activated_c2v[layer][(check, vertex)]
                derivative_term * gradient_term
            end
            for (check, vertex) in bpnn.base.edges
        )

        # d. Compute ∂ m^(l+1)_(v -> c) / ∂ b^(t)_(v*) = ∑_(c' ∈ N(v) - c) W^(l)_(v,c ; c',v) * ∂ m^l_(c' -> v) / ∂ b^(t)_(v*) : for all (v,c) pairs.
        if layer < n_layers # We only need to compute this for l < n_layers, since the messages at layer n_layers do not affect any messages at higher layers.
            grad_v2c[layer + 1] = Dict{Tuple{Int,Int}, Float32}(
                (vertex, check) => sum(
                    bpnn.weights_c2v_v2c[(layer, vertex, check, check_prime, vertex)] * grad_c2v[layer][(check_prime, vertex)] 
                    for check_prime in bpnn.base.neighbors_of_vertex[vertex] if check_prime != check
                , init=0.0f0) for (check, vertex) in bpnn.base.edges
            )
        end
    end    
    
    return grad_c2v
end

function grad_llrs_wrt_bias(
    bpnn::NachmaniNeuralBP, 
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    time::Int, # t*
    derivative_wrt_vertex::Int # v*,
)
    """
    Compute the gradient of the final LLRs with respect to the learnable weights for the channel LLRs: `weights_llrs`.
    We want to compute the derivative: ∂ μ^l_v / ∂ b^(t)_(v*) for all layers l >= t.

    ∂ μ^l_v / ∂ b^(t)_(v*) = l_v δ(v = v*) * δ(t = t*) + sum_(c ∈ N(v)) W^(l)_(v;c,v) * ∂ m^l_(c->v) / ∂ b^(t)_(v*).

    We can compute ∂ m^l_(c->v) / ∂ b^(t)_(v*) using the `grad_message_c2v_wrt_bias` function defined above.
    """
    grad_c2v = grad_message_c2v_wrt_bias(
        bpnn, 
        intermediate_c2v_messages, 
        intermediate_v2c_messages, 
        time, 
        derivative_wrt_vertex
    )
    n_layers = bpnn.base.n_layers
    grad_llrs = Dict{Tuple{Int,Int}, Float32}(
        (layer, vertex) => (layer < time) ? 0.0f0 : (bpnn.initial_llrs[vertex] * (vertex == derivative_wrt_vertex) + sum(
            bpnn.weights_c2v_readout[(layer, vertex, check, vertex)] * grad_c2v[layer][(check, vertex)] for check in bpnn.base.neighbors_of_vertex[vertex]
        )) for layer in time:n_layers for vertex in 1:bpnn.base.code_n_bits
    )
    return grad_llrs
end
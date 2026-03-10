function (bpnn::NachmaniNeuralBP)(
    initial_llrs::Vector{Float32}, 
    syndrome::BitVector
)::Tuple{Matrix{Float32}, Vector{Dict{Tuple{Int,Int}, Float32}}, Vector{Dict{Tuple{Int,Int}, Float32}}}
    """
    Forward pass through the Neural Network for Nachmani et al. architecture: https://arxiv.org/abs/1607.04793 and https://arxiv.org/abs/1811.07835.
    
    We have the following steps for t from 1 to T:
    
    1. Variable-to-Check messages:
       m^t_(v->c) = l_v * b_v + sum_(c' ∈ N(v) - c) W^(t-1)_(v,c ; c',v) * m^t_(c'->v)
       
    2. Check-to-Variable messages:
       a(m^t_(c->v)) = i * π * s_c + sum_(v' ∈ N(c) - v) a(m^t_(v' -> c))
       which can be rewritten as:
       m^t_(c->v) = 2 * atanh(exp(sum_(v' ∈ N(c) - v) a(m^t_(v' -> c)))) * (-1)^(s_c)
       
    3. Final LLRs:
       μ^t_v = l_v * b_v + sum_(c ∈ N(v)) W^(t)_(v;c,v) * m^t_(c->v)
    
    Where:
    - m^t_(v->c): messages from variable nodes to check nodes at iteration t (V2C neurons)
    - m^t_(c->v): messages from check nodes to variable nodes at iteration t (C2V neurons)  
    - μ^t_v: final LLRs for variable node v at iteration t
    - a(x) = log(tanh(x/2)): activation function for V2C neurons
    - l_v: channel LLRs
    - b_v: learnable weights for the channel LLRs
    - W_(v,c ; c',v): learnable weights for connections from C2V to V2C neurons
    - s_c: syndrome bit for check node c

    The initial messages m^0_(c->v) are set to zero.
    
    Returns:
    - LLRs computed at each layer as a matrix of shape (n_bits, n_layers)
    - C2V messages for each layer
    - V2C messages for each layer
    """
    
    # Input validation
    if eltype(initial_llrs) != Float32
        throw(ArgumentError("Initial LLRs must be of type Float32. Found: $(eltype(initial_llrs))"))
    end

    # Initialize data structures
    n_layers = bpnn.base.n_layers
    
    # Initialize message storage for each layer
    messages_c2v = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    messages_v2c = Vector{Dict{Tuple{Int,Int}, Float32}}(undef, n_layers)
    
    # Initialize first layer C2V messages to zero
    messages_c2v[1] = Dict{Tuple{Int,Int}, Float32}(
        (check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges
    )
    
    # Initialize first layer V2C messages to zero  
    messages_v2c[1] = Dict{Tuple{Int,Int}, Float32}(
        (vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges
    )
    
    # Temporary storage for activated messages
    activated_messages_c2v = Dict{Tuple{Int,Int}, ComplexPi32}(
        (check, vertex) => ComplexPi32(0.0f0, 0) for (check, vertex) in bpnn.base.edges
    )
    activated_messages_v2c = Dict{Tuple{Int,Int}, ComplexPi32}(
        (vertex, check) => ComplexPi32(0.0f0, 0) for (check, vertex) in bpnn.base.edges
    )
    
    # Storage for final LLRs at each layer
    final_llrs_by_layers = zeros(Float32, n_layers, bpnn.base.code_n_bits)
    
    # Main belief propagation loop
    for t = 1:n_layers
        
        #=============================================================================
                                    STEP 1: V2C MESSAGES
        =============================================================================#
        
        # Initialize V2C message dictionary for current layer
        messages_v2c[t] = Dict{Tuple{Int,Int}, Float32}(
            (vertex, check) => 0.0f0 for (check, vertex) in bpnn.base.edges
        )

        # Compute variable-to-check messages
        for (check, vertex) in bpnn.base.edges
            sum_term = 0.0f0
            
            if t == 1
                # At the first layer, C2V messages are zero
                sum_term = 0.0f0
            else
                # Sum weighted messages from neighboring checks
                for neighbor_check in bpnn.base.neighbors_of_vertex[vertex]
                    if neighbor_check != check
                        weight = bpnn.weights_c2v_v2c[(t-1, vertex, check, neighbor_check, vertex)]
                        sum_term += weight * messages_c2v[t-1][(neighbor_check, vertex)]
                    end
                end
            end
            
            # Compute V2C message: weighted LLR + sum of weighted neighbor messages
            messages_v2c[t][(vertex, check)] = (
                bpnn.weights_llrs[(t, vertex)] * initial_llrs[vertex] + sum_term
            )
            
            # Apply activation function
            activated_messages_v2c[(vertex, check)] = (
                bpnn.base.activation_function(messages_v2c[t][(vertex, check)])
            )
        end

        #=============================================================================
                                    STEP 2: C2V MESSAGES  
        =============================================================================#
        
        # Initialize C2V message dictionary for current layer
        messages_c2v[t] = Dict{Tuple{Int,Int}, Float32}(
            (check, vertex) => 0.0f0 for (check, vertex) in bpnn.base.edges
        )

        # Compute check-to-variable messages
        for (check, vertex) in bpnn.base.edges
            sum_term = ComplexPi32(0.0f0, 0)
            
            # Sum activated V2C messages from neighboring variables
            for neighbor_vertex in bpnn.base.neighbors_of_check[check]
                if neighbor_vertex != vertex
                    sum_term += activated_messages_v2c[(neighbor_vertex, check)]
                end
            end
            
            # Add syndrome contribution and apply inverse activation
            activated_messages_c2v[(check, vertex)] = sum_term + ComplexPi32(0.0f0, syndrome[check])
            messages_c2v[t][(check, vertex)] = real(
                bpnn.base.inverse_activation_function(activated_messages_c2v[(check, vertex)])
            )
        end

        #=============================================================================
                                    STEP 3: FINAL LLRS
        =============================================================================#
        
        # Compute intermediate LLRs for each variable node
        for vertex in 1:bpnn.base.code_n_bits
            sum_term = 0.0f0
            
            # Sum weighted C2V messages from neighboring checks
            for check in bpnn.base.neighbors_of_vertex[vertex]
                weight = bpnn.weights_c2v_readout[(t, vertex, check, vertex)]
                sum_term += weight * messages_c2v[t][(check, vertex)]
            end
            
            # Final LLR: weighted initial LLR + sum of weighted C2V messages
            final_llrs_by_layers[t, vertex] = (
                bpnn.weights_llrs[(t, vertex)] * initial_llrs[vertex] + sum_term
            )
        end
    end

    # Return results
    return (final_llrs_by_layers, messages_c2v, messages_v2c)
end
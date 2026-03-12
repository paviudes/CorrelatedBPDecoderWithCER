function derivative_iid_loss_at_layer_t_wrt_weight(
    bpnn::NachmaniNeuralBP,
    layer::Int,
    llrs_at_layer_t::Vector{Float32},
    expected_recovery::BitVector,
    derivatives_llr_wrt_weight::Dict{Tuple{Int, Int}, Float32}
)
    """
    Compute the derivative of the Loss function with respect to a specific weight W^(t*)_(v*,c*; c*',v*) in the NachmaniNeuralBP model.
    Recall that the Loss function is defined by
        L(μ^t) = ∑_(c=1)^(N_c) f( (H^⟂ * (σ(μ^t) + e)) )_c )
    where
        - μ^t is finall LLR at time t
        - σ(μ^t) is the predicted error vector at time t, and σ(x) = 1 / (1 + exp(-x)) is the sigmoid function,
        - e is the expected recovery vector,
        - H^⟂ is the parity-check matrix of the dual code, and
        - f(x) = |sin(π x / 2)| is the function applied to the commutation relations.
    
    Note that `(H^⟂ * (σ(μ^t) + e)) )_c` denotes the c-th syndrome bit of the residual error after adding the predicted error and the expected recovery.
    
    The derivative of the Loss function with respect to the weight W^(t*)_(v*,c*; c*',v*) is given by:
        ∂L / ∂W^(t*)_(v*,c*; c*',v*) = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^t) + e)) )_c ) * ∑_(v'=1)^Nv H^⟂_(c,v') * σ'(μ^t_v') * ∂μ^t_v' / ∂W^(t*)_(v*,c*; c*',v*)
    where
        - f'(x) = (π / 2) * cos(π x / 2) * sign(sin(π x / 2)) is the derivative of f(x),
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - ∂μ^t_v' / ∂W^(t*)_(v*,c*; c*',v*) is given by `grad_llrs_wrt_weight(bpnn, intermediate_c2v_messages, intermediate_v2c_messages, t*, (c*, c*'), v*)`.
    """
    # Compute the syndrome of the residual error with respect to the dual code.
    residual_error_syndrome = bpnn.base.parity_check_matrix_dual * (sigmoid.(llrs_at_layer_t) + expected_recovery)
    loss_derivative = 0.0f0
    for c in 1:bpnn.base.code_n_checks
        # Compute f'( (H^⟂ * (σ(μ^t) + e)) )_c )
        derivative_f = (π / 2) * cos(π * residual_error_syndrome[c] / 2) * sign(sin(π * residual_error_syndrome[c] / 2))
        sum_over_v = 0.0f0
        for v in 1:bpnn.base.code_n_bits
            # Compute H^⟂_(c,v) * σ'(μ^t_v) * ∂μ^t_v / ∂W^(t*)_(v*,c*; c*',v*)
            derivative_sigmoid = - sigmoid(llrs_at_layer_t[v]) * (1 - sigmoid(llrs_at_layer_t[v]))
            derivative_llr = derivatives_llr_wrt_weight[(layer, v)]
            sum_over_v += bpnn.base.parity_check_matrix_dual[c, v] * derivative_sigmoid * derivative_llr
         end
        loss_derivative += derivative_f * sum_over_v
    end
    return loss_derivative    
end

function derivative_correlated_loss_at_layer_t_wrt_weight(
    bpnn::NachmaniNeuralBP,
    layer::Int,
    llrs_at_layer_t::Vector{Float32},
    derivatives_llr_wrt_weight::Dict{Tuple{Int, Int}, Float32}
)
    """
    Compute the derivative of the correlated part of the Loss function with respect to a specific weight W^(t*)_(v*,c*; c*',v*) in the NachmaniNeuralBP model.
    
    Recall that the correlations are specified by a correlation graph G = (V, E), where V is the set of vertices corresponding to the bits of the code, and E is the set of edges corresponding to the correlations between the bits.
    Each edge e = (u, v) ∈ E is associated with a correlation weight w_e, which quantifies the strength of the correlation between the two bits connected by the edge.
    
    The correlated part of the Loss function is defined by
        L_corr(μ^t) = ∑_(e=(u, v) ∈ E) w_e * σ(μ^t_u) * (1 - σ(μ^t_v))
    where
    - μ^t is the final LLR at time t,
    - σ(μ^t_u) is the predicted error probability for bit u at time t, and
    - (1 - σ(μ^t_v)) is the predicted probability that there is no error on bit v at time t.

    Hence, the derivative of the correlated part of the Loss function with respect to the weight W^(t*)_(v*,c*; c*',v*) is given by:
    ∂L_corr / ∂W^(t*)_(v*,c*; c*',v*) = ∑_(e=(u, v) ∈ E) w_e * ( ∂σ(μ^t_u) / ∂W^(t*)_(v*,c*; c*',v*) * (1 - σ(μ^t_v)) + σ(μ^t_u) * ∂(1 - σ(μ^t_v)) / ∂W^(t*)_(v*,c*; c*',v*) )
    where
        - ∂σ(μ^t_u) / ∂W^(t*)_(v*,c*; c*',v*) = σ'(μ^t_u) * ∂μ^t_u / ∂W^(t*)_(v*,c*; c*',v*)
        - ∂(1 - σ(μ^t_v)) / ∂W^(t*)_(v*,c*; c*',v*) = - σ'(μ^t_v) * ∂μ^t_v / ∂W^(t*)_(v*,c*; c*',v*)
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - ∂μ^t_u / ∂W^(t*)_(v*,c*; c*',v*) and ∂μ^t_v / ∂W^(t*)_(v*,c*; c*',v*) are given by `grad_llrs_wrt_weight(bpnn, intermediate_c2v_messages, intermediate_v2c_messages, t*, (c*, c*'), v*)`.
    """

    loss_derivative = 0.0f0
    for (i, edge) in enumerate(eachrow(bpnn.base.connectivity_edges))
        u, v = edge
        w_e = bpnn.base.correlation_strengths[i]
        derivative_sigma_u = - sigmoid(llrs_at_layer_t[u]) * (1 - sigmoid(llrs_at_layer_t[u])) * derivatives_llr_wrt_weight[(layer, u)]
        derivative_one_minus_sigma_v = sigmoid(llrs_at_layer_t[v]) * (1 - sigmoid(llrs_at_layer_t[v])) * derivatives_llr_wrt_weight[(layer, v)]
        loss_derivative += w_e * (derivative_sigma_u * (1 - sigmoid(llrs_at_layer_t[v])) + sigmoid(llrs_at_layer_t[u]) * derivative_one_minus_sigma_v)
    end

    return loss_derivative
end

function derivative_total_loss_wrt_weight(
    bpnn::NachmaniNeuralBP,
    expected_recovery::BitVector,
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_llrs::Matrix{Float32},
    time::Int, # The time step t* in W^(t*)_(v*,c*; c*',v*)
    derivative_check_nodes::Tuple{Int, Int}, # The check nodes (c*, c*') in W^(t*)_(v*,c*; c*',v*)
    derivative_vertex::Int; # The vertex v* in W^(t*)_(v*,c*; c*',v*)
    is_correlated::Bool=false
)
    """
    Compute the derivative of the Loss function with respect to a specific weight W^(t*)_(v*,c*; c*',v*) in the NachmaniNeuralBP model, given the intermediate LLRs at each layer.
    This function is a wrapper around the previous `derivative_loss_wrt_weight` function, which computes the derivative of the Loss per layer.
    We can sum over the derivatives for each layer to get the total derivative of the Loss with respect to the weight W^(t*)_(v*,c*; c*',v*).
    """
    # Precompute the derivatives of the LLRs at all times with respect to the weight W^(t*)_(v*,c*; c*',v*)
    derivatives_llr_wrt_weight = grad_llrs_wrt_weight(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        time,
        derivative_check_nodes,
        derivative_vertex
    )

    # println("∂ μ^t_v / ∂W^($(time))_(v$(derivative_vertex),c$(derivative_check_nodes[1]); c$(derivative_check_nodes[2]),v$(derivative_vertex)) for all t: ", derivatives_llr_wrt_weight)
    
    total_derivative = 0.0f0
    for loss_layer in time:bpnn.base.n_layers # Loss at time `time` only depends on the weights between times 1 to `time`.
        llrs_at_layer_t = intermediate_llrs[loss_layer, :]
        derivative_iid_loss_at_layer_t = derivative_iid_loss_at_layer_t_wrt_weight(
            bpnn,
            loss_layer,
            llrs_at_layer_t,
            expected_recovery,
            derivatives_llr_wrt_weight
        )
        if is_correlated
            derivative_correlated_loss_at_layer_t = derivative_correlated_loss_at_layer_t_wrt_weight(
                bpnn,
                loss_layer,
                llrs_at_layer_t,
                derivatives_llr_wrt_weight
            )
        else
            derivative_correlated_loss_at_layer_t = 0.0f0
        end
        
        total_derivative += derivative_iid_loss_at_layer_t + derivative_correlated_loss_at_layer_t
    end
    return total_derivative
end

function derivative_iid_loss_at_layer_t_wrt_bias(
    bpnn::NachmaniNeuralBP,
    layer::Int,
    expected_recovery::BitVector,
    llrs_at_layer_t::Vector{Float32},
    derivatives_llr_wrt_bias::Dict{Tuple{Int, Int}, Float32}
)
    """
    Compute the derivative of the Loss function with respect to a specific bias weight b^(t*)_(v*) in the NachmaniNeuralBP model.
    The derivative of the Loss function with respect to the bias weight b^(t*)_(v*) is given by:
        ∂L / ∂b^(t*)_(v*) = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^t) + e)) )_c ) * ∑_(v'=1)^Nv H^⟂_(c,v') * σ'(μ^t_v') * ∂μ^t_v' / ∂b^(t*)_(v*)
    where
        - f'(x) = (π / 2) * cos(π x / 2) * sign(sin(π x / 2)) is the derivative of f(x),
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - ∂μ^t_v' / ∂b^(t*)_(v*) is given by `grad_llr_wrt_bias(bpnn, intermediate_c2v_messages, intermediate_v2c_messages, t*, v*)`.
    """
    # Compute the syndrome of the residual error with respect to the dual code.
    residual_error_syndrome = bpnn.base.parity_check_matrix_dual * (sigmoid.(llrs_at_layer_t) + expected_recovery)
    loss_derivative = 0.0f0
    for c in 1:bpnn.base.code_n_checks
        # Compute f'( (H^⟂ * (σ(μ^t) + e)) )_c )
        derivative_f = (π / 2) * cos(π * residual_error_syndrome[c] / 2) * sign(sin(π * residual_error_syndrome[c] / 2))
        sum_over_v = 0.0f0
        for v in 1:bpnn.base.code_n_bits
            # Compute H^⟂_(c,v) * σ'(μ^t_v) * ∂μ^t_v / ∂b^(t*)_(v*)
            derivative_sigmoid = - sigmoid(llrs_at_layer_t[v]) * (1 - sigmoid(llrs_at_layer_t[v]))
            derivative_llr = derivatives_llr_wrt_bias[(layer, v)]
            sum_over_v += bpnn.base.parity_check_matrix_dual[c, v] * derivative_sigmoid * derivative_llr
         end
        loss_derivative += derivative_f * sum_over_v
    end        
    return loss_derivative    
end

function derivative_correlated_loss_at_layer_t_wrt_bias(
    bpnn::NachmaniNeuralBP,
    layer::Int,
    llrs_at_layer_t::Vector{Float32},
    derivatives_llr_wrt_bias::Dict{Tuple{Int, Int}, Float32}
)
    """
    Compute the derivative of the correlated part of the Loss function with respect to a specific bias weight b^(t*)_(v*) in the NachmaniNeuralBP model.
    
    Recall that the correlations are specified by a correlation graph G = (V, E), where V is the set of vertices corresponding to the bits of the code, and E is the set of edges corresponding to the correlations between the bits.
    Each edge e = (u, v) ∈ E is associated with a correlation weight w_e, which quantifies the strength of the correlation between the two bits connected by the edge.
    
    The correlated part of the Loss function is defined by
        L_corr(μ^t) = ∑_(e=(u, v) ∈ E) w_e * σ(μ^t_u) * (1 - σ(μ^t_v))
    where
    - μ^t is the final LLR at time t
    - σ(μ^t_u) is the predicted error probability for bit u at time t, and
    - (1 - σ(μ^t_v)) is the predicted probability that there is no error on bit v at time t.

    Hence the derivative of the correlated part of the Loss function with respect to the bias weight b^(t*)_(v*) is given by:
        ∂L_corr / ∂b^(t*)_(v*) = ∑_(e=(u, v) ∈ E) w_e * ( ∂σ(μ^t_u) / ∂b^(t*)_(v*) * (1 - σ(μ^t_v)) + σ(μ^t_u) * ∂(1 - σ(μ^t_v)) / ∂b^(t*)_(v*) )
    where
        - ∂σ(μ^t_u) / ∂b^(t*)_(v*) = σ'(μ^t_u) * ∂μ^t_u / ∂b^(t*)_(v*)
        - ∂(1 - σ(μ^t_v)) / ∂b^(t*)_(v*) = - σ'(μ^t_v) * ∂μ^t_v / ∂b^(t*)_(v*)
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - ∂μ^t_u / ∂b^(t*)_(v*) and ∂μ^t_v / ∂b^(t*)_(v*) are given by `grad_llr_wrt_bias(bpnn, intermediate_c2v_messages, intermediate_v2c_messages, t*, v*)`.
    """
    loss_derivative = 0.0f0
    for (i, edge) in enumerate(eachrow(bpnn.base.connectivity_edges))
        u, v = edge
        w_e = bpnn.base.correlation_strengths[i]
        derivative_sigma_u = - sigmoid(llrs_at_layer_t[u]) * (1 - sigmoid(llrs_at_layer_t[u])) * derivatives_llr_wrt_bias[(layer, u)]
        derivative_one_minus_sigma_v = sigmoid(llrs_at_layer_t[v]) * (1 - sigmoid(llrs_at_layer_t[v])) * derivatives_llr_wrt_bias[(layer, v)]
        loss_derivative += w_e * (derivative_sigma_u * (1 - sigmoid(llrs_at_layer_t[v])) + sigmoid(llrs_at_layer_t[u]) * derivative_one_minus_sigma_v)
    end
    return loss_derivative
end

function derivative_total_loss_wrt_bias(
    bpnn::NachmaniNeuralBP,
    expected_recovery::BitVector,
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_llrs::Matrix{Float32},
    time::Int,
    derivative_vertex::Int;
    is_correlated::Bool=false
)
    """
    Compute the derivative of the Loss function with respect to a specific bias weight b^(t*)_(v*) in the NachmaniNeuralBP model, given the intermediate LLRs at each layer.
    This function is a wrapper around the previous `derivative_loss_wrt_bias` function, which computes the derivative of the Loss per layer.
    We can sum over the derivatives for each layer to get the total derivative of the Loss with respect to the bias weight b^(t*)_(v*).
    """
    # Precompute the derivatives of the LLRs at layer t with respect to the bias weight b^(t*)_(v*)
    derivatives_llr_wrt_bias = grad_llrs_wrt_bias(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        time,
        derivative_vertex
    )
    
    total_derivative = 0.0f0
    for t in time:bpnn.base.n_layers
        llrs_at_layer_t = intermediate_llrs[t, :]
        derivative_iid_loss_at_layer_t = derivative_iid_loss_at_layer_t_wrt_bias(
            bpnn,
            t,
            expected_recovery,
            llrs_at_layer_t,
            derivatives_llr_wrt_bias
        )
        if is_correlated
            derivative_correlated_loss_at_layer_t = derivative_correlated_loss_at_layer_t_wrt_bias(
                bpnn,
                t,
                llrs_at_layer_t,
                derivatives_llr_wrt_bias
            )
        else
            derivative_correlated_loss_at_layer_t = 0.0f0
        end
        total_derivative += derivative_iid_loss_at_layer_t + derivative_correlated_loss_at_layer_t
    end
    return total_derivative
end

function derivative_iid_loss_wrt_c2v_readout_weight(
    bpnn::NachmaniNeuralBP,
    expected_recovery::BitVector,
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_llrs::Matrix{Float32},
    derivative_wrt_layer::Int, # The time step t* in W^(t*)_(v*; c*, v*)
    derivative_wrt_check::Int, # The check nodes c* in W^(t*)_(v*; c*, v*)
    derivative_wrt_vertex::Int; # The vertex v* in W^(t*)_(v*; c*, v*)
)
    """
    Compute the derivative of the Loss function with respect to a specific weight W^(t*)_(v*; c*, v*) in the NachmaniNeuralBP model, given the intermediate LLRs at each layer.
    
    The derivative is given by
    
    ∂ L / ∂W^(t)_(v*; c*, v*) = ∑_(l=1)^(n_layers) ∂L_l / ∂W^(t)_(v*; c*, v*)
    
    where L_l is the Loss at layer l:
    
    L_l = ∑_(c=1)^(N_c) f( (H^⟂ * (σ(μ^l) + e)) )_c )
    
    and the derivative of L_l with respect to W^(t)_(v*; c*, v*) is given by
    
    ∂ L_l / ∂W^(t)_(v*; c*, v*) = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∂ ( (H^⟂ * (σ(μ^l) + e)) )_c ) / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∑_(v=1)^Nv H^⟂_(c,v) * σ'(μ^l_v) * ∂μ^l_v / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∑_(v=1)^Nv H^⟂_(c, v) * σ'(μ^l_v) ∑_(c' ∈ N(v)) m^l_(c' -> v) ∂W^(l)_(v; c', v) / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * H^⟂_(c, v*) * σ'(μ^l_v*) * m^l_(c* -> v*) * δ(t = l)
    
    where
        - f'(x) = (π / 2) * cos(π x / 2) * sign(sin(π x / 2)) is the derivative of f(x),
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - δ(t = l) is the Kronecker delta function, which is 1 if t = l and 0 otherwise.
    """
    llrs_at_layer = intermediate_llrs[derivative_wrt_layer, :]
    n_checks_dual = size(bpnn.base.parity_check_matrix_dual, 1)

    # Compute H^⟂ * (σ(μ^l) + e)) for the current layer.
    residual_syndrome = bpnn.base.parity_check_matrix_dual * (sigmoid.(llrs_at_layer) + expected_recovery)
    
    loss_derivative = 0.0f0
    for check in 1:n_checks_dual
        # Compute f'( (H^⟂ * (σ(μ^l) + e))_c* )
        derivative_f = (π / 2) * cos(π * residual_syndrome[check] / 2) * sign(sin(π * residual_syndrome[check] / 2))

        # Compute H^⟂_(c*, v*) * σ'(μ^l_v*) * m^l_(c* -> v*)
        h_dual_c_v = bpnn.base.parity_check_matrix_dual[check, derivative_wrt_vertex]
        derivative_sigmoid = - sigmoid(llrs_at_layer[derivative_wrt_vertex]) * (1 - sigmoid(llrs_at_layer[derivative_wrt_vertex]))
        message_c2v = intermediate_c2v_messages[derivative_wrt_layer][(derivative_wrt_check, derivative_wrt_vertex)]

        # Putting it together: f'( (H^⟂ * (σ(μ^l) + e))_c* ) H^⟂_(c*, v*) * σ'(μ^l_v*) * m^l_(c* -> v*)
        loss_derivative += derivative_f * h_dual_c_v * derivative_sigmoid * message_c2v
    end
    return loss_derivative
end

function derivative_correlated_loss_wrt_c2v_readout_weight(
    bpnn::NachmaniNeuralBP,
    intermediate_llrs::Matrix{Float32},
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    derivative_wrt_layer::Int, # The time step t* in W^(t*)_(v*; c*, v*)
    derivative_wrt_check::Int, # The check nodes c* in W^(t*)_(v*; c*, v*)
    derivative_wrt_vertex::Int; # The vertex v* in W^(t*)_(v*; c*, v*)
)
    """
    Compute the derivative of the correlated Loss function with respect to a specific weight W^(t*)_(v*; c*, v*) in the NachmaniNeuralBP model, given the intermediate LLRs at each layer.
    The derivative is given by
    ∂ L_corr / ∂W^(t*)_(v*; c*, v*) = ∑_(e=(u, v) ∈ E) w_e * ( ∂σ(μ^t_u) / ∂W^(t*)_(v*; c*, v*) * (1 - σ(μ^t_v)) + σ(μ^t_u) * ∂(1 - σ(μ^t_v)) / ∂W^(t*)_(v*; c*, v*) )
    where
        - ∂σ(μ^t_u) / ∂W^(t*)_(v*; c*, v*) = σ'(μ^t_u) * ∂μ^t_u / ∂W^(t*)_(v*; c*, v*)
        - ∂(1 - σ(μ^t_v)) / ∂W^(t*)_(v*; c*, v*) = - σ'(μ^t_v) * ∂μ^t_v / ∂W^(t*)_(v*; c*, v*)
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
        - ∂μ^t_u / ∂W^(t*)_(v*; c*, v*) = m^t(c* -> v*) * δ(u = v*) * δ(t = t*) and ∂μ^t_v / ∂W^(t*)_(v*; c*, v*) = m^t(c* -> v*) * δ(v = v*) * δ(t = t*), and
        - m^t(c* -> v*) is the message from check node c* to variable node v* at time t.
    """
    loss_derivative = 0.0f0
    for (i, edge) in enumerate(eachrow(bpnn.base.connectivity_edges))
        u, v = edge
        w_e = bpnn.base.correlation_strengths[i]
        derivative_sigma_u = - sigmoid(intermediate_llrs[derivative_wrt_layer, u]) * (1 - sigmoid(intermediate_llrs[derivative_wrt_layer, u])) * intermediate_c2v_messages[derivative_wrt_layer][(derivative_wrt_check, derivative_wrt_vertex)]
        derivative_one_minus_sigma_v = sigmoid(intermediate_llrs[derivative_wrt_layer, v]) * (1 - sigmoid(intermediate_llrs[derivative_wrt_layer, v])) * intermediate_c2v_messages[derivative_wrt_layer][(derivative_wrt_check, derivative_wrt_vertex)]
        loss_derivative += w_e * (derivative_sigma_u * (1 - sigmoid(intermediate_llrs[derivative_wrt_layer, v])) + sigmoid(intermediate_llrs[derivative_wrt_layer, u]) * derivative_one_minus_sigma_v)
    end
    return loss_derivative
end


function nachmani_loss_jacobian_wrt_c2v_readout_weights(
    bpnn::NachmaniNeuralBP,
    expected_recovery::BitVector,
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_llrs::Matrix{Float32};
    is_correlated::Bool=false
)
    """
    Compute the derivative of the Loss function with respect to all the weights of the form W^(t)_(v*; c*, v*) in the NachmaniNeuralBP model.
    The derivative is given by
    
    ∂ L / ∂W^(t)_(v*; c*, v*) = ∑_(l=1)^(n_layers) ∂L_l / ∂W^(t)_(v*; c*, v*)
    
    where L_l is the Loss at layer l:
    
    L_l = ∑_(c=1)^(N_c) f( (H^⟂ * (σ(μ^l) + e)) )_c )
    
    and the derivative of L_l with respect to W^(t)_(v*; c*, v*) is given by
    
    ∂ L_l / ∂W^(t)_(v*; c*, v*) = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∂ ( (H^⟂ * (σ(μ^l) + e)) )_c ) / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∑_(v=1)^Nv H^⟂_(c,v) * σ'(μ^l_v) * ∂μ^l_v / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * ∑_(v=1)^Nv H^⟂_(c, v) * σ'(μ^l_v) ∑_(c' ∈ N(v)) m^l_(c' -> v) ∂W^(l)_(v; c', v) / ∂W^(t)_(v*; c*, v*)
                                = ∑_(c=1)^(N_c) f'( (H^⟂ * (σ(μ^l) + e)) )_c ) * H^⟂_(c, v*) * σ'(μ^l_v*) * m^l_(c* -> v*) * δ(t = l)
    
    where
        - f'(x) = (π / 2) * cos(π x / 2) * sign(sin(π x / 2)) is the derivative of f(x),
        - σ'(x) = - σ(x) * (1 - σ(x)) is the derivative of the sigmoid function, and
    
    We will compute the derivative with respect to each weight W^(t)_(v*; c*, v*).
    """
    # Initialize the Jacobian dictionary with zeros for all possible keys to ensure that we have an entry for each weight, even if the derivative is zero.
    loss_jacobian_wrt_c2v_readout_weights = Dict{NTuple{4,Int}, Float32}(
        ((layer, vertex, check, vertex), 0.0f0) for layer in 1:bpnn.base.n_layers for (check, vertex) in bpnn.base.edges
    )

    for derivative_wrt_layer in 1:bpnn.base.n_layers
        # llrs_at_layer = intermediate_llrs[derivative_wrt_layer, :]

        for (derivative_wrt_check, derivative_wrt_vertex) in bpnn.base.edges
            iid_loss_derivative = derivative_iid_loss_wrt_c2v_readout_weight(
                bpnn,
                expected_recovery,
                intermediate_c2v_messages,
                intermediate_llrs,
                derivative_wrt_layer,
                derivative_wrt_check,
                derivative_wrt_vertex
            )
            if (is_correlated)
                correlated_loss_derivative = derivative_correlated_loss_wrt_c2v_readout_weight(
                    bpnn,
                    intermediate_llrs,
                    intermediate_c2v_messages,
                    derivative_wrt_layer,
                    derivative_wrt_check,
                    derivative_wrt_vertex
                )
            else
                correlated_loss_derivative = 0.0f0
            end

            loss_jacobian_wrt_c2v_readout_weights[(derivative_wrt_layer, derivative_wrt_vertex, derivative_wrt_check, derivative_wrt_vertex)] = iid_loss_derivative + correlated_loss_derivative
        end
    end
    return loss_jacobian_wrt_c2v_readout_weights
end

function nachmani_loss_jacobian(
    bpnn::NachmaniNeuralBP,
    expected_recovery::BitVector,
    intermediate_c2v_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_v2c_messages::Vector{Dict{Tuple{Int,Int}, Float32}},
    intermediate_llrs::Matrix{Float32};
    is_correlated::Bool=false,
)
    """
    Compute the Jacobian of the Loss function with respect to all learnable parameters in the NachmaniNeuralBP model, which include:
        - weights_c2v_v2c
        - weights_llrs
        - weights_c2v_readout
        - loss_weights
    The Jacobian is a dictionary that maps each learnable parameter to its corresponding derivative of the Loss function.
    """
    # start = time()
    # Compute the Jacobian with respect to weights_c2v_v2c
    jacobian_weights_c2v_v2c = Dict{NTuple{5,Int}, Float32}(
        (layer, vertex, check, check_prime, vertex) => derivative_total_loss_wrt_weight(
            bpnn,
            expected_recovery,
            intermediate_c2v_messages,
            intermediate_v2c_messages,
            intermediate_llrs,
            layer,
            (check, check_prime),
            vertex;
            is_correlated=is_correlated,
        )
        for (check, vertex) in bpnn.base.edges for check_prime in bpnn.base.neighbors_of_vertex[vertex] if check_prime != check for layer in 1:(bpnn.base.n_layers - 1)
    )

    # elapsed_time_c2v_v2c = time() - start
    # println("[", round(elapsed_time_c2v_v2c, digits=3), " seconds] Completed Jacobian computation for weights_c2v_v2c.")

    # Compute the Jacobian with respect to weights_llrs
    jacobian_weights_llrs = Dict{NTuple{2,Int}, Float32}(
        (layer, vertex) => derivative_total_loss_wrt_bias(
            bpnn,
            expected_recovery,
            intermediate_c2v_messages,
            intermediate_v2c_messages,
            intermediate_llrs,
            layer,
            vertex;
            is_correlated=is_correlated,
        )
        for layer in 1:bpnn.base.n_layers for vertex in 1:bpnn.base.code_n_bits
    )

    # println("Computed Jacobian for weights_llrs.\n", jacobian_weights_llrs)

        # elapsed_time_llrs = time() - start - elapsed_time_c2v_v2c
        # println("[", round(elapsed_time_llrs, digits=3), " seconds] Completed Jacobian computation for weights_llrs.")

    # Compute the Jacobian with respect to weights_c2v_readout
    jacobian_weights_c2v_readout = nachmani_loss_jacobian_wrt_c2v_readout_weights(
        bpnn,
        expected_recovery,
        intermediate_c2v_messages,
        intermediate_llrs;
        is_correlated=is_correlated
    )
    # println("Computed Jacobian for weights_c2v_readout.\n", jacobian_weights_c2v_readout)

    # Compute the Jacobian with respect to the loss weights.
    jacobian_loss_weights = Dict{NTuple{1, Int}, Float32}(
        (layer,) => compute_loss_per_layer(
            intermediate_llrs[layer, :],
            expected_recovery,
            bpnn.base.parity_check_matrix_dual;
            is_correlated=is_correlated,
            connectivity_edges=bpnn.base.connectivity_edges,
            correlation_strengths=bpnn.base.correlation_strengths
        )
        for layer in 1:bpnn.base.n_layers
    )

    # elapsed_time_c2v_readout = time() - start - elapsed_time_c2v_v2c - elapsed_time_llrs
    # println("[", round(elapsed_time_c2v_readout, digits=3), " seconds] Completed Jacobian computation for weights_c2v_readout.")

    # println("===============================\nTotal time for Jacobian computation: ", round(time() - start, digits=3), " seconds.\n================================")

    # Combine all Jacobians into a single dictionary #TODO: we also consider other output formats, depending on what is more convenient for the optimization step.
    jacobian = Dict{NachmaniLearnableParameterGroup, Dict{NTuple, Float32}}()
    jacobian[C2V_V2C] = jacobian_weights_c2v_v2c
    jacobian[LLRS] = jacobian_weights_llrs
    jacobian[C2V_READOUT] = jacobian_weights_c2v_readout
    jacobian[LOSS_WEIGHTS] = jacobian_loss_weights
    return jacobian
end
using DelimitedFiles
using CorrelatedBPDecoderWithCER

function test_neural_BP()
    """
    Test the neural belief propagation decoder on a simple parity-check matrix of the Hamming code.
    H = [0 0 0 1 1 1 1;
         0 1 1 0 0 1 1;
         1 0 1 0 1 0 1]
    syndrome = [1, 0, 1] (indicating errors on qubits 1, 3, and 4)
    The expected output is the decoded message [1, 0, 1, 1, 0, 0, 0].

    We will compare the output LLRs of the Neural BP decoder with those of the classical BP decoder after K iterations.
    """
    H = [0 0 0 1 1 1 1;
         0 1 1 0 0 1 1;
         1 0 1 0 1 0 1]
    n_bits = size(H, 2)
    H_dual = copy(H)  # For Hamming code, the dual is the same
    # Example syndrome
    syndrome = [1, 0, 1]  # Indicates errors on qubits 1, 3, and 4
    initial_llrs = log(9) .* ones(n_bits)  # Assuming an initial bit-flip probability of 0.1
    n_iterations = 1

    ## Run the standard BP decoder
    (final_llrs_standard_bp, _) = run_bp("SumProduct", H, 4, syndrome, initial_llrs, n_iterations)
    # println("Final LLRs from standard BP after $(n_iterations) iterations: ", final_llrs_standard_bp)

    # println("--------------------------------------------------")

    ## Run the Neural BP decoder
    # Initialize the NeuralBP model
    base = NeuralBPBase(
        H,
        H_dual,
        n_iterations
    )
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Set the weights to be all ones
    standard_NBP = reset_to_standard_BP(bpnn)
    
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP/", standard_NBP)

    # Perform `n_iterations` forward passes: this corresponds to N iterations of standard BP
    # println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    (final_llrs_neural_bp, _, _) = standard_NBP(convert.(Float32, initial_llrs), convert.(Bool, syndrome))

    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs_neural_bp[end, :], final_llrs_standard_bp, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values.")
    else
        println("LLRs after $(n_iterations) iterations do not match the expected values.")
        println("Expected: ", final_llrs_standard_bp)
        println("Got: ", final_llrs_neural_bp[end, :])
    end
end

function test_intermediate_messages()
    """
    Test the intermediate messages of the Neural BP decoder on a simple parity-check matrix of the Hamming code.
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0] (indicating an error on qubit 2)

    We should have
    1. m^1_(c1 -> v5) = a^-1 ( i π s_1 + a(b^1_(v1) l_(v1)) + a(b^1_(v3) l_(v3)) + a(b^1_(v7) l_(v7)) )
    2. m^1_(c3 -> v5) = a^-1 ( i π s_3 + a(b^1_(v4) l_(v4)) + a(b^1_(v6) l_(v6)) + a(b^1_(v7) l_(v7)) )
    3. m^2_(v5 -> c2) = b^2_(v5) l_(v5) + m^1_(c1 -> v5) W^1_(v5,c2; c1,v5) + m^1_(c3 -> v5) W^1_(v5,c2; c3,v5)
    """
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0]  # Indicates an error on qubit 2
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2))  # Assuming an initial bit-flip probability of 0.1

    expected_message_c1_v5 = safe_atanh_exp(
        convert(Float32, im * π * syndrome[1]) + 
        safe_log_tanh(initial_llrs[1]) + 
        safe_log_tanh(initial_llrs[3]) + 
        safe_log_tanh(initial_llrs[7])
    )
    expected_message_c3_v5 = safe_atanh_exp(
        convert(Float32, im * π * syndrome[3]) + 
        safe_log_tanh(initial_llrs[4]) + 
        safe_log_tanh(initial_llrs[6]) + 
        safe_log_tanh(initial_llrs[7])
    )
    expected_message_v5_c2 = initial_llrs[5] + expected_message_c1_v5 + expected_message_c3_v5 # since all weights are 1 in standard BP

    # Compute the intermediate messages using the Neural BP decoder
    n_layers = 2
    dual_H = copy(H) # The dual is not necessary for this test since we are only interested in the intermediate messages.
    base = NeuralBPBase(
        H,
        dual_H,
        n_layers # number of layers (rounds of BP)
    )
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    standard_NBP = reset_to_standard_BP(bpnn)
    (_, intermediate_c2v_messages, intermediate_v2c_messages) = standard_NBP(initial_llrs, convert.(Bool, syndrome))

    # Compare the expected and actual intermediate messages
    if (isapprox(intermediate_c2v_messages[1][(1, 5)], expected_message_c1_v5, atol=1e-6))
        println("Message from c1 to v5 at layer 1 matches the expected value.")
    else
        println("Message from c1 to v5 at layer 1 does not match the expected value.")
        println("Expected: ", expected_message_c1_v5)
        println("Actual: ", intermediate_c2v_messages[1][(1, 5)])
    end

    if (isapprox(intermediate_c2v_messages[1][(3, 5)], expected_message_c3_v5, atol=1e-6))
        println("Message from c3 to v5 at layer 1 matches the expected value.")
    else
        println("Message from c3 to v5 at layer 1 does not match the expected value.")
        println("Expected: ", expected_message_c3_v5)
        println("Actual: ", intermediate_c2v_messages[1][(3, 5)])
    end

    println("Edges in the code: ", bpnn.base.edges)
    println("Intermediate V2C messages at layer 2: ", intermediate_v2c_messages[2])
    if (isapprox(intermediate_v2c_messages[2][(5, 2)], expected_message_v5_c2, atol=1e-6))
        println("Message from v5 to c2 at layer 2 matches the expected value.")
    else
        println("Message from v5 to c2 at layer 2 does not match the expected value.")
        println("Expected: ", expected_message_v5_c2)
        println("Actual: ", intermediate_v2c_messages[2][(5, 2)])
    end
end

function test_derivatives_wrt_weights()
    """
    Test the computation of the derivatives of the message from a check node to a variable node with respect to one of the weights in the Neural BP model.
    We will consider the following parity check matrix to define the code:
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 1;
         0 0 0 1 1 1 1]
    We will use the syndrome
    s = [0, 1, 0].

    We want to compute the derivative: ∂ m^2_(c2 -> v3) / ∂ W^1_(v5,c2; c1,v5). Using the chain rule and the message update equations, we have
    ∂ m^2_(c2 -> v3) / ∂ W^1_(v5,c2; c1,v5) = (a^-1)'( a(m^2_(c2 -> v3)) ) * ∂ a(m^2_(c2 -> v3)) / ∂ W^1_(v5,c2; c1,v5)
    where
    ∂ a(m^2_(c2 -> v3)) / ∂ W^1_(v5,c2; c1,v5) = a'(m^2(v5 -> c2)) * m^1_(c1 -> v5)

    Note that since the activation function is a(x) = log(tanh(x/2)), we have
        a'(x) = 1/sinh(x)
        (a^-1)(x) = 2 atanh(exp(x))
        (a^-1)'(x) = ∂ (2 atanh(exp(x))) / ∂x = -1/sinh(x)

    We will assume all the weights to be 1 and all the initial LLRs to be log(9) (corresponding to a bit-flip probability of 0.1) to compute the expected value of this derivative explicitly.
    
    Finally, we will compare this expected derivative with the one computed using the `grad_message_c2v_wrt_weight` function in `nachmani.jl`.
    """
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0]

    # We will need to construct a NeuralBP model and set the weights and LLRs accordingly to compute this derivative.
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The dual is not necessary for this test since we are only interested in the derivative.
    n_layers = 2
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    # Initial (channel) LLRs
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2))
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Set the weights to be all ones.
    standard_NBP = reset_to_standard_BP(bpnn)

    # Debugging step: print the `weights_c2v_v2c` dictionary.
    # println("Weights for deriving m_(v->c) from m_(c->v): ", bpnn.weights_c2v_v2c)
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP/", standard_NBP)
    
    # Compute the derivative using the function `grad_message_c2v_wrt_weight` in `nachmani.jl`
    (_, intermediate_c2v_messages, intermediate_v2c_messages) = standard_NBP(convert.(Float32, initial_llrs), convert.(Bool, syndrome))

    # Compute the expected derivative
    # ∂ m^2_(c2 -> v3) / ∂ W^1_(v5,c2; c1,v5) = (a^-1)'( a(m^2_(c2 -> v3)) ) * a'(m^2(v5 -> c2)) * m^1_(c1 -> v5)
    message_c2_v3 = intermediate_c2v_messages[2][(2, 3)]
    activated_message_c2_v3 = standard_NBP.base.activation_function(message_c2_v3)
    derivative_activation_inverse = standard_NBP.base.derivative_inverse_activation_function(activated_message_c2_v3)

    message_v5_c2 = intermediate_v2c_messages[2][(5, 2)]
    derivative_activated_message_v5_c2 = standard_NBP.base.derivative_activation_function(message_v5_c2) # since a'(x) = 1/sinh(x)
    
    message_c1_v5 = intermediate_c2v_messages[1][(1, 5)]

    expected_derivative = derivative_activation_inverse * derivative_activated_message_v5_c2 * message_c1_v5

    # Compute the actual derivative using the `grad_message_c2v_wrt_weight` function
    actual_derivatives = grad_message_c2v_wrt_weight(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        1, # t = 1
        (2, 1), # c* = 2, c*' = 1
        5 # v* = 5
    )
    actual_derivative_c2_v3_wrt_weight = actual_derivatives[2][(2, 3)] # since we want the derivative of m^2_(c2 -> v3)
    
    # Compare the expected and actual derivatives
    if (isapprox(real(actual_derivative_c2_v3_wrt_weight), real(expected_derivative), atol=1e-6) &&
        isapprox(imag(actual_derivative_c2_v3_wrt_weight), imag(expected_derivative), atol=1e-6))
        println("Derivative matches the expected value.")
    else
        println("Derivative does not match the expected value.")
        println("Expected: ", expected_derivative)
        println("Actual: ", actual_derivative_c2_v3_wrt_weight)
    end
end

function test_loss_derivative_wrt_weights()
    """
    Test the computation of the derivative of the loss function with respect to one of the weights in the Neural BP model.
    We will consider the following parity check matrix to define the code:
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 1;
         0 0 0 1 1 1 1]
    We will use the syndrome
    s = [0, 1, 0].

    We will compute the derivative of the loss with respect to the weight W^1_(v5,c2; c1,v5).

    It turns out (refer to the handwritten calculations) that
    ∂ L(μ^2) / ∂ W^1_(v5,c2; c1,v5) = ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^2) + e) )_c ) * ( ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^2_v) * ∂ μ^2_v / ∂ W^1_(v5,c2; c1,v5) )
    where
        - μ^2_v are the final LLRs for variable node v at iteration 2,
        - σ(x) = 1 / (1 + exp(x)) is the sigmoid function,
        - f(x) = |sin(π x / 2)| is the function used in the loss computation,
        - ∂ μ^2_v / ∂ W^1_(v5,c2; c1,v5) can be computed using the `grad_final_llr_wrt_weight` function in `nachmani.jl`.
    """
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0] # Indicates an error on qubit 2
    expected_recovery = [0, 1, 0, 0, 0, 0, 0] # We expect to recover an error on qubit 2
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2)) # Assuming an initial bit-flip probability of 0.1
    n_layers = 2
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The specific value of the dual isn't necessary, as long as it is consistent.
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Set the weights to be all ones.
    standard_NBP = reset_to_standard_BP(bpnn)
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP", standard_NBP)

    # Compute the intermediate messages using the forward propagation routine of the Neural BP model
    (intermediate_llrs, intermediate_c2v_messages, intermediate_v2c_messages) = standard_NBP(initial_llrs, convert.(Bool, syndrome))

    # Compute the derivative of the loss with respect to the weight W^1_(v5,c2; c1,v5)
    derivative_loss = grad_llrs_wrt_weight(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        1, # layer t* = 1
        (2, 1), # check nodes c* = 2, c*' = 1
        5 # vertex v* = 5
    )

    # println("Derivative of the loss with respect to the weight W^1_(v5,c2; c1,v5) computed using `grad_llrs_wrt_weight`: ", derivative_loss)

    n_checks_dual = size(H_dual, 1)
    expected_derivative_loss_wrt_weight = 0.0f0
    for layer in 1:n_layers
        for c_dual = 1:n_checks_dual
            # Compute f'( ( H^⟂ (σ(μ^t) + e) )_c )
            commutation_relation_c = sum( H_dual[c_dual, v] * (sigmoid(intermediate_llrs[layer, v]) + expected_recovery[v]) for v in 1:bpnn.base.code_n_bits )
            f_derivative = (π / 2) * cos(π * commutation_relation_c / 2) * sign(sin(π * commutation_relation_c / 2))

            # Compute the inner sum ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^t_v) * ∂ μ^t_v / ∂ W^1_(v5,c2; c1,v5)
            inner_sum = 0.0f0
            for v in 1:bpnn.base.code_n_bits # loop over variable nodes
                sigmoid_derivative = sigmoid(intermediate_llrs[layer, v]) * (1 - sigmoid(intermediate_llrs[layer, v])) # σ'(μ^t_v)
                derivative_mu_v = derivative_loss[(layer, v)] # ∂ μ^t_v / ∂ W^1_(v5,c2; c1,v5)
                inner_sum += H_dual[c_dual, v] * sigmoid_derivative * derivative_mu_v
            end

            expected_derivative_loss_wrt_weight += f_derivative * inner_sum
        end
    end

    # Compute the derivative of the loss with respect to the weight W^1_(v5,c2; c1,v5) using the `derivative_total_loss_wrt_weight` function in `backprop.jl`.
    actual_derivative_loss = derivative_total_loss_wrt_weight(
        bpnn,
        convert.(Bool, expected_recovery),
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        intermediate_llrs,
        1, # layer t* = 1
        (2, 1), # check nodes c* = 2, c*' = 1
        5 # vertex v* = 5
    )

    # println("Actual derivative of the loss with respect to the weight W^1_(v5,c2; c1,v5) computed using `derivative_total_loss_wrt_weight`: ", actual_derivative_loss)

    # Compare the expected and actual derivatives
    if (isapprox(real(actual_derivative_loss), real(expected_derivative_loss_wrt_weight), atol=1e-6) &&
        isapprox(imag(actual_derivative_loss), imag(expected_derivative_loss_wrt_weight), atol=1e-6))
        println("Derivative of the loss with respect to the weight matches the expected value. Their value is : ", expected_derivative_loss_wrt_weight)
    else
        println("Derivative of the loss with respect to the weight does not match the expected value.")
        println("Expected: ", expected_derivative_loss_wrt_weight)
        println("Actual: ", actual_derivative_loss)
    end
end

function test_loss_derivative_wrt_biases()
    """
    Test the computation of the derivative of the loss function with respect to one of the biases in the Neural BP model.
    We will consider the following parity check matrix to define the code:
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 1;
         0 0 0 1 1 1 1]
    We will use the syndrome
    s = [0, 1, 0].

    We will compute the derivative of the loss with respect to the bias b^1_(v7).

    It turns out (refer to the handwritten calculations) that
    ∂ L(μ^2) / ∂ b^1_(v7) = ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^2) + e) )_c ) * ( ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^2_v) * ∂ μ^2_v / ∂ b^1_(v7) )
    where
        - μ^2_v are the final LLRs for variable node v at iteration 2,
        - σ(x) = 1 / (1 + exp(x)) is the sigmoid function,
        - f(x) = |sin(π x / 2)| is the function used in the loss computation,
        - ∂ μ^2_v / ∂ b^1_(v7) can be computed using the `grad_final_llr_wrt_bias` function in `nachmani.jl`.
    """
    # The setup for this test is very similar to that of `test_loss_derivative_wrt_weights`, except that we will compute derivatives with respect to a bias instead of a weight. We can reuse most of the code from that test, and just change the relevant parts to compute derivatives with respect to biases instead of weights.
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0] # Indicates an error on qubit 2
    expected_recovery = [0, 1, 0, 0, 0, 0, 0] # We expect to recover an error on qubit 2
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2)) # Assuming an initial bit-flip probability of 0.1
    n_layers = 2
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The specific value of the dual isn't necessary, as long as it is consistent.
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Set the weights to be all ones.
    standard_NBP = reset_to_standard_BP(bpnn)
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP", standard_NBP)

    # Compute the intermediate messages using the forward propagation routine of the Neural BP model
    (intermediate_llrs, intermediate_c2v_messages, intermediate_v2c_messages) = standard_NBP(initial_llrs, convert.(Bool, syndrome))
    # Compute the derivative of the loss with respect to the bias b^1_(v7)
    derivative_loss = grad_llrs_wrt_bias(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        1, # layer t* = 2
        7 # vertex v* = 7
    )
    # We can compute the expected derivative using a similar approach as in `test_loss_derivative_wrt_weights`, but using the derivatives with respect to biases instead of weights.
    n_checks_dual = size(H_dual, 1)
    expected_derivative_loss_wrt_bias = 0.0f0
    for layer in 1:n_layers
        for c_dual = 1:n_checks_dual
            # Compute f'( ( H^⟂ (σ(μ^t) + e) )_c )
            commutation_relation_c = sum( H_dual[c_dual, v] * (sigmoid(intermediate_llrs[layer, v]) + expected_recovery[v]) for v in 1:bpnn.base.code_n_bits )
            f_derivative = (π / 2) * cos(π * commutation_relation_c / 2) * sign(sin(π * commutation_relation_c / 2))
            # Compute the inner sum ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^t_v) * ∂ μ^t_v / ∂ b^1_(v7)
            inner_sum = 0.0f0
            for v in 1:bpnn.base.code_n_bits # loop over variable nodes
                sigmoid_derivative = sigmoid(intermediate_llrs[layer, v]) * (1 - sigmoid(intermediate_llrs[layer, v])) # σ'(μ^t_v)
                derivative_mu_v = derivative_loss[(layer, v)] # ∂ μ^t_v / ∂ b^1_(v7)
                inner_sum += H_dual[c_dual, v] * sigmoid_derivative * derivative_mu_v
            end
            expected_derivative_loss_wrt_bias += f_derivative * inner_sum
        end
    end

    # Compute the derivative of the loss with respect to the bias b^1_(v7) using the `derivative_total_loss_wrt_bias` function in `backprop.jl`.
    actual_derivative_loss = derivative_total_loss_wrt_bias(
        bpnn,
        convert.(Bool, expected_recovery),
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        intermediate_llrs,
        1, # layer t* = 2
        7 # vertex v* = 7
    )
    
    # Compare the expected and actual derivatives
    if (isapprox(real(actual_derivative_loss), real(expected_derivative_loss_wrt_bias), atol=1e-6) &&
        isapprox(imag(actual_derivative_loss), imag(expected_derivative_loss_wrt_bias), atol=1e-6))
        println("Derivative of the loss with respect to the bias matches the expected value, which is ", expected_derivative_loss_wrt_bias, ".")
    else
        println("Derivative of the loss with respect to the bias does not match the expected value.")
        println("Expected: ", expected_derivative_loss_wrt_bias)
        println("Actual: ", actual_derivative_loss)
    end
end

function test_derivative_wrt_biases()
    """
    Test the computation of the derivatives of the message from a check node to a variable node with respect to one of the biases in the Neural BP model.
    We will consider the following parity check matrix to define the code:
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 1;
         0 0 0 1 1 1 1]
    We will use the syndrome
    s = [0, 1, 0].

    It turns out that (refer to the handwritten calculations):
    ∂ m^2_(c1 -> v3) / ∂ b^1_(v7) = (a^-1)'( a(m^2_(c1 -> v3)) ) * ∂ a(m^2_(c1 -> v3)) / ∂ b^1_(v7)
    
    where
    
    ∂ a(m^2_(c1 -> v3)) / ∂ b^1_(v7) = a'(m^2_(v5 -> c3)) W^1_(v5, c1; c3, v5) * (a^-1)'(a(m^1_(c_3 -> v5))) * a'(m^1_(v7 -> c3)) * ( l_(v7) )
    
    and
        - a(x) is the activation function used in the Neural BP model, which is a(x) = log(tanh(x/2)),
        - a'(x) = ∂ a(x) / ∂ x
        - (a^-1)'(x) = ∂ a^-1(x) / ∂ x
    """
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0]
    # We will need to construct a NeuralBP model and set the weights and LLRs accordingly to compute this derivative.
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The dual is not necessary for this test since we are only interested in the derivative.
    n_layers = 2
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    # Initial (channel) LLRs
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2))
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Set the weights to be all ones.
    standard_NBP = reset_to_standard_BP(bpnn)
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP/", standard_NBP)

    # Compute the intermediate messages using the forward propagation routine of the Neural BP model
    (_, intermediate_c2v_messages, intermediate_v2c_messages) = standard_NBP(initial_llrs, convert.(Bool, syndrome))

    # Compute the expected derivative
    # 1. a'(m^2_(v5 -> c3))
    m2_from_v5_to_c3 = intermediate_v2c_messages[2][(5, 3)] # m^2_(v5 -> c3)
    activation_derivative_m2_v5_c3 = bpnn.base.derivative_activation_function(m2_from_v5_to_c3) # a'(m^2_(v5 -> c3))
    
    # 2. (a^-1)'(a(m^1_(c_3 -> v5)))
    m1_from_c3_to_v5 = intermediate_c2v_messages[1][(3, 5)] # m^1_(c3 -> v5)
    activation_m1_c3_v5 = bpnn.base.activation_function(m1_from_c3_to_v5) # a(m^1_(c3 -> v5))
    inverse_activation_derivative_m1_c3_v5 = bpnn.base.derivative_inverse_activation_function(activation_m1_c3_v5) # (a^-1)'(a(m^1_(c3 -> v5)))
    
    # 3. a'(m^1_(v7 -> c3))
    m1_from_v7_to_c3 = intermediate_v2c_messages[1][(7, 3)] # m^1_(v7 -> c3)
    activation_derivative_m1_v7_c3 = bpnn.base.derivative_activation_function(m1_from_v7_to_c3) # a'(m^1_(v7 -> c3))

    # 4. (a^-1)'( a(m^2_(c1 -> v3)) )
    m2_from_c1_to_v3 = intermediate_c2v_messages[2][(1, 3)] # m^2_(c1 -> v3)
    activation_m2_c1_v3 = bpnn.base.activation_function(m2_from_c1_to_v3) # a(m^2_(c1 -> v3))
    inverse_activation_derivative_m2_c1_v3 = bpnn.base.derivative_inverse_activation_function(activation_m2_c1_v3) # (a^-1)'( a(m^2_(c1 -> v3)) )

    ## Put it all together
    derivative_activated_m2_c1_v3 = activation_derivative_m2_v5_c3 * bpnn.weights_c2v_v2c[(1, 5, 1, 3, 5)] * inverse_activation_derivative_m1_c3_v5 * activation_derivative_m1_v7_c3 * initial_llrs[7]
    expected_derivative = inverse_activation_derivative_m2_c1_v3 * derivative_activated_m2_c1_v3

    # Compute the actual derivative using the `grad_message_c2v_wrt_bias` function
    actual_derivatives = grad_message_c2v_wrt_bias(
        bpnn,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        1, # t = 1
        7 # v* = 7
    )
    actual_derivative = actual_derivatives[2][(1, 3)]

    # Compare the expected and actual derivatives
    if (isapprox(real(actual_derivative), real(expected_derivative), atol=1e-6) &&
        isapprox(imag(actual_derivative), imag(expected_derivative), atol=1e-6))
        println("Derivative with respect to bias matches the expected value, which is ", expected_derivative, ".")
    else
        println("Derivative with respect to bias does not match the expected value.")
        println("Expected: ", expected_derivative)
        println("Actual: ", actual_derivative)
    end
end

function test_derivative_wrt_readout()
    """
    Test the computation of the derivatives of the Loss function with respect to the readout weights in the Neural BP model.
    We will consider the following parity check matrix to define the code:
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    We will use the syndrome
    s = [0, 1, 0].

    We want to compute
    ∂ L / ∂ W^2_(v3; c1) = ∑_(l=1)^T ∂ L_l / ∂ W^2_(v3; c1)
                         = ∑_(l=1)^T ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^l) + e) )_c ) * ∂ ( H^⟂ (σ(μ^l) + e) )_c / ∂ W^2_(v3; c1)
                         = ∑_(l=1)^T ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^l) + e) )_c ) * ( ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^l_v) * ∂ μ^l_v / ∂ W^2_(v3; c1) )
                         = ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^2) + e) )_c ) * ( ∑_(v=1)^N_v H^⟂_(c,v) * σ'(μ^2_v) * ∂ μ^2_v / ∂ W^2_(v3; c1) )
                         = ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^2) + e) )_c ) * ( H^⟂_(c, 3) * σ'(μ^2_3) * ∂ μ^2_3 / ∂ W^2_(v3; c1) )
                         = ∑_(c=1)^N_c f'( ( H^⟂ (σ(μ^2) + e) )_c ) * ( H^⟂_(c, 3) * σ'(μ^2_3) * m^2_(c1 -> v3) )
    where
        - μ^2_(v3) is the final LLR for variable node v3 at layer 2.
        - m^2_(c1 -> v3) is the message from check node c1 to variable node v3 at layer 2, which can be computed using the forward propagation routine of the Neural BP model.
    """
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    syndrome = [0, 1, 0]
    expected_recovery = [0, 1, 0, 0, 0, 0, 0] # We expect to recover an error on qubit 2

    # We will need to construct a NeuralBP model and set the weights and LLRs accordingly to compute this derivative.
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The dual is not necessary for this test since we are only interested in the derivative.
    n_layers = 2
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    # Initial (channel) LLRs
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2))
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    
    # Set the weights to be all ones.
    standard_NBP = reset_to_standard_BP(bpnn)
    
    # Save the BP model to a file for inspection.
    save_NBP("./../data/test_neural_BP/", standard_NBP)

    # Compute the intermediate messages using the forward propagation routine of the Neural BP model
    (intermediate_llrs, intermediate_c2v_messages, _) = standard_NBP(initial_llrs, convert.(Bool, syndrome))

    # Compute the expected derivative
    expected_derivative = 0.0f0
    n_checks_dual = size(H_dual, 1)
    for c_dual in 1:n_checks_dual
        # Compute f'( ( H^⟂ (σ(μ^2) + e) )_c )
        residual_syndrome_bit = sum(H_dual[c_dual, v] * (sigmoid(intermediate_llrs[2, v]) + expected_recovery[v]) for v in 1:bpnn.base.code_n_bits)
        f_derivative = (π / 2) * cos(π * residual_syndrome_bit / 2) * sign(sin(π * residual_syndrome_bit / 2))
        # Compute H^⟂_(c, 3) * σ'(μ^2_3) * m^2_(c1 -> v3)
        h_dual_c_3 = H_dual[c_dual, 3]
        sigmoid_derivative_v3 = sigmoid(intermediate_llrs[2, 3]) * (1 - sigmoid(intermediate_llrs[2, 3]))
        message_c1_v3 = intermediate_c2v_messages[2][(1, 3)]
        # Putting it all together
        expected_derivative += f_derivative * h_dual_c_3 * sigmoid_derivative_v3 * message_c1_v3
    end

    # Compute the actual derivative using the `grad_final_llr_wrt_readout_weight` function
    actual_jacobian = nachmani_loss_jacobian_wrt_c2v_readout_weights(
        bpnn,
        convert.(Bool, expected_recovery),
        intermediate_c2v_messages,
        intermediate_llrs
    )
    actual_derivative = actual_jacobian[(2, 3, 1, 3)]

    # Compare the expected and actual derivatives
    if (isapprox(real(actual_derivative), real(expected_derivative), atol=1e-6) &&
        isapprox(imag(actual_derivative), imag(expected_derivative), atol=1e-6))
        println("Derivative with respect to readout weight matches the expected value, which is ", expected_derivative, ".")
    else
        println("Derivative with respect to readout weight does not match the expected value.")
        println("Expected: ", expected_derivative)
        println("Actual: ", actual_derivative)
    end
end

function test_forward_propagation()
    """
    We want to test the forward propagation of the NeuralBP model.
    We will define a small parity-check matrix, a syndrome and initial LLRs.
    We will then perform a forward pass through the network and print the output.
    """
    # Define the parity-check matrix
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Read from the files `data/<example_name>/HX.txt` and `data/<example_name>/LX.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    println("Parity-check matrix H of $(size(H))")
    # show(stdout, "text/plain", H)
    println()
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    # show(stdout, "text/plain", H_dual)
    # println()
    
    # Number of layers (rounds of BP)
    n_layers = 1
    
    # Initialize the NeuralBP model
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2)) # Initial LLRs corresponding to p=0.1
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    # Explicitly define weights for testing, to be all ones since that corresponds to standard BP.
    standard_NBP = reset_to_standard_BP(bpnn)
    # print_neuralbp_info(bpnn)

    # Define a syndrome
    error = convert.(Bool, vec([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))  # Error on qubit 1
    syndrome = convert.(Bool, (H * error) .% 2)
    
    # Perform a forward pass
    println("Performing forward pass through the NeuralBP model on syndrome: ", syndrome, " and with initial LLRs: ", initial_llrs, ".")
    (final_llrs, _, _) = standard_NBP(initial_llrs, syndrome)

    println("Output LLRs from forward pass:")
    show(stdout, "text/plain", final_llrs[end, :])
end

function test_loss()
    """
    Test the computation of the loss function for the NeuralBP model.
    We will define a parity-check matrix, a syndrome, initial LLRs, and expected recoveries.
    We will then compute the loss using the `compute_loss_error_from_llrs` function.
    H = [
        0 0 0 1 1 1 1 0;
        0 1 1 0 0 1 1 0;
        1 0 1 0 1 0 1 0
        ]
    H^⟂ = [
            1 0 0 0 0 1 1 0;
            0 1 0 0 1 0 1 0;
            0 0 1 0 1 1 0 0;
            0 0 0 1 1 1 1 0
          ]
    syndrome = [1, 0, 1] (indicating errors on qubits 1, 3, and 4)
    errors = [1, 0, 1, 1, 0, 0, 0] (the actual error pattern)
    posterior_llrs = [1.0663514264498881; 1.0663514264498881; 3.3280977282225512; 2.1972245773362196; 1.0663514264498881; -0.06452172443644333; 2.1972245773362196; 1.0663514264498881]
    The Loss function is given by
        L(e) = 1/T ∑_(t=1)^T L(μ^t, e)
        L(μ^t, e) = ∑_i  f ( ∑_(j) H^⟂_ij [ e_j + σ(μ^t_j)]) + ∑_((u,v) ∈ E) w_e * σ(μ^t_u) * (1 - σ(μ^t_v))
    where
        - σ(μ^t_k) = 1 / (1 + exp(-μ^t_k))
        - f(x) = |sin(π x / 2)|
        - H^⟂ is the parity-check matrix of the dual code.
        - E is the set of correlated qubit pairs, and w_e is the strength of the correlation for the pair e.
    """
    # Define the parity-check matrix
    H = [0 0 0 1 1 1 1 0;
         0 1 1 0 0 1 1 0;
         1 0 1 0 1 0 1 0]
    H_dual = [1 0 0 0 0 1 1 0;
              0 1 0 0 1 0 1 0;
              0 0 1 0 1 1 0 0;
              0 0 0 1 1 1 1 0]

    # Define correlated qubit pairs
    connectivity_edges = [1 2; 3 4; 5 6; 7 8] # Pairs of qubits that are encouraged to be correlated
    correlation_strengths = [0.5f0 ; 0.5f0 ; 0.5f0 ; 0.5f0] # Strength of the correlation penalty for each pair
    
    # Expected recovery
    expected_recovery = convert.(Bool, [1; 0; 1; 1; 0; 0; 0; 0;])
    
    # Posterior LLRs for each qubit at each layer.
    posterior_llrs = [
        2.1f0   1.1f0   -3.3f0   -2.2f0   -0.066f0   -0.064f0   2.2f0   1.1f0;;
    ]

    # Compute the expected loss using the explicit formula for the loss function
    expected_loss = 0.0f0
    n_checks_dual = size(H_dual, 1)
    n_bits = size(H_dual, 2)
    n_layers = size(posterior_llrs, 1)
    for layer in 1:n_layers
        # IID contribution to the loss
        expected_iid_loss = 0.0f0
        for c_dual in 1:n_checks_dual
            commutation_relation_c = sum(H_dual[c_dual, v] * (sigmoid(posterior_llrs[layer, v]) + expected_recovery[v]) for v in 1:n_bits)
            expected_iid_loss += abs(sin(π * commutation_relation_c / 2))
        end
        # Add the correlation loss
        expected_correlated_loss = 0.0f0
        for (idx, (u, v)) in enumerate(eachrow(connectivity_edges))
            correlation_penalty = correlation_strengths[idx] * sigmoid(posterior_llrs[layer, u]) * (1 - sigmoid(posterior_llrs[layer, v]))
            expected_correlated_loss += correlation_penalty
        end
        expected_loss = expected_iid_loss + expected_correlated_loss
    end
    # Average over layers
    expected_loss /= n_layers

    # Compute the Loss function.
    actual_loss = compute_loss_from_llrs(
        posterior_llrs, 
        expected_recovery, 
        convert.(Bool, H_dual);
        is_correlated=true,
        connectivity_edges=connectivity_edges,
        correlation_strengths=correlation_strengths
    )

    # Compare the expected and actual loss
    if isapprox(actual_loss, expected_loss, atol=1e-6)
        println("Loss computed by the function matches the expected value, which is ", expected_loss, ".")
    else
        println("Loss computed by the function does not match the expected value.")
        println("Expected loss: ", expected_loss)
        println("Actual loss: ", actual_loss)
    end
end

function test_training_step()
    """
    We want to test a single training step of the NeuralBP model, which involves computing the gradients of the loss with respect to the weights and biases, and then updating the weights and biases accordingly.
    We will define a small parity-check matrix, a syndrome, initial LLRs, expected recoveries, and a simple NeuralBP model with a small number of layers and neurons.
    We will then perform a forward pass to compute the loss, compute the gradients using backpropagation, and update the weights and biases using a simple gradient descent step.
    Finally, we will check if the weights and biases have been updated in the expected direction (i.e., if the loss has decreased after the update).
    """
    # Define the parity-check matrix
    H = [1 0 1 0 1 0 1;
         0 1 1 0 1 1 0;
         0 0 0 1 1 1 1]
    H_dual = copy(H) # For this test, we can take the dual to be the same as H. The specific value of the dual isn't necessary, as long as it is consistent.
    
    # Define a simple NeuralBP model
    n_layers = 2
    base = NeuralBPBase(
        H,
        H_dual,
        n_layers
    )
    initial_llrs = log(9.0f0) .* ones(Float32, size(H, 2))
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    
    # Define a syndrome and an expected recovery
    syndrome::BitVector = convert.(Bool, [1; 0; 1]) # Indicates errors on qubits 1, 3, and 4
    expected_recovery::BitVector = convert.(Bool, [1; 0; 1; 1; 0; 0; 0]) # We expect to recover errors on qubits 1, 3, and 4

    # Build the vectorization map for training, used to turn the learnable parameters and their corresponding gradients into a 1D vector for optimization.
    build_vectorization_maps!(bpnn)

    # Compute the loss before the update
    (intermediate_llrs, _, _) = bpnn(initial_llrs, syndrome)
    loss_before = compute_loss_including_correlations(
        intermediate_llrs,
        expected_recovery,
        convert.(Bool, H_dual),
        Int[;;], # no correlations for this test
        0.0f0;
        is_correlated = false
    )
    println("Loss before training step: ", loss_before)
    
    # Perform the training step
    train_step!(
        bpnn,
        initial_llrs,
        syndrome,
        expected_recovery;
        is_correlated = false
    )

    # After the training step, we can perform a forward pass to compute the new loss and check if it has decreased compared to the loss before the update.
    # Compute the loss after the update
    (intermediate_llrs_after, _, _) = bpnn(initial_llrs, syndrome)
    loss_after = compute_loss_including_correlations(
        intermediate_llrs_after,
        expected_recovery,
        convert.(Bool, H_dual),
        Int[;;], # no correlations for this test
        0.0f0;
        is_correlated = false
    )
    println("Loss after training step: ", loss_after)

    if loss_after <= loss_before
        println("Loss has decreased after the training step, as expected.")
    else
        @warn "Loss has increased after the training step. This may indicate an issue with the gradient computation or the update step."
        println("Loss before training step: ", loss_before)
        println("Loss after training step: ", loss_after)
    end
end

function test_training_Nachmani_BP()
    """
    Test the training of the Nachmani Neural BP model on a small code defined by a parity-check matrix H and its dual H^⟂.
    We will test on a small set of syndromes and expected recoveries, and check if the model is able to learn to decode correctly after training.
    """
    prefix::String = "./../data/hamming"
    
    # Define the parity-check matrix
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    logicals = readdlm("$(prefix)/LX.txt", Int)
    parity_check_matrix_dual = vcat(parity_check_matrix, logicals)

    # Load the connectivity matrix and the correlation strengths.
    connectivity_edges = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    correlation_strengths = 0.5f0 .* ones(Float32, size(connectivity_edges, 1)) # For testing, we can set all correlation strengths to be the same.

    # Define a simple NeuralBP model
    n_layers = 10
    base = NeuralBPBase(
        parity_check_matrix,
        parity_check_matrix_dual,
        n_layers;
        connectivity_edges=connectivity_edges,
        correlation_strengths=correlation_strengths
    )
    initial_llrs = log(9.0f0) .* ones(Float32, size(parity_check_matrix, 2))
    bpnn = NachmaniNeuralBP(base, initial_llrs)
    
    # Build the vectorization map for training, used to turn the learnable parameters and their corresponding gradients into a 1D vector for optimization.
    build_vectorization_maps!(bpnn)
    
    # Generate training data
    generate_data = false # Set to true to generate new training data using an i.i.d error model, or false to load pre-generated training data from files.
    if (generate_data == true)
        # Generate training data using an i.i.d error model
        n_samples = 10
        error_probability = 0.1
        training_syndromes, expected_recoveries = generate_training_data(parity_check_matrix, n_samples, error_probability)
    else
        # Load pre-generated training data from files
        expected_recoveries = convert.(Bool, readdlm("$(prefix)/basis_vectors.txt", Int))
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(parity_check_matrix * expected_recoveries, 2))
    end

    # Train the model
    train_minibatch!(bpnn, training_syndromes, expected_recoveries; n_epochs=3, batch_size=32, is_correlated=true)
    # Temporary: Restore to standard BP.
    # bpnn = reset_to_standard_BP(bpnn)

    # Save the trained model to a file for inspection.
    save_NBP("./../data/test_neural_BP", bpnn)

    # Test the model
    # test_error_patterns = convert.(Bool, readdlm("$(prefix)/basis_vectors.txt", Int))[:, 1:1] # Test only the first basis vector
    test_error_patterns = expected_recoveries
    test_syndromes = convert.(Bool, mod.(parity_check_matrix * test_error_patterns, 2))
    # Check if the predicted recoveries match the expected recoveries
    failures = predict_and_validate(bpnn, convert.(Int, parity_check_matrix_dual), test_syndromes, test_error_patterns)
    println("Out of ", size(test_error_patterns, 2), " test samples, ", sum(failures), " were incorrectly decoded.")
end
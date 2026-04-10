using SparseArrays
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
    syndrome = [0, 1, 1]  # Indicates errors on qubits 1, 3, and 4
    initial_llrs = log(9) .* ones(n_bits)  # Assuming an initial bit-flip probability of 0.1
    n_iterations = 3
    n_layers = n_iterations

    ## Run the standard BP decoder
    (final_llrs_standard_bp, _) = run_bp("SumProduct", H, 4, syndrome, initial_llrs, n_iterations; verbose=false)
    # println("Final LLRs from standard BP after $(n_iterations) iterations: ", final_llrs_standard_bp)

    # println("--------------------------------------------------")

    ## Run the Neural BP decoder
    
    # Initialize the NeuralBP model
    base = NeuralBPBase(
        H,
        H_dual,
        convert.(Float32, initial_llrs),
        n_layers
    )
    # Explicitly define weights for testing, to be all ones since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * n_layers)
    weights_llrs = ones(Float32, n_bits * n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)

    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout
    )
    # print_neuralbp_info(bpnn)

    # define the batch of syndromes (in this case, just one syndrome)
    syndromes = convert.(Bool, repeat(syndrome, 1, 1))  # single sample
    # define initial LLRs batch
    initial_llrs_batch = convert.(Float32, repeat(initial_llrs, 1, 1))  # single sample

    # Perform `n_iterations` forward passes: this corresponds to N iterations of standard BP
    # println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    llrs_neural_bp = bpnn(initial_llrs_batch, syndromes)
    
    final_llrs_neural_bp = llrs_neural_bp[:, :, n_layers]  # Get the final layer's LLRs from the 3D tensor output

    # Check if the final LLRs match the expected values
    # println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs_neural_bp, final_llrs_standard_bp, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values:", final_llrs_neural_bp)
    else
        println("LLRs after $(n_iterations) iterations do not match the expected values.")
        println("Expected: ", final_llrs_standard_bp)
        println("Got: ", final_llrs_neural_bp)
    end
end

function test_forward_propagation()
    """
    We want to test the forward propagation of the NeuralBP model.
    We will define a small parity-check matrix, a syndrome and initial LLRs.
    We will then perform a forward pass through the network and print the output.
    There are two ways to forward propagate through the network. One is an efficient in-place version that uses pre-allocated arrays to store intermediate results,
    and the other is a functional version that constructs new arrays at each step. We will test both versions and check that they give the same output.
    """
    # Define the parity-check matrix
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Read from the files `data/<example_name>/HX.txt` and `data/<example_name>/LX.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    # println("Parity-check matrix H of $(size(H))")
    # show(stdout, "text/plain", H)
    # println()
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    # println("Dual parity-check matrix H^⟂ of size $(size(H_dual))")
    # show(stdout, "text/plain", H_dual)
    # println()
    n_bits = size(H, 2)
    # Compute the number of neurons per layer: sum of ones in H
    nb_neurons_per_layer = sum(H)
    # println("Number of neurons per layer: ", nb_neurons_per_layer)

    # Define a syndrome to be a random binary vector of size equal to the number of rows of H
    syndrome = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    # Number of layers (rounds of BP)
    n_layers = 2
    
    # Set the initial LLRS
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, size(H, 2)) # Initial LLRs corresponding to p=0.1

    # Initialize the NeuralBP model with random weights around 1.0.
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_layers
    )
    #=
    weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * n_layers]; scale=0.01f0)
    weights_llrs = random_values_around_one([n_bits * n_layers]; scale=0.01f0)
    weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0)
    weights_loss_layers = random_values_around_one([n_layers]; scale=0.01f0)
    =#
    # Set all weights to 1.0 for testing, since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * n_layers)
    weights_llrs = ones(Float32, n_bits * n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)
    weights_loss_layers = ones(Float32, n_layers)
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout,
        weights_loss_layers=weights_loss_layers
    )

    # syndrome = convert.(Bool, zeros(Int, size(H, 1)))  # No errors
    syndromes_batch = repeat(convert.(Bool, syndrome), 1, 1)  # single sample

    # Define initial LLRs batch
    initial_llrs_batch = repeat(initial_llrs, 1, 1) # Initial LLRs corresponding to p=0.1

    # println("Syndrome shape: ", size(syndromes_batch))
    # println("Initial LLRs shape: ", size(initial_llrs_batch))

    # Perform a forward pass
    output_llrs_inplace_version = bpnn(initial_llrs_batch, syndromes_batch)
    output_llrs_functional_version = forward_pass_with_weights(bpnn, initial_llrs_batch, syndromes_batch)
    
    # Check if the outputs match
    println("Syndrome: ", syndrome)
    if all(isapprox.(output_llrs_inplace_version, output_llrs_functional_version, atol=1e-6))
        println("Forward pass outputs from both versions match, and they produce the LLRS: ", output_llrs_inplace_version, ".")
    else
        println("Forward pass outputs from both versions do not match.")
        # print the posterior LLRs from both versions, at each layer, to see where they start to differ.
        for layer in 1:n_layers
            println("Layer ", layer, ":")
            # Check if the outputs match at this layer
            if all(isapprox.(output_llrs_inplace_version[:, :, layer], output_llrs_functional_version[:, :, layer], atol=1e-6))
                println("LLRs at layer ", layer, " match: ", output_llrs_inplace_version[:, :, layer])
            else
                println("LLRs at layer ", layer, " do not match.")
                println("In-place version LLRs: ", output_llrs_inplace_version[:, :, layer])
                println("Functional version LLRs: ", output_llrs_functional_version[:, :, layer])
            end
            println("----------------------------------------------")
        end
    end

    # Run the standard BP decoder for `n_layers` iterations to get the expected LLRs after `n_layers` iterations.
    n_iterations = n_layers
    (final_llrs_standard_bp, _) = run_bp("SumProduct", H, size(H, 1) + 1, syndrome, convert.(Float64, initial_llrs), n_iterations; verbose=false)

    # Check if the final LLRs from the functional form of the Neural BP match the expected LLRs from the standard BP after `n_layers` iterations.
    if all(isapprox.(output_llrs_functional_version[:, :, n_layers], final_llrs_standard_bp, atol=1e-6))
        println("Final LLRs from the functional version of Neural BP after $(n_layers) iterations match the expected values from standard BP:", output_llrs_functional_version[:, :, n_layers])
    else
        println("Final LLRs from the functional version of Neural BP after $(n_layers) iterations do not match the expected values from standard BP.")
        println("Expected: ", final_llrs_standard_bp)
        println("Got: ", output_llrs_functional_version[:, :, n_layers])
    end
end

function test_activation_functions()
    """
    Test the two different implementations of the activation functions
    (i) `safe_log_tanh_split` and `safe_log_tanh_split!`
    (ii) `safe_atanh_exp_signed` and `safe_atanh_exp_signed!`
    In each case, while one is a functional version that constructs new arrays at each step, the other is an in-place version that modifies existing arrays.
    We will test both versions and check that they give the same output.
    We will define a sample input matrix, compute the outputs of both versions of the functions, and check that they match.
    """
    # Define a sample input matrix
    matrix = randn(Float32, 5, 5)  # 5 x 5 matrix of random values

    # Test safe_log_tanh_split and safe_log_tanh_split!
    magnitudes_functional, signs_functional = safe_log_tanh_split(matrix)
    # Define the buffers for the in-place version
    magnitudes_inplace = similar(magnitudes_functional)
    signs_inplace = similar(signs_functional)
    safe_log_tanh_split!(magnitudes_inplace, signs_inplace, matrix)

    # Check if the outputs match
    if all(isapprox.(magnitudes_functional, magnitudes_inplace, atol=1e-6)) && all(signs_functional .== signs_inplace)
        println("Outputs of safe_log_tanh_split and safe_log_tanh_split! match.")
    else
        println("Outputs of safe_log_tanh_split and safe_log_tanh_split! do not match.")
        println("Functional version magnitudes: ", magnitudes_functional)
        println("In-place version magnitudes: ", magnitudes_inplace)
        println("Functional version signs: ", signs_functional)
        println("In-place version signs: ", signs_inplace)
    end

    # Test safe_atanh_exp_signed and safe_atanh_exp_signed!
    # Use the same magnitudes and signs from the output of `safe_log_tanh_split`. Ideally we should get back the original matrix `matrix`.
    output_matrix_functional = safe_atanh_exp_signed(magnitudes_functional, signs_functional)
    # Define the buffer for the in-place version
    output_matrix_inplace = similar(output_matrix_functional)
    safe_atanh_exp_signed!(output_matrix_inplace, magnitudes_functional, signs_functional)

    # Check if the outputs match
    if all(isapprox.(output_matrix_functional, output_matrix_inplace, atol=1e-6))
        println("Outputs of safe_atanh_exp_signed and safe_atanh_exp_signed! match.")
    else
        println("Outputs of safe_atanh_exp_signed and safe_atanh_exp_signed! do not match.")
        println("Functional version output: ", output_matrix_functional)
        println("In-place version output: ", output_matrix_inplace)
    end
end

function test_messages_c2v_to_v2c()
    """
    Test the functions that compute messages from check to vertex, using the messages from vertex to check, using two different implementations:
    (i) c2v_to_v2c(activated_m_v2c_magnitudes, activated_m_v2c_signs, syndromes_batch, base): functional version that constructs new arrays at each step
    (ii) c2v_to_v2c!(m_c2v, activated_m_c2v_magnitudes, activated_m_c2v_signs, activated_m_v2c_magnitudes, activated_m_v2c_signs, syndromes_batch, base): in-place version that modifies existing arrays
    We will define random sample values for the inputs, run both versions of the function, and check that the outputs match.
    """
    # Define random sample values for the inputs
    example_name = "test_neural_BP"
    prefix = "./../data/$(example_name)"
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm("$(prefix)/code/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/code/LX.txt", Int)
    H_dual = vcat(H, logicals)
    
    ## Initialize the NeuralBP model
    n_bits = size(H, 2)
    n_layers = 5  # Number of BP iterations / layers
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define connectivity matrix and the correlation strengths
    correlation_strengths_file = "$(prefix)/correlated_weights/correlated_weights_p_0.001_q_0.3_s_1.txt"
    (connectivity_matrix, correlation_strengths) = parse_correlation_strengths_connectivity(correlation_strengths_file)

    # Explicitly define weights associated to computing the messages from V2C to C2V, since we don't want to run into DomainError issues with atanh during training.
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )
    
    n_samples = 1
    # Define sample random values for the messages from check to vertex.
    m_c2v_previous = randn(Float32, base.nb_neurons_per_layer, n_samples)  # random messages from check to vertex from the previous iteration
    
    # Set the weights to random values around 1.0, to avoid DomainError issues with atanh during testing.
    weights_llrs = random_values_around_one([n_bits]; scale=0.01f0)
    weights_messages = random_values_around_one([base.nb_weights_c2v_v2c]; scale=0.01f0)

    # Run the functional version of the function to compute messages from check to vertex
    (activated_m_v2c_magnitudes, activated_m_v2c_signs) = c2v_to_v2c(
        m_c2v_previous,
        weights_llrs,
        weights_messages,
        initial_llrs,
        base
    )

    # Run the in-place version of the function to compute messages from check to vertex
    messages_v2c_inplace = similar(activated_m_v2c_magnitudes)
    activated_m_v2c_magnitudes_inplace = similar(activated_m_v2c_magnitudes)
    activated_m_v2c_signs_inplace = similar(activated_m_v2c_signs)
    weighted_channel_llrs = similar(initial_llrs)
    weight_matrix = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_messages,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )
    c2v_to_v2c!(
        activated_m_v2c_magnitudes_inplace,        # output: magnitudes (log-domain)
        activated_m_v2c_signs_inplace,             # output: sign bits
        messages_v2c_inplace,                   # buffer (pre-activation real values)
        weighted_channel_llrs,   # buffer (same size as channel_llrs)
        m_c2v_previous,
        weights_llrs,
        weights_messages,
        initial_llrs,
        base,
        weight_matrix
    )

    # Check if the messages from check to vertex computed by both versions match
    if all(isapprox.(activated_m_v2c_magnitudes, activated_m_v2c_magnitudes_inplace, atol=1e-6)) &&
       all(activated_m_v2c_signs .== activated_m_v2c_signs_inplace)
        println("Messages from check to vertex computed by both versions match, and they are: ", activated_m_v2c_magnitudes, ".")
    else
        println("Messages from check to vertex computed by both versions do not match.")
        println("Functional version messages: ", activated_m_v2c_magnitudes, " (magnitudes), ", activated_m_v2c_signs, " (signs)")
        println("In-place version messages: ", activated_m_v2c_magnitudes_inplace, " (magnitudes), ", activated_m_v2c_signs_inplace, " (signs)")
    end
end

function test_messages_v2c_c2v()
    """
    Test the functions that compute messages from vertex to check, using the messages from check to vertex, using two different implementations:
    (i) v2c_to_c2v(
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            syndromes_batch,
            base
        ): functional version that constructs new arrays at each step
    
    (ii) v2c_to_c2v!(
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            m_v2c,
            weighted_channel_llrs,
            m_c2v_previous,
            weights_llrs,
            weights_messages,
            channel_llrs,
            base,
            weight_matrix
        ): in-place version that modifies existing arrays
    
    We will define random sample values for the inputs, run both versions of the function, and check that the outputs match.
    """
    # Define random sample values for the inputs
    example_name = "test_neural_BP"
    prefix = "./../data/$(example_name)"
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm("$(prefix)/code/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/code/LX.txt", Int)
    H_dual = vcat(H, logicals)
    
    ## Initialize the NeuralBP model
    n_bits = size(H, 2)
    n_layers = 5  # Number of BP iterations / layers
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define connectivity matrix and the correlation strengths
    correlation_strengths_file = "$(prefix)/correlated_weights/correlated_weights_p_0.001_q_0.3_s_1.txt"
    (connectivity_matrix, correlation_strengths) = parse_correlation_strengths_connectivity(correlation_strengths_file)

    # Explicitly define weights associated to computing the messages from V2C to C2V, since we don't want to run into DomainError issues with atanh during training.
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )
    
    n_samples = 1
    # Define sample random values for the messages from vertex to check, and for the syndromes batch.
    syndrome = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    syndromes_batch = repeat(convert.(Bool, syndrome), 1, n_samples)  # single sample
    
    # Compute the messages from check to vertex using the functional version of the function.
    activated_m_v2c_magnitudes = randn(Float32, base.nb_neurons_per_layer, n_samples)
    activated_m_v2c_signs = rand(Bool, base.nb_neurons_per_layer, n_samples)
    messages_c2v_functional = v2c_to_c2v(
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base
    )

    # Compute the messages from vertex to check using the in-place version of the function.
    messages_c2v_inplace = similar(messages_c2v_functional)
    activated_m_c2v_magnitudes_inplace = similar(messages_c2v_functional)
    activated_m_c2v_signs_inplace = rand(Bool, size(messages_c2v_functional))
    v2c_to_c2v!(
        messages_c2v_inplace,
        activated_m_c2v_magnitudes_inplace,
        activated_m_c2v_signs_inplace,
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base
    )

    # Check if the messages from vertex to check computed by both versions match
    if all(isapprox.(messages_c2v_functional, messages_c2v_inplace, atol=1e-6))
        println("Messages from vertex to check computed by both versions match, and they are: ", messages_c2v_functional, ".")
    else
        println("Messages from vertex to check computed by both versions do not match.")
        println("Functional version messages: ", messages_c2v_functional)
        println("In-place version messages: ", messages_c2v_inplace)
    end
end

function test_readout()
    """
    Test the readout function of the NeuralBP model, which computes the posterior LLRs from the messages from check to vertex and the channel LLRs.
    There are two versions of the readout function:
    (i) readout(m_c2v, weights_readout, weights_llrs, channel_llrs, base): functional version that constructs new arrays at each step
    (ii) function readout!(posterior_llrs, m_c2v, weights_readout, weights_llrs, channel_llrs, base, weight_matrix): in-place version that modifies existing arrays
    We will define random sample values for the inputs, run both versions of the function, and check that the outputs match.
    """
    # Define random sample values for the inputs
    example_name = "test_neural_BP"
    prefix = "./../data/$(example_name)"
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm("$(prefix)/code/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/code/LX.txt", Int)
    H_dual = vcat(H, logicals)
    
    ## Initialize the NeuralBP model
    n_bits = size(H, 2)
    n_layers = 5  # Number of BP iterations / layers
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define connectivity matrix and the correlation strengths
    correlation_strengths_file = "$(prefix)/correlated_weights/correlated_weights_p_0.001_q_0.3_s_1.txt"
    (connectivity_matrix, correlation_strengths) = parse_correlation_strengths_connectivity(correlation_strengths_file)

    # Explicitly define weights associated to computing the messages from V2C to C2V, since we don't want to run into DomainError issues with atanh during training.
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )

    n_samples = 1
    # Define sample random values for the messages from check to vertex.
    m_c2v = randn(Float32, base.nb_neurons_per_layer, n_samples)  # random messages from check to vertex from the previous iteration
    
    # Set the weights to random values around 1.0, to avoid DomainError issues with atanh during testing.
    weights_llrs = random_values_around_one([n_bits]; scale=0.01f0)
    weights_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0)

    # Compute the posterior LLRs using the functional version of the readout function
    posterior_llrs_functional = readout(
        m_c2v,
        weights_readout,
        weights_llrs,
        initial_llrs,
        base
    )

    # Compute the posterior LLRs using the in-place version of the readout function
    posterior_llrs_inplace = similar(posterior_llrs_functional)
    weight_matrix = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        weights_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )
    readout!(
        posterior_llrs_inplace,
        m_c2v,
        weights_readout,
        weights_llrs,
        initial_llrs,
        base,
        weight_matrix
    )

    # Check if the posterior LLRs computed by both versions match
    if all(isapprox.(posterior_llrs_functional, posterior_llrs_inplace, atol=1e-6))
        println("Posterior LLRs computed by both versions match, and they are: ", posterior_llrs_functional, ".")
    else
        println("Posterior LLRs computed by both versions do not match.")
        println("Functional version posterior LLRs: ", posterior_llrs_functional)
        println("In-place version posterior LLRs: ", posterior_llrs_inplace)
    end
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
        L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij M_(jk) [ e_k + σ(μ_k)])
    where
        - σ(μ_k) = 1 / (1 + exp(μ_k))
        - f(x) = |sin(π x / 2)|
        - M = [0 I ; I 0] is the symplectic matrix
        - H^⟂ is the parity-check matrix of the dual code.
    """
    # Define the parity-check matrix
    H = [0 0 0 1 1 1 1 0;
         0 1 1 0 0 1 1 0;
         1 0 1 0 1 0 1 0]
    H_dual = [1 0 0 0 0 1 1 0;
              0 1 0 0 1 0 1 0;
              0 0 1 0 1 1 0 0;
              0 0 0 1 1 1 1 0]
    # Example syndrome
    # syndrome = [1; 0; 1;;]  # Indicates errors on qubits 1, 3, and 4
    # Expected recovery
    errors = convert.(Bool, [1; 0; 1; 1; 0; 0; 0; 0;;])
    # Initial LLRs
    posterior_llrs = convert.(Float32, [2.0663514264498881; 1.0663514264498881; -3.3280977282225512; -2.1972245773362196; -0.0663514264498881; -0.06452172443644333; 2.1972245773362196; 1.0663514264498881;;])
    
    # Compute the Loss function.
    actual_loss = compute_loss_error_from_llrs(posterior_llrs, errors, convert.(Bool, H_dual))

    println("Computed Loss:", actual_loss)
end

function test_correlation_loss()
    """
    Test the computation of the loss function that comes from encouraging correlations between errors.
    The correlation loss is given by
        L_corr(μ) = λ * ∑_((qi, qj) ∈ C) [ e_(qi) XOR e_(qj) ]
    where
        - λ is a hyperparameter that controls the strength of the correlation penalty.
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.
    
    We will define a small set of correlated qubit pairs, predicted LLRs, and compute the correlation loss explicitly using the formula above.
    Additionally, we will compute the correlation loss using the `compute_additional_loss_from_ising_correlations` function and compare the results.
    """
    H_dual = [1 0 0 0 0 1 1 0;
              0 1 0 0 1 0 1 0;
              0 0 1 0 1 1 0 0;
              0 0 0 1 1 1 1 0]
    posterior_llrs = convert.(Float32, [
        2.0663514264498881;     # qubit 1
        1.0663514264498881;     # qubit 2
        -3.3280977282225512;    # qubit 3
        -2.1972245773362196;    # qubit 4
        -0.0663514264498881;    # qubit 5
        -0.06452172443644333;   # qubit 6
        2.1972245773362196;     # qubit 7
        1.0663514264498881;;    # qubit 8
    ])
    # Define correlated qubit pairs
    connectivity_matrix = [1 2; 3 4; 5 6; 7 8]
    correlation_strength = 0.5f0
    # Expected recoveries
    expected_recoveries = convert.(Bool, [1; 0; 1; 1; 0; 0; 0; 0;;])
    
    # Compute the correlation loss explicitly
    expected_corr_loss = 0.0f0
    for (qi, qj) in eachrow(connectivity_matrix)
        e_qi = 1 / (1 + exp(posterior_llrs[qi]))
        e_qj = 1 / (1 + exp(posterior_llrs[qj]))
        xor_value = e_qi + e_qj - 2 * e_qi * e_qj
        expected_corr_loss += xor_value
    end
    expected_corr_loss *= correlation_strength
    
    expected_bare_loss = compute_loss_error_from_llrs(
        posterior_llrs, 
        expected_recoveries,
        convert.(Bool, H_dual)
    )
    expected_loss = expected_bare_loss + expected_corr_loss
    
    # Now compute the correlation loss using the function
    actual_loss = compute_loss_including_correlations(
        posterior_llrs, 
        expected_recoveries,
        convert.(Bool, H_dual),
        connectivity_matrix, 
        correlation_strength,
        true
    )

    # Compare the two results
    if isapprox(actual_loss, expected_loss, atol=1e-6)
        println("Correlation loss computed by the function matches the expected value.")
    else
        println("Correlation loss computed by the function does not match the expected value.")
        println("Explicitly computed loss: ", expected_loss)
        println("Computed loss from function: ", actual_loss)
    end
end

function test_training_Nachmani_BP()
    """
    We will test the NeuralBP implementation on an example in `data/neural_example/`.
    We will generate training data with a certain error probability, train the NeuralBP model, and then test it on some test syndromes.
    
    """
    example_name = "test_neural_BP"
    prefix = "./../data/$(example_name)"
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm("$(prefix)/code/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/code/LX.txt", Int)
    H_dual = vcat(H, logicals)
    
    # generate training data
    generate_data = false
    if (generate_data == true)
        # Generate training data using an i.i.d error model
        n_samples = 10000
        error_probability = 0.1
        training_syndromes, expected_recoveries = generate_training_data(H, n_samples, error_probability)
    else
        # Load pre-generated training data from files
        expected_recoveries = convert.(Bool, readdlm("$(prefix)/training_data/sq_errors.txt", Int))
        n_samples = size(expected_recoveries, 2)
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(H * expected_recoveries, 2))
    end

    ## Initialize the NeuralBP model
    n_bits = size(H, 2)
    n_layers = 2  # Number of BP iterations / layers
    
    # Train the model
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define connectivity matrix and the correlation strengths
    correlation_strengths_file = "$(prefix)/correlated_weights/correlated_weights_p_0.001_q_0.3_s_1.txt"
    (connectivity_matrix, correlation_strengths) = parse_correlation_strengths_connectivity(correlation_strengths_file)

    # Explicitly define weights associated to computing the messages from V2C to C2V, since we don't want to run into DomainError issues with atanh during training.
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )
    #=
    # Explicitly define weights for testing, to be all ones since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * n_layers)
    weights_llrs = ones(Float32, n_bits * n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)
    weights_loss_layers=ones(Float32, n_layers)
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout,
        weights_loss_layers=weights_loss_layers
    )
    =#
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=random_values_around_one([base.nb_weights_c2v_v2c * n_layers]; scale=0.01f0),
        weights_llrs=random_values_around_one([n_bits * n_layers]; scale=0.01f0),
        weights_c2v_readout=random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0),
        weights_loss_layers=random_values_around_one([n_layers]; scale=0.01f0)
    )
    
    println("Going to train the Nachmani Neural BP model with $(base.nb_weights_c2v_v2c * n_layers + base.code_n_bits * n_layers + base.nb_weights_c2v_readout + n_layers) weights.")

    train_neuralbp!(bpnn, training_syndromes, expected_recoveries; n_epochs=1, batch_size=2)

    # Save the trained weights to a file
    save_trained_neuralbp_model("./../data/test_neural_BP/models/trained_weights.json", bpnn)
    
    # Debugging purpose: check if any element of the weights is NaN or `null`
    if any(isnan.(bpnn.weights_c2v_v2c)) || any(isnan.(bpnn.weights_llrs)) || any(isnan.(bpnn.weights_c2v_readout)) || any(isnan.(bpnn.weights_loss_layers))
        error("Trained weights contain NaN values. Please check the training process.")
    end

    # Test the model
    test_error_patterns = convert.(Bool, readdlm("$(prefix)/testing_data/sq_errors.txt", Int))
    println("Test error patterns shape: ", size(test_error_patterns))
    test_syndromes = convert.(Bool, mod.(H * test_error_patterns, 2))
    println("Test syndromes shape: ", size(test_syndromes))
    predicted_recoveries = predict_neuralbp(bpnn, test_syndromes)
    println("Predicted recoveries shape: ", size(predicted_recoveries))
    # println("Predicted recoveries for test syndromes:", predicted_recoveries)

    # Check if the predicted recoveries match the expected recoveries
    is_correct = check_bp_solutions(convert.(Int, H), test_error_patterns, predicted_recoveries)
    # println("Do the predicted recoveries match the expected recoveries? ", is_correct)
    println("Out of ", size(test_error_patterns, 2), " test samples, ", sum(is_correct), " were correctly decoded.")
end
using SparseArrays
using DelimitedFiles
using CorrelatedBPDecoderWithCER

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
    
    (i) c2v_to_v2c(
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            syndromes_batch,
            base
        ): functional version that constructs new arrays at each step
    
    (ii) c2v_to_v2c_with_weights!(
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            messages_v2c,
            weighted_channel_llrs,
            messages_c2v_previous,
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
    c2v_to_v2c_with_weights!(
        activated_m_v2c_magnitudes_inplace,        # output: magnitudes (log-domain)
        activated_m_v2c_signs_inplace,             # output: sign bits
        messages_v2c_inplace,                   # buffer (pre-activation real values)
        m_c2v_previous,
        weighted_channel_llrs,   # buffer (same size as channel_llrs)
        weights_llrs,
        weights_messages,
        initial_llrs,
        base
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
    
    (i) readout(
            m_c2v,
            weights_readout,
            weights_llrs,
            channel_llrs,
            base
        ): functional version that constructs new arrays at each step
    
    (ii) readout_with_weights!(
            posterior_llrs,
            m_c2v,
            weights_readout,
            weights_llrs,
            channel_llrs,
            base
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
    readout_with_weights!(
        posterior_llrs_inplace,
        m_c2v,
        weights_readout,
        weights_llrs,
        initial_llrs,
        base
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
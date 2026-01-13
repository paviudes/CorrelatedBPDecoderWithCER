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
    initial_llrs = log.((1 .- 0.1) ./ 0.1) .* ones(n_bits)  # Assuming an initial bit-flip probability of 0.1
    n_iterations = 1

    ## Run the standard BP decoder
    (final_llrs_standard_bp, _) = run_bp("SumProduct", H, 4, syndrome, initial_llrs, n_iterations)
    # println("Final LLRs from standard BP after $(n_iterations) iterations: ", final_llrs_standard_bp)

    # println("--------------------------------------------------")

    ## Run the Neural BP decoder
    nb_neurons_per_layer = sum(H)
    # println("Number of neurons per layer: ", nb_neurons_per_layer)

    # Explicitly define weights for testing, to be all ones since that corresponds to standard BP.
    weights_v2c_c2v = ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
    weights_c2v_v2c = ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
    weights_c2v_readout = ones(Float32, n_bits, nb_neurons_per_layer)

    # Initialize the NeuralBP model
    bpnn = NeuralBP(
        H,
        H_dual;
        weights_v2c_c2v=weights_v2c_c2v,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_c2v_readout=weights_c2v_readout
    )
    # print_neuralbp_info(bpnn)

    # define the batch of syndromes (in this case, just one syndrome)
    syndromes = convert.(Bool, repeat(syndrome, 1, 1))  # single sample
    # define initial LLRs batch
    initial_llrs_batch = repeat(initial_llrs, 1, 1)  # single sample

    # Perform `n_iterations` forward passes: this corresponds to N iterations of standard BP
    # println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    final_llrs_neural_bp = bpnn(initial_llrs_batch, syndromes; n_layers=n_iterations)

    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
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
    """
    # Define the parity-check matrix
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Read from the files `data/<example_name>/HX.txt` and `data/<example_name>/LX.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    println("Parity-check matrix H of $(size(H))")
    show(stdout, "text/plain", H)
    println()
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    println("Dual parity-check matrix H^⟂ of size $(size(H_dual))")
    show(stdout, "text/plain", H_dual)
    println()
    n_bits = size(H, 2)
    # Compute the number of neurons per layer: sum of ones in H
    nb_neurons_per_layer = sum(H)
    println("Number of neurons per layer: ", nb_neurons_per_layer)

    # Number of layers (rounds of BP)
    n_layers = 2
    
    # Explicitly define weights for testing, to be all ones since that corresponds to standard BP.
    weights_v2c_c2v = ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer, n_layers)
    weights_c2v_v2c = ones(Float32, nb_neurons_per_layer, nb_neurons_per_layer, n_layers)
    weights_c2v_readout = ones(Float32, n_bits, nb_neurons_per_layer)

    # Initialize the NeuralBP model
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, size(H, 2)) # Initial LLRs corresponding to p=0.1
    bpnn = NeuralBP(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        weights_v2c_c2v=weights_v2c_c2v,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_c2v_readout=weights_c2v_readout
    )
    # print_neuralbp_info(bpnn)

    # Define a syndrome
    syndrome = convert.(Bool, zeros(Int, size(H, 1)))
    syndromes_batch = repeat(syndrome, 1, 1)  # single sample

    # Define initial LLRs batch
    initial_llrs_batch = repeat(initial_llrs, 1, 1) # Initial LLRs corresponding to p=0.1

    # println("Syndrome shape: ", size(syndromes_batch))
    # println("Initial LLRs shape: ", size(initial_llrs_batch))

    # Perform a forward pass
    println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes_batch[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    output_llrs = bpnn(initial_llrs_batch, syndromes_batch)

    println("Output LLRs from forward pass:", output_llrs)
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

function test_training_BP()
    """
    We will test the NeuralBP implementation on an example in `data/neural_example/`.
    We will generate training data with a certain error probability, train the NeuralBP model, and then test it on some test syndromes.
    
    """
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    
    # generate training data
    generate_data = false
    if (generate_data == true)
        # Generate training data using an i.i.d error model
        n_samples = 10
        error_probability = 0.1
        training_syndromes, expected_recoveries = generate_training_data(H, n_samples, error_probability)
    else
        # Load pre-generated training data from files
        expected_recoveries = convert.(Bool, readdlm("$(prefix)/ballistic_training_data.txt", Int))
        n_samples = size(expected_recoveries, 2)
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(H * expected_recoveries, 2))
    end

    ## Initialize the NeuralBP model
    nb_neurons_per_layer = sum(H)
    n_layers = 5  # Number of BP iterations / layers
    
    # Train the model
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, size(H, 2)) # Initial LLRs corresponding to p=0.1

    # Define connectivity matrix for correlated errors (for testing purposes, we can define some arbitrary pairs)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)

    # Explicitly define weights associated to computing the messages from V2C to C2V, since we don't want to run into DomainError issues with atanh during training.
    bpnn = NeuralBP(
        H,
        H_dual,
        initial_llrs,
        n_layers;
        weights_v2c_c2v=random_values_around_one([nb_neurons_per_layer, nb_neurons_per_layer, n_layers]; scale=0.01f0),
        weights_c2v_v2c=random_values_around_one([nb_neurons_per_layer, nb_neurons_per_layer, n_layers]; scale=0.01f0),
        weights_c2v_readout=random_values_around_one([size(H, 2), nb_neurons_per_layer]; scale=0.01f0),
        connectivity=connectivity_matrix,
        correlation_strength=0.5f0
    )
    # print_neuralbp_info(bpnn)

    train_neuralbp!(bpnn, training_syndromes, expected_recoveries; n_epochs=1, batch_size=2)

    # Debugging purpose: check if any element of the weights is NaN or `null`
    for layer in 1:n_layers
        if any(isnan.(bpnn.weights_v2c_c2v[:, :, layer])) || any(isnan.(bpnn.weights_c2v_v2c[:, :, layer])) || any(isnan.(bpnn.weights_c2v_readout))
            error("Trained weights contain NaN values. Please check the training process.")
        end
    end

    # Test the model
    test_error_patterns = convert.(Bool, readdlm("$(prefix)/test_error_patterns_Z_th_0.995_nb_flip_0.01.txt", Int))
    println("Test error patterns shape: ", size(test_error_patterns))
    test_syndromes = convert.(Bool, mod.(H * test_error_patterns, 2))
    println("Test syndromes shape: ", size(test_syndromes))
    predicted_recoveries = predict_neuralbp(bpnn, test_syndromes)
    println("Predicted recoveries shape: ", size(predicted_recoveries))
    # println("Predicted recoveries for test syndromes:", predicted_recoveries)

    # Check if the predicted recoveries match the expected recoveries
    is_correct = check_bp_solutions(convert.(Int, H), test_error_patterns, convert.(Bool, predicted_recoveries))
    # println("Do the predicted recoveries match the expected recoveries? ", is_correct)
    println("Out of ", size(test_error_patterns, 2), " test samples, ", sum(is_correct), " were correctly decoded.")
end
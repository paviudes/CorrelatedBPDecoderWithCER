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
    println("Final LLRs from standard BP after $(n_iterations) iterations: ", final_llrs_standard_bp)

    println("--------------------------------------------------")

    ## Run the Neural BP decoder
    nb_neurons_per_layer = sum(H)
    println("Number of neurons per layer: ", nb_neurons_per_layer)

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
    println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    final_llrs_neural_bp = bpnn(initial_llrs_batch, syndromes; n_layers=n_iterations)

    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs_neural_bp, final_llrs_standard_bp, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values.")
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
    H = [1 0 0 0 1;
         0 1 0 1 0;
         0 0 1 1 1]
    n_bits = size(H, 2)
    
    H_dual = [1 1 0 1 1;
              1 0 1 0 1]
    
    # Compute the number of neurons per layer: sum of ones in H
    nb_neurons_per_layer = sum(H)
    println("Number of neurons per layer: ", nb_neurons_per_layer)

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
    print_neuralbp_info(bpnn)
    
    # Define a syndrome
    syndrome = convert.(Bool, [1; 0; 0])
    syndromes = repeat(syndrome, 1, 1)  # single sample

    # Define initial LLRs
    initial_llrs = -log(9) .* ones(Float32, size(H, 2), 1) # Initial LLRs corresponding to p=0.1

    # Perform a forward pass
    println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs[:, 1], ".")
    output_llrs = bpnn(initial_llrs, syndromes; n_layers=1)

    println("Output LLRs from forward pass:", output_llrs)
end

function test_training_BP()
    """
    We will test the NeuralBP implementation on a small example.
    H = [1 1 0 1 0;
         0 1 1 0 1;
         1 0 1 0 1]
    Syndrome = [1; 0; 1]
    Expected recovery = [1; 0; 1; 0; 0]
    """
    # Define the parity-check matrix
    H = [1 1 0 1 0;
         0 1 1 0 1;
         1 0 1 0 1]
    
    # generate training data
    n_samples = 10
    error_probability = 0.2
    syndromes, expected_recoveries = generate_training_data(H, n_samples, error_probability)

    # Initialize the NeuralBP model
    bpnn = NeuralBP(H, H)
    print_neuralbp_info(bpnn)

    # Train the model
    initial_llrs = log(9) .* ones(Float32, size(expected_recoveries)) # Initial LLRs corresponding to p=0.9

    train_neuralbp!(bpnn, syndromes, expected_recoveries; initial_llrs=initial_llrs, n_epochs=5, batch_size=2)

    # Test the model
    test_error_patterns = [1 0 1 0 0;
                           0 1 0 1 0;
                           1 1 0 0 1]
    test_syndromes = mod.(H * test_error_patterns, 2)
    predicted_recoveries = predict_neuralbp(bpnn, test_syndromes)

    println("Predicted recoveries:", predicted_recoveries)

    # Compare with expected recoveries
    for i in 1:eachcol(test_error_patterns)
        println("Test sample $i:")
        println("  Error: ", test_error_patterns[:, i])
        println("  Predicted: ", predicted_recoveries[:, i])
        println("---------------------------")
    end
end
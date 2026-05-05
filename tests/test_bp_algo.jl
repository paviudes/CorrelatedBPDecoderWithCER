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
    n_layers = 10  # Number of BP iterations / layers
    
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
        weights_c2v_readout=random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0)
    )
    
    println("Going to train the Nachmani Neural BP model with $(base.nb_weights_c2v_v2c * n_layers + base.code_n_bits * n_layers + base.nb_weights_c2v_readout + n_layers) weights.")

    train_neuralbp_enzyme!(bpnn, training_syndromes, expected_recoveries; learning_rate=1f-1, n_epochs=1, batch_size=2)

    # Save the trained weights to a file
    save_trained_neuralbp_model("./../data/test_neural_BP/models/trained_weights.json", bpnn)
    
    # Debugging purpose: check if any element of the weights is NaN or `null`
    if any(isnan.(bpnn.weights_c2v_v2c)) || any(isnan.(bpnn.weights_llrs)) || any(isnan.(bpnn.weights_c2v_readout))
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
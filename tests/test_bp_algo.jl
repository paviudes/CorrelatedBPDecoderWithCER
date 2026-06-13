using SparseArrays
using DelimitedFiles
using CorrelatedBPDecoderWithCER

@inline function load_neural_BP_model()::NeuralBP
    """
    Load a Neural BP model from a file.
    """
    # Define the parity-check matrix
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Read from the files `data/<example_name>/HX.txt` and `data/<example_name>/LX.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    n_bits = size(H, 2)
    
    # Number of layers (rounds of BP)
    n_layers = 3
    
    # Initialize the NeuralBP model with random weights around 1.0.
    base = NeuralBPBase(
        H,
        H_dual,
        zeros(Float32, n_bits), # default initial LLRs corresponding to p=0.5
        n_layers
    )
    
    #=
    weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * n_layers]; scale=0.01f0)
    weights_llrs = random_values_around_one([n_bits * n_layers]; scale=0.01f0)
    weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0)
    =#

    # Set all weights to 1.0 for testing, since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * n_layers)
    weights_llrs = ones(Float32, n_bits * n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout
    )

    return bpnn

end

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
    bpnn = load_neural_BP_model()
    (n_checks, n_bits) = size(bpnn.base.parity_check_matrix)
    n_layers = bpnn.base.n_layers
    
    # Define a syndrome to be a random binary vector of size equal to the number of rows of H
    syndrome = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    
    # Define initial LLRs batch
    initial_llrs = convert.(Float64, log(9)) .* ones(Float64, n_bits) # Initial LLRs corresponding to p=0.1
    
    n_iterations = n_layers
    
    ## Run the standard BP decoder
    parity_check_matrix_int = convert.(Int, bpnn.base.parity_check_matrix)
    (final_llrs_standard_bp, _) = run_bp("SumProduct", parity_check_matrix_int, 4, syndrome, initial_llrs, n_iterations; verbose=false)
    # println("Final LLRs from standard BP after $(n_iterations) iterations: ", final_llrs_standard_bp)

    # println("--------------------------------------------------")

    ## Run the Neural BP decoder
    # define the batch of syndromes (in this case, just one syndrome)
    syndromes_batch = repeat(convert.(Bool, syndrome), 1, 1)  # single sample
    # define initial LLRs batch
    initial_llrs_batch = repeat(convert.(Float32, initial_llrs), 1, 1) # Initial LLRs corresponding to p=0.1

    # Perform `n_iterations` forward passes: this corresponds to N iterations of standard BP
    # println("Performing forward pass through the NeuralBP model on syndrome: ", syndromes[:, 1], " and with initial LLRs: ", initial_llrs_batch[:, 1], ".")
    llrs_neural_bp = bpnn(initial_llrs_batch, syndromes_batch) # shape (n_bits, n_samples, n_layers)
    
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
    prefix = "./../data/72q_BB_p_0.010_q_0.001_std_0.01_data"
    parity_check_matrix_file = "$(prefix)/code/HZ.txt"
    logicals_file = "$(prefix)/code/LZ.txt"
    correlation_strengths_file = "$(prefix)/correlated_weights/correlated_weights_p_0.01_q_0.001_s_1.txt"
    training_errors_file = "$(prefix)/training_data/train_ballistic_p_0.01_q_0.001_s_1.txt"
    n_layers = 5
    
    start = time()
    
    # Load the base model and initialize the weights.
    base = load_base_BP_model(parity_check_matrix_file, logicals_file, n_layers; correlation_strengths_file=correlation_strengths_file)
    
    # Extract hyperparameters from file or use defaults
    hyperparams_file = "default_hyperparams.toml"
    hyperparams = parse_hyper_parameters(hyperparams_file; prefix=prefix)
    n_epochs = hyperparams["n_epochs"]

    # Define initial conditions for the learnable parameters: all weights are initialized to Gaussian random values around 1, with a standard deviation of σ specified in the hyperparameters.
    weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale=hyperparams["initial_conditions_scale"])
    weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale=hyperparams["initial_conditions_scale"])
    weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale=hyperparams["initial_conditions_scale"])
    
    #=
    # Explicitly define weights for debugging, to be all ones since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * base.n_layers)
    weights_llrs = ones(Float32, base.code_n_bits * base.n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)
    =#
    initial_conditions = Dict{String, Vector{Float32}}(
        "weights_c2v_v2c" => weights_c2v_v2c,
        "weights_llrs" => weights_llrs,
        "weights_c2v_readout" => weights_c2v_readout
    )
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=initial_conditions["weights_c2v_v2c"],
        weights_llrs=initial_conditions["weights_llrs"],
        weights_c2v_readout=initial_conditions["weights_c2v_readout"]
    )
    
    # Train the model using the training data and the specified hyperparameters.
    n_weights = length(weights_c2v_v2c) + length(weights_llrs) + length(weights_c2v_readout)
    println("Going to train the Nachmani Neural BP model with $(n_weights) weights.")
    bpnn = train_Nachmani_neuralbp(
        base,
        training_errors_file,
        hyperparams;
        initial_conditions=initial_conditions,
        prefix=prefix
    )
    
    # Save the trained weights to a file
    save_trained_neuralbp_model(
        "$(prefix)/models/trained_weights_$(n_layers)_layers_$(n_epochs)_epochs.json",
        bpnn
    )
    
    # Debugging purpose: check if any element of the weights is NaN or `null`
    if any(isnan.(bpnn.weights_c2v_v2c)) || any(isnan.(bpnn.weights_llrs)) || any(isnan.(bpnn.weights_c2v_readout))
        error("Trained weights contain NaN values. Please check the training process.")
    end
    
    # Test the model.
    test_errors_file = "$(prefix)/testing_data/test_ballistic_p_0.01_q_0.001_s_1.txt"
    is_correct = neuralbp_test_predictions(bpnn, test_errors_file)
    n_error_patterns = size(is_correct, 1)
    n_successful_decodings = sum(is_correct)
    
    runtime = time() - start
    
    println(
        "[", round(runtime, digits=2), "s] Out of ",
        n_error_patterns, " test samples, ",
        n_successful_decodings, " were correctly decoded."
    )
end
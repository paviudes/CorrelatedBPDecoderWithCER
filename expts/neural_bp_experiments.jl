using DelimitedFiles
using CorrelatedBPDecoderWithCER

function generate_ballistic_training_data(
    prefix::String,
    ballistic_per_qubit_error_probs::AbstractVector{Float64},
    ballistic_neighbour_error_probs::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/ballistic_training_data.txt"
)::String
    """
    Generate training data for the Ballistic error model.
    The training data consists of error patterns generated according to the Ballistic error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, p_neighbour) for p_qubit in ballistic_per_qubit_error_probs for p_neighbour in ballistic_neighbour_error_probs]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate
    
    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)
    # Iterate over all combinations of error parameters
    for (i, (ballistic_per_qubit_error_prob, ballistic_neighbour_error_prob)) in enumerate(error_rates)
        errormodel = BallisticErrorModel(ballistic_per_qubit_error_prob, ballistic_neighbour_error_prob; correlations=connectivity_matrix, name="Ballistic Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1
    
    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')
    return output_errors_file
end

function generate_randomwalk_training_data(
    prefix::String,
    randomwalk_per_qubit_error_probs::AbstractVector{Float64},
    randomwalk_lengths::AbstractVector{Int},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/randomwalk_training_data.txt"
)::String
    """
    Generate training data for the Random Walk error model.
    The training data consists of error patterns generated according to the Random Walk error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, length) for p_qubit in randomwalk_per_qubit_error_probs for length in randomwalk_lengths]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate
    
    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)
    # Iterate over all combinations of error parameters
    for (i, (randomwalk_per_qubit_error_prob, randomwalk_length)) in enumerate(error_rates)
        errormodel = RandomWalkErrorModel(randomwalk_per_qubit_error_prob, randomwalk_length, nqubits; correlations=connectivity_matrix, name="Random Walk Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1
    
    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')
    return output_errors_file
end

function generate_randomwalk_testing_data(
    prefix::String,
    randomwalk_per_qubit_error_probs::AbstractVector{Float64},
    randomwalk_lengths::AbstractVector{Int},
    samples_per_error_rate::Int;
    output_errors_dir::String="./../data/hamming"
)::Vector{String}
    """
    Generate training data for the Random Walk error model.
    The training data consists of error patterns generated according to the Random Walk error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, length) for p_qubit in randomwalk_per_qubit_error_probs for length in randomwalk_lengths]
    
    # Define the error model and generate error patterns
    output_error_files = String[]
    # Iterate over all combinations of error parameters
    for (randomwalk_per_qubit_error_prob, randomwalk_length) in error_rates
        errormodel = RandomWalkErrorModel(randomwalk_per_qubit_error_prob, randomwalk_length, nqubits; correlations=connectivity_matrix, name="Random Walk Error Model")
        error_patterns = sample_errors(errormodel, nqubits, samples_per_error_rate)
        # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
        # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
        error_patterns[error_patterns .== 1] .= 0
        error_patterns[error_patterns .== 2] .= 1
        error_patterns[error_patterns .== 3] .= 1
        # Save the generated error patterns to a file
        output_errors_file = "$(output_errors_dir)/test_error_patterns_Z_p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length).txt"
        writedlm(output_errors_file, error_patterns', ' ')
        push!(output_error_files, output_errors_file)
        # Print the command to run the test with this error patterns file
        println("julia --project=./../ neural_bp_experiments.jl " *
                "--codename hamming " *
                "--n_hidden_layers 50 " *
                "--n_epochs 5 " *
                "--batch_size 32 " *
                "--retrain false " *
                "--train randomwalk_training_data.txt " *
                "--test test_error_patterns_Z_p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length).txt " *
                "--correlation_strength 0.5")
        println("echo \"Testing done for p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length)\" >&2")
    end
    return output_error_files
end

function generate_regenerative_training_data(
    prefix::String,
    regenerative_block_sizes::AbstractVector{Int},
    regenerative_block_probabilities::AbstractVector{Float64},
    regenerative_error_probs_within_block::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/regenerative_training_data.txt"
)::String
    """
    Generate training data for the Regenerative error model.
    The training data consists of error patterns generated according to the Regenerative error model for a small range of error parameters.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    # connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)

    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [
        (block_size, block_prob, error_prob_within_block) 
        for block_size in regenerative_block_sizes 
        for block_prob in regenerative_block_probabilities 
        for error_prob_within_block in regenerative_error_probs_within_block
    ]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate

    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)

    # Iterate over all combinations of error parameters
    for (i, (block_size, block_prob, error_prob_within_block)) in enumerate(error_rates)
        errormodel = RegenerativeErrorModel(block_size, block_prob, error_prob_within_block, nqubits; name="Regenerative Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1

    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')

    return output_errors_file
end

function generate_regenerative_testing_data(
    prefix::String,
    regenerative_block_sizes::AbstractVector{Int},
    regenerative_block_probabilities::AbstractVector{Float64},
    regenerative_error_probs_within_block::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_dir::String="./../data/hamming"
)::Vector{String}
    """
    Generate testing data for the Regenerative error model.
    The testing data consists of error patterns generated according to the Regenerative error model for a small range of error parameters.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    # connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)

    # Determine the number of qubits
    nqubits = size(parity_check_matrix, 2)
    error_rates = [
        (block_size, block_prob, error_prob_within_block) 
        for block_size in regenerative_block_sizes 
        for block_prob in regenerative_block_probabilities 
        for error_prob_within_block in regenerative_error_probs_within_block
    ]

    output_error_files = String[]

    # Iterate over all combinations of error parameters
    for (block_size, block_prob, error_prob_within_block) in error_rates
        errormodel = RegenerativeErrorModel(block_size, block_prob, error_prob_within_block, nqubits; name="Regenerative Error Model")
        error_patterns = sample_errors(errormodel, nqubits, samples_per_error_rate)

        # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
        # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
        error_patterns[error_patterns .== 1] .= 0
        error_patterns[error_patterns .== 2] .= 1
        error_patterns[error_patterns .== 3] .= 1

        # Save the generated error patterns to a file
        output_errors_file = "$(output_errors_dir)/test_error_patterns_Z_bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block).txt"
        writedlm(output_errors_file, error_patterns', ' ')
        push!(output_error_files, output_errors_file)

        # Print the command to run the test with this error patterns file
        println("julia --project=./../ neural_bp_experiments.jl " *
                "--codename hamming " *
                "--n_hidden_layers 50 " *
                "--n_epochs 5 " *
                "--batch_size 32 " *
                "--retrain false " *
                "--train regenerative_training_data.txt " *
                "--test test_error_patterns_Z_bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block).txt " *
                "--correlation_strength 0.5")
        println("echo \"Testing done for bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block)\" >&2")
    end

    return output_error_files
end

function train_neuralbp(
    parity_check_matrix_file::String,
    logicals_file::String;
    connectivity_matrix::Matrix{Int}=zeros(Int,0,0),
    correlation_strength::Float64=0.5,
    n_hidden_layers::Int=2,
    n_epochs::Int=5,
    training_errors_file::String="",
    n_samples::Int=1000,
    batch_size::Int=2,
    prefix::String="./../data/models",
    retrain::Bool=false
)
    """
    Train a Neural Belief Propagation decoder for the given parity-check matrix.
    The trained model consists of weights (coefficients) for each pair of connected neurons in the neural BP network.
    We will save the weights into a file for later use.
    If this weights file already exists, we will load the weights from the file instead of training a new model.
    """
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm(parity_check_matrix_file, Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm(logicals_file, Int)
    H_dual = vcat(H, logicals)

    # Define the initial LLRs for the variable nodes
    n_bits = size(H, 2)
    nb_neurons_per_layer = sum(H)
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define the Neural BP model
    bpnn = NeuralBP(
        H,
        H_dual,
        initial_llrs,
        n_hidden_layers;
        weights_v2c_c2v=random_values_around_one([nb_neurons_per_layer, nb_neurons_per_layer, n_hidden_layers]; scale=0.01f0),
        weights_c2v_v2c=random_values_around_one([nb_neurons_per_layer, nb_neurons_per_layer, n_hidden_layers]; scale=0.01f0),
        weights_c2v_readout=random_values_around_one([n_bits, nb_neurons_per_layer]; scale=0.01f0),
        connectivity=connectivity_matrix,
        correlation_strength=convert(Float32, correlation_strength)
    )
    # println("Correlation strength set to: ", bpnn.correlation_strength)
    # print_neuralbp_info(bpnn)

    # Check if the weights file already exists
    weights_filename = "$(prefix)/neuralbp_weights_nlayers_$(n_hidden_layers)_epochs_$(n_epochs)_correlation_strength_$(correlation_strength).json"
    if isfile(weights_filename) && !retrain
        # println("Loading existing weights from file: $weights_filename")
        bpnn = load_trained_neuralbp_model(weights_filename, bpnn)
    else
        # println("Training Neural BP model for parity-check matrix from file: $parity_check_matrix_file")
        # Generate training data if not provided
        if training_errors_file == ""
            error_probability = 0.1
            (__, expected_recoveries) = generate_training_data(H, n_samples, error_probability)
            training_errors_file = "$(prefix)/training_data.txt"
            # Save the generated training data to a file using `DelimitedFiles.writedlm`
            writedlm(training_errors_file, expected_recoveries, ',')
        end

        # Read errors from the training errors file
        expected_recoveries = convert.(Bool, readdlm(training_errors_file, Int))
        n_samples = size(expected_recoveries, 2)
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(H * expected_recoveries, 2))
        
        # Train the Neural BP model
        train_neuralbp!(bpnn, training_syndromes, expected_recoveries; n_epochs=n_epochs, batch_size=batch_size)

        # Save the trained weights to a file
        save_trained_neuralbp_model(weights_filename, bpnn)

        #TODO: Save a report of the training to a file. The file name should start with `training_report_`

        # println("Trained weights saved to file: $weights_filename")
    end
    return bpnn
end

function neuralbp_test_predictions(bpnn::NeuralBP, test_errors_file::String)::BitVector
    """
    Predict the recoveries for the given test syndromes using the trained Neural BP model.
    Test these predictions to see if they match the expected recoveries.
    """
    test_errors = convert.(Bool, readdlm(test_errors_file, Int))
    test_syndromes = convert.(Bool, mod.(bpnn.parity_check_matrix * test_errors, 2))
    predicted_recoveries = predict_neuralbp(bpnn, test_syndromes)
    
    # Check if the predicted recoveries match the expected recoveries
    is_correct = check_bp_solutions(convert.(Int, bpnn.parity_check_matrix), test_errors, convert.(Bool, predicted_recoveries))

    #TODO: Save a report of the testing to a file. The file name should start with `testing_report_`
    
    # println("Out of ", size(test_errors, 2), " test samples, ", sum(is_correct), " were correctly decoded.")
    return is_correct
end

function postprocess_neuralbp_results(summary_json_file::String; output_csv_file::String="./../data/hamming/neuralbp_decoder_statistics.csv")::String
    """
    Post-process the results of the Neural BP experiments.
    This function can be used to aggregate results, and produce a CSV summary of the experiments.
    """
    # Collect decoder statistics from the summary JSON file
    decoder_stats = collect_decoder_statistics(summary_json_file)
    # Save the decoder statistics to a CSV file
    output_csv_file = save_decoder_dataframe(decoder_stats, output_csv_file)
    return output_csv_file
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    """
    Run a complete experiment to train and test a Neural BP decoder.
    """
    
    # Parse command-line arguments
	args_dict = parse_command_line_args_NN(;prefix="./../data")
    # print_arguments(args_dict; io=stdout)

    # Extract arguments
    prefix = "./../data/$(args_dict["codename"])"
    parity_check_matrix_file = "$(prefix)/HX.txt"
    logicals_file = "$(prefix)/LX.txt"
    connectivity_matrix_file = "$(prefix)/connectivity_matrix.txt"
    connectivity_matrix = readdlm(connectivity_matrix_file, Int)
    correlation_strength = args_dict["correlation_strength"]
    n_hidden_layers = args_dict["n_hidden_layers"]
    n_epochs = args_dict["n_epochs"]
    batch_size = args_dict["batch_size"]
    training_errors_file = "$(prefix)/$(args_dict["train"])"
    n_samples = args_dict["n_samples"]
    retrain = args_dict["retrain"]
    # Train the Neural BP model
    start = time()
    bpnn = train_neuralbp(
        parity_check_matrix_file,
        logicals_file;
        connectivity_matrix=connectivity_matrix,
        correlation_strength=correlation_strength,
        n_hidden_layers=n_hidden_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        training_errors_file=training_errors_file,
        n_samples=n_samples,
        prefix=prefix,
        retrain=retrain
    )
    runtime = time() - start

    # Test the Neural BP model predictions
    test_errors_file = "$(prefix)/$(args_dict["test"])"
    is_correct = neuralbp_test_predictions(bpnn, test_errors_file)
    failures = collect(.!is_correct)

    # println("Out of ", size(is_correct), " test samples, ", sum(is_correct), " were correctly decoded.")

    # Load the results on to the `DecoderStatistics` structure.
    stats = DecoderStatistics(
        "NN",
        "ExplicitErrorModel",
        test_errors_file,
        size(is_correct, 1),
        n_hidden_layers,
        n_epochs,
        convert(Float64, bpnn.correlation_strength);
        num_failures = count(failures),
        failures = failures,
        runtime = runtime
    )

    # Print the statistics in JSON format to shell.
    record_decoder_statistics(stats)
end
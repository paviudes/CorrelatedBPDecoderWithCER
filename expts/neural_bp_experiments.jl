using CSV
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER

function train_Nachmani_neuralbp(
    parity_check_matrix_file::String,
    logicals_file::String;
    # Hyperparameters for the Neural BP model
    connectivity_matrix::Matrix{Int}=zeros(Int,0,0),
    correlation_strengths::AbstractVector{Float32}=[],
    n_hidden_layers::Int=2,
    n_epochs::Int=5,
    training_errors_file::String="",
    n_samples::Int=1000,
    batch_size::Int=2,
    prefix::String="./../data",
    # Hyperparameters for training the Neural BP model
    retrain::Bool=false,
    learning_rate::Float32=1f-1,
    max_grad_norm::Float32=2.0f0
)
    """
    Train a Neural Belief Propagation decoder for the given parity-check matrix.
    The trained model consists of weights (coefficients) for each pair of connected neurons in the neural BP network.
    We will save the weights into a file for later use.
    If this weights file already exists, we will load the weights from the file instead of training a new model.
    """
    
    # Create the models and results directories if they don't exist
    models_dir = "$(prefix)/models"
    if !isdir(models_dir)
        mkdir(models_dir)
    end
    
    # Define the parity-check matrix
    # Read from the files `data/neural_example/H.txt` and `data/neural_example/H_dual.txt`
    H = readdlm(parity_check_matrix_file, Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm(logicals_file, Int)
    H_dual = vcat(H, logicals)

    # Define the initial LLRs for the variable nodes
    n_bits = size(H, 2)
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1

    # Define the Neural BP model
    base = NeuralBPBase(
        H,
        H_dual,
        initial_llrs,
        n_hidden_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=random_values_around_one([base.nb_weights_c2v_v2c * n_hidden_layers]; scale=0.1f0),
        weights_llrs=random_values_around_one([n_bits * n_hidden_layers]; scale=0.1f0),
        weights_c2v_readout=random_values_around_one([base.nb_weights_c2v_readout]; scale=0.1f0)
    )
    
    # Extract the name of the training file name to include in the weights file name for clarity on what data the model was trained on.
    # We only want the filename without the path and extension. For example, if the training file is `data/hamming/training_data.txt`, we want to extract `training_data`.
    training_source = splitext(basename(training_errors_file))[1]
    # Check if the weights file already exists
    weights_filename = "$(models_dir)/neuralbp_weights_nlayers_$(n_hidden_layers)_epochs_$(n_epochs)_trained_using_$(training_source).json"
    if isfile(weights_filename) && !retrain
        # println("Loading existing weights from file: $weights_filename")
        bpnn = load_trained_neuralbp_model(weights_filename, bpnn)
    else
        # println("Training Neural BP model for parity-check matrix from file: $parity_check_matrix_file")
        # Generate training data if not provided
        if training_errors_file == ""
            error_probability = 0.1
            (__, expected_recoveries) = generate_training_data(H, n_samples, error_probability)
            training_errors_file = "$(prefix)/training_data/training_errors_p_$(error_probability).txt"
            # Create the training_data directory if it doesn't exist
            training_data_dir = "$(prefix)/training_data"
            if !isdir(training_data_dir)
                mkdir(training_data_dir)
            end
            # Save the generated training data to a file using `DelimitedFiles.writedlm`
            writedlm(training_errors_file, expected_recoveries, ',')
        end

        # Read errors from the training errors file
        expected_recoveries = convert.(Bool, readdlm(training_errors_file, Int))
        n_samples = size(expected_recoveries, 2)
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(H * expected_recoveries, 2))
        
        # Train the Neural BP model
        train_neuralbp_enzyme!(
            bpnn, 
            training_syndromes, 
            expected_recoveries; 
            learning_rate=learning_rate, 
            n_epochs=n_epochs, 
            batch_size=batch_size,
            max_grad_norm=max_grad_norm
        )

        # Save the trained weights to a file
        save_trained_neuralbp_model(weights_filename, bpnn)
    end
    return bpnn
end

function neuralbp_test_predictions(bpnn::NeuralBP, test_errors_file::String)::BitVector
    """
    Predict the recoveries for the given test syndromes using the trained Neural BP model.
    Test these predictions to see if they match the expected recoveries.
    """
    test_errors = convert.(Bool, readdlm(test_errors_file, Int))
    test_syndromes = convert.(Bool, mod.(bpnn.base.parity_check_matrix * test_errors, 2))
    start = time()
    is_correct = predict_and_check_neuralbp(bpnn, test_syndromes, test_errors; batch_size=4096)
    runtime = time() - start
    println("[", runtime, "s] elapsed. Predicted recoveries computed and verified.")
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

function extract_parameters_from_description(description::String)::Tuple{Float64, Float64, Int}
    """
    Extract the error parameters p, q, and sample index from the `error_model_parameters_description` string.
    The description string is in the format: <prefix>/testing_data/test_ballistic_p_<p>_q_<q_mean>_s_<sample>.txt
    We want to extract the values of p, q_mean, and sample from this string.
    """
    filename = basename(description)
    m = match(r"test_ballistic_p_(\d+\.\d+)_q_(\d+\.\d+)_s_(\d+)\.txt", filename)
    if m !== nothing
        p = parse(Float64, m.captures[1])
        q = parse(Float64, m.captures[2])
        sample = parse(Int, m.captures[3])
        return (p, q, sample)
    else
        @warn ("Filename does not match expected format: $(filename). Returning default values (-1.0, -1.0, -1).")
        return (-1.0, -1.0, -1)
    end
end

function identify_best_performing_samples(neuralbp_results::DataFrame, standardbp_results::DataFrame; performance_threshold::Float64=10.0, prefix::String="./../data")::DataFrame
    """
    Identify the best performing samples where the ratio of the logical error rate of the standard decoder to the logical error rate of the neural BP decoder is higher than a given threshold.
    """
    # Create a new dataframe that contains the error parameters, sample index, and the neural BP and standard decoder logical error rates.
    # The `error_model_parameters_description` column contains a string description of the error parameters in the format
    # <prefix>/testing_data/test_ballistic_p_<p>_q_<q_mean>_s_<sample>.txt
    # So we just want to extract the basename of the file: test_ballistic_p_<p>_q_<q>_s_<sample>.txt and then extract the values of p, q, and sample from this string.
    compare_results = DataFrame(
        p = zeros(Float64, nrow(neuralbp_results)),
        q = zeros(Float64, nrow(neuralbp_results)),
        sample = zeros(Int, nrow(neuralbp_results)),
        logical_error_rate_neural = zeros(Float64, nrow(neuralbp_results)),
        logical_error_rate_standard = zeros(Float64, nrow(neuralbp_results)),
        performance_ratio = zeros(Float64, nrow(neuralbp_results)),
    )

    # Iterate over each row of the standard and neural BP results dataframes and extract the error parameters.
    for i in 1:nrow(neuralbp_results)
        description = neuralbp_results[i, :error_model_parameters_description]
        (p, q, sample) = extract_parameters_from_description(description)
        compare_results[i, :p] = p
        compare_results[i, :q] = q
        compare_results[i, :sample] = sample

        # Assign logical error rates.
        compare_results[i, :logical_error_rate_neural] = neuralbp_results[i, :average_logical_error_rate]
        
        # Search for the corresponding entry in the standard BP results dataframe with the same p, q, and sample index in `error_model_parameters_description`
        # There should be only one such entry.
        standard_entry = filter(row -> extract_parameters_from_description(row[:error_model_parameters_description]) == (p, q, sample), standardbp_results)
        if nrow(standard_entry) >= 1
            compare_results[i, :logical_error_rate_standard] = standard_entry[1, :average_logical_error_rate]
            # Set the performance ratio to be the logical error rate of the standard decoder divided by the logical error rate of the neural BP decoder.
            compare_results[i, :performance_ratio] = compare_results[i, :logical_error_rate_standard] / compare_results[i, :logical_error_rate_neural]
        else
            @warn ("No matching entry found in standard BP results for description: $(description).")
            # Set the performance ratio to -1
            compare_results[i, :performance_ratio] = -1.0
        end
    end

    # Remove all entries where the performance ratio is -1, as these correspond to entries where we could not find a matching entry in the standard BP results dataframe.
    compare_results = filter(row -> row[:performance_ratio] != -1.0, compare_results)
    
    # Save the performance comparison results to a CSV file for later analysis.
    performance_comparison_csv_file = "$(prefix)/results/performance_comparison_neuralbp_vs_standardbp.csv"
    save_decoder_dataframe(compare_results, performance_comparison_csv_file)
    println("Performance comparison saved to file: $performance_comparison_csv_file")

    # Filter the samples where the performance ratio is higher than the threshold
    best_samples = filter(row -> row[:performance_ratio] >= performance_threshold, compare_results)
    
    return best_samples
end

function collect_results()
    """
    Collect results from the Neural BP experiments and save them to a CSV file.
    """
    #=
    codename = "aps_7q_Hamm_code_data"
    per_qubit_error_probs = 0.001:0.001:0.005
    neighbour_error_probs = 0.3:0.04:0.66
    =#

    per_qubit_error_probs = [0.006, 0.008]
    neighbour_error_probs = [0.1]
    n_samples = 56
    codename = "7q_Hamm_code_data_10000_train"

    prefix = "./../data/$(codename)"
    n_hidden_layers = 100
    n_epochs = 20

    # Create the plots directory if it doesn't exist
    plots_dir = "$(prefix)/plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
    end

    # Collect results for the Neural BP decoder. If the results file already exists, load it instead of re-computing.
    output_csv_file_neural = "$(prefix)/results/decoder_statistics_ballistic.csv"
    if (isfile(output_csv_file_neural))
        # Load the dataframe from the existing CSV file
        neuralbp_results = CSV.read(output_csv_file_neural, DataFrame)
    else
        neuralbp_results = collect_decoder_statistics_for_ballistic_data(
            per_qubit_error_probs, 
            neighbour_error_probs, 
            n_samples, 
            n_hidden_layers, 
            n_epochs; 
            prefix=prefix
        )
        save_decoder_dataframe(neuralbp_results, output_csv_file_neural)
        println("Decoder statistics saved to file: $output_csv_file_neural")
    end

    # Collect results for the standard decoder. If the results file already exists, load it instead of re-computing.
    output_csv_file_standard = "$(prefix)/results/standard_decoder_statistics_ballistic.csv"
    if (isfile(output_csv_file_standard))
        # Load the dataframe from the existing CSV file
        standardbp_results = CSV.read(output_csv_file_standard, DataFrame)
    else
        standardbp_results = collect_standard_decoder_statistics_for_ballistic_data(
            prefix; 
            standard_BP_output_file="7q_Hamm_BP+OSD_failure_rates_OSD_E_order_2.txt"
        )
        save_decoder_dataframe(standardbp_results, output_csv_file_standard)
        println("Standard decoder statistics saved to file: $output_csv_file_standard")
    end

    #=
    # Plot results for the neural BP decoder
    plot_statistics_for_ballistic_error_model(
        neuralbp_results, 
        per_qubit_error_probs, 
        neighbour_error_probs; 
        prefix="$(prefix)/plots", 
        data_to_compare=standardbp_results
    )
    =#
    
    # Violin plots to show the spread of the logical error rates across different samples for a given set of error parameters.
    violin_error_parameters = [
        (0.006, 0.1),
        (0.008, 0.1)
    ]
    plot_performance_spread(
        neuralbp_results, 
        standardbp_results,
        violin_error_parameters;
        prefix="$(prefix)/plots"
    )

    # Identify the best performing samples: for which the ratio of the logical error rate of the standard decoder to the logical error rate of the neural BP decoder is higher than 10X.
    best_samples = identify_best_performing_samples(neuralbp_results, standardbp_results; performance_threshold=10.0, prefix=prefix)
    println("Best performing samples (where standard decoder performs >10X worse than neural BP):")
    println(best_samples)
    return nothing
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    """
    Run a complete experiment to train and test a Neural BP decoder.

    Example run command:
    ```sh
    julia --project="./../" neural_bp_experiments.jl --codename <codename> --n_hidden_layers <n_hidden_layers> --n_epochs <n_epochs> --batch_size <batch_size> --correlation_strengths_file <correlation_strengths_file> --train <training_errors_file> --test <testing_errors_file> --retrain <retrain> --learning_rate <learning_rate> --max_grad_norm <max_grad_norm>
    ```
    Where:
    ```
    <codename>: (String) The codename for the experiment.
    <n_hidden_layers>: (Int) Number of hidden layers in the Neural BP model.
    <n_epochs>: (Int) Number of training epochs.
    <batch_size>: (Int) Batch size for training.
    <correlation_strengths_file>: (String) File containing correlation strengths.
    <training_errors_file>: (String) File containing training errors.
    <testing_errors_file>: (String) File containing testing errors.
    <retrain>: (Bool) Whether to retrain the model (true/false).
    <learning_rate>: (Float) Learning rate for training.
    <max_grad_norm>: (Float) Maximum gradient norm for training.
    ```

    Before running this experiment, ensure that there is a directory in `./../data/` with the name corresponding to `<codename>` that contains:
    - code/ : containing `HX.txt`, `HZ.txt`, `LX.txt` and `LZ.txt`
    - correlated_weights/ : containing the correlation strengths file specified by `<correlation_strengths_file>`
    - training_data/ : containing the training errors file specified by `<training_errors_file>`
    - testing_data/ : containing the testing errors file specified by `<testing_errors_file>`

    The trained Neural BP model will be saved in `./../data/<codename>/models/` and the results of the experiment will be saved in `./../data/<codename>/results/`.
    """

    # If no arguments are provided, print a message and exit.
    if length(ARGS) == 0
        println("No command-line arguments provided. Please provide the necessary arguments to run the experiment.")
        println("Example run command:")
        println("julia --project=\"./../\" neural_bp_experiments.jl --codename hamming --n_hidden_layers 5 --n_epochs 5 --batch_size 2 --correlation_strengths_file correlation_strengths.txt --train training_errors.txt --test testing_errors.txt --retrain false --learning_rate 0.1 --max_grad_norm 2.0")
        println("Please refer to the script for details on the required and optional command-line arguments.")
        exit(1)
    end
    
    # Parse command-line arguments
	args_dict = parse_command_line_args_NN(;prefix="./../data")
    # print_arguments(args_dict; io=stdout)

    # Extract arguments
    prefix = "./../data/$(args_dict["codename"])"
    parity_check_matrix_file = "$(prefix)/code/HX.txt"
    logicals_file = "$(prefix)/code/LX.txt"
    
    # connectivity_matrix_file = "$(prefix)/code/connectivity_matrix.txt"
    # connectivity_matrix = readdlm(connectivity_matrix_file, Int)
    correlation_strengths_file = "$(prefix)/correlated_weights/$(args_dict["correlation_strengths_file"])"
    (connectivity_matrix, correlation_strengths) = parse_correlation_strengths_connectivity(correlation_strengths_file)

    n_hidden_layers = args_dict["n_hidden_layers"]
    n_epochs = args_dict["n_epochs"]
    batch_size = args_dict["batch_size"]
    training_errors_file = "$(prefix)/training_data/$(args_dict["train"])"
    n_samples = args_dict["n_samples"]

    # Hyperparameters for training the Neural BP model
    retrain = args_dict["retrain"]
    learning_rate = args_dict["learning_rate"]
    max_grad_norm = args_dict["max_grad_norm"]

    # Train the Neural BP model
    start = time()
    bpnn = train_Nachmani_neuralbp(
        parity_check_matrix_file,
        logicals_file;
        # Hyperparameters for the Neural BP model
        connectivity_matrix=connectivity_matrix,
        correlation_strengths=correlation_strengths,
        n_hidden_layers=n_hidden_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        training_errors_file=training_errors_file,
        n_samples=n_samples,
        prefix=prefix,
        retrain=retrain,
        # Hyperparameters for training the Neural BP model
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm
    )

    # Test the Neural BP model predictions
    
    # If no test file is provided, skip testing of the Neural BP model and exit.
    if args_dict["test"] == ""
        println("No test file provided. Skipping testing of the Neural BP model.")
        exit(0)
    end
    
    test_errors_file = "$(prefix)/testing_data/$(args_dict["test"])"
    is_correct = neuralbp_test_predictions(bpnn, test_errors_file)
    failures = collect(.!is_correct)

    println("Out of ", size(is_correct), " test samples, ", sum(is_correct), " were correctly decoded.")

    runtime = time() - start
    
    # Load the results on to the `DecoderStatistics` structure.
    average_correlation_strength = Float64(sum(correlation_strengths) / length(correlation_strengths))
    stats = DecoderStatistics(
        "NN",
        "ExplicitErrorModel",
        test_errors_file,
        size(is_correct, 1),
        n_hidden_layers,
        n_epochs,
        average_correlation_strength;
        num_failures = count(failures),
        failures = failures,
        runtime = runtime
    )

    # Save the decoder statistics to a CSV file for later analysis.
    results_dir = "$(prefix)/results"
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    
    # The filename to save the results is:
    results_dir = "$(prefix)/results"
    if !isdir(results_dir)
        mkdir(results_dir)
    end

    # Extract the name of the test and training files without the path and extension to include in the results file name.
    test_errors_file_name = splitext(basename(test_errors_file))[1]
    train_errors_file_name = splitext(basename(training_errors_file))[1]
    
    results_file = "$(results_dir)/simulation_results_$(test_errors_file_name)_nlayers_$(n_hidden_layers)_epochs_$(n_epochs)_trained_using_$(train_errors_file_name).csv"
    record_decoder_statistics(stats, results_file)
end
using CSV
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER

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

    per_qubit_error_probs = [0.006]
    neighbour_error_probs = [0.1]
    n_samples = 56
    codename = "90q_BB_p_0.006_q_0.1_std_0.1"

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
            standard_BP_output_file="90q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt"
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
        (0.006, 0.1)
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
    julia --project="./../" neural_bp_experiments.jl --codename hamming --n_hidden_layers 5 --correlation_strengths_file correlation_strengths.txt --train training_errors.txt --test testing_errors.txt --hyperparams default_hyperparams.json
    ```
    """

    # If no arguments are provided, print a message and exit.
    if length(ARGS) == 0
        println("No command-line arguments provided. Please provide the necessary arguments to run the experiment.")
        println("Example run command:")
        println("julia --project=\"./../\" neural_bp_experiments.jl --codename hamming --n_hidden_layers 5 --correlation_strengths_file correlation_strengths.txt --train training_errors.txt --test testing_errors.txt --hyperparams default_hyperparams.json")
        exit(1)
    end

    # Parse command-line arguments
    args_dict = parse_command_line_args_NN(;prefix="./../data")

    # Extract arguments
    prefix = "./../data/$(args_dict["codename"])"
    parity_check_matrix_file = "$(prefix)/code/HZ.txt"
    logicals_file = "$(prefix)/code/LZ.txt"
    correlation_strengths_file = "$(prefix)/correlated_weights/$(args_dict["correlation_strengths_file"])"
    training_errors_file = "$(prefix)/training_data/$(args_dict["train"])"
    n_hidden_layers = args_dict["n_hidden_layers"]
    is_debug = args_dict["debug"]
    is_quiet = args_dict["quiet"]

    # Extract hyperparameters from file or use defaults
    hyperparams_file = args_dict["hyperparams"]
    hyperparams = parse_hyper_parameters(hyperparams_file; prefix=prefix)
    n_epochs = hyperparams["n_epochs"]
    
    # Train the Neural BP model
    base = load_base_BP_model(parity_check_matrix_file, logicals_file, n_hidden_layers; correlation_strengths_file=correlation_strengths_file)
    initial_conditions = Dict{String, Vector{Float32}}(
        "weights_c2v_v2c" => random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale=hyperparams["initial_conditions_scale"]),
        "weights_llrs" => random_values_around_one([base.code_n_bits * base.n_layers]; scale=hyperparams["initial_conditions_scale"]),
        "weights_c2v_readout" => random_values_around_one([base.nb_weights_c2v_readout]; scale=hyperparams["initial_conditions_scale"])
    )
    start = time()
    bpnn = train_Nachmani_neuralbp(
        base,
        training_errors_file,
        hyperparams;
        initial_conditions=initial_conditions,
        prefix=prefix,
        is_debug=is_debug,
        is_quiet=is_quiet
    )

    # Test the Neural BP model predictions
    results_dir = "$(prefix)/results"
    if !isdir(results_dir)
        mkdir(results_dir)
    end
    
    # If no test file is provided, skip testing of the Neural BP model and exit.
    if args_dict["test"] == ""
        println("No test file provided. Skipping testing of the Neural BP model.")
        exit(0)
    end
    
    test_errors_file = "$(prefix)/testing_data/$(args_dict["test"])"
    
    # The filename to save the results is:
    training_source = splitext(basename(training_errors_file))[1]
    testing_source = splitext(basename(test_errors_file))[1]
    results_file = "$(results_dir)/simulation_results_" *
               "$(testing_source)_nlayers_" *
               "$(n_hidden_layers)_epochs_" *
               "$(n_epochs)_trained_using_" *
               "$(training_source).csv"
    
    if isfile(results_file)
        println("Results file already exists: $(results_file). Skipping testing of the Neural BP model and loading results from file.")
        results_df = collect_decoder_statistics(results_file)
        println(results_df)
        exit(0)
    end
    
    is_correct = neuralbp_test_predictions(bpnn, test_errors_file)
    failures = collect(.!is_correct)

    println("Out of ", size(is_correct), " test samples, ", sum(is_correct), " were correctly decoded.")

    runtime = time() - start

    #= #################################
                Debugging
    ################################# =#
    # Save which of the test samples were correctly decoded and which were not to a CSV file for later analysis.
    # We want to save the index of the test sample, whether it was correctly decoded or not, and the weight of the error for each test sample that failed.
    if is_debug
        test_errors = convert.(Bool, readdlm(test_errors_file, Int))
        failed_error_indices = findall(failures)
        test_samples_df = DataFrame(
            sample_index = failed_error_indices,
            error_weight = vec(sum(test_errors[:, failed_error_indices], dims=1)) # Sum the number of bit flips in each error pattern to get the error weight
        )
        test_filename = splitext(basename(test_errors_file))[1]
        test_samples_csv_file = "$(prefix)/results/failures_$(test_filename).csv"
        CSV.write(test_samples_csv_file, test_samples_df)
        println("Test sample results saved to file: $(test_samples_csv_file)")
    end
    #################################
    
    # Load the results on to the `DecoderStatistics` structure.
    stats = DecoderStatistics(
        "NN",
        "ExplicitErrorModel",
        test_errors_file,
        size(is_correct, 1),
        n_hidden_layers,
        n_epochs,
        0.0;
        num_failures = count(failures),
        failures = failures,
        runtime = runtime
    )

    # Save the decoder statistics to a CSV file for later analysis.
    results_df = record_decoder_statistics(stats, results_file)
end
#=
For batch runs, copy paste the following command in the terminal.
parallel --jobs 56 --bar '
julia --project="./../" neural_bp_experiments.jl \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 100 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_{}.txt \
  --train train_ballistic_p_0.01_q_0.001_s_{}.txt \
  --test test_ballistic_p_0.01_q_0.001_s_{}.txt
' ::: $(seq 1 56)

parallel --jobs 6 --bar 'julia --project="./../" neural_bp_experiments.jl --codename 90q_BB_p_0.010_q_0.001_std_0.01_data --n_hidden_layers 100 --hyperparams default_hyperparams.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_{}.txt --train train_ballistic_p_0.01_q_0.001_s_{}.txt' ::: $(seq 1 6)

For single runs, copy paste the following command in the terminal.
julia --project="./../" neural_bp_experiments.jl \
  --codename 90q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 100 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test test_ballistic_p_0.01_q_0.001_s_1.txt
=#
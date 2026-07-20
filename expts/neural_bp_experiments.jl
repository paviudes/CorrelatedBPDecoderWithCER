using CSV
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER

# NOTE: postprocess_neuralbp_results and collect_results moved to
# PlotsForBPDecoder/src/collect.jl (analysis-only; they need the plotting stack).

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    """
    Run a complete experiment to train and test a Neural BP decoder.

    Example run command:
    ```sh
    julia --project="./../" neural_bp_experiments.jl --workdir ./../data --codename hamming --n_hidden_layers 5 --correlation_strengths_file correlation_strengths.txt --train training_errors.txt --test testing_errors.txt --hyperparams default_hyperparams.json
    ```
    """

    # If no arguments are provided, print a message and exit.
    if length(ARGS) == 0
        println("No command-line arguments provided. Please provide the necessary arguments to run the experiment.")
        println("Example run command:")
        println("julia --project=\"./../\" neural_bp_experiments.jl --workdir ./../data --codename hamming --n_hidden_layers 5 --correlation_strengths_file correlation_strengths.txt --train training_errors.txt --test testing_errors.txt --hyperparams default_hyperparams.json")
        exit(1)
    end

    # Parse command-line arguments
    args_dict = parse_command_line_args_NN()

    # Extract arguments
    work_dir = args_dict["workdir"]
    prefix = "$(work_dir)/$(args_dict["codename"])"
    parity_check_matrix_file = "$(prefix)/code/HZ.txt"
    logicals_file = "$(prefix)/code/LZ.txt"
    correlation_strengths_file = "$(prefix)/correlated_weights/$(args_dict["correlation_strengths_file"])"
    training_errors_file = "$(prefix)/training_data/$(args_dict["train"])"
    n_hidden_layers = args_dict["n_hidden_layers"]
    n_samples = args_dict["n_samples"]
    is_debug = args_dict["isdebug"]
    is_quiet = args_dict["quiet"]

    # Extract hyperparameters from file or use defaults
    hyperparams_file = args_dict["hyperparams"]
    hyperparams = parse_hyper_parameters(hyperparams_file; prefix=prefix)
    n_epochs = hyperparams["n_epochs"]
    online_training = hyperparams["online_training"]
    
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
        is_quiet=is_quiet,
        online_training=online_training, # Set to true if you want to simulate online training by generating random batches of training samples on the fly instead of reading from a file. Note: we don't have an actual implementation for online training yet, so this will just read random batches from the training dataset to simulate the online training scenario.
        n_gradient_updates_per_epoch = hyperparams["n_gradient_updates_per_epoch"]
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
    results_file = "$(results_dir)/simulation_results_$(testing_source)_" *
               "nlayers_$(n_hidden_layers)_" *
               "epochs_$(n_epochs)_" *
               "trained_using_$(training_source).csv"
    
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

To train only without testing, copy paste the following command in the terminal.
parallel --jobs 6 --bar 'julia --project="./../" neural_bp_experiments.jl --codename 90q_BB_p_0.010_q_0.001_std_0.01_data --n_hidden_layers 100 --hyperparams default_hyperparams.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_{}.txt --train train_ballistic_p_0.01_q_0.001_s_{}.txt' ::: $(seq 1 6)

To test after training, copy paste the following command in the terminal.
After using `export USE_GPU="1"`
parallel --jobs 1 --bar 'julia --project="./../" neural_bp_experiments.jl --codename 72q_BB_p_0.010_q_0.001_std_0.01_data --n_hidden_layers 100 --hyperparams default_hyperparams.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_{}.txt --train train_ballistic_p_0.01_q_0.001_s_{}.txt --test test_ballistic_p_0.01_q_0.001_s_{}.txt' ::: $(seq 1 56)

For single runs, copy paste the following command in the terminal.
julia --project="./../" neural_bp_experiments.jl \
  --codename 90q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 100 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test test_ballistic_p_0.01_q_0.001_s_1.txt
=#
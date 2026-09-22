using CSV
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER

# Standard BP, run through the neural decoder's own forward pass with unit
# weights (see src/standard_bp.jl). Structurally this is neural_bp_experiments.jl
# with the training half removed: no training file, no epochs, no seed, no
# initial conditions, no weights on disk.
#
# `--n_iterations_BP` is the alias for `--n_hidden_layers`: for standard BP that
# quantity is the number of BP iterations. ArgParse keys both on the first name,
# so the parsed entry is "n_hidden_layers" whichever is typed.

if abspath(PROGRAM_FILE) == @__FILE__
    """
    Decode a test set with standard belief propagation.

    Example run command:
    ```sh
    julia --project="./../" standard_bp_experiments.jl --workdir ./../data \
      --codename 72q_BB_cycles_1_spread_comparison --n_iterations_BP 90 \
      --cer_data correlated_weights_p_0.0005_sig_0.001_s_1.txt \
      --test test_p_0.0005_sig_0.001_s_1.txt --diagnose true
    ```
    """

    if length(ARGS) == 0
        println("No command-line arguments provided. Please provide the necessary arguments to run the experiment.")
        println("Example run command:")
        println("julia --project=\"./../\" standard_bp_experiments.jl --workdir ./../data --codename 72q_BB_cycles_1_spread_comparison --n_iterations_BP 90 --cer_data correlated_weights_p_0.0005_sig_0.001_s_1.txt --test test_p_0.0005_sig_0.001_s_1.txt")
        exit(1)
    end

    arguments::Dict{String, Any} = parse_command_line_args_NN()

    work_directory::String = arguments["workdir"]
    prefix::String = "$(work_directory)/$(arguments["codename"])"
    parity_check_matrix_file::String = "$(prefix)/code/HZ.txt"
    logicals_file::String = "$(prefix)/code/LZ.txt"
    cer_data_file::String = "$(prefix)/correlated_weights/$(arguments["cer_data"])"
    n_iterations_bp::Int = arguments["n_hidden_layers"]
    is_debug::Bool = arguments["isdebug"]

    if arguments["test"] == ""
        println("No test file provided. Standard BP has nothing to do without one.")
        exit(1)
    end
    test_errors_file::String = "$(prefix)/testing_data/$(arguments["test"])"

    # The hyperparameters file is still read, but ONLY for the knobs that shape
    # the decoder's inputs and batching. Everything about training is ignored.
    hyperparameters::Dict{String, Any} = parse_hyper_parameters(
        arguments["hyperparams"]; prefix = prefix
    )

    use_cer::Bool = get(hyperparameters, "use_CER", true)
    cer_tag::String = ""
    if !use_cer
        cer_tag = "_no_cer"
    end

    run_tag::String = String(get(hyperparameters, "run_tag", ""))
    prior_llr_clip::Float32 = Float32(get(hyperparameters, "prior_llr_clip", 0.0f0))

    if prior_llr_clip > 0.0f0
        print_info("[prior_llr_clip=$(prior_llr_clip)] capping |initial LLR|.")
    end
    if !use_cer
        print_info("[use_CER=false] Ignoring correlated_weights/: preset p=0.1 priors, outputs tagged `_no_cer`.")
    end

    # The CER data enters standard BP exactly where it enters the neural decoder.
    # The single-qubit rates always enter through the channel LLRs. The
    # two-qubit couplings enter ONLY with `check_node = "enriched"`, where they
    # sit inside each check factor scaled by the fixed `coupling_scale_init`
    # (alpha = 1 is the Bayesian rule, alpha = 0 is the standard rule). With the
    # default "tanh" rule they play no part here at all.
    check_node::String = String(get(hyperparameters, "check_node", "tanh"))
    check_node_code(check_node)
    coupling_scale::Float32 = Float32(get(hyperparameters, "coupling_scale_init", 1.0f0))
    if check_node == "enriched"
        print_info("[check_node=enriched] couplings enter the check factor with fixed α = $(coupling_scale).")
        # The results filename carries no check-node tag (only `run_tag` may
        # extend a name), so without one an enriched run would find the tanh
        # run's results file and "skip the decode" with the wrong numbers.
        if run_tag == ""
            throw(ArgumentError(
                "check_node = \"enriched\" needs a non-empty `run_tag` in the " *
                "hyperparameters TOML (e.g. \"_cnenriched\"), or its results file " *
                "collides with the standard rule's."))
        end
    end
    # The layer schedule on α, fixed at (T₀, w) from the TOML: this is the
    # classical scan over the schedule, nothing trained. Same filename rule as
    # the check node, so a step run needs its own `run_tag`.
    coupling_schedule::String = String(get(hyperparameters, "coupling_schedule", "constant"))
    coupling_schedule_code(coupling_schedule)
    coupling_schedule_layer::Float32 =
        Float32(get(hyperparameters, "coupling_schedule_layer_init", Float32(n_iterations_bp)))
    coupling_schedule_width::Float32 =
        Float32(get(hyperparameters, "coupling_schedule_width_init", 3.0f0))
    if coupling_schedule == "step"
        print_info("[coupling_schedule=step] α_t = α · d(t), fixed step at layer T₀ = $(coupling_schedule_layer), " *
                   "width w = $(coupling_schedule_width) layers.")
        if run_tag == ""
            throw(ArgumentError(
                "coupling_schedule = \"step\" needs a non-empty `run_tag` in the " *
                "hyperparameters TOML (e.g. \"_sch12w3F\"), or its results file " *
                "collides with the constant schedule's."))
        end
    end
    base::NeuralBPBase = load_base_BP_model(
        parity_check_matrix_file,
        logicals_file,
        n_iterations_bp;
        cer_data_file = cer_data_file,
        use_cer = use_cer,
        prior_llr_clip = prior_llr_clip,
        single_qubit_rescale = Float32(get(hyperparameters, "single_qubit_rescale", 0.0f0)),
        require_correlations = Bool(get(hyperparameters, "require_correlations", false)),
        check_node = check_node,
        coupling_schedule = coupling_schedule,
    )

    results_directory::String = "$(prefix)/results"
    if !isdir(results_directory)
        mkdir(results_directory)
    end

    # No `trained_using_`, no `epochs_`, no seed: standard BP is deterministic and
    # has no training set, so none of those identify a run. `standard_bp` in the
    # name also keeps these files out of the neural sweep collector's glob.
    testing_source::String = splitext(basename(test_errors_file))[1]
    results_file::String = "$(results_directory)/simulation_results_$(testing_source)_" *
                           "standard_bp_iters_$(n_iterations_bp)$(cer_tag)$(run_tag).csv"

    if isfile(results_file)
        println("Results file already exists: $(results_file). Skipping the decode and loading results from file.")
        existing_results::DataFrame = collect_decoder_statistics(results_file)
        print_simulation_results_to_console(existing_results)
        exit(0)
    end

    diagnose::Bool = arguments["diagnose"]
    start_time::Float64 = time()
    prediction_outcome::Union{BitVector, NamedTuple} = standard_bp_test_predictions(
        base,
        test_errors_file;
        coupling_scale = coupling_scale,
        coupling_schedule_layer = coupling_schedule_layer,
        coupling_schedule_width = coupling_schedule_width,
        batch_size = Int(get(hyperparameters, "prediction_batch_size", 0)),
        gpu_memory = String(get(hyperparameters, "gpu_memory", "")),
        diagnose = diagnose
    )

    # `prediction_outcome` is a BitVector when diagnose = false and a NamedTuple
    # when it is true, so `is_correct` must be assigned INSIDE the branch. The
    # annotation is declared against an empty BitVector first: writing
    # `is_correct::BitVector = prediction_outcome` unconditionally makes Julia
    # convert the Union eagerly, which throws on the NamedTuple in diagnostic mode.
    diagnosis::Union{NamedTuple, Nothing} = nothing
    is_correct::BitVector = falses(0)
    if diagnose
        diagnosis = prediction_outcome
        is_correct = diagnosis.is_correct
    else
        is_correct = prediction_outcome
    end
    failures::Vector{Bool} = collect(.!is_correct)
    runtime::Float64 = time() - start_time

    if is_debug
        test_errors::BitMatrix = convert.(Bool, readdlm(test_errors_file, Int))
        failed_error_indices::Vector{Int} = findall(failures)
        failure_detail::DataFrame = DataFrame(
            sample_index = failed_error_indices,
            error_weight = vec(sum(test_errors[:, failed_error_indices], dims = 1))
        )
        failure_detail_file::String =
            "$(results_directory)/failures_standard_bp_$(testing_source).csv"
        CSV.write(failure_detail_file, failure_detail)
        println("Test sample results saved to file: $(failure_detail_file)")
    end

    # `n_epochs = 0` because nothing was trained. The field is kept so standard-BP
    # and neural results share one schema and one reader.
    statistics::NeuralBPDecoderStatistics = NeuralBPDecoderStatistics(
        "BP",
        "ExplicitErrorModel",
        test_errors_file,
        size(is_correct, 1),
        n_iterations_bp,
        0;
        num_failures = count(failures),
        failures = failures,
        runtime = runtime
    )

    extra_result_columns::Vector{Pair{String, Any}} = Pair{String, Any}[]
    push!(extra_result_columns, "check_node" => check_node)
    push!(extra_result_columns, "coupling_scale" => coupling_scale)
    push!(extra_result_columns, "coupling_schedule" => coupling_schedule)
    push!(extra_result_columns, "coupling_schedule_layer" => coupling_schedule_layer)
    push!(extra_result_columns, "coupling_schedule_width" => coupling_schedule_width)
    if diagnosis !== nothing
        push!(extra_result_columns, "num_syndrome_cleared" => diagnosis.n_syndrome_cleared)
        push!(extra_result_columns, "num_coset_failures" => diagnosis.n_coset_failures)
        push!(extra_result_columns, "num_convergence_failures" => diagnosis.n_convergence_failures)
        push!(extra_result_columns, "mean_committed_layer" => mean_committed_layer(diagnosis))
    end
    results_dataframe::DataFrame = record_decoder_statistics(
        statistics, results_file; extra_columns = extra_result_columns
    )

    if diagnosis !== nothing
        written_files::NamedTuple = write_failure_diagnostics(diagnosis, results_file)
        print_info("Failure detail  -> $(basename(written_files.failures_file)) " *
                   "($(written_files.n_failure_rows) rows: " *
                   "$(diagnosis.n_coset_failures) coset, $(diagnosis.n_convergence_failures) convergence)")
        print_info("Layer histogram -> $(basename(written_files.layer_profile_file)) " *
                   "($(diagnosis.n_layers) iterations; covers the $(diagnosis.n_correct) successes)")
    end
    print_simulation_results_to_console(results_dataframe)
end
#=
Batch runs mirror the neural driver, minus the training arguments:

parallel --jobs 8 --bar '
julia --project="./../" standard_bp_experiments.jl \
  --codename 72q_BB_cycles_1_spread_comparison \
  --n_iterations_BP 90 \
  --hyperparams hyperparams_epochs_5_corrs.toml \
  --cer_data correlated_weights_p_0.0005_sig_0.001_s_{}.txt \
  --test test_p_0.0005_sig_0.001_s_{}.txt \
  --diagnose true
' ::: $(seq 1 3)

Set `export USE_GPU="1"` to decode on the GPU; the batch sizer and the device
selection are shared with the neural path and need nothing extra here.
=#

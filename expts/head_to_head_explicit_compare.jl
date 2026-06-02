using ArgParse
using Dates
using DelimitedFiles
using JSON
using Random

using CorrelatedBPDecoderWithCER

const DATASETS = Dict(
    "72q" => (
        subdir = "72q_BB_p_0.006_q_0.1_std_0.1_data",
        code_dir = "72q_BB_code",
        default_sample = 52,
        bp_osd_results_file = "72q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    ),
    "90q" => (
        subdir = "90q_BB_p_0.006_q_0.1_std_0.1_data",
        code_dir = "90q_BB_code",
        default_sample = 36,
        bp_osd_results_file = "90q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    ),
)


function parse_args()
    settings = ArgParseSettings(
        description = "Train and evaluate the Julia neural BP decoder on one explicit Standard_BP_OSD sample.",
    )
    @add_arg_table! settings begin
        "--dataset"
            help = "Dataset alias: 72q or 90q."
            default = "72q"
        "--sample"
            help = "Sample index within the dataset. Use 0 to pick the dataset default."
            arg_type = Int
            default = 0
        "--n-layers"
            help = "Number of unfolded BP layers."
            arg_type = Int
            default = 50
        "--n-epochs"
            help = "Number of training epochs."
            arg_type = Int
            default = 5
        "--batch-size"
            help = "Training batch size."
            arg_type = Int
            default = 100
        "--eval-batch-size"
            help = "Evaluation batch size."
            arg_type = Int
            default = 1024
        "--max-train-samples"
            help = "If positive, limit the training matrix to its first N columns."
            arg_type = Int
            default = 0
        "--max-test-samples"
            help = "If positive, limit the test matrix to its first N columns."
            arg_type = Int
            default = 0
        "--seed"
            help = "Random seed."
            arg_type = Int
            default = 0
        "--run-label"
            help = "Optional suffix to keep multiple comparison runs separate."
            arg_type = String
            default = ""
        "--initial-weights-file"
            help = "Optional JSON file containing initial weights in the Julia flat format."
            arg_type = String
            default = ""
        "--initial-conditions-scale"
            help = "Uniform initialization half-width around 1.0."
            arg_type = Float64
            default = 0.3
        "--learning-rate"
            help = "Adam/AdamW learning rate."
            arg_type = Float64
            default = 1e-1
        "--weight-decay"
            help = "AdamW weight decay."
            arg_type = Float64
            default = 1e-4
        "--max-grad-norm"
            help = "Gradient clipping threshold."
            arg_type = Float64
            default = 2.0
        "--adam-eps"
            help = "Adam epsilon."
            arg_type = Float64
            default = 1e-4
        "--nan-skip-count"
            help = "Maximum non-finite batch skips before rolling back an epoch."
            arg_type = Int
            default = 5
        "--warmup-layers"
            help = "Initial layers excluded from the loss."
            arg_type = Int
            default = 10
        "--loss-layer-temperature-min"
            arg_type = Float64
            default = 0.1
        "--loss-layer-temperature-max"
            arg_type = Float64
            default = 5.0
        "--loss-layer-temperature-decay"
            arg_type = Float64
            default = 0.9
        "--correlation-importance-min"
            arg_type = Float64
            default = 0.1
        "--correlation-importance-max"
            arg_type = Float64
            default = 1.0
        "--correlation-importance-decay"
            arg_type = Float64
            default = 0.1
        "--llr-certainty-importance-min"
            arg_type = Float64
            default = 0.001
        "--llr-certainty-importance-max"
            arg_type = Float64
            default = 0.01
        "--llr-certainty-importance-decay"
            arg_type = Float64
            default = 0.1
        "--sparsity-importance-min"
            arg_type = Float64
            default = 0.0
        "--sparsity-importance-max"
            arg_type = Float64
            default = 0.01
        "--sparsity-importance-decay"
            arg_type = Float64
            default = 0.5
    end
    return ArgParse.parse_args(settings)
end


function build_paths(dataset::String, spec, sample::Int)
    pavi_root = normpath(joinpath(@__DIR__, ".."))
    dataset_root = joinpath(pavi_root, "Standard_BP_OSD", spec.subdir)
    code_dir = joinpath(dataset_root, spec.code_dir)
    return (
        dataset_root = dataset_root,
        parity_check = joinpath(code_dir, "HZ.txt"),
        logicals = joinpath(code_dir, "LZ.txt"),
        train_errors = joinpath(
            dataset_root,
            "training_data",
            "train_ballistic_p_0.006_q_0.1_s_$(sample).txt",
        ),
        test_errors = joinpath(
            dataset_root,
            "testing_data",
            "test_ballistic_p_0.006_q_0.1_s_$(sample).txt",
        ),
        correlated_weights = joinpath(
            dataset_root,
            "correlated_weights",
            "correlated_weights_p_0.006_q_0.1_s_$(sample).txt",
        ),
        bp_osd_results = joinpath(dataset_root, spec.bp_osd_results_file),
    )
end


function load_binary_subset(path::String, max_samples::Int)
    matrix = Int.(readdlm(path, Int))
    total_samples = size(matrix, 2)
    if max_samples > 0
        matrix = matrix[:, 1:min(max_samples, total_samples)]
    end
    return matrix, total_samples
end


function argget(args::Dict{String, Any}, key::String)
    if haskey(args, key)
        return args[key]
    end
    return args[replace(key, "_" => "-")]
end


function build_hyperparameters(args, effective_warmup::Int)
    return Dict{String, Any}(
        "n_epochs" => argget(args, "n_epochs"),
        "batch_size" => argget(args, "batch_size"),
        "learning_rate" => Float32(argget(args, "learning_rate")),
        "weight_decay" => Float32(argget(args, "weight_decay")),
        "max_grad_norm" => Float32(argget(args, "max_grad_norm")),
        "adam_eps" => Float32(argget(args, "adam_eps")),
        "nanskip" => argget(args, "nan_skip_count"),
        "warmup_layers" => effective_warmup,
        "loss_layer_temperature" => Dict(
            "min" => Float32(argget(args, "loss_layer_temperature_min")),
            "max" => Float32(argget(args, "loss_layer_temperature_max")),
            "decay" => Float32(argget(args, "loss_layer_temperature_decay")),
            "direction" => :down,
        ),
        "correlation_importance" => Dict(
            "min" => Float32(argget(args, "correlation_importance_min")),
            "max" => Float32(argget(args, "correlation_importance_max")),
            "decay" => Float32(argget(args, "correlation_importance_decay")),
            "direction" => :down,
        ),
        "llr_certainty_importance" => Dict(
            "min" => Float32(argget(args, "llr_certainty_importance_min")),
            "max" => Float32(argget(args, "llr_certainty_importance_max")),
            "decay" => Float32(argget(args, "llr_certainty_importance_decay")),
            "direction" => :down,
        ),
        "sparsity_importance" => Dict(
            "min" => Float32(argget(args, "sparsity_importance_min")),
            "max" => Float32(argget(args, "sparsity_importance_max")),
            "decay" => Float32(argget(args, "sparsity_importance_decay")),
            "direction" => :up,
        ),
    )
end


function evaluate_model(bpnn::NachmaniNeuralBP, syndromes::BitMatrix, errors::BitMatrix; batch_size::Int)
    n_samples = size(errors, 2)
    parity_validation = Int.(bpnn.base.parity_check_matrix)
    dual_validation = Int.(bpnn.base.parity_check_matrix_dual)
    parity_correct = 0
    dual_correct = 0

    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)
        chunk_syndromes = syndromes[:, start:stop]
        chunk_errors = errors[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, size(chunk_syndromes, 2))
        posterior = forward_pass_gpu(bpnn, chunk_llrs, chunk_syndromes)
        recoveries = Array(posterior .< 0)

        parity_correct += count(
            check_bp_solutions(parity_validation, chunk_errors, recoveries)
        )
        dual_correct += count(
            check_bp_solutions(dual_validation, chunk_errors, recoveries)
        )
    end

    return Dict(
        "parity_n_correct" => parity_correct,
        "parity_success_rate" => parity_correct / n_samples,
        "dual_n_correct" => dual_correct,
        "dual_success_rate" => dual_correct / n_samples,
    )
end


function load_bp_osd_reference(
    path::String,
    sample::Int,
    test_samples_used::Int,
    full_test_samples::Int,
)
    if !isfile(path)
        return nothing
    end

    for line in eachline(path)
        stripped = strip(line)
        if isempty(stripped) || startswith(stripped, "#")
            continue
        end
        fields = split(stripped)
        if length(fields) < 4
            continue
        end
        if parse(Int, fields[3]) == sample
            failures = parse(Int, fields[4])
            return Dict(
                "results_file" => path,
                "sample" => sample,
                "full_test_samples" => full_test_samples,
                "logical_failures" => failures,
                "success_rate" => 1.0 - failures / full_test_samples,
                "comparable_to_full_test_run" => test_samples_used == full_test_samples,
            )
        end
    end
    return nothing
end


function main()
    args = parse_args()
    dataset = argget(args, "dataset")
    if !haskey(DATASETS, dataset)
        error("Unsupported dataset alias: $(dataset)")
    end

    spec = DATASETS[dataset]
    sample = argget(args, "sample") <= 0 ? spec.default_sample : argget(args, "sample")
    n_layers = argget(args, "n_layers")
    if n_layers < 2
        error("n-layers must be at least 2 so at least one loss layer remains.")
    end

    Random.seed!(argget(args, "seed"))
    paths = build_paths(dataset, spec, sample)

    train_errors_int, total_train_samples = load_binary_subset(
        paths.train_errors,
        argget(args, "max_train_samples"),
    )
    test_errors_int, total_test_samples = load_binary_subset(
        paths.test_errors,
        argget(args, "max_test_samples"),
    )
    train_errors = Bool.(train_errors_int)
    test_errors = Bool.(test_errors_int)

    parity_check = Int.(readdlm(paths.parity_check, Int))
    train_syndromes = Bool.(mod.(parity_check * train_errors_int, 2))
    test_syndromes = Bool.(mod.(parity_check * test_errors_int, 2))

    base = load_base_BP_model(
        paths.parity_check,
        paths.logicals,
        n_layers;
        correlation_strengths_file = paths.correlated_weights,
    )

    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c = random_values_around_one(
            [base.nb_weights_c2v_v2c * base.n_layers];
            scale = Float32(argget(args, "initial_conditions_scale")),
        ),
        weights_llrs = random_values_around_one(
            [base.code_n_bits * base.n_layers];
            scale = Float32(argget(args, "initial_conditions_scale")),
        ),
        weights_c2v_readout = random_values_around_one(
            [base.nb_weights_c2v_readout];
            scale = Float32(argget(args, "initial_conditions_scale")),
        ),
    )
    initial_weights_file = argget(args, "initial_weights_file")
    if !isempty(initial_weights_file)
        bpnn = load_trained_neuralbp_model(initial_weights_file, bpnn)
        Random.seed!(argget(args, "seed"))
    end

    effective_warmup = min(argget(args, "warmup_layers"), n_layers - 1)
    hyperparameters = build_hyperparameters(args, effective_warmup)

    models_dir = joinpath(paths.dataset_root, "models")
    results_dir = joinpath(paths.dataset_root, "results")
    mkpath(models_dir)
    mkpath(results_dir)

    run_tag = "$(dataset)_sample_$(sample)_nlayers_$(n_layers)_epochs_$(argget(args, "n_epochs"))"
    run_label = argget(args, "run_label")
    if !isempty(run_label)
        run_tag *= "_$(run_label)"
    end
    weights_path = joinpath(models_dir, "head_to_head_julia_$(run_tag).json")
    summary_path = joinpath(results_dir, "head_to_head_julia_$(run_tag).json")
    debug_logfile = joinpath(models_dir, "head_to_head_julia_$(run_tag)_debug")

    train_start = time()
    train_neuralbp_enzyme!(
        bpnn,
        train_syndromes,
        train_errors,
        hyperparameters;
        debugging_logfile = debug_logfile,
    )
    training_time_s = time() - train_start

    save_trained_neuralbp_model(weights_path, bpnn)

    eval_start = time()
    evaluation = evaluate_model(
        bpnn,
        test_syndromes,
        test_errors;
        batch_size = argget(args, "eval_batch_size"),
    )
    evaluation_time_s = time() - eval_start

    payload = Dict(
        "implementation" => "julia",
        "script" => @__FILE__,
        "dataset" => dataset,
        "dataset_root" => paths.dataset_root,
        "sample" => sample,
        "seed" => argget(args, "seed"),
        "n_layers" => n_layers,
        "n_epochs" => argget(args, "n_epochs"),
        "run_label" => run_label,
        "batch_size" => argget(args, "batch_size"),
        "eval_batch_size" => argget(args, "eval_batch_size"),
        "initial_conditions_scale" => argget(args, "initial_conditions_scale"),
        "initial_weights_file" => initial_weights_file,
        "learning_rate" => argget(args, "learning_rate"),
        "weight_decay" => argget(args, "weight_decay"),
        "max_grad_norm" => argget(args, "max_grad_norm"),
        "adam_eps" => argget(args, "adam_eps"),
        "warmup_layers_requested" => argget(args, "warmup_layers"),
        "warmup_layers_used" => effective_warmup,
        "train_samples_used" => size(train_errors, 2),
        "train_samples_available" => total_train_samples,
        "test_samples_used" => size(test_errors, 2),
        "test_samples_available" => total_test_samples,
        "training_time_s" => training_time_s,
        "evaluation_time_s" => evaluation_time_s,
        "weights_file" => weights_path,
        "parity_n_correct" => evaluation["parity_n_correct"],
        "parity_success_rate" => evaluation["parity_success_rate"],
        "dual_n_correct" => evaluation["dual_n_correct"],
        "dual_success_rate" => evaluation["dual_success_rate"],
        "bp_osd_reference" => load_bp_osd_reference(
            paths.bp_osd_results,
            sample,
            size(test_errors, 2),
            total_test_samples,
        ),
        "timestamp_local" => string(Dates.now()),
    )

    open(summary_path, "w") do io
        JSON.print(io, payload, 2)
        println(io)
    end
    JSON.print(stdout, payload, 2)
    println()
    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

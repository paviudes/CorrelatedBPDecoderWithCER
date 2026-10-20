# ============================================================================
# batch_commands.jl — dataset keys, the commands file, and backend dispatch
# ============================================================================
#
# DATASET NAMES. Every input of one dataset lives under <workdir>/<codename>/:
#
#     training_data/train_<key>.txt
#     testing_data/test_<key>.txt
#     correlated_weights/correlated_weights_<key>.txt
#
# with  <key> = p_<p>_sig_<sigma>_s_<sample>, e.g. p_0.0015_sig_0.003_s_1. The
# numbers are written the way the Python data generator writes them (its repr):
# 0.0015, 0.003, 0.0. This replaces the old "Ising" (p_..._q_..., zero-padded)
# and "Circuit" (p_...) conventions, which this pipeline no longer reads.
#
# A submission TRAINS on one dataset and TESTS every trained model on a list of
# datasets. Each is named by a (p, sigma) pair, sigma being a number or a
# multiple of p ("0", "p", "2p"), plus a sample index shared by the list.
#
# Exposes:
#   python_float_text, resolve_sigma, dataset_key
#   DatasetPair, parse_dataset_pair, dataset_pairs, pair_dataset_key, test_dataset_keys
#   ExperimentCommand                 one julia neural_bp_experiments.jl invocation
#   missing_dataset_inputs            submit-time check that every input exists
#   expected_weights_name             the weights file a command trains or loads
#   generate_parallel_commands        write commands_<mode>_<ts>.txt and dispatch it
#
# `include`d by batch_run.jl, which also `include`s the three backend files.
# Deliberately free of `using CorrelatedBPDecoderWithCER` — see batch_run.jl.
# ============================================================================

using Printf
using Dates      # timestamped commands filenames, so concurrent submissions can't collide
import TOML

"""
    python_float_text(value) -> String

The text Python's `repr` gives `value`, which is how the data generator names
its files: `0.0015`, `0.003`, `0.0`, and `1e-05` below 1e-4. Julia's shortest
round-trip printing agrees with Python's digit for digit; only the exponent form
differs (Julia `1.0e-5`, Python `1e-05`), so that is rewritten.
"""
function python_float_text(value::Float64)::String
    julia_text::String = string(value)
    if !occursin('e', julia_text)
        return julia_text
    end
    exponent_parts::Vector{SubString{String}} = split(julia_text, 'e')
    mantissa_text::String = String(exponent_parts[1])
    if endswith(mantissa_text, ".0")
        mantissa_text = mantissa_text[1:end-2]
    end
    exponent_value::Int = parse(Int, exponent_parts[2])
    python_text::String = mantissa_text * "e" * @sprintf("%+03d", exponent_value)
    return python_text
end

"""
    resolve_sigma(sigma_spec, p_value) -> Float64

`sigma_spec` is a number, used as it is, or text: a number such as "0" or
"0.003", or a multiple of p written "p", "2p", "0.5p". The product is computed
as the data generator computes it (coefficient × p in Float64), so "2p" at
p = 0.0007 names the file `..._sig_0.0014_...`.
"""
function resolve_sigma(sigma_spec::Union{Real, AbstractString}, p_value::Float64)::Float64
    sigma_value::Float64 = 0.0
    if sigma_spec isa Real
        sigma_value = Float64(sigma_spec)
    else
        sigma_text::String = lowercase(strip(String(sigma_spec)))
        parsed_value::Union{Float64, Nothing} = nothing
        if endswith(sigma_text, "p")
            coefficient_text::String = strip(sigma_text[1:end-1])
            coefficient::Union{Float64, Nothing} = 1.0
            if !isempty(coefficient_text)
                coefficient = tryparse(Float64, coefficient_text)
            end
            if coefficient !== nothing
                parsed_value = coefficient * p_value
            end
        else
            parsed_value = tryparse(Float64, sigma_text)
        end
        if parsed_value === nothing
            throw(ArgumentError("sigma $(repr(sigma_spec)): write a number (\"0\", \"0.003\") or a " *
                                "multiple of p (\"p\", \"2p\", \"0.5p\")."))
        end
        sigma_value = parsed_value
    end
    if sigma_value < 0.0
        throw(ArgumentError("sigma $(repr(sigma_spec)) at p = $(p_value) is negative."))
    end
    return sigma_value
end

"""
    dataset_key(p_value, sigma_value, sample_index) -> String

`p_<p>_sig_<sigma>_s_<sample>`, the key every input file of a dataset carries.
"""
function dataset_key(p_value::Float64, sigma_value::Float64, sample_index::Int)::String
    key::String = "p_$(python_float_text(p_value))_sig_$(python_float_text(sigma_value))_s_$(sample_index)"
    return key
end

"""
One dataset named by its noise parameters: p, and sigma as written — a number,
or a multiple of p ("p", "2p") that `resolve_sigma` turns into one.
"""
const DatasetPair = Tuple{Float64, Union{Float64, String}}

"""
    parse_dataset_pair(pair_spec, setting_name) -> DatasetPair

One (p, sigma) pair as the settings file writes it, a two-element array
`[0.0015, "2p"]`, or as the command line does, the text `0.0015,2p`. p is a
positive number; sigma is a number or a multiple of p. The sigma is resolved
here once, so a typo fails before anything is written. `setting_name` names the
setting in error messages.
"""
function parse_dataset_pair(pair_spec::Any, setting_name::String)::DatasetPair
    pair_items::Vector{Any} = Any[]
    if pair_spec isa AbstractVector
        pair_items = collect(Any, pair_spec)
    elseif pair_spec isa AbstractString && occursin(',', pair_spec)
        pair_items = Any[String(strip(part)) for part in split(pair_spec, ',')]
    end
    if length(pair_items) != 2
        throw(ArgumentError("$(setting_name): $(repr(pair_spec)) is not a (p, sigma) pair. Write it as " *
                            "[0.0015, \"2p\"] in the settings file, or 0.0015,2p on the command line."))
    end

    p_item::Any = pair_items[1]
    p_value::Union{Float64, Nothing} = nothing
    if p_item isa Real
        p_value = Float64(p_item)
    elseif p_item isa AbstractString
        p_value = tryparse(Float64, strip(p_item))
    end
    if p_value === nothing || p_value <= 0.0
        throw(ArgumentError("$(setting_name): p in $(repr(pair_spec)) must be a positive number."))
    end

    sigma_item::Any = pair_items[2]
    sigma_spec::Union{Float64, String} = 0.0
    if sigma_item isa Real
        sigma_spec = Float64(sigma_item)
    elseif sigma_item isa AbstractString
        sigma_spec = String(strip(sigma_item))
    else
        throw(ArgumentError("$(setting_name): sigma in $(repr(pair_spec)) must be a number or a multiple of p " *
                            "(\"0\", \"p\", \"2p\")."))
    end
    resolve_sigma(sigma_spec, p_value)

    parsed_pair::DatasetPair = (p_value, sigma_spec)
    return parsed_pair
end

"""
    dataset_pairs(pairs_spec, setting_name) -> Vector{DatasetPair}

A list of (p, sigma) pairs: `[[0.0001, 0], [0.0001, "p"], ...]` in the settings
file, or `0.0001,0 0.0001,p ...` on the command line. Only the listed pairs are
used — no p is combined with another pair's sigma.
"""
function dataset_pairs(pairs_spec::Any, setting_name::String)::Vector{DatasetPair}
    if !(pairs_spec isa AbstractVector)
        throw(ArgumentError("$(setting_name) must be a list of (p, sigma) pairs, e.g. " *
                            "$(setting_name) = [[0.0001, 0], [0.0001, \"p\"]]; got $(repr(pairs_spec))."))
    end
    parsed_pairs::Vector{DatasetPair} = DatasetPair[]
    for pair_spec in pairs_spec
        is_pair_shaped::Bool = pair_spec isa AbstractVector || pair_spec isa AbstractString
        if !is_pair_shaped
            throw(ArgumentError("$(setting_name): every entry must be a [p, sigma] pair, and " *
                                "$(repr(pair_spec)) is not. One pair still needs the outer brackets: " *
                                "$(setting_name) = [[0.0015, \"2p\"]]."))
        end
        push!(parsed_pairs, parse_dataset_pair(pair_spec, setting_name))
    end
    return parsed_pairs
end

"""
    pair_dataset_key(pair, sample_index) -> String

The key of the dataset `pair` names at sample index `sample_index`, e.g.
(0.0015, "2p") at 1 -> `p_0.0015_sig_0.003_s_1`.
"""
function pair_dataset_key(pair::DatasetPair, sample_index::Int)::String
    key::String = dataset_key(pair[1], resolve_sigma(pair[2], pair[1]), sample_index)
    return key
end

"""
    test_dataset_keys(test_pairs, test_sample) -> Vector{String}

The key of every test pair at sample index `test_sample`, in the order given.
Two spellings of one dataset (sigma "0" and 0.0, or "p" and the number) are
kept once.
"""
function test_dataset_keys(test_pairs::AbstractVector{DatasetPair}, test_sample::Int)::Vector{String}
    keys_in_order::Vector{String} = String[]
    for pair in test_pairs
        key::String = pair_dataset_key(pair, test_sample)
        if !(key in keys_in_order)
            push!(keys_in_order, key)
        end
    end
    return keys_in_order
end

"""
One `julia neural_bp_experiments.jl` invocation. `test_file` is empty for a
training command. `seed` is passed as `--seed` (it beats the TOML's) unless it
is `nothing`, in which case the hyperparameters file's own seed applies.
"""
struct ExperimentCommand
    arm::String                    # "cer" or "nocer"
    hyperparams_file::String       # under models/
    cer_file::String               # under correlated_weights/
    train_file::String             # under training_data/; names the model at test time too
    test_file::String              # under testing_data/; "" for a training command
    seed::Union{Int, Nothing}
end

"""
    command_text(command; ...) -> String

The command line for one invocation. Training lines carry `--isdebug true`
(logs/debugging_*.csv: per-batch losses, NaN skips — what
misc/check_training_health.sh reads); test lines carry `--diagnose true` (coset
vs convergence failures, and the per-sample failure files).
"""
function command_text(command::ExperimentCommand; julia_project::String, heap_size_hint::String,
                      working_dir::String, codename::String, n_hidden_layers::Int)::String
    parts::Vector{String} = [
        "julia --project=\"$(julia_project)\" --heap-size-hint=$(heap_size_hint) neural_bp_experiments.jl",
        "--workdir $(working_dir)",
        "--codename $(codename)",
        "--n_hidden_layers $(n_hidden_layers)",
        "--hyperparams $(command.hyperparams_file)",
        "--cer_data $(command.cer_file)",
        "--quiet true",
    ]
    if command.seed !== nothing
        push!(parts, "--seed $(command.seed)")
    end
    if isempty(command.test_file)
        push!(parts, "--isdebug true")
        push!(parts, "--train $(command.train_file)")
    else
        push!(parts, "--diagnose true")
        push!(parts, "--train $(command.train_file)")
        push!(parts, "--test $(command.test_file)")
    end
    line::String = join(parts, " ")
    return line
end

"""
    required_input_files(command; bethe_tables) -> Vector{String}

Paths, relative to the dataset directory, that `command` reads: the training
file for a training command, the test file for a test command (which only names
the training file, to find the model), the CER file when the arm uses CER, and
with `bethe_tables` (an enriched node on the "bethe" table) the CER arm's
`bethe_table_<key>.txt` next to its CER file, as src/soft_constraints.jl's
`bethe_table_path` names it.
"""
function required_input_files(command::ExperimentCommand; bethe_tables::Bool=false)::Vector{String}
    required_files::Vector{String} = String[]
    if isempty(command.test_file)
        push!(required_files, joinpath("training_data", command.train_file))
    else
        push!(required_files, joinpath("testing_data", command.test_file))
    end
    if command.arm == "cer"
        push!(required_files, joinpath("correlated_weights", command.cer_file))
        if bethe_tables
            table_file::String = "bethe_table_" * replace(command.cer_file, r"^correlated_weights_" => "")
            push!(required_files, joinpath("correlated_weights", table_file))
        end
    end
    return required_files
end

"""
    missing_dataset_inputs(dataset_dir, commands; bethe_tables) -> Vector{String}

Every input some command reads that is not under `dataset_dir`, each followed by
the files that do exist for the same p when there are any — the usual cause is
a sigma written differently from the generator's file names.
"""
function missing_dataset_inputs(dataset_dir::String, commands::AbstractVector{ExperimentCommand};
                                bethe_tables::Bool=false)::Vector{String}
    missing_reports::Vector{String} = String[]
    if isempty(commands)
        return missing_reports
    end
    required_files::Vector{String} =
        unique(reduce(vcat, [required_input_files(command; bethe_tables=bethe_tables) for command in commands]))
    for relative_path in required_files
        if isfile(joinpath(dataset_dir, relative_path))
            continue
        end
        report::String = relative_path
        # Files of the same p, if any: "testing_data/test_p_0.0007_sig_" etc.
        p_prefix_match::Union{RegexMatch, Nothing} = match(r"^(.*_p_[^_]+_sig_)", basename(relative_path))
        folder::String = joinpath(dataset_dir, dirname(relative_path))
        if p_prefix_match !== nothing && isdir(folder)
            same_p_files::Vector{String} = filter(name -> startswith(name, p_prefix_match.captures[1]), readdir(folder))
            if !isempty(same_p_files)
                report *= "   (present for this p: " * join(first(same_p_files, 6), ", ") * ")"
            end
        end
        push!(missing_reports, report)
    end
    return missing_reports
end

"""
    expected_weights_name(n_hidden_layers, n_epochs, train_file, use_cer, run_tag, seed) -> String

The weights file a training command writes and a test command loads, as
`train_Nachmani_neuralbp` (src/train.jl) names it:
    neuralbp_weights_nlayers_<L>_epochs_<E>_trained_using_<training source><cer_tag><run_tag><seed_tag>.json
with `_no_cer` iff `use_cer` is false and `_seed_<n>` iff a seed is set (the
command's --seed, else the hyperparameters file's). Duplicated rather than
imported, like everything in this file: keep the two in sync.
"""
function expected_weights_name(n_hidden_layers::Int, n_epochs::Int, train_file::String, use_cer::Bool,
                               run_tag::String, seed::Union{Int, Nothing})::String
    cer_tag::String = ""
    if !use_cer
        cer_tag = "_no_cer"
    end
    seed_tag::String = ""
    if seed !== nothing
        seed_tag = "_seed_$(seed)"
    end
    training_source::String = splitext(basename(train_file))[1]
    weights_filename::String =
        "neuralbp_weights_nlayers_$(n_hidden_layers)_epochs_$(n_epochs)_" *
        "trained_using_$(training_source)$(cer_tag)$(run_tag)$(seed_tag).json"
    return weights_filename
end

"""
    generate_parallel_commands(commands, codename; ...) -> String

Write <working_dir>/<codename>/cluster/commands_<mode>_<timestamp>.txt, one line
per command, and hand it to the backend `cluster_backend` names. `model_names`
(test jobs) is the weights file each command loads, in the same order; it is
written alongside as models_<mode>_<timestamp>.txt, and every SLURM task checks
its slice of it after the stage-in. Returns the path of the script the backend
wrote.
"""
function generate_parallel_commands(
    commands::AbstractVector{ExperimentCommand},
    codename::String;
    n_hidden_layers::Int=90,
    heap_size_hint::String="4G",
    julia_project::String="./../",
    output_file::String="simulation_results.log",
    working_dir::String=joinpath(@__DIR__, ".."),
    ncpus::Int=12,
    mem_per_cpu::String="4G",
    max_nodes::Int=8,
    wall_time::String="1:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    cluster_backend::String="SLURM",
    account::String="def-jemerson_gpu",
    mode::Symbol=:train,
    use_gpu::Bool=true,
    jobs_per_gpu::Int=1,
    n_gpus_per_node::Int=1,
    gpu_type::String="",
    cuda_module::String="cuda",
    mem_per_gpu::String="",
    vram_per_gpu::Int=0,
    seconds_per_round::Int=0,     # measured time of one round of a task, for the walltime preflight; 0 = unknown
    startup_seconds::Int=0,       # per-job startup in the same estimate
    timestamp::String=Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"),   # one per submission, shared by its jobs
    model_names::AbstractVector{String}=String[],   # test jobs: the weights file of each command, in order
    depot_tag::String="",         # SLURM jobs sharing a tag prepare the Julia depot once between them
)::String
    commands_dir::String = joinpath(working_dir, codename, "cluster")
    mkpath(commands_dir)
    if !isempty(model_names) && length(model_names) != length(commands)
        throw(ArgumentError("generate_parallel_commands: $(length(model_names)) model name(s) for " *
                            "$(length(commands)) command(s)."))
    end

    # A new file per submission and job: the job reads it when it STARTS, possibly
    # hours later, so a fixed name let a second submission rewrite the first one's
    # work, and the mode keeps the two jobs of one submission apart.
    commands_file::String = "commands_$(mode)_$(timestamp).txt"
    commands_file_path::String = joinpath(commands_dir, commands_file)
    open(commands_file_path, "w") do io
        for command in commands
            println(io, command_text(command; julia_project=julia_project, heap_size_hint=heap_size_hint,
                                     working_dir=working_dir, codename=codename, n_hidden_layers=n_hidden_layers))
        end
    end
    models_file_path::String = ""
    if !isempty(model_names)
        models_file_path = joinpath(commands_dir, "models_$(mode)_$(timestamp).txt")
        open(models_file_path, "w") do io
            for model_name in model_names
                println(io, model_name)
            end
        end
    end
    n_simulations::Int = length(commands)
    println("[$(codename)] wrote $(n_simulations) $(mode) command(s) to $(commands_file_path)")

    backend::String = lowercase(cluster_backend)
    script_path::String = ""
    if backend == "google_vm"
        script_path = run_on_Google_VM(commands_file_path, output_file, min(ncpus, n_simulations))
    elseif backend == "slurm"
        script_path = run_on_SLURM(
            commands_file_path,
            n_simulations,
            codename;
            n_cpus          = ncpus,
            mem_per_cpu     = mem_per_cpu,
            max_nodes       = max_nodes,
            wall_time       = wall_time,
            email_address   = email_address,
            working_dir     = working_dir,
            account         = account,
            mode            = mode,
            use_gpu         = use_gpu,
            jobs_per_gpu    = jobs_per_gpu,
            n_gpus_per_node = n_gpus_per_node,
            gpu_type        = gpu_type,
            cuda_module     = cuda_module,
            mem_per_gpu     = mem_per_gpu,
            vram_per_gpu    = vram_per_gpu,
            seconds_per_round = seconds_per_round,
            startup_seconds = startup_seconds,
            timestamp       = timestamp,
            models_file     = models_file_path,
            depot_tag       = depot_tag,
        )
    elseif backend == "local"
        # train_local_<ts>.sh / test_local_<ts>.sh, detecting the GPU at run time
        # (Metal on Apple Silicon, CUDA on Linux) when use_gpu is set.
        script_path = run_locally(
            commands_file_path,
            n_simulations,
            codename;
            n_cpus          = min(ncpus, n_simulations),
            working_dir     = working_dir,
            mode            = mode,
            use_gpu         = use_gpu,
            n_gpus_per_node = n_gpus_per_node,
            gpu_type        = gpu_type,
        )
    else
        throw(ArgumentError("Unknown cluster_backend=\"$(cluster_backend)\". " *
                            "Must be one of: \"SLURM\", \"local\", \"Google_VM\"."))
    end
    return script_path
end

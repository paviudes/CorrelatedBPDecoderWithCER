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
# A submission TRAINS on one dataset instance, (p, sigma, sample), and TESTS
# every trained model on a list of (p, sigma) pairs at a list of samples. p and
# sigma are the numbers the file names carry; nothing symbolic ("2p") is read.
#
# Exposes (with dataset_settings.jl, which it includes: python_float_text,
# dataset_key, DatasetPair, DatasetInstance, instance_key, parse_dataset_pair,
# parse_dataset_instance, dataset_pairs, sample_indices, test_instance_list):
#   same_p_hint
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

# The dataset names and the settings that give them (p, sigma, sample), shared
# with postprocess.jl.
include("dataset_settings.jl")

"""
    same_p_hint(dataset_dir, relative_path) -> String

"   (present for this p: …)", naming up to six files in the path's folder with
the same p, or "" when there are none: the usual cause of a missing file is a
sigma or sample the data does not have.
"""
function same_p_hint(dataset_dir::String, relative_path::String)::String
    hint::String = ""
    p_prefix_match::Union{RegexMatch, Nothing} = match(r"^(.*_p_[^_]+_sig_)", basename(relative_path))
    folder::String = joinpath(dataset_dir, dirname(relative_path))
    if p_prefix_match !== nothing && isdir(folder)
        p_prefix::String = String(p_prefix_match.captures[1])
        same_p_files::Vector{String} = filter(name -> startswith(name, p_prefix), readdir(folder))
        if !isempty(same_p_files)
            hint = "   (present for this p: " * join(first(same_p_files, 6), ", ") * ")"
        end
    end
    return hint
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
        report::String = relative_path * same_p_hint(dataset_dir, relative_path)
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

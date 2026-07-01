# ============================================================================
# batch_run.jl — top-level submission driver
# ============================================================================
#
# Splits its work across four included files (in ./submission/) so this
# driver stays small:
#   * submission/batch_commands.jl — build commands.txt and dispatch to a backend
#   * submission/slurm.jl          — SLURM (Alliance Canada) job-script generator
#   * submission/local_runs.jl     — local runner with runtime GPU detection
#   * submission/google_vm.jl      — Google Cloud VM runner + shutdown wrapper
#
# Two entry points:
#   1. `main(...)` — programmatic, called by the ArgParse block below or by
#                    other scripts (`include("batch_run.jl"); main(...)`).
#   2. Running this file directly:  `julia --project=./../ batch_run.jl [flags]`
#
# You can supply flags either as individual --foo values or via a single
# TOML file with `--settings <path>.toml`; CLI flags override TOML values,
# which override built-in defaults. The `submit.sh` helper writes a defaults
# TOML, opens it in $EDITOR, and runs this driver with --settings.
# ============================================================================

using Dates
using Printf
using ArgParse
using LinearAlgebra
using TOML

# `disable_retrain_in_hyperparams` (sed-based TOML editor) is exported from
# src/command_line.jl. Used by generate_batch_runs() when --test is passed so the test job
# loads the trained weights instead of retraining.
using CorrelatedBPDecoderWithCER

# The four extracted modules live in the submission/ subfolder.
include("submission/slurm.jl")
include("submission/google_vm.jl")
include("submission/local_runs.jl")
include("submission/batch_commands.jl")

# ----------------------------------------------------------------------------
# generate_batch_runs — generate a submission script (SLURM / local / Google VM) for one or
# more codenames. In test mode, flips retrain=false in each hyperparams TOML
# so the generated commands load existing weights instead of retraining.
# ----------------------------------------------------------------------------
function generate_batch_runs(;
    codenames::AbstractVector{<:String}=["72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data"],
    pvals::AbstractVector{<:Real}=[0.01],
    qvals::AbstractVector{<:Real}=[0.001],
    n_samples::Int=64,
    hyperparams_file::String="hyperparams_epochs_10.toml",
    n_hidden_layers::Int=200,
    # --- cluster args ---
    cluster_backend::String="SLURM",   # "SLURM", "local", "Google_VM"
    n_cpus::Int=64,
    mem_per_cpu::String="4G",
    wall_time::String="1:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    max_nodes::Int=1,
    data_dir::String="./../data",
    # --- test mode + GPU args ---
    test::Bool=false,
    account::String="def-jemerson",
    n_gpus_per_node::Int=1,
    gpu_type::String="",
    cuda_module::String="cuda",
)
    """
    Generate a submission script for the given codenames, pvals, and qvals.
    In test mode, flips retrain=false in each hyperparams TOML so the generated commands load existing weights instead of retraining.
    """
    mode = test ? :test : :train
    skip_testing = !test    # in train mode we omit --test from the constructed commands

    for codename in codenames
        if test
            hp_path = joinpath(data_dir, codename, "models", hyperparams_file)
            println("[--test] disabling retrain in $(hp_path)")
            disable_retrain_in_hyperparams(hp_path)
        end

        generate_parallel_commands(
            pvals, qvals, n_samples, codename;
            n_hidden_layers  = n_hidden_layers,
            hyperparams_file = hyperparams_file,
            julia_project    = "./../",
            commands_file    = "commands.txt",
            output_file      = "simulation_results.log",
            working_dir      = "$(data_dir)",
            ncpus            = n_cpus,
            mem_per_cpu      = mem_per_cpu,
            max_nodes        = max_nodes,
            wall_time        = wall_time,
            email_address    = email_address,
            cluster_backend  = cluster_backend,
            skip_testing     = skip_testing,
            account          = account,
            mode             = mode,
            n_gpus_per_node  = n_gpus_per_node,
            gpu_type         = gpu_type,
            cuda_module      = cuda_module,
        )
    end
end

# ----------------------------------------------------------------------------
# TOML-overlay helpers for --settings support.
# ----------------------------------------------------------------------------

function _is_flag_passed(flag_name::String)
    """
    Return true when the user actually typed `--<flag_name>` on the command line
    (with or without an `=value`), false when the value in `parsed_args` came
    from ArgParse's default. We scan `ARGS` directly because ArgParse doesn't
    expose "was this explicitly provided" and we need CLI > TOML > default
    precedence.
    """
    prefix = "--$(flag_name)"
    for arg in ARGS
        if arg == prefix || startswith(arg, prefix * "=")
            return true
        end
    end
    return false
end

function _overlay_toml!(parsed_args::Dict, toml_path::String)
    """
    Read `toml_path` and, for every key it contains, replace `parsed_args[key]`
    UNLESS the user already passed that flag on the command line. Unknown TOML
    keys (not in `parsed_args`) are ignored with a warning.
    """
    if !isfile(toml_path)
        error("Settings TOML not found: $(toml_path)")
    end
    settings = TOML.parsefile(toml_path)
    for (key, val) in settings
        if !haskey(parsed_args, key)
            @warn "Ignoring unknown key in settings TOML: $(key) = $(repr(val))"
            continue
        end
        if _is_flag_passed(key)
            # Don't override the value since the user explicitly passed it on the command line.
            continue
        end
        parsed_args[key] = val
    end
    return parsed_args
end

# ----------------------------------------------------------------------------
# ArgParse entry point.
# ----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--settings"
            help = "Path to a TOML file with any/all of the flags below. Individual --foo " *
                   "flags on the command line override values in the TOML; TOML values " *
                   "override built-in defaults."
            arg_type = String
            default = ""
        "--working_dir"
            help = "Working directory for the simulations."
            arg_type = String
            default = "./../data"
        "--dirnames"
            help = "List of directory names for different simulation settings."
            nargs = '+'
            default = ["72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data"]
        "--pvals"
            help = "List of p values for the simulations."
            nargs = '+'
            default = [0.01]
        "--qvals"
            help = "List of q values for the simulations."
            nargs = '+'
            default = [0.001]
        "--n_samples"
            help = "Number of samples per (p, q) pair."
            arg_type = Int
            default = 64
        "--hyperparams_file"
            help = "Path to the hyperparameters file."
            arg_type = String
            default = "hyperparams_epochs_10.toml"
        "--n_hidden_layers"
            help = "Number of hidden layers in the neural BP model."
            arg_type = Int
            default = 200
        "--cluster_backend"
            help = "Cluster backend: \"SLURM\", \"local\", or \"Google_VM\"."
            arg_type = String
            default = "SLURM"
        "--n_cpus"
            help = "Number of CPUs to use for parallel execution."
            arg_type = Int
            default = 64
        "--mem_per_cpu"
            help = "Memory per CPU for the SLURM job."
            arg_type = String
            default = "4G"
        "--wall_time"
            help = "Wall time for the SLURM job."
            arg_type = String
            default = "1:00:00"
        "--email"
            help = "Email address for SLURM job notifications."
            arg_type = String
            default = "pavithran.sridhar@gmail.com"
        "--max_nodes"
            help = "Maximum number of nodes to use for the SLURM job."
            arg_type = Int
            default = 1
        "--test"
            help = "Generate a TEST script (GPU where available) instead of a TRAIN script. " *
                   "Also flips retrain=true to retrain=false in the hyperparams TOML."
            action = :store_true
        "--account"
            help = "SLURM account string (e.g. def-jemerson, or def-<sponsor>_gpu on some clusters)."
            arg_type = String
            default = "def-jemerson"
        "--n_gpus_per_node"
            help = "GPUs per array task in test mode. On local backend, upper-bounds concurrency."
            arg_type = Int
            default = 1
        "--gpu_type"
            help = "Alliance Canada GPU model specifier: h100, a100, l40s, h200, mi300a, v100. " *
                   "Empty string means \"any\", but per the docs this may cause SLURM job rejection."
            arg_type = String
            default = ""
        "--cuda_module"
            help = "Name of the CUDA module to load in SLURM test mode (e.g. cuda, cuda/12.2)."
            arg_type = String
            default = "cuda"
    end

    parsed_args = parse_args(settings)

    # Overlay values from the TOML settings file if --settings was given.
    if !isempty(parsed_args["settings"])
        println("[settings] loading $(parsed_args["settings"])")
        _overlay_toml!(parsed_args, parsed_args["settings"])
    end

    dirnames = String.(parsed_args["dirnames"])
    # TOML numeric values come through as Float64/Int directly; command-line
    # values come through as strings when nargs='+'. Coerce both to Float64.
    pvals = [val isa Number ? Float64(val) : parse(Float64, string(val)) for val in parsed_args["pvals"]]
    qvals = [val isa Number ? Float64(val) : parse(Float64, string(val)) for val in parsed_args["qvals"]]

    generate_batch_runs(;
        data_dir         = parsed_args["working_dir"],
        codenames        = dirnames,
        pvals            = pvals,
        qvals            = qvals,
        n_samples        = parsed_args["n_samples"],
        hyperparams_file = parsed_args["hyperparams_file"],
        n_hidden_layers  = parsed_args["n_hidden_layers"],
        cluster_backend  = parsed_args["cluster_backend"],
        n_cpus           = parsed_args["n_cpus"],
        mem_per_cpu      = parsed_args["mem_per_cpu"],
        wall_time        = parsed_args["wall_time"],
        email_address    = parsed_args["email"],
        max_nodes        = parsed_args["max_nodes"],
        test             = parsed_args["test"],
        account          = parsed_args["account"],
        n_gpus_per_node  = parsed_args["n_gpus_per_node"],
        gpu_type         = parsed_args["gpu_type"],
        cuda_module      = parsed_args["cuda_module"],
    )

    # Example usage (from Shell in the `expts` directory):
    #   Train on SLURM:
    #     julia --project="./../" batch_run.jl --dirnames 72q_BB_... --n_cpus 64 --wall_time 3:00:00
    #   Test on SLURM+GPU:
    #     julia --project="./../" batch_run.jl --test --dirnames 72q_BB_... --gpu_type h100 --cuda_module cuda
    #   Test locally (Metal on Mac, CUDA on Linux):
    #     julia --project="./../" batch_run.jl --test --dirnames 72q_BB_... --cluster_backend local --n_cpus 4
    #   With a settings TOML written by submit.sh:
    #     julia --project="./../" batch_run.jl --settings submission_settings_2026_07_02_090000.toml
end
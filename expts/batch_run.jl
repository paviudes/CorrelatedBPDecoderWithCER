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

# NOTE: we deliberately do NOT `using CorrelatedBPDecoderWithCER` here. That
# package pulls in Enzyme + Flux + Zygote + DataFrames + Plots — great for
# training/testing, catastrophic for a submission-side driver that runs on an
# HPC login node where precompile invalidation triggers `lld` SIGBUS under
# the login node's resource throttling.
#
# The only symbol we'd get from the package is `disable_retrain_in_hyperparams`
# (see src/command_line.jl for the canonical copy). We reproduce it inline
# below — it's a tiny sed shell-out, worth duplicating to keep this driver
# loadable from anywhere with just TOML + ArgParse + Dates.

"""
    disable_retrain_in_hyperparams(hyperparams_file::String)

Stream-edit the TOML at `hyperparams_file` to flip `retrain = true` to
`retrain = false`, preserving comments and key ordering. Uses `sed -E` so
it works on both BSD sed (macOS) and GNU sed (Linux clusters).

Kept in sync with `src/command_line.jl`'s canonical copy — if you change one,
change the other.
"""
function disable_retrain_in_hyperparams(hyperparams_file::String)
    if !isfile(hyperparams_file)
        error("Hyperparams file not found: $(hyperparams_file)")
    end
    sed_expr = raw"s|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true([[:space:]]*(#.*)?)$|\1false\2|"
    new_contents = read(`sed -E $(sed_expr) $(hyperparams_file)`, String)
    write(hyperparams_file, new_contents)
    return hyperparams_file
end

# The four extracted modules live in the submission/ subfolder.
include("submission/slurm.jl")
include("submission/google_vm.jl")
include("submission/local_runs.jl")
include("submission/batch_commands.jl")

"""
    _use_cer_from_hyperparams(hyperparams_path::String) -> Bool

Read `use_CER` from a hyperparams TOML for the submit-time CER preflight.
Defaults to `true` when the file is missing, the key is absent, or the file
can't be parsed — mirroring `parse_hyper_parameters` (CER is the default). Like
the rest of this driver, it reads TOML directly rather than loading the heavy
package, so submission stays lightweight on the login node.
"""
function _use_cer_from_hyperparams(hyperparams_path::String)::Bool
    if !isfile(hyperparams_path)
        return true
    end
    try
        return get(TOML.parsefile(hyperparams_path), "use_CER", true) === true
    catch
        return true
    end
end

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
    # Error model selecting the filename convention (case-insensitive):
    #   "Ising"   → two-parameter correlated model, uses both pvals and qvals.
    #   "Circuit" → single-parameter circuit-level model, uses pvals only.
    error_model::String="Ising",
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
    mem_per_gpu::String="",   # SLURM --mem-per-gpu (test mode only). Empty ⇒ use --mem-per-cpu instead.
)
    """
    Generate a submission script for the given codenames, pvals, and qvals.
    In test mode, flips retrain=false in each hyperparams TOML so the generated commands load existing weights instead of retraining.
    """
    mode = test ? :test : :train
    skip_testing = !test    # in train mode we omit --test from the constructed commands

    # Normalise the error-model selector once (case-insensitive per the CLI/TOML
    # contract) and fail loudly on anything we don't recognise, rather than
    # silently defaulting to one model and generating the wrong filenames.
    em = lowercase(error_model)
    if em ∉ ("ising", "circuit")
        error("Unknown error_model=\"$(error_model)\". Must be \"Ising\" or \"Circuit\" " *
              "(case-insensitive).")
    end

    for codename in codenames
        # CER preflight — run BEFORE any side effects (e.g. the retrain flip). If
        # this codename's hyperparams enable CER, its correlated_weights/ folder
        # must exist; error before submitting anything if it doesn't. With
        # use_CER = false we intentionally do NOT require the folder (the run
        # pretends it's absent: preset p=0.1 priors, correlation loss dropped).
        hp_path = joinpath(data_dir, codename, "models", hyperparams_file)
        if _use_cer_from_hyperparams(hp_path)
            cer_dir = joinpath(data_dir, codename, "correlated_weights")
            if !isdir(cer_dir)
                error("[use_CER=true] correlated_weights/ not found for codename \"$(codename)\": $(cer_dir).\n" *
                      "Provide the folder, or set `use_CER = false` in $(hyperparams_file) to run without CER " *
                      "priors (preset p=0.1, correlation loss dropped, outputs tagged `_no_cer`).")
            end
        end

        if test
            println("[--test] disabling retrain in $(hp_path)")
            disable_retrain_in_hyperparams(hp_path)
        end

        # Shared keyword args — identical for both models. Only the positional
        # parameter grid differs (Ising: p×q, Circuit: p only), so we build the
        # kwargs once and splat them into whichever grid form we dispatch to.
        common_kwargs = (
            n_hidden_layers  = n_hidden_layers,
            hyperparams_file = hyperparams_file,
            julia_project    = "./../",
            commands_file    = "",  # "" => auto commands_<timestamp>.txt (see batch_commands.jl)
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
            mem_per_gpu      = mem_per_gpu,
        )

        if em == "ising"
            generate_parallel_commands_Ising(
                pvals, qvals, n_samples, codename; common_kwargs...
            )
        else # "circuit" — qvals deliberately unused
            generate_parallel_commands_Circuit(
                pvals, n_samples, codename; common_kwargs...
            )
        end
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

function _expand_numeric_spec(spec)::Vector{Float64}
    """
    Turn a p/q specification into a Float64 vector, accepting:
      - a number                        -> [x]
      - a plain numeric string "0.01"   -> [0.01]
      - a Julia-style range STRING      -> collect(start:step:stop) (or start:stop)
        e.g. "0.0001:0.0002:0.0009" -> [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
      - a vector mixing any of the above (flattened),
        e.g. [0.001, "0.01:0.01:0.05"]

    Bare `a:b:c` is NOT valid TOML, so in the settings file the range must be
    QUOTED:  pvals = "0.0001:0.0002:0.0009". On the command line `--pvals a:b:c`
    also works, since nargs='+' delivers each token as a string.
    """
    if spec isa Number
        return [Float64(spec)]
    elseif spec isa AbstractString
        s = strip(spec)
        if occursin(':', s)
            parts = parse.(Float64, split(s, ':'))
            if length(parts) == 3
                return collect(parts[1]:parts[2]:parts[3])
            elseif length(parts) == 2
                return collect(parts[1]:parts[2])   # step defaults to 1.0
            else
                error("Range shorthand \"$(spec)\" must be start:step:stop or start:stop.")
            end
        else
            return [parse(Float64, s)]
        end
    elseif spec isa AbstractVector
        return isempty(spec) ? Float64[] : reduce(vcat, _expand_numeric_spec.(spec))
    else
        error("Cannot interpret $(repr(spec)) as a number, range string, or list.")
    end
end

function _overlay_toml!(parsed_args::Dict, toml_path::String)
    """
    Read `toml_path` and, for every key it contains, replace `parsed_args[key]`
    UNLESS the user already passed that flag on the command line. Any key in the
    TOML that is not a recognised flag is a hard ERROR: we fail loudly rather than
    silently falling back to a default (which used to hide typos like `workdir`
    vs `working_dir`). The message lists every valid key so it's actionable.
    """
    if !isfile(toml_path)
        error("Settings TOML not found: $(toml_path)")
    end
    settings = TOML.parsefile(toml_path)

    # Reject unknown keys up front, reporting all of them at once (rather than
    # erroring on the first, which would force fix-one-rerun-repeat).
    unknown_keys = sort([key for key in keys(settings) if !haskey(parsed_args, key)])
    if !isempty(unknown_keys)
        valid_keys = sort([key for key in keys(parsed_args) if key != "settings"])
        error("Unknown key(s) in settings TOML $(toml_path): $(join(unknown_keys, ", ")).\n" *
              "Valid keys are: $(join(valid_keys, ", ")).")
    end

    for (key, val) in settings
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
        "--workdir"
            help = "Working directory for the simulations (the data root; matches " *
                   "the --workdir passed to neural_bp_experiments.jl and postprocess.jl)."
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
        "--error_model"
            help = "Error model / filename convention (case-insensitive): " *
                   "\"Ising\" (two-parameter correlated, uses p and q) or " *
                   "\"Circuit\" (single-parameter circuit-level, uses p only)."
            arg_type = String
            default = "Ising"
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
        "--mem_per_gpu"
            help = "SLURM --mem-per-gpu memory allocation (test mode only, e.g. \"16G\"). " *
                   "When set, overrides --mem-per-cpu (SLURM disallows both at once). " *
                   "Empty (default) = use --mem-per-cpu instead."
            arg_type = String
            default = ""
    end

    parsed_args = parse_args(settings)

    # Overlay values from the TOML settings file if --settings was given.
    if !isempty(parsed_args["settings"])
        println("[settings] loading $(parsed_args["settings"])")
        _overlay_toml!(parsed_args, parsed_args["settings"])
    end

    dirnames = String.(parsed_args["dirnames"])
    # Coerce p/q specs to Float64 vectors. Values may arrive as TOML numbers/
    # arrays, command-line strings (nargs='+'), or a QUOTED range shorthand like
    # "0.0001:0.0002:0.0009" — `_expand_numeric_spec` handles all of these.
    pvals = _expand_numeric_spec(parsed_args["pvals"])
    qvals = _expand_numeric_spec(parsed_args["qvals"])

    generate_batch_runs(;
        data_dir         = parsed_args["workdir"],
        codenames        = dirnames,
        pvals            = pvals,
        qvals            = qvals,
        n_samples        = parsed_args["n_samples"],
        error_model      = parsed_args["error_model"],
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
        mem_per_gpu      = parsed_args["mem_per_gpu"],
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
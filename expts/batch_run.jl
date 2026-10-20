# ============================================================================
# batch_run.jl — top-level submission driver
# ============================================================================
#
# Trains on ONE dataset and tests every trained model on a LIST of datasets, for
# a CER arm, a no-CER arm or both, and one model per seed. Datasets are named by
# key p_<p>_sig_<sigma>_s_<sample> (see submission/batch_commands.jl), and the
# settings give the numbers the file names carry:
#
#   training   train_instance = [p, sigma, sample]                -> one key
#   testing    test_pairs = [[p, sigma], ...] at test_samples = [s, ...]
#                                                                 -> one key per pair and sample
#
#     test_pairs = [[0.0001, 0.0], [0.0001, 0.0001], [0.0001, 0.0002], ...]
#     test_samples = [1]
#
# Every listed pair is tested at every listed sample; a combination the data does
# not have (sigma = 0 has samples 1-3 only) is skipped with a warning, and a pair
# the data has at none of the samples is refused (a mistyped number).
#
# On SLURM a walltime preflight estimates every array task from measured timings
# (see the constants below) and refuses to write a job that cannot finish.
#
# Which jobs: `train = true` writes the training job, `test = true` the test job.
# Both (the usual run) writes the two together and chains them: the test job starts
# only when every training task has exited 0, through one submit script,
# cluster/submit_<ts>.sh. `test` alone tests models already on disk. Either job can
# run on GPUs (gpu_for_training, gpu_for_testing).
#
# Run tags: the CER arm's files carry `_cer`, the no-CER arm's `_nocer` (after the
# code's own `_no_cer`), then `_seed_<n>`; `run_label` adds `_<label>` to both, so a
# second experiment in the same data directory does not overwrite the first.
#
# Splits its work across four included files (in ./submission/):
#   * submission/batch_commands.jl — dataset keys, commands file, dispatch
#   * submission/slurm.jl          — SLURM job script (fills slurm_job_template.sh)
#   * submission/local_runs.jl     — local runner with runtime GPU detection
#   * submission/google_vm.jl      — Google Cloud VM runner + shutdown wrapper
#
# Run it with `julia --project=./../ batch_run.jl [flags]`, or with a single TOML
# via `--settings <path>.toml` (what submit.sh writes); CLI flags override TOML
# values, which override the built-in defaults.
# ============================================================================

using Dates
using Printf
using ArgParse
using TOML

# NOTE: we deliberately do NOT `using CorrelatedBPDecoderWithCER` here. That
# package pulls in Enzyme, CUDA, DataFrames and more — fine for training and
# testing, but a submission driver runs on an HPC login node, where a
# precompile invalidation of that stack triggers `lld` SIGBUS under the login
# node's resource throttling. Everything here needs only TOML + ArgParse + Dates.

# The four extracted modules live in the submission/ subfolder.
include("submission/slurm.jl")
include("submission/google_vm.jl")
include("submission/local_runs.jl")
include("submission/batch_commands.jl")

"""
    megabytes_of(size_text) -> Int

"4G", "32G", "2048M" or a bare number (megabytes, SLURM's default unit) in MB.
"""
function megabytes_of(size_text::String)::Int
    size_match::Union{RegexMatch, Nothing} = match(r"^\s*([0-9.]+)\s*([GgMmTt]?)[Bb]?\s*$", size_text)
    if size_match === nothing
        throw(ArgumentError("cannot read the memory size $(repr(size_text)); write it like \"4G\" or \"2048M\"."))
    end
    amount::Float64 = parse(Float64, size_match.captures[1])
    unit::String = uppercase(String(size_match.captures[2]))
    megabytes_per_unit::Dict{String, Float64} = Dict("" => 1.0, "M" => 1.0, "G" => 1024.0, "T" => 1024.0^2)
    megabytes::Int = round(Int, amount * megabytes_per_unit[unit])
    return megabytes
end

# ----------------------------------------------------------------------------
# Walltime preflight. Measured on Narval A100s at 90 layers, 500 gradient updates
# per epoch, a batch of 20 and 10^6-sample test files. A task that runs out of
# walltime loses every command it had not finished, so slurm.jl refuses a job
# whose estimate does not fit. Raise these if the code or the data grow.
# ----------------------------------------------------------------------------
"One epoch of a wave of up to 4 tanh trainings sharing an A100 (~12 min for 5 epochs, 2026-10-15)."
const SECONDS_PER_EPOCH_TANH_WAVE = 144
"One epoch of a wave holding enriched-node trainings (~25 min for 5 epochs, 2026-10-19)."
const SECONDS_PER_EPOCH_ENRICHED_WAVE = 300
"One 10^6-sample test on an A100, Julia startup included (~2.3 min, 2026-10-19)."
const SECONDS_PER_GPU_TEST = 150
"Per job: depot preparation (the CUDA_Runtime_jll rebuild took 6 min) and the stage-in."
const JOB_STARTUP_SECONDS = 600

"""
    measured_round_seconds(mode, use_gpu, hyperparams, n_hidden_layers) -> Int

Measured time of one round of an array task in the job being written: a wave of
trainings sharing the cards, or one test per card. Training is scaled from the
measured point by epochs × gradient updates per epoch × layers. 0 for a job on
CPUs, which has no measurement; slurm.jl then skips the check and says so.
"""
function measured_round_seconds(mode::Symbol, use_gpu::Bool, hyperparams::Dict{String, Any},
                                n_hidden_layers::Int)::Int
    round_seconds::Int = 0
    if use_gpu && mode == :test
        round_seconds = SECONDS_PER_GPU_TEST
    elseif use_gpu && mode == :train
        seconds_per_epoch::Int = SECONDS_PER_EPOCH_TANH_WAVE
        if String(get(hyperparams, "check_node", "tanh")) == "enriched"
            seconds_per_epoch = SECONDS_PER_EPOCH_ENRICHED_WAVE
        end
        n_epochs::Int = Int(get(hyperparams, "n_epochs", 5))
        updates_per_epoch::Int = Int(get(hyperparams, "n_gradient_updates_per_epoch", 500))
        size_scale::Float64 = (updates_per_epoch / 500) * (n_hidden_layers / 90)
        round_seconds = ceil(Int, n_epochs * seconds_per_epoch * size_scale)
    end
    return round_seconds
end

"""
    submission_hyperparams_path(timestamp, base_file, arm, mode) -> String

The hyperparameters file one arm's commands of one job read, relative to
models/: `submissions/<timestamp>/<base stem>_<arm>_<mode>.toml`, written by
`write_arm_hyperparams`. One folder per submission: a job reads its file when it
STARTS — hours later for a test job chained on training — and a fixed name let
the next submission rewrite it in the meantime. The folder is also the record
of what each submission ran.
"""
function submission_hyperparams_path(timestamp::String, base_file::String, arm::String, mode::Symbol)::String
    derived_name::String = "$(splitext(basename(base_file))[1])_$(arm)_$(mode).toml"
    relative_path::String = joinpath("submissions", timestamp, derived_name)
    return relative_path
end

"""
    arm_run_tag(arm, run_label) -> String

The `run_tag` of one arm's models and results: `_cer` or `_nocer`, and `_<label>`
after it when `run_label` is set. The code adds `_seed_<n>` after the tag, and
`_no_cer` before it on the no-CER arm.
"""
function arm_run_tag(arm::String, run_label::String)::String
    run_tag::String = "_$(arm)"
    if !isempty(run_label)
        run_tag *= "_$(run_label)"
    end
    return run_tag
end

"""
    write_arm_hyperparams(models_dir, base_file, derived_file, arm, mode, run_label, timestamp) -> Nothing

Write `derived_file` (relative to `models_dir`, see `submission_hyperparams_path`):
the base hyperparameters file with `use_CER` set for the arm (true for "cer",
false for "nocer"), `retrain` set for the job (true to train; false to test,
which loads the trained weights) and `run_tag` set to `arm_run_tag`. The no-CER
arm also gets `check_node = "tanh"`: it has no couplings for the enriched node,
which refuses to run without them. These keys go FIRST, in the root table, and
the base's own lines for them are dropped. The base file is never modified — the
old pipeline flipped `retrain` in it, so a later training job silently loaded
weights instead of training.
"""
function write_arm_hyperparams(models_dir::String, base_file::String, derived_file::String, arm::String,
                               mode::Symbol, run_label::String, timestamp::String)::Nothing
    base_lines::Vector{String} = readlines(joinpath(models_dir, base_file))
    overridden_keys::Regex = r"^\s*(use_CER|retrain|run_tag)\s*="
    if arm == "nocer"
        overridden_keys = r"^\s*(use_CER|retrain|run_tag|check_node)\s*="
    end
    kept_lines::Vector{String} = filter(line -> match(overridden_keys, line) === nothing, base_lines)
    use_cer_text::String = "true"
    if arm == "nocer"
        use_cer_text = "false"
    end
    retrain_text::String = "true"
    if mode == :test
        retrain_text = "false"
    end
    run_tag::String = arm_run_tag(arm, run_label)
    derived_path::String = joinpath(models_dir, derived_file)
    mkpath(dirname(derived_path))
    open(derived_path, "w") do io
        println(io, "# Written by batch_run.jl on $(timestamp) from $(base_file): arm \"$(arm)\", $(mode) job.")
        println(io, "# The jobs of that submission read it; edit $(base_file) for the next one.")
        println(io, "use_CER = $(use_cer_text)")
        println(io, "retrain = $(retrain_text)")
        println(io, "run_tag = \"$(run_tag)\"")
        if arm == "nocer"
            println(io, "check_node = \"tanh\"")
        end
        println(io, "")
        for line in kept_lines
            println(io, line)
        end
    end
    # Read it back: a file the experiment cannot parse must fail here, not on a node.
    derived_hyperparams::Dict{String, Any} = TOML.parsefile(derived_path)
    use_cer_read::Bool = derived_hyperparams["use_CER"]
    retrain_read::Bool = derived_hyperparams["retrain"]
    run_tag_read::String = derived_hyperparams["run_tag"]
    check_node_read::String = String(get(derived_hyperparams, "check_node", "tanh"))
    is_as_written::Bool = use_cer_read == (arm == "cer") && retrain_read == (mode == :train) && run_tag_read == run_tag
    if arm == "nocer" && check_node_read != "tanh"
        is_as_written = false
    end
    if !is_as_written
        throw(ErrorException("batch_run.jl: $(derived_path) does not read back as written."))
    end
    return nothing
end

"""
    write_chained_submission(commands_dir, timestamp, train_script, test_script) -> String

cluster/submit_<timestamp>.sh: submits the training job, then the test job with
`--dependency=afterok` on it, so testing starts only when EVERY training task has
exited 0 (afterok on an array is all-or-nothing) and is cancelled, not left
pending, if any has not (`--kill-on-invalid-dep=yes`). The job scripts' paths are
relative to expts/, so the script changes there first.
"""
function write_chained_submission(commands_dir::String, timestamp::String, train_script::String,
                                  test_script::String)::String
    submit_path::String = joinpath(commands_dir, "submit_$(timestamp).sh")
    expts_dir::String = pwd()
    open(submit_path, "w") do io
        println(io, "#!/bin/bash")
        println(io, "# Written by batch_run.jl on $(timestamp): the training job, then the test job chained on it.")
        println(io, "set -euo pipefail")
        println(io, "cd \"$(expts_dir)\"")
        println(io, "TRAIN_ID=\$(sbatch --parsable \"$(train_script)\")")
        println(io, "TRAIN_ID=\${TRAIN_ID%%;*}")
        println(io, "echo \"training job: \$TRAIN_ID\"")
        println(io, "TEST_ID=\$(sbatch --parsable --dependency=afterok:\$TRAIN_ID --kill-on-invalid-dep=yes " *
                    "\"$(test_script)\")")
        println(io, "echo \"test job:     \${TEST_ID%%;*} (starts when every training task has exited 0)\"")
    end
    chmod(submit_path, 0o755)
    return submit_path
end

"One job of a submission: what it runs, on what, and the model each command trains or loads."
struct SubmissionJob
    mode::Symbol                          # :train or :test
    use_gpu::Bool
    jobs_per_gpu::Int                     # processes sharing one card
    commands::Vector{ExperimentCommand}
    model_names::Vector{String}           # weights file each command writes (:train) or loads (:test), in order
end

"""
    run_label_problem(run_label) -> String

Why `run_label` cannot go into file names, or "" when it can: letters, digits,
`-` and `_` only, and not the word "seed" (every name ends in `_seed_<n>`, which
the collectors look for).
"""
function run_label_problem(run_label::String)::String
    problem::String = ""
    if match(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", run_label) === nothing
        problem = "use letters, digits, - and _ only (it goes into file names)"
    elseif occursin("seed", lowercase(run_label))
        problem = "leave out the word \"seed\" (every name ends in _seed_<n>)"
    end
    return problem
end

"""
    print_submission(codename, backend, jobs, script_paths, submit_path) -> Nothing

How to run what was written: the chained submit script, one sbatch, or the
local scripts in order.
"""
function print_submission(codename::String, backend::String, jobs::Vector{SubmissionJob},
                          script_paths::Vector{String}, submit_path::String)::Nothing
    if backend == "slurm"
        println("\n=== Submission (from expts/, after `bash misc/precompile_depot.sh` on a login node) ===")
        if !isempty(submit_path)
            println("  bash $(submit_path)")
            println("  It submits the training job, then the test job with --dependency=afterok on it: testing")
            println("  starts when EVERY training task has exited 0, and is cancelled if any has not.")
        else
            println("  sbatch $(script_paths[1])")
        end
    elseif length(script_paths) == 2
        println("\n=== To run: training, then testing if every training command succeeded ===")
        println("  bash $(script_paths[1]) && bash $(script_paths[2])")
    end
    if length(jobs) == 1 && jobs[1].mode == :train
        println("[$(codename)] Training only. To test these models later: the same settings with train = false, " *
                "test = true.")
    end
    println()
    return nothing
end

# ----------------------------------------------------------------------------
# generate_batch_runs — write the commands and the job scripts (SLURM / local /
# Google VM) of one submission for one or more dataset directories.
# ----------------------------------------------------------------------------
function generate_batch_runs(;
    codenames::AbstractVector{<:AbstractString}=["72q_BB_cycles_1_sweep"],
    # --- training dataset: one (p, sigma, sample) ---
    train_instance::Union{DatasetInstance, Nothing}=nothing,
    # --- test datasets: every pair at every sample; none = the training pair / sample ---
    test_pairs::AbstractVector{DatasetPair}=DatasetPair[],
    test_samples::AbstractVector{Int}=Int[],
    # --- arms and seeds: one model per (arm, seed); run_label names the experiment ---
    cer_arms::AbstractVector{<:AbstractString}=["cer", "nocer"],
    seeds::AbstractVector{Int}=Int[],
    run_label::String="",
    # --- model ---
    hyperparams_file::String="hyperparams_baseline.toml",
    n_hidden_layers::Int=90,
    heap_size_hint::String="4G",
    # --- cluster ---
    cluster_backend::String="SLURM",
    n_cpus::Int=12,
    mem_per_cpu::String="4G",
    wall_time::String="1:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    max_nodes::Int=8,
    data_dir::String="./../data",
    # --- which jobs, and on what ---
    train::Bool=true,
    test::Bool=true,
    gpu_for_training::Bool=true,
    gpu_for_testing::Bool=true,
    trainings_per_gpu::Int=4,
    account::String="def-jemerson_gpu",
    n_gpus_per_node::Int=1,
    gpu_type::String="a100",
    cuda_module::String="cuda",
    mem_per_gpu::String="",
    vram_per_gpu::Int=0,
)::Nothing
    """
    Write the jobs of one submission for each codename: the training job
    (train = true), the test job (test = true), or both, in which case the test
    job is chained on the training job. Every input file is checked, and every
    SLURM job's walltime estimated, before anything is written. A test job on its
    own includes only the commands whose trained model is on disk.
    """
    if !train && !test
        throw(ArgumentError("train = false and test = false: there is no job to write."))
    end
    backend::String = lowercase(cluster_backend)
    if !(backend in ("slurm", "local", "google_vm"))
        throw(ArgumentError("Unknown cluster_backend=\"$(cluster_backend)\". Must be one of: \"SLURM\", \"local\", " *
                            "\"Google_VM\"."))
    end
    is_chained::Bool = train && test
    if is_chained && backend == "google_vm"
        throw(ArgumentError("cluster_backend = \"Google_VM\" writes one job per submission (its script shuts the " *
                            "VM down when it ends): set train = true or test = true, not both."))
    end
    if !isempty(run_label)
        label_problem::String = run_label_problem(run_label)
        if !isempty(label_problem)
            throw(ArgumentError("run_label = $(repr(run_label)): $(label_problem)."))
        end
    end
    modes::Vector{Symbol} = Symbol[]
    if train
        push!(modes, :train)
    end
    if test
        push!(modes, :test)
    end

    # ---- arms, seeds and dataset keys ---------------------------------------
    arm_names::Vector{String} = String.(cer_arms)
    if isempty(arm_names)
        throw(ArgumentError("cer_arms is empty: list \"cer\", \"nocer\" or both."))
    end
    unknown_arms::Vector{String} = filter(arm -> !(arm in ("cer", "nocer")), arm_names)
    if !isempty(unknown_arms)
        throw(ArgumentError("cer_arms: $(join(unknown_arms, ", ")) — use \"cer\" (CER priors, use_CER = true) " *
                            "or \"nocer\" (uniform priors, use_CER = false)."))
    end
    if length(unique(arm_names)) != length(arm_names)
        throw(ArgumentError("cer_arms lists an arm twice: $(arm_names)."))
    end
    seed_choices::Vector{Union{Int, Nothing}} = Union{Int, Nothing}[seed for seed in seeds]
    if isempty(seed_choices)
        push!(seed_choices, nothing)   # one model, with the hyperparameters file's own seed
    end
    if train_instance === nothing
        throw(ArgumentError("no training dataset: set train_instance = [p, sigma, sample], e.g. " *
                            "train_instance = [0.0015, 0.003, 1]."))
    end
    train_key::String = instance_key(train_instance)
    train_file::String = "train_$(train_key).txt"
    # Tests default to the training pair and the training sample.
    test_pair_list::Vector{DatasetPair} = DatasetPair[(train_instance[1], train_instance[2])]
    if !isempty(test_pairs)
        test_pair_list = collect(DatasetPair, test_pairs)
    end
    test_sample_list::Vector{Int} = Int[train_instance[3]]
    if !isempty(test_samples)
        test_sample_list = collect(Int, test_samples)
    end
    requested_test_instances::Vector{DatasetInstance} = test_instance_list(test_pair_list, test_sample_list)

    # ---- what one array task can hold (training) -----------------------------
    # Every training is a single-threaded host process (on a GPU job it drives the
    # card), each with up to heap_size_hint of heap on top of the dataset it reads.
    if train
        heap_mb::Int = megabytes_of(heap_size_hint)
        processes_per_task::Int = n_cpus
        task_memory_mb::Int = n_cpus * megabytes_of(mem_per_cpu)
        if gpu_for_training
            processes_per_task = n_gpus_per_node * trainings_per_gpu
            if !isempty(mem_per_gpu)
                task_memory_mb = n_gpus_per_node * megabytes_of(mem_per_gpu)
            end
            if n_cpus < processes_per_task
                throw(ArgumentError("n_cpus = $(n_cpus) < n_gpus_per_node × trainings_per_gpu = $(processes_per_task): " *
                                    "every training needs a core of its own. Raise n_cpus or lower trainings_per_gpu."))
            end
        end
        if processes_per_task * heap_mb > task_memory_mb
            throw(ArgumentError("$(processes_per_task) trainings × heap_size_hint $(heap_size_hint) exceed the " *
                                "$(task_memory_mb) MB one array task gets. Raise mem_per_gpu (or mem_per_cpu), or " *
                                "lower trainings_per_gpu."))
        end
    end

    timestamp::String = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    for codename in codenames
        dataset_dir::String = joinpath(data_dir, codename)
        models_dir::String = joinpath(dataset_dir, "models")
        commands_dir::String = joinpath(dataset_dir, "cluster")
        base_path::String = joinpath(models_dir, hyperparams_file)
        if !isfile(base_path)
            throw(ArgumentError("[$(codename)] hyperparameters file not found: $(base_path)"))
        end
        base_hyperparams::Dict{String, Any} = TOML.parsefile(base_path)
        # The GPU training path is the hand-written backward pass; Enzyme cannot
        # differentiate through device arrays.
        if train && gpu_for_training && get(base_hyperparams, "manual_backprop", false) !== true
            throw(ArgumentError("[$(codename)] training on the GPU needs manual_backprop = true in " *
                                "$(hyperparams_file) (Enzyme cannot use the GPU). Set it, or set " *
                                "gpu_for_training = false."))
        end
        # What src/train.jl will call each model: n_epochs and the seed as the
        # experiment reads them (5 is parse_hyper_parameters' default n_epochs; a
        # command's --seed beats the file's seed).
        n_epochs::Int = Int(get(base_hyperparams, "n_epochs", 5))
        base_seed::Union{Int, Nothing} = nothing
        if get(base_hyperparams, "seed", nothing) !== nothing
            base_seed = Int(base_hyperparams["seed"])
        end

        # ---- the test instances the data has --------------------------------
        # Every pair at every sample, less the combinations without a test file
        # (sigma = 0 comes with samples 1-3 only); a pair with a test file at none
        # of the samples is a mistyped number and is refused.
        tested_instances::Vector{DatasetInstance} = DatasetInstance[]
        if test
            skipped_test_files::Vector{String} = String[]
            for instance in requested_test_instances
                test_path::String = joinpath("testing_data", "test_$(instance_key(instance)).txt")
                if isfile(joinpath(dataset_dir, test_path))
                    push!(tested_instances, instance)
                else
                    push!(skipped_test_files, test_path)
                end
            end
            unmatched_pairs::Vector{DatasetPair} =
                filter(pair -> !any(instance -> (instance[1], instance[2]) == pair, tested_instances), test_pair_list)
            if !isempty(unmatched_pairs)
                unmatched_reports::Vector{String} = String[]
                for pair in unmatched_pairs
                    first_path::String = joinpath("testing_data",
                                                  "test_$(instance_key((pair[1], pair[2], test_sample_list[1]))).txt")
                    push!(unmatched_reports, "(p, sigma) = ($(pair[1]), $(pair[2]))" *
                                             same_p_hint(dataset_dir, first_path))
                end
                throw(ArgumentError("[$(codename)] no test file for these pair(s) at any of the samples " *
                                    "$(test_sample_list) — check the numbers:\n  " * join(unmatched_reports, "\n  ")))
            end
            if !isempty(skipped_test_files)
                @warn "[$(codename)] $(length(skipped_test_files)) of $(length(requested_test_instances)) test " *
                      "instance(s) are not in the data and are skipped:\n  " * join(skipped_test_files, "\n  ")
            end
        end

        # ---- the jobs: one command per (arm, seed), times every test instance in a test job
        jobs::Vector{SubmissionJob} = SubmissionJob[]
        for mode in modes
            use_gpu::Bool = gpu_for_training
            jobs_per_gpu::Int = trainings_per_gpu
            if mode == :test
                use_gpu = gpu_for_testing
                jobs_per_gpu = 1    # a test process sizes its batch to the whole card
            end
            commands::Vector{ExperimentCommand} = ExperimentCommand[]
            model_names::Vector{String} = String[]
            for arm in arm_names
                arm_file::String = submission_hyperparams_path(timestamp, hyperparams_file, arm, mode)
                for seed in seed_choices
                    model_seed::Union{Int, Nothing} = base_seed
                    if seed !== nothing
                        model_seed = seed
                    end
                    model_name::String = expected_weights_name(n_hidden_layers, n_epochs, train_file, arm == "cer",
                                                               arm_run_tag(arm, run_label), model_seed)
                    if mode == :train
                        push!(commands, ExperimentCommand(arm, arm_file, "correlated_weights_$(train_key).txt",
                                                          train_file, "", seed))
                        push!(model_names, model_name)
                    else
                        for instance in tested_instances
                            test_key::String = instance_key(instance)
                            push!(commands, ExperimentCommand(arm, arm_file, "correlated_weights_$(test_key).txt",
                                                              train_file, "test_$(test_key).txt", seed))
                            push!(model_names, model_name)
                        end
                    end
                end
            end
            push!(jobs, SubmissionJob(mode, use_gpu, jobs_per_gpu, commands, model_names))
        end

        # ---- every input on disk before anything is written or queued ---------
        # The enriched node on the "bethe" table also reads bethe_table_<key>.txt
        # (the CER arm; the no-CER arm runs tanh).
        uses_bethe_tables::Bool = String(get(base_hyperparams, "check_node", "tanh")) == "enriched" &&
                                  String(get(base_hyperparams, "enriched_table", "pairwise")) == "bethe"
        all_commands::Vector{ExperimentCommand} = reduce(vcat, [job.commands for job in jobs])
        missing_reports::Vector{String} = missing_dataset_inputs(dataset_dir, all_commands;
                                                                 bethe_tables=uses_bethe_tables)
        if !isempty(missing_reports)
            bethe_hint::String = ""
            if any(report -> occursin("bethe_table_", report), missing_reports)
                bethe_hint = "\nBuild a Bethe table with: python3 misc/build_bethe_tables.py --workdir $(codename) " *
                             "--key <key>"
            end
            throw(ArgumentError("[$(codename)] missing dataset input(s) under $(dataset_dir):\n  " *
                                join(missing_reports, "\n  ") * "\n" *
                                "Check train_instance and test_pairs / test_samples against the file names." *
                                bethe_hint))
        end

        # ---- a test job on its own: only the models that are on disk ---------
        # (Chained on training, the models do not exist yet; the test job checks
        # them itself after its stage-in.)
        if test && !train
            test_job::SubmissionJob = jobs[end]
            kept_commands::Vector{ExperimentCommand} = ExperimentCommand[]
            kept_model_names::Vector{String} = String[]
            missing_models::Vector{String} = String[]
            for (command, model_name) in zip(test_job.commands, test_job.model_names)
                if isfile(joinpath(models_dir, model_name))
                    push!(kept_commands, command)
                    push!(kept_model_names, model_name)
                elseif !(model_name in missing_models)
                    push!(missing_models, model_name)
                end
            end
            if !isempty(missing_models)
                @warn "[$(codename)] $(length(missing_models)) trained model(s) not found under $(models_dir); " *
                      "their tests are left out (a test would otherwise train a model of its own). Missing:\n  " *
                      join(missing_models, "\n  ")
            end
            if isempty(kept_commands)
                throw(ErrorException("[$(codename)] no trained model for any test command under $(models_dir) — " *
                                     "nothing to run. Has the training job finished and staged out? A test job " *
                                     "names the models by the same hyperparameters file, arms, seeds, run_label and " *
                                     "n_hidden_layers as the training job."))
            end
            jobs[end] = SubmissionJob(:test, test_job.use_gpu, test_job.jobs_per_gpu, kept_commands, kept_model_names)
        end

        # ---- models a training job would replace ------------------------------
        if train
            existing_models::Vector{String} = filter(name -> isfile(joinpath(models_dir, name)),
                                                     unique(jobs[1].model_names))
            if !isempty(existing_models)
                @warn "[$(codename)] $(length(existing_models)) model(s) of this submission already exist and the " *
                      "training job will REPLACE them (retrain = true), e.g. $(existing_models[1]). Set run_label to " *
                      "keep them."
            end
        end

        # ---- SLURM preflight for every job before writing any ------------------
        # Walltime, and the card's VRAM that a GPU test job sizes its batch to.
        round_seconds_by_job::Vector{Int} =
            Int[measured_round_seconds(job.mode, job.use_gpu, base_hyperparams, n_hidden_layers) for job in jobs]
        if backend == "slurm"
            for (job_index, job) in enumerate(jobs)
                job_layout::SlurmTaskLayout = slurm_task_layout(length(job.commands), job.use_gpu, n_cpus,
                                                                n_gpus_per_node, job.jobs_per_gpu, max_nodes)
                walltime_estimate(String(job.mode), length(job.commands), job_layout, max_nodes, wall_time,
                                  round_seconds_by_job[job_index], JOB_STARTUP_SECONDS)
                is_unsized_gpu_test::Bool = job.mode == :test && job.use_gpu && vram_per_gpu <= 0 &&
                                            card_vram_gb(gpu_type) <= 0
                if is_unsized_gpu_test
                    throw(ArgumentError("no VRAM known for gpu_type = \"$(gpu_type)\": set vram_per_gpu (GB per " *
                                        "card) in the settings. Nothing was written."))
                end
            end
        end

        # ---- write: the arms' hyperparameters, then each job ------------------
        for job in jobs
            for arm in arm_names
                write_arm_hyperparams(models_dir, hyperparams_file,
                                      submission_hyperparams_path(timestamp, hyperparams_file, arm, job.mode),
                                      arm, job.mode, run_label, timestamp)
            end
        end
        job_description::String = "$(jobs[1].mode) job"
        if is_chained
            job_description = "training job, then the test job chained on it"
        end
        println("\n[$(codename)] submission $(timestamp): $(job_description)")
        println("  trained on : $(train_file)")
        if test
            println("  tested on  : $(length(tested_instances)) dataset(s) = $(length(test_pair_list)) (p, sigma) " *
                    "pair(s) at sample(s) $(join(test_sample_list, ", ")), " *
                    "$(length(requested_test_instances) - length(tested_instances)) not in the data")
        end
        arm_labels::Vector{String} = ["$(arm) (run_tag $(arm_run_tag(arm, run_label)))" for arm in arm_names]
        seed_labels::Vector{String} = [string(something(seed, "(from $(hyperparams_file))")) for seed in seed_choices]
        println("  arms       : $(join(arm_labels, ", "))")
        println("  seeds      : $(join(seed_labels, ", "))")
        println("  models     : $(length(unique(jobs[1].model_names))), e.g. $(jobs[1].model_names[1])")
        println("  settings   : $(joinpath(models_dir, "submissions", timestamp))/")

        script_paths::Vector{String} = String[]
        for (job_index, job) in enumerate(jobs)
            checked_model_names::Vector{String} = String[]
            if job.mode == :test
                checked_model_names = job.model_names
            end
            # Jobs on the same kind of device share one depot preparation: the test
            # job chained on a GPU training job finds the CUDA rebuild done.
            device_name::String = "cpu"
            if job.use_gpu
                device_name = "gpu"
            end
            script_path::String = generate_parallel_commands(
                job.commands,
                codename;
                n_hidden_layers  = n_hidden_layers,
                heap_size_hint   = heap_size_hint,
                julia_project    = "./../",
                output_file      = "simulation_results.log",
                working_dir      = data_dir,
                ncpus            = n_cpus,
                mem_per_cpu      = mem_per_cpu,
                max_nodes        = max_nodes,
                wall_time        = wall_time,
                email_address    = email_address,
                cluster_backend  = cluster_backend,
                account          = account,
                mode             = job.mode,
                use_gpu          = job.use_gpu,
                jobs_per_gpu     = job.jobs_per_gpu,
                n_gpus_per_node  = n_gpus_per_node,
                gpu_type         = gpu_type,
                cuda_module      = cuda_module,
                mem_per_gpu      = mem_per_gpu,
                vram_per_gpu     = vram_per_gpu,
                seconds_per_round = round_seconds_by_job[job_index],
                startup_seconds  = JOB_STARTUP_SECONDS,
                timestamp        = timestamp,
                model_names      = checked_model_names,
                depot_tag        = "submission_$(timestamp)_$(device_name)",
            )
            push!(script_paths, script_path)
        end
        submit_path::String = ""
        if is_chained && backend == "slurm"
            submit_path = write_chained_submission(commands_dir, timestamp, script_paths[1], script_paths[2])
        end
        print_submission(codename, backend, jobs, script_paths, submit_path)
    end
    return nothing
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

"""
    _as_list(value) -> Vector{Any}

A list setting as a vector even when the settings file gives one bare value
(`cer_arms = "cer"`, `seeds = 1`), which would otherwise be iterated character
by character, or broadcast as a scalar.
"""
function _as_list(value)::Vector{Any}
    list_value::Vector{Any} = Any[value]
    if value isa AbstractVector
        list_value = collect(Any, value)
    end
    return list_value
end

"""
    _integer_list(spec) -> Vector{Int}

Seeds from the settings file (TOML integers) or the command line (strings).
"""
function _integer_list(spec)::Vector{Int}
    integers::Vector{Int} = Int[]
    for item in _as_list(spec)
        if item isa Integer
            push!(integers, Int(item))
        else
            push!(integers, parse(Int, strip(String(item))))
        end
    end
    return integers
end

function _overlay_toml!(parsed_args::Dict, toml_path::String)::Dict
    """
    Read `toml_path` and, for every key it contains, replace `parsed_args[key]`
    UNLESS the user already passed that flag on the command line. Any key in the
    TOML that is not a recognised flag is a hard ERROR: we fail loudly rather than
    silently falling back to a default (which used to hide typos like `workdir`
    vs `working_dir`). The message lists every valid key so it's actionable.
    """
    if !isfile(toml_path)
        throw(ArgumentError("Settings TOML not found: $(toml_path)"))
    end
    settings::Dict{String, Any} = TOML.parsefile(toml_path)

    # Reject unknown keys up front, reporting all of them at once (rather than
    # erroring on the first, which would force fix-one-rerun-repeat).
    unknown_keys::Vector{String} = sort([key for key in keys(settings) if !haskey(parsed_args, key)])
    if !isempty(unknown_keys)
        valid_keys::Vector{String} = sort([key for key in keys(parsed_args) if key != "settings"])
        removed_keys::Vector{String} = filter(key -> key in ("pvals", "qvals", "n_samples", "error_model",
                                                             "train_p", "train_sigma", "test_pvals", "test_sigmas",
                                                             "train_pair", "train_sample", "test_sample"),
                                              unknown_keys)
        removed_note::String = ""
        if !isempty(removed_keys)
            removed_note = "\n$(join(removed_keys, ", ")) belonged to an older settings format. Datasets are given " *
                           "by their numbers now: train_instance = [0.0015, 0.003, 1] (p, sigma, sample) for " *
                           "training; test_pairs = [[0.0001, 0.0], [0.0001, 0.0001], ...] at test_samples = [1] " *
                           "for testing (bash submit.sh writes a fresh file)."
        end
        throw(ArgumentError("Unknown key(s) in settings TOML $(toml_path): $(join(unknown_keys, ", ")).\n" *
                            "Valid keys are: $(join(valid_keys, ", "))." * removed_note))
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
            help = "The data root holding one directory per codename (matches the --workdir " *
                   "passed to neural_bp_experiments.jl)."
            arg_type = String
            default = "./../data"
        "--dirnames"
            help = "Dataset directories (codenames) under --workdir."
            nargs = '+'
            default = ["72q_BB_cycles_1_sweep"]
        "--train_instance"
            help = "The ONE training dataset as p,sigma,sample, e.g. 0.0015,0.003,1 (the numbers of " *
                   "train_p_0.0015_sig_0.003_s_1.txt). In the settings file: train_instance = [0.0015, 0.003, 1]."
            arg_type = String
            default = ""
        "--test_pairs"
            help = "Test datasets as p,sigma pairs, e.g. 0.0001,0 0.0001,0.0001 0.0001,0.0002; each is tested at " *
                   "every sample in --test_samples. Empty = the training pair only."
            nargs = '+'
            default = Any[]
        "--test_samples"
            help = "Sample indices s at which every test pair is tested, e.g. 1 2 3. A pair the data lacks at a " *
                   "sample is skipped there. Empty = the training sample only."
            nargs = '+'
            default = Any[]
        "--cer_arms"
            help = "\"cer\" (CER priors, use_CER = true), \"nocer\" (uniform priors) or both: one model per arm and seed."
            nargs = '+'
            default = Any["cer", "nocer"]
        "--seeds"
            help = "Training seeds, passed as --seed (one model each). Empty = the hyperparameters file's seed."
            nargs = '+'
            default = Any[]
        "--run_label"
            help = "Names the experiment: the arms' run tags become _cer_<label> / _nocer_<label>, so its models " *
                   "and results do not replace another experiment's in the same data directory. Empty = _cer / _nocer."
            arg_type = String
            default = ""
        "--hyperparams_file"
            help = "Base hyperparameters file under <workdir>/<codename>/models/. Per-arm copies are written to " *
                   "models/submissions/<timestamp>/<stem>_<arm>_<train|test>.toml; it is not modified."
            arg_type = String
            default = "hyperparams_baseline.toml"
        "--n_hidden_layers"
            help = "Number of layers of the neural BP model."
            arg_type = Int
            default = 90
        "--heap_size_hint"
            help = "julia --heap-size-hint for every process (the GC otherwise sizes its heap to the whole node)."
            arg_type = String
            default = "4G"
        "--cluster_backend"
            help = "Cluster backend: \"SLURM\", \"local\", or \"Google_VM\"."
            arg_type = String
            default = "SLURM"
        "--n_cpus"
            help = "CPU cores per array task (on a GPU job: the cores that come with the cards)."
            arg_type = Int
            default = 12
        "--mem_per_cpu"
            help = "SLURM --mem-per-cpu, used when the job requests no GPU or mem_per_gpu is empty."
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
            help = "Maximum number of SLURM array tasks."
            arg_type = Int
            default = 8
        "--train"
            help = "Write the training job (one model per arm and seed)."
            arg_type = Bool
            default = true
        "--test"
            help = "Write the test job (every model on every test dataset). With --train true as well, the test " *
                   "job is chained on the training job and starts when every training task has exited 0; alone, " *
                   "it tests the models already on disk."
            arg_type = Bool
            default = true
        "--gpu_for_training"
            help = "Train on GPUs (needs manual_backprop = true in the hyperparameters file)."
            arg_type = Bool
            default = true
        "--gpu_for_testing"
            help = "Test on GPUs."
            arg_type = Bool
            default = true
        "--trainings_per_gpu"
            help = "Trainings sharing one card in a GPU training job (each keeps an A100 ~20% busy)."
            arg_type = Int
            default = 4
        "--account"
            help = "SLURM account string (e.g. def-jemerson_gpu for GPU jobs on Narval)."
            arg_type = String
            default = "def-jemerson_gpu"
        "--n_gpus_per_node"
            help = "GPUs per array task (GPU jobs)."
            arg_type = Int
            default = 1
        "--gpu_type"
            help = "Alliance Canada GPU model specifier: a100, a100_3g.20gb, h100, ..."
            arg_type = String
            default = "a100"
        "--cuda_module"
            help = "Name of the CUDA module to load in GPU jobs (e.g. cuda, cuda/12.2)."
            arg_type = String
            default = "cuda"
        "--mem_per_gpu"
            help = "SLURM --mem-per-gpu (HOST memory per GPU, e.g. \"32G\"). Empty = use --mem-per-cpu."
            arg_type = String
            default = ""
        "--vram_per_gpu"
            help = "GB of VRAM per card, for the test batch sizer. 0 = from gpu_type."
            arg_type = Int
            default = 0
    end

    parsed_args = parse_args(settings)

    # Overlay values from the TOML settings file if --settings was given.
    if !isempty(parsed_args["settings"])
        println("[settings] loading $(parsed_args["settings"])")
        _overlay_toml!(parsed_args, parsed_args["settings"])
    end

    # Values arrive as TOML types from the settings file and as strings from the
    # command line; normalise both.
    train_instance_spec::Any = parsed_args["train_instance"]
    if train_instance_spec isa AbstractString && isempty(strip(train_instance_spec))
        throw(ArgumentError("no training dataset: set train_instance = [p, sigma, sample] in the settings file " *
                            "(e.g. [0.0015, 0.003, 1]), or pass --train_instance 0.0015,0.003,1."))
    end
    train_instance_value::DatasetInstance = parse_dataset_instance(train_instance_spec, "train_instance")
    test_pair_values::Vector{DatasetPair} = dataset_pairs(_as_list(parsed_args["test_pairs"]), "test_pairs")
    test_sample_values::Vector{Int} = sample_indices(parsed_args["test_samples"], "test_samples")

    generate_batch_runs(;
        codenames         = String[String(name) for name in _as_list(parsed_args["dirnames"])],
        train_instance    = train_instance_value,
        test_pairs        = test_pair_values,
        test_samples      = test_sample_values,
        cer_arms          = String[String(arm) for arm in _as_list(parsed_args["cer_arms"])],
        seeds             = _integer_list(parsed_args["seeds"]),
        run_label         = String(parsed_args["run_label"]),
        hyperparams_file  = parsed_args["hyperparams_file"],
        n_hidden_layers   = Int(parsed_args["n_hidden_layers"]),
        heap_size_hint    = parsed_args["heap_size_hint"],
        cluster_backend   = parsed_args["cluster_backend"],
        n_cpus            = Int(parsed_args["n_cpus"]),
        mem_per_cpu       = parsed_args["mem_per_cpu"],
        wall_time         = parsed_args["wall_time"],
        email_address     = parsed_args["email"],
        max_nodes         = Int(parsed_args["max_nodes"]),
        data_dir          = parsed_args["workdir"],
        train             = Bool(parsed_args["train"]),
        test              = Bool(parsed_args["test"]),
        gpu_for_training  = Bool(parsed_args["gpu_for_training"]),
        gpu_for_testing   = Bool(parsed_args["gpu_for_testing"]),
        trainings_per_gpu = Int(parsed_args["trainings_per_gpu"]),
        account           = parsed_args["account"],
        n_gpus_per_node   = Int(parsed_args["n_gpus_per_node"]),
        gpu_type          = parsed_args["gpu_type"],
        cuda_module       = parsed_args["cuda_module"],
        mem_per_gpu       = parsed_args["mem_per_gpu"],
        vram_per_gpu      = Int(parsed_args["vram_per_gpu"]),
    )

    # Example usage (from Shell in the `expts` directory):
    #   Training and the test job chained on it (train = test = true), then submit both:
    #     julia --project="./../" batch_run.jl --settings scripts/submission_settings_<ts>.toml
    #     bash ./../data/<codename>/cluster/submit_<ts>.sh
    #   Only test models already trained (same settings otherwise):
    #     julia --project="./../" batch_run.jl --settings scripts/submission_settings_<ts>.toml --train false
    #   Without a settings file:
    #     julia --project="./../" batch_run.jl --dirnames 72q_BB_cycles_1_sweep --train_instance 0.0015,0.003,1 \
    #         --test_pairs 0.0015,0 0.0015,0.0015 0.0015,0.003 --test_samples 1 2 3 --seeds 1 \
    #         --gpu_type a100 --mem_per_gpu 32G
end

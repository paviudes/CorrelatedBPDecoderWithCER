# ============================================================================
# slurm.jl — SLURM job-script generator for Alliance Canada GPU/CPU clusters
# ============================================================================
#
# Exposes:  run_on_SLURM(commands_file, n_commands, codename; ...)
#
# Called from batch_commands.jl when cluster_backend == "SLURM". Emits a shell
# script named train_<ts>.sh or test_<ts>.sh in <working_dir>/<codename>/cluster/
# and prints an `sbatch <script>` submission hint. This file is `include`d by
# batch_run.jl.
# ============================================================================

using Dates
using Printf

function run_on_SLURM(
    commands_file::String,
    n_commands::Int,
    codename::String;
    n_cpus::Int=10,
    mem_per_cpu::String="4G",
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    working_dir::String=joinpath(@__DIR__, ".."),
    account::String="def-jemerson",
    mode::Symbol=:train,         # :train (CPU multi-threaded) or :test (NVIDIA GPU per array task)
    n_gpus_per_node::Int=1,      # only used when mode == :test
    gpu_type::String="",         # Alliance model specifier: "h100", "a100", "l40s", "h200", "mi300a", "v100", "" (any)
    cuda_module::String="cuda",  # cluster's CUDA module name (e.g. "cuda/12")
    mem_per_gpu::String="",      # SLURM `--mem-per-gpu` (test mode only). Empty ⇒ use `--mem-per-cpu` instead.
)
    """
    Run the commands in `commands_file` in parallel on a SLURM cluster.

    Two modes:
      :train  — CPU compute nodes. GNU parallel fans out across `n_cpus` cores per
                array task. Output script is `train_<timestamp>.sh`. USE_GPU=0.
      :test   — Each array task gets `n_gpus_per_node` NVIDIA GPU(s). GNU parallel
                runs `--jobs n_gpus_per_node` so each command holds one GPU
                (via `CUDA_VISIBLE_DEVICES` per slot, per Alliance Canada docs).
                Output script is `test_<timestamp>.sh`. USE_GPU=1, GPU_BACKEND=cuda.
                Requires CUDA.jl in the project; `module load \$(cuda_module)`.

    Per the Alliance Canada GPU docs:
      - `--gpus-per-node=<model>:<n>` is preferred over `--gres=gpu:<n>`.
      - Omitting the model specifier "may cause the job to be rejected or be
        sent to an arbitrary GPU" — so pass `gpu_type` (e.g. "h100", "a100").
      - Recommended max CPU cores per full GPU: Fir 12, Narval 12, Nibi 14,
        Rorqual 16 (cluster-dependent — see the docs).
    """
    if !(mode in (:train, :test))
        error("mode must be :train or :test, got :$(mode)")
    end

    is_test_mode = (mode == :test)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")

    target_dir = joinpath(working_dir, codename)
    commands_dir = joinpath(target_dir, "cluster")
    logs_dir = joinpath(commands_dir, "logs")
    if !isdir(logs_dir)
        mkpath(logs_dir)
    end

    # Script and log filenames carry the mode prefix so train/test artifacts coexist.
    script_prefix = is_test_mode ? "test" : "train"
    slurm_script_file = joinpath(commands_dir, "$(script_prefix)_$(timestamp).sh")
    output_file = joinpath(commands_dir, "$(script_prefix)_$(timestamp).out")
    error_file  = joinpath(commands_dir, "$(script_prefix)_$(timestamp).err")

    # In test mode each parallel job owns a GPU, so the concurrency limit per
    # node is `n_gpus_per_node`, not `n_cpus`.
    jobs_per_node = is_test_mode ? n_gpus_per_node : n_cpus
    n_nodes_needed = ceil(Int, n_commands / jobs_per_node)
    n_nodes = min(n_nodes_needed, max_nodes)
    commands_per_node = ceil(Int, n_commands / n_nodes)

    jobname = "$(script_prefix)_$(timestamp)"

    # --- SBATCH header (mode-dependent) ---
    # Memory directive: SLURM's `--mem-per-gpu`, `--mem-per-cpu`, `--mem`, and
    # `--mem-per-node` are mutually exclusive — specifying two causes the job to
    # be rejected or behave unpredictably. When the caller provides `mem_per_gpu`
    # AND we're in test mode (so GPUs are actually requested), we emit
    # `--mem-per-gpu` INSTEAD of `--mem-per-cpu`. Alliance Canada docs list
    # `--mem-per-gpu` as a supported GPU-resource directive.
    use_mem_per_gpu = is_test_mode && !isempty(mem_per_gpu)
    mem_directive = use_mem_per_gpu ? "#SBATCH --mem-per-gpu=$(mem_per_gpu)" :
                                      "#SBATCH --mem-per-cpu=$(mem_per_cpu)"

    sbatch_header = [
        "#!/bin/bash",
        "#SBATCH --account=$(account)",
        "#SBATCH --job-name=$(jobname)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --array=0-$(n_nodes-1)",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
        mem_directive,
        "#SBATCH --time=$(wall_time)",
        # --signal=B:TERM@<sec> asks SLURM to send SIGTERM to the batch shell
        # (B: → the batch script itself, not its children) `<sec>` seconds
        # BEFORE the wall clock expires.  We trap TERM in the body below to
        # tar-pipe partial results back to \$SCRATCH before SLURM's follow-up
        # SIGKILL wipes \$SLURM_TMPDIR.  300 s is a starting point — bump to
        # 600 if the codename's results/models trees are large enough that
        # tar-pipe over NFS takes longer than 5 minutes.
        "#SBATCH --signal=B:TERM@300",
    ]
    if is_test_mode
        # Per the Alliance Canada docs: `--gpus-per-node=<model_specifier>:<n>` is
        # preferred. Omitting the specifier "may cause the job to be rejected or
        # be sent to an arbitrary GPU". Warn if no gpu_type was supplied.
        gpus_spec = isempty(gpu_type) ? string(n_gpus_per_node) :
                                        "$(gpu_type):$(n_gpus_per_node)"
        push!(sbatch_header, "#SBATCH --gpus-per-node=$(gpus_spec)")
        if isempty(gpu_type)
            @warn """run_on_SLURM(mode=:test) called without `gpu_type`.
                    Alliance Canada may reject the job or assign an arbitrary GPU.
                    Pass e.g. gpu_type=\"h100\", \"a100\", \"l40s\", or \"h200\"."""
        end
    elseif !isempty(mem_per_gpu)
        # Non-test mode ignores mem_per_gpu since no GPU is requested. Flag it
        # so the user notices the value is being ignored rather than applied.
        @warn """`mem_per_gpu` = $(repr(mem_per_gpu)) is ignored outside test mode
                (train mode requests no GPU, so `--mem-per-gpu` is meaningless).
                Using `--mem-per-cpu=$(mem_per_cpu)` instead."""
    end
    append!(sbatch_header, [
        "",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=$(email_address)",
        "",
    ])

    module_load_lines = ["module load julia/1.12.5"]
    if is_test_mode
        push!(module_load_lines, "module load $(cuda_module)")
    end

    gpu_env_lines = if is_test_mode
        [
            "# Enable CUDA backend in CorrelatedBPDecoderWithCER",
            "export USE_GPU=1",
            "export GPU_BACKEND=cuda",
        ]
    else
        [
            "# Disable GPU usage",
            "export USE_GPU=0",
        ]
    end

    # In train mode we serialize each Julia process to 1 thread (GNU parallel spawns
    # n_cpus of them). In test mode, n_gpus_per_node Julia processes run concurrently,
    # each getting floor(n_cpus / n_gpus_per_node) threads to avoid oversubscription.
    threads_per_job = max(1, n_cpus ÷ n_gpus_per_node)
    thread_lines = if is_test_mode
        [
            "# Test mode: $(n_gpus_per_node) Julia process(es) per node, one per GPU.",
            "# Each one gets floor(cpus-per-task / gpus-per-node) = $(threads_per_job) threads.",
            "export JULIA_NUM_THREADS=$(threads_per_job)",
            "export OMP_NUM_THREADS=$(threads_per_job)",
            "export JULIA_NUM_PRECOMPILE_TASKS=1",
        ]
    else
        [
            "# Force 1 thread per process to avoid CPU thrashing with GNU parallel",
            "export JULIA_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export BLAS_NUM_THREADS=1",
            "export JULIA_NUM_PRECOMPILE_TASKS=1",
        ]
    end

    # (A) Force-recompile CUDA_Runtime_jll on this compute node BEFORE the
    # regular Pkg.precompile() step. Why:
    #   * The shared depot at $JULIA_DEPOT_PATH is used across nodes and jobs.
    #   * If a previous job died mid-precompile OR precompiled CUDA_Runtime_jll
    #     in an environment where the NVIDIA driver wasn't visible to the
    #     precompile subprocess (e.g. login node, or a compute node where the
    #     subprocess had a stripped env), the depot ends up with a .ji that
    #     hard-codes "no runtime available".
    #   * Pkg.precompile() below would treat that .ji as up-to-date (source
    #     hash matches) and never rebuild it — so CUDA.functional() would
    #     return false at runtime, even though nvidia-smi on this node shows
    #     working GPUs. This is precisely how we lost job 64946651.
    #   * Base.compilecache() ignores cache validity and rebuilds unconditionally.
    # Cost: ~10-70s per test-mode job start, one-time.
    cuda_prep_lines = is_test_mode ? [
        "",
        "######################################################################",
        "# CUDA_Runtime_jll cache refresh (mitigates stale-depot failures)",
        "######################################################################",
        "echo \"[cuda] force-recompiling CUDA_Runtime_jll on \$(hostname)...\"",
        "if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'pkg = Base.PkgId(Base.UUID(\"76a88914-d11a-5bdc-97e0-2f5a05c973a2\"), \"CUDA_Runtime_jll\"); Base.compilecache(pkg); println(\"[cuda] CUDA_Runtime_jll recompiled.\")'; then",
        "    echo \"[cuda] Base.compilecache(CUDA_Runtime_jll) failed — depot is unhealthy. Aborting job to save wall time.\" >&2",
        "    exit 1",
        "fi",
    ] : String[]

    # (B) CUDA visibility + strict sanity check. If CUDA.functional() returns
    # false we hard-exit BEFORE parallel launches, so the remaining wall time
    # isn't burned running 64 doomed julia invocations. The heredoc dumps a
    # diagnostic block into .err so postmortems don't require reading the
    # SLURM script — the recovery steps are visible in the failure log itself.
    cuda_sanity_lines = is_test_mode ? [
        "",
        "# --- CUDA visibility diagnostics (surface in .out for postmortems) ---",
        "# CUDA_Runtime_jll is configured via LocalPreferences.toml at the project",
        "# root to use the system CUDA (local_toolkit=true) — no artifact download",
        "# attempted. That's why `module load cuda` above must precede julia.",
        "echo \"[cuda] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}\"",
        "echo \"[cuda] nvcc version:\"",
        "nvcc --version 2>&1 | tail -1 | sed 's/^/[cuda]   /'",
        "if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @assert CUDA.functional(); println(\"CUDA OK: \", CUDA.name(CUDA.device()), \" (runtime v\", CUDA.runtime_version(), \")\")'; then",
        "    cat >&2 <<EOF",
        "",
        "===============================================================",
        "[cuda] SANITY CHECK FAILED on \$(hostname).",
        "[cuda] CUDA.functional() returned false EVEN AFTER the force-recompile",
        "[cuda] of CUDA_Runtime_jll above. Likely causes, in order of likelihood:",
        "[cuda]   1. LocalPreferences.toml at the project root is missing or",
        "[cuda]      does not contain [CUDA_Runtime_jll] local_toolkit = true.",
        "[cuda]      Check with:  cat \$SLURM_SUBMIT_DIR/../LocalPreferences.toml",
        "[cuda]   2. 'module load cuda' didn't populate CUDA_HOME / LD_LIBRARY_PATH.",
        "[cuda]      Check with:  echo \\\$CUDA_HOME  ;  which nvcc",
        "[cuda]   3. Driver / runtime version mismatch on this node.",
        "[cuda] Manual recovery (from a compute node via salloc):",
        "[cuda]     rm -rf \$JULIA_DEPOT_PATH/compiled/v1.12/CUDA*",
        "[cuda]     julia --project=. -e 'using Pkg; Pkg.precompile()'",
        "[cuda]     julia --project=. -e 'using CUDA; @assert CUDA.functional()'",
        "[cuda] Aborting to save the remaining wall time — no parallel commands launched.",
        "===============================================================",
        "EOF",
        "    exit 1",
        "fi",
    ] : String[]

    slurm_script_lines = vcat(
        sbatch_header,
        [
            "echo \"Running SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID}\"",
            "",
            "# Record exact start date and epoch time",
            "START_TIME_SEC=\$(date +%s)",
            "echo \"=========================================\"",
            "echo \"Job started at: \$(date)\"",
            "echo \"Mode: $(mode)\"",
            "echo \"=========================================\"",
            "",
            "# Determine line range for this task",
            "START=\$((SLURM_ARRAY_TASK_ID * $(commands_per_node) + 1))",
            "END=\$((START + $(commands_per_node) - 1))",
            "",
            "# Extract commands for this node and write them to the network target directory",
            "sed -n \"\${START},\${END}p\" $(commands_file) > $(commands_dir)/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
            "",
            "# Load necessary modules and set up environment",
        ],
        module_load_lines,
        [
            "# Copy the depot into node-local storage so precompilation and package",
            "# Depot policy: use \$SCRATCH/.julia (or the caller's JULIA_DEPOT_PATH)",
            "# DIRECTLY — no copy to \$SLURM_TMPDIR. Two reasons:",
            "#   1. Skipping the cp saves 1-3 min of job startup for a full depot.",
            "#   2. Any .ji files newly baked during this job (e.g. after a Julia or",
            "#      CUDA module version bump) persist to \$SCRATCH so the next job",
            "#      doesn't repeat the work. \$SLURM_TMPDIR would wipe them.",
            "# Julia's own file-lock (mkpidlock) protects against concurrent-job",
            "# precompile races on the shared depot.",
            "if [ -n \"\$JULIA_DEPOT_PATH\" ]; then",
            "    :  # respect what the caller (login-shell .bashrc) already set",
            "elif [ -n \"\$SCRATCH\" ] && [ -d \"\$SCRATCH/.julia\" ]; then",
            "    export JULIA_DEPOT_PATH=\"\$SCRATCH/.julia\"",
            "else",
            "    export JULIA_DEPOT_PATH=\"\$HOME/.julia\"",
            "fi",
            "echo \"[depot] JULIA_DEPOT_PATH=\$JULIA_DEPOT_PATH\"",
            "",
        ],
        gpu_env_lines,
        [
            "",
            "######################################################################",
            "# Thread Safety & Precompilation",
            "######################################################################",
        ],
        thread_lines,
        # Force-recompile CUDA_Runtime_jll (test mode only) BEFORE the generic
        # Pkg.precompile() step, so any stale .ji from a previous session gets
        # discarded rather than reused. See the `cuda_prep_lines` definition
        # above for the full rationale. In train mode this expands to nothing.
        cuda_prep_lines,
        [
            "",
            "# Safely instantiate and precompile on the local drive before launching parallel workers",
            "julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'",
            "# Disable concurrent auto-precompilation on the compute nodes to avoid deadlocks",
            "export JULIA_PKG_PRECOMPILE_AUTO=0",
        ],
        cuda_sanity_lines,
        [
            "",
            "######################################################################",
            "# 1. STAGE IN: Mirror Targeted Project Folder to Fast Local Storage",
            "######################################################################",
            "# `rsync -a` on many small files is dominated by per-file metadata",
            "# round-trips (open/stat/close per file), which is why the test-mode",
            "# codenames dir takes ~15 minutes to sync. `tar -cf - src | tar -xf -`",
            "# reads/writes the source once in bulk — typically 5-15× faster on",
            "# NFS/Lustre. No `-z` on purpose: over Alliance Canada's fast",
            "# interconnect gzip is CPU-bound and slower than the raw network.",
            "# Switch to `-czf | -xzf` if your link is < ~500 MB/s.",
            "LOCAL_WORK_DIR=\"\$SLURM_TMPDIR/$(codename)\"",
            "mkdir -p \"\$SLURM_TMPDIR\"",
            "",
            "echo \"Staging $(codename) into \$SLURM_TMPDIR via tar-pipe...\"",
            "STAGE_IN_START=\$(date +%s)",
            "tar -cf - -C \"\$(dirname $(target_dir))\" \"\$(basename $(target_dir))\" \\",
            "    | tar -xf - -C \"\$SLURM_TMPDIR\"",
            "echo \"[stage-in] done in \$(( \$(date +%s) - STAGE_IN_START ))s\"",
            "",
            "# Dynamically replace whatever string follows --workdir with the fast local path.",
            "sed -i \"s|--workdir [^ ]*|--workdir \$SLURM_TMPDIR|g\" \$LOCAL_WORK_DIR/cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
            "",
            "######################################################################",
            "# 2. COMPUTE",
            "######################################################################",
            "LOCAL_LOGS=\"\$LOCAL_WORK_DIR/cluster/logs/node_\${SLURM_ARRAY_TASK_ID}\"",
            "mkdir -p \$LOCAL_LOGS",
            "",
            "######################################################################",
            "# 2a. STAGE-OUT FUNCTION (fires on normal exit OR on SLURM SIGTERM)",
            "######################################################################",
            "# `stage_out` copies whatever's currently in \$LOCAL_WORK_DIR/{results,",
            "# models, cluster/logs} back to the persistent \$SCRATCH tree via",
            "# tar-pipe. Called from two places (both via `trap`):",
            "#   * EXIT: fires on normal script exit (compute finished cleanly).",
            "#   * TERM: fires when SLURM sends SIGTERM \$300s before wall time",
            "#           (see #SBATCH --signal=B:TERM@300 above).  Without this",
            "#           trap, cancellation-by-timeout wipes \$SLURM_TMPDIR and",
            "#           every partial result JSON with it — that's how we lost",
            "#           the entire 5h run on 2026-07-08.",
            "# The `stage_out_done` guard means an EXIT that follows a TERM only",
            "# does the copy once.",
            "stage_out_done=0",
            "stage_out() {",
            "    [ \"\$stage_out_done\" = \"1\" ] && return 0",
            "    stage_out_done=1",
            "    echo \"[stage-out] triggered at \$(date '+%F %T')\"",
            "    STAGE_OUT_START=\$(date +%s)",
            "    STAGE_OUT_DIRS=()",
            "    for d in results models cluster/logs; do",
            "        [ -d \"\$LOCAL_WORK_DIR/\$d\" ] && STAGE_OUT_DIRS+=(\"\$d\")",
            "    done",
            "    if [ \${#STAGE_OUT_DIRS[@]} -gt 0 ]; then",
            "        echo \"[stage-out] copying back: \${STAGE_OUT_DIRS[*]}\"",
            "        tar -cf - -C \"\$LOCAL_WORK_DIR\" \"\${STAGE_OUT_DIRS[@]}\" \\",
            "            | tar -xf - -C \"$(target_dir)\"",
            "    else",
            "        echo \"[stage-out] no new artifacts detected under \$LOCAL_WORK_DIR — nothing to copy.\"",
            "    fi",
            "    echo \"[stage-out] done in \$(( \$(date +%s) - STAGE_OUT_START ))s\"",
            "}",
            "# `parallel` is launched in the BACKGROUND and `wait`ed on (below), so",
            "# TERM (from --signal=B:TERM@300, 5 min before wall) interrupts the",
            "# `wait` and runs this handler immediately. `exit` after stage_out",
            "# stops us resuming the wait. A FOREGROUND `parallel` would instead",
            "# DEFER the trap until it returned — which on a slow/contended node it",
            "# never does before the wall SIGKILL, so stage_out never runs and the",
            "# whole localscratch (all partial results) is wiped (the 2026-07-27 loss).",
            "term_handler() { stage_out; exit 0; }",
            "trap term_handler TERM",
            "trap stage_out EXIT",
            "",
            "# Anchor execution to the exact directory where you ran 'sbatch'",
            "cd \$SLURM_SUBMIT_DIR",
            "",
            "echo \"Running parallel computations...\"",
            "# In test mode with n_gpus_per_node > 1 we set CUDA_VISIBLE_DEVICES",
            "# per parallel-slot so concurrent workers don't fight over the same",
            "# GPU. See Alliance Canada docs: \"Packing single-GPU jobs within one",
            "# SLURM job\". Two subtleties are baked into the override below —",
            "#",
            "# 1. MIG-safe slicing (was a bug pre-2026-07-09).  SLURM exposes an",
            "#    allocation of N MIG partitions as CUDA_VISIBLE_DEVICES=\"MIG-uuid1,",
            "#    MIG-uuid2,...\".  Overriding that with an integer index like \"0\"",
            "#    hides the allocated MIG partition from the CUDA runtime, and the",
            "#    julia process silently falls back to CPU — no error, wall time",
            "#    burned.  We instead capture SLURM's original list into",
            "#    SLURM_CUDA_VISIBLE_DEVICES and hand each worker its own field",
            "#    from that list via `cut -d, -f{%}`.  This works for BOTH cases:",
            "#      * whole GPUs: SLURM gives \"0,1\" → worker 1 gets \"0\", worker 2 gets \"1\"",
            "#      * MIG:        SLURM gives \"MIG-a,MIG-b\" → worker 1 gets \"MIG-a\", worker 2 gets \"MIG-b\"",
            "#",
            "# 2. Shell-quoting via `bash -c {}`.  GNU parallel shell-quotes {}",
            "#    before substitution, so without `bash -c` the shell would try",
            "#    to exec the whole julia command line as a single filename and",
            "#    fail with \"No such file or directory\".  `bash -c '<quoted-cmd>'`",
            "#    re-parses it as a shell command.",
            "#",
            "# For n_gpus_per_node == 1 (whether whole GPU or MIG), we don't",
            "# override — SLURM's single CUDA_VISIBLE_DEVICES value is already",
            "# correct for the one julia process we're going to spawn.",
            is_test_mode && n_gpus_per_node > 1 ?
                "export SLURM_CUDA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\"" :
                "",
            is_test_mode && n_gpus_per_node > 1 ?
                "echo \"[cuda] per-slot slicing from SLURM_CUDA_VISIBLE_DEVICES=\$SLURM_CUDA_VISIBLE_DEVICES\"" :
                "",
            # Background the run and wait on it, so the TERM trap above can fire
            # and rescue partial results before the wall SIGKILL. `wait $!` returns
            # the instant a trapped signal arrives; a foreground `parallel` would
            # swallow the signal until it finished. `wait` also propagates the
            # backgrounded exit status, so a genuine failure still fails the job.
            is_test_mode && n_gpus_per_node > 1 ?
                "parallel --jobs $(jobs_per_node) --results \$LOCAL_LOGS 'CUDA_VISIBLE_DEVICES=\$(echo \"\$SLURM_CUDA_VISIBLE_DEVICES\" | cut -d, -f{%}) bash -c {}' :::: \$LOCAL_WORK_DIR/cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt &" :
                "parallel --jobs $(jobs_per_node) --results \$LOCAL_LOGS < \$LOCAL_WORK_DIR/cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt &",
            "wait \$!",
            "",
            "# NOTE: stage-out is NOT done inline here.  The `stage_out` bash",
            "# function defined above is wired to the EXIT trap, so it fires",
            "# when the script exits normally (after this timing block runs)",
            "# AND to the TERM trap, so it also fires when SLURM cancels us",
            "# 300 s before wall-time expires.  See section 2a for the trap.",
            "echo \"Computation finished. Timing block will run now; stage-out fires on exit.\"",
            "",
            "######################################################################",
            "# 3. Log elapsed time.",
            "######################################################################",
            "END_TIME_SEC=\$(date +%s)",
            "DURATION_SEC=\$((END_TIME_SEC - START_TIME_SEC))",
            "HOURS=\$((DURATION_SEC / 3600))",
            "MINUTES=\$(((DURATION_SEC % 3600) / 60))",
            "SECONDS=\$((DURATION_SEC % 60))",
            "echo \"\"",
            "echo \"=========================================\"",
            "echo \"Job finished at: \$(date)\"",
            "echo \"Total job execution time: \${HOURS}h \${MINUTES}m \${SECONDS}s\"",
            "echo \"=========================================\"",
        ],
    )
    
    open(slurm_script_file, "w") do io
        println(io, join(slurm_script_lines, "\n"))
    end

    println("\n=== Job Details ===")
    println("  Mode             : $(mode)")
    println("  Job name         : $jobname")
    println("  Target codename  : $codename")
    println("  Total commands   : $n_commands")
    println("  CPUs per task    : $n_cpus")
    if is_test_mode
        println("  GPUs per node    : $n_gpus_per_node")
        println("  CUDA module      : $cuda_module")
    end
    if use_mem_per_gpu
        println("  Mem per GPU      : $mem_per_gpu  (overrides mem_per_cpu in test mode)")
    else
        println("  Mem per CPU      : $mem_per_cpu")
    end
    println("  Nodes in use     : $n_nodes")
    println("  Commands per node: $commands_per_node")
    println("  Jobs per node    : $jobs_per_node")
    println("  Wall time        : $wall_time")
    println("\n=== Output Files ===")
    println("  Commands file    : $commands_file")
    println("  SLURM script     : $slurm_script_file")
    println("\n=== Submission ===")
    println("  sbatch $slurm_script_file\n")

    return slurm_script_file
end

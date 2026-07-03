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
    isdir(logs_dir) || mkpath(logs_dir)

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
    sbatch_header = [
        "#!/bin/bash",
        "#SBATCH --account=$(account)",
        "#SBATCH --job-name=$(jobname)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --array=0-$(n_nodes-1)",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
        "#SBATCH --mem-per-cpu=$(mem_per_cpu)",
        "#SBATCH --time=$(wall_time)",
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
            "export USE_GPU=\"1\"",
            "export GPU_BACKEND=\"cuda\"",
        ]
    else
        [
            "# Disable GPU usage",
            "export USE_GPU=\"0\"",
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

    cuda_sanity_lines = is_test_mode ? [
        "",
        "# Quick CUDA sanity check — fails loudly if CUDA.jl can't see a GPU.",
        "julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @assert CUDA.functional(); println(\"CUDA OK: \", CUDA.name(CUDA.device()))'",
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
            "cp -r ~/.julia \$SLURM_TMPDIR/",
            "export JULIA_DEPOT_PATH=\"\$SLURM_TMPDIR/.julia\"",
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
            "# Anchor execution to the exact directory where you ran 'sbatch'",
            "cd \$SLURM_SUBMIT_DIR",
            "",
            "echo \"Running parallel computations...\"",
            "# In test mode we set CUDA_VISIBLE_DEVICES per parallel-slot (Alliance",
            "# Canada docs: \"Packing single-GPU jobs within one SLURM job\") so",
            "# concurrent jobs don't fight over GPU 0. {%} is parallel's slot index",
            "# (1..n_gpus_per_node); single quotes defer \$((...)) evaluation.",
            "# `bash -c {}` is required: GNU parallel shell-quotes {} before",
            "# substitution, so without `bash -c` the shell tries to exec the whole",
            "# julia command line as a single filename and errors with \"No such file",
            "# or directory\". `bash -c '<quoted-cmd>'` re-parses it as a shell command.",
            is_test_mode ?
                "parallel --jobs $(jobs_per_node) --results \$LOCAL_LOGS 'CUDA_VISIBLE_DEVICES=\$(({%} - 1)) bash -c {}' :::: \$LOCAL_WORK_DIR/cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt" :
                "parallel --jobs $(jobs_per_node) --results \$LOCAL_LOGS < \$LOCAL_WORK_DIR/cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
            "",
            "######################################################################",
            "# 3. STAGE OUT: tar-pipe only the new/modified subdirs back",
            "######################################################################",
            "# Only a handful of subdirectories actually change during compute —",
            "# staging the whole codename dir back is wasted work (and the source",
            "# of the ~15-minute rsync). We tar-pipe just the paths that could",
            "# have new content: results/ (test outputs), models/ (train weights),",
            "# cluster/logs/ (per-parallel-worker stdout+stderr). Missing dirs are",
            "# skipped silently — this handles both train and test modes without",
            "# a mode check here.",
            "echo \"Computation finished. Staging new artifacts back via tar-pipe...\"",
            "STAGE_OUT_START=\$(date +%s)",
            "STAGE_OUT_DIRS=()",
            "for d in results models cluster/logs; do",
            "    [ -d \"\$LOCAL_WORK_DIR/\$d\" ] && STAGE_OUT_DIRS+=(\"\$d\")",
            "done",
            "if [ \${#STAGE_OUT_DIRS[@]} -gt 0 ]; then",
            "    echo \"[stage-out] copying back: \${STAGE_OUT_DIRS[*]}\"",
            "    tar -cf - -C \"\$LOCAL_WORK_DIR\" \"\${STAGE_OUT_DIRS[@]}\" \\",
            "        | tar -xf - -C \"$(target_dir)\"",
            "else",
            "    echo \"[stage-out] no new artifacts detected under \$LOCAL_WORK_DIR — nothing to copy.\"",
            "fi",
            "echo \"[stage-out] done in \$(( \$(date +%s) - STAGE_OUT_START ))s\"",
            "",
            "echo \"Job completed and data safely transferred.\"",
            "",
            "######################################################################",
            "# 4. Log elapsed time.",
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
    println("  Mem per CPU      : $mem_per_cpu")
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

# hp_sweep/cluster.sh -- the CLUSTER side: the cluster profile in, the two SLURM
# job scripts out.
#
# Sourced by sweep_hyperparams.sh; defines functions only. Everything here is the
# same for every sweep: how one array task is requested (one card, its cores and
# host RAM, from the profile), what it copies to the node, how the Julia depot and
# CUDA are prepared, how GNU parallel runs the points, and what is copied back.
# The sweep reaches it only through the lists hp_sweep/sweep.sh writes (TRAIN_CMDS,
# TEST_CMDS, EXPECTED_MODELS, the point TOMLs) and four numbers of the sweep file:
# train_array_tasks, train_wall_time, test_array_tasks, test_wall_time.
#
# Both jobs run on GPU nodes (2026-10-15). Training uses the hand-written backward
# pass on the card (manual_backprop = true, which the base TOML must say, and
# USE_GPU=1), train_jobs trainings sharing one card; testing runs one process per
# card. The test job is chained on the training job with afterok.

# Every key a cluster profile holds; all of them are required (account_gpu,
# stdenv_module, vram_per_gpu, jobs_write_under and submit_host may be ""). Described
# in clusters/narval.toml; clusters/trillium.toml uses the site-specific ones.
CLUSTER_KEYS="account_gpu email stdenv_module julia_module cuda_module heap_size_hint gpu_type gpu_request_style n_gpus_per_node request_cpus_and_memory mem_per_gpu vram_per_gpu train_jobs train_cpus test_jobs test_cpus jobs_write_under submit_host seconds_per_epoch_tanh_wave seconds_per_epoch_enriched_wave seconds_per_test job_startup_seconds"

cluster_value() { settings_value "$CLUSTER_FILE" "$1"; }

megabytes_of() {   # <size like 4G, 32G, 2048M>
    echo "$1" | awk '{u=toupper($0); v=u; gsub(/[^0-9.]/,"",v);
        if (u ~ /G/) printf "%d", v*1024; else printf "%d", v}'
}

# ------------------------------------------------------ the cluster profile ---
# Reads the profile named by the sweep file's `cluster` key into globals and
# checks that one array task can hold what the jobs put on it. Sets ACCOUNT_GPU,
# EMAIL, STDENV_MODULE, JULIA_MODULE, CUDA_MODULE, HEAP, GPU_TYPE, GPU_REQUEST_STYLE,
# N_GPUS, REQUEST_CPUS_AND_MEMORY, MEM_PER_GPU, VRAM_PER_GPU, TRAIN_JOBS, TRAIN_CPUS,
# TEST_JOBS, TEST_CPUS, JOBS_WRITE_UNDER, SUBMIT_HOST, the measured costs, and the
# derived GPU_SBATCH_LINE, GPU_MEMORY_MB and JOB_DEPOT_PATH.
read_cluster_profile() {
    check_settings_keys "$CLUSTER_FILE" "cluster profile" "$CLUSTER_KEYS" "$CLUSTER_KEYS" \
        "$SWEEP_KEYS" "the sweep file ($SWEEP_FILE)" "clusters/narval.toml"
    ACCOUNT_GPU="$(cluster_value account_gpu)"
    EMAIL="$(cluster_value email)"
    STDENV_MODULE="$(cluster_value stdenv_module)"
    JULIA_MODULE="$(cluster_value julia_module)"
    CUDA_MODULE="$(cluster_value cuda_module)"
    HEAP="$(cluster_value heap_size_hint)"
    GPU_TYPE="$(cluster_value gpu_type)"
    GPU_REQUEST_STYLE="$(cluster_value gpu_request_style)"
    N_GPUS="$(cluster_value n_gpus_per_node)"
    REQUEST_CPUS_AND_MEMORY="$(cluster_value request_cpus_and_memory)"
    MEM_PER_GPU="$(cluster_value mem_per_gpu)"
    VRAM_PER_GPU="$(cluster_value vram_per_gpu)"
    TRAIN_JOBS="$(cluster_value train_jobs)"
    TRAIN_CPUS="$(cluster_value train_cpus)"
    TEST_JOBS="$(cluster_value test_jobs)"
    TEST_CPUS="$(cluster_value test_cpus)"
    JOBS_WRITE_UNDER="$(cluster_value jobs_write_under)"
    SUBMIT_HOST="$(cluster_value submit_host)"
    SECONDS_PER_EPOCH_TANH_WAVE="$(cluster_value seconds_per_epoch_tanh_wave)"
    SECONDS_PER_EPOCH_ENRICHED_WAVE="$(cluster_value seconds_per_epoch_enriched_wave)"
    SECONDS_PER_TEST="$(cluster_value seconds_per_test)"
    JOB_STARTUP_SECONDS="$(cluster_value job_startup_seconds)"

    case "$REQUEST_CPUS_AND_MEMORY" in
        true|false) ;;
        *) echo "request_cpus_and_memory = '$REQUEST_CPUS_AND_MEMORY' in $CLUSTER_PROFILE: use true or false." >&2; exit 1 ;;
    esac
    if [ -n "$JOBS_WRITE_UNDER" ] && ! echo "$JOBS_WRITE_UNDER" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*$'; then
        echo "jobs_write_under = '$JOBS_WRITE_UNDER' in $CLUSTER_PROFILE: the NAME of an environment variable, e.g. SCRATCH." >&2
        exit 1
    fi
    local text_key
    for text_key in email julia_module cuda_module heap_size_hint gpu_type mem_per_gpu; do
        if [ -z "$(cluster_value "$text_key")" ]; then
            echo "$text_key is empty in $CLUSTER_PROFILE." >&2
            exit 1
        fi
    done
    local integer_key integer_value
    for integer_key in n_gpus_per_node train_jobs train_cpus test_jobs test_cpus \
                       seconds_per_epoch_tanh_wave seconds_per_epoch_enriched_wave \
                       seconds_per_test job_startup_seconds; do
        integer_value="$(cluster_value "$integer_key")"
        if ! echo "$integer_value" | grep -qE '^[1-9][0-9]*$'; then
            echo "$integer_key = '$integer_value' in $CLUSTER_PROFILE must be a positive integer." >&2
            exit 1
        fi
    done

    # A training task gets the cores ONE card comes with, as a test task does. A
    # larger number is almost certainly a CPU-node value (54), which SLURM would
    # refuse for this allocation.
    if [ "$TRAIN_CPUS" -gt $(( TEST_CPUS * N_GPUS )) ]; then
        echo "train_cpus = $TRAIN_CPUS is more than the $(( TEST_CPUS * N_GPUS )) cores of n_gpus_per_node = $N_GPUS" >&2
        echo "  card(s) (test_cpus each) in $CLUSTER_PROFILE. Training runs on GPU nodes; set train_cpus <= test_cpus." >&2
        exit 1
    fi
    if [ "$TRAIN_CPUS" -lt "$TRAIN_JOBS" ]; then
        echo "train_cpus = $TRAIN_CPUS < train_jobs = $TRAIN_JOBS in $CLUSTER_PROFILE: every training needs a core" >&2
        echo "  of its own for its host-side work. Raise train_cpus or lower train_jobs." >&2
        exit 1
    fi
    # The training job runs train_jobs julia processes in ONE GPU allocation, whose
    # host RAM is n_gpus_per_node x mem_per_gpu. Each may grow to its heap_size_hint
    # (and reads the 1e6-sample training file), so their sum must fit. Getting this
    # wrong is not a slow job but processes killed at OOM after the queue wait.
    local heap_megabytes train_host_megabytes
    heap_megabytes="$(megabytes_of "$HEAP")"
    train_host_megabytes=$(( N_GPUS * $(megabytes_of "$MEM_PER_GPU") ))
    if [ $(( TRAIN_JOBS * heap_megabytes )) -gt "$train_host_megabytes" ]; then
        echo "train_jobs x heap_size_hint = ${TRAIN_JOBS} x ${HEAP} exceeds the training task's host RAM," >&2
        echo "  n_gpus_per_node x mem_per_gpu = ${N_GPUS} x ${MEM_PER_GPU} ($CLUSTER_PROFILE)." >&2
        echo "  Lower train_jobs or raise mem_per_gpu." >&2
        exit 1
    fi

    # VRAM per card, for the prediction batch sizer. This is NOT --mem-per-gpu,
    # which is host RAM: GPU_MEMORY has to fit the CARD or the batch is sized too
    # large and the run dies at cuDevicePrimaryCtxRetain.
    if [ -z "$VRAM_PER_GPU" ]; then
        case "$GPU_TYPE" in
            h100_1g.10gb) VRAM_PER_GPU=10 ;;
            h100_2g.20gb) VRAM_PER_GPU=20 ;;
            h100_3g.40gb) VRAM_PER_GPU=40 ;;
            h100_80gb)    VRAM_PER_GPU=80 ;;
            a100_1g.5gb)  VRAM_PER_GPU=5  ;;
            a100_2g.10gb) VRAM_PER_GPU=10 ;;
            a100_3g.20gb) VRAM_PER_GPU=20 ;;
            a100)         VRAM_PER_GPU=40 ;;
            h100)         VRAM_PER_GPU=80 ;;
            v100*)        VRAM_PER_GPU=32 ;;
            *) echo "unknown gpu_type '$GPU_TYPE' in $CLUSTER_PROFILE: set vram_per_gpu explicitly." >&2; exit 1 ;;
        esac
    fi
    if ! echo "$VRAM_PER_GPU" | grep -qE '^[1-9][0-9]*$'; then
        echo "vram_per_gpu = '$VRAM_PER_GPU' in $CLUSTER_PROFILE must be a whole number of GB (or \"\")." >&2
        exit 1
    fi

    # Nibi requests MIG instances with "--gpus=<name>:1" and full cards with
    # "--gpus-per-node=<name>:<n>". A MIG type is one naming a <k>g.<m>gb slice.
    # Trillium takes a bare count, "--gpus-per-node=1" (1 or 4; no type, no MIG).
    case "$GPU_REQUEST_STYLE" in
        gpus-per-node) GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
        gpus)          GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
        count)         GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${N_GPUS}" ;;
        auto)
            case "$GPU_TYPE" in
                *g.*gb) GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
                *)      GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
            esac ;;
        *) echo "unknown gpu_request_style '$GPU_REQUEST_STYLE' in $CLUSTER_PROFILE: use auto, gpus, gpus-per-node or count." >&2; exit 1 ;;
    esac

    # 85% of one card, shared by the test processes assigned to it.
    local jobs_per_gpu=$(( TEST_JOBS / N_GPUS ))
    if [ "$jobs_per_gpu" -lt 1 ]; then
        jobs_per_gpu=1
    fi
    GPU_MEMORY_MB=$(( VRAM_PER_GPU * 1024 * 85 / (100 * jobs_per_gpu) ))

    # The Julia depot the jobs use, decided HERE and written into the job scripts:
    # JULIA_DEPOT_PATH as this shell has it (the shell that ran precompile_depot.sh),
    # else $SCRATCH/.julia followed by Julia's bundled depots (the trailing ":"), which
    # is precompile_depot.sh's own default. Written in rather than left to the job's
    # environment, because a cluster may start jobs with a fresh one (Trillium's
    # debugjob does) and a job that falls back to another depot recompiles everything
    # on the compute node -- the 2026-10-22 failure. Empty when this shell has neither
    # variable (a laptop): the jobs then decide at run time, as before.
    JOB_DEPOT_PATH=""
    if [ -n "${JULIA_DEPOT_PATH:-}" ]; then
        JOB_DEPOT_PATH="$JULIA_DEPOT_PATH"
    elif [ -n "${SCRATCH:-}" ]; then
        JOB_DEPOT_PATH="$SCRATCH/.julia:"
    fi
}

# ------------------------------------------------ where jobs may write, and from where to submit ---
# On a cluster whose compute nodes can write under one file system only (Trillium:
# $HOME and $PROJECT are read-only from jobs, jobs_write_under = "SCRATCH"), refuses a
# sweep whose run directory or Julia depot lies elsewhere: the jobs write the stage-out,
# their .out/.err files and the depot lock into the run directory, and Julia writes
# its caches into the depot. A submit directory elsewhere only gets a warning (the
# jobs cd there and read the code, but write nothing in it).
path_is_under() {   # <path> <root>
    case "$1/" in
        "$2"/*) return 0 ;;
        *) return 1 ;;
    esac
}

check_job_write_locations() {
    if [ -z "$JOBS_WRITE_UNDER" ]; then
        return 0
    fi
    local writable_root=""
    eval "writable_root=\"\${$JOBS_WRITE_UNDER:-}\""
    if [ -z "$writable_root" ] || [ ! -d "$writable_root" ]; then
        echo "jobs_write_under = \"$JOBS_WRITE_UNDER\" ($CLUSTER_PROFILE), but \$$JOBS_WRITE_UNDER is not a directory" >&2
        echo "  in this shell. Generate this sweep on $CLUSTER_NAME, where it is." >&2
        exit 1
    fi
    local resolved_root resolved_run_directory depot resolved_depot problems=""
    resolved_root="$(cd "$writable_root" && pwd -P)"
    resolved_run_directory="$(cd "$WORKDIR/$CODENAME" && pwd -P)"
    if ! path_is_under "$resolved_run_directory" "$resolved_root"; then
        problems="${problems}    run directory  $resolved_run_directory
"
    fi
    depot="${JOB_DEPOT_PATH%%:*}"
    resolved_depot="$depot"
    if [ -d "$depot" ]; then
        resolved_depot="$(cd "$depot" && pwd -P)"
    fi
    if [ -z "$depot" ]; then
        problems="${problems}    Julia depot    (none: JULIA_DEPOT_PATH and SCRATCH are both unset)
"
    elif ! path_is_under "$resolved_depot" "$resolved_root" && ! path_is_under "$depot" "$writable_root"; then
        problems="${problems}    Julia depot    $depot   (the first entry of JULIA_DEPOT_PATH)
"
    fi
    if [ -n "$problems" ]; then
        echo "On $CLUSTER_NAME the jobs can write only under \$$JOBS_WRITE_UNDER ($writable_root), but these are elsewhere:" >&2
        printf '%s' "$problems" >&2
        echo "  Put the run directory there (the repository, or its data/ as a link to" >&2
        echo "  \$$JOBS_WRITE_UNDER/...), and the depot: unset JULIA_DEPOT_PATH for \$$JOBS_WRITE_UNDER/.julia, or" >&2
        echo "  point it there. Warm it with misc/precompile_depot.sh in the same shell." >&2
        exit 1
    fi
    if ! path_is_under "$(pwd -P)" "$resolved_root"; then
        echo "[hp_sweep] note: you are in $(pwd -P), outside \$$JOBS_WRITE_UNDER; $CLUSTER_NAME's documentation asks" >&2
        echo "           for jobs to be submitted from \$$JOBS_WRITE_UNDER. The jobs write nothing here, but" >&2
        echo "           moving the repository there is the safe choice." >&2
    fi
}

# Some clusters take GPU jobs only from one login node (Trillium: trig-login01).
check_submit_host() {
    if [ -z "$SUBMIT_HOST" ]; then
        return 0
    fi
    local this_host
    this_host="$(hostname -s 2>/dev/null || hostname)"
    case "$this_host" in
        $SUBMIT_HOST) ;;
        *)
            echo "[hp_sweep] note: $CLUSTER_NAME takes these GPU jobs from $SUBMIT_HOST; this is $this_host." >&2
            echo "           Run the sbatch lines (and misc/precompile_depot.sh) there." >&2
            ;;
    esac
}

# ------------------------------------------------------- stage-in lists ---
# What each job copies to the node, as paths under $WORKDIR (a directory goes
# whole): the code, the CER files, the sweep's own TOMLs, and the training files
# (training job) or the test files and trained models (test job). Both jobs used
# to copy the whole codename, ~30 GB of data sets per task for
# 72q_BB_cycles_1_sweep (208 files of 144 MB).
write_stage_lists() {
    local key tested_keys="$TEST_KEYS"
    if [ -z "$(echo $tested_keys)" ]; then
        tested_keys="$DATASETS $REF_DATASETS"
    fi
    {
        echo "$CODENAME/code"
        echo "$CODENAME/correlated_weights"
        for key in $DATASETS $REF_DATASETS; do
            echo "$CODENAME/training_data/train_${key}.txt"
        done
        sed "s|^|$CODENAME/models/|" "$EMITTED_TOML_NAMES"
    } | awk '!seen[$0]++' > "$TRAIN_STAGE_LIST"
    {
        echo "$CODENAME/code"
        echo "$CODENAME/correlated_weights"
        for key in $tested_keys; do
            echo "$CODENAME/testing_data/test_${key}.txt"
        done
        sed "s|^|$CODENAME/models/|" "$EMITTED_TOML_NAMES"
        sed "s|^|$CODENAME/models/|" "$EXPECTED_MODELS"
    } | awk '!seen[$0]++' > "$TEST_STAGE_LIST"
}

# ------------------------------------------------------ walltime preflight ---
# A job that runs out of walltime loses whatever it had not staged out, and the
# loss is SILENT: the 2026-10-01 test job returned 175 of 280 results and the tasks
# that never ran left .out files with one line in them. Estimate here from the
# profile's measured costs, and refuse to write a job that cannot finish.
walltime_to_seconds() {   # <[[D-]HH:]MM:SS>
    echo "$1" | awk -F: '{
        if (NF == 3) { split($1, dayhour, "-");
                       if (length(dayhour) == 2) { print ((dayhour[1]*24 + dayhour[2])*60 + $2)*60 + $3 }
                       else                      { print ($1*60 + $2)*60 + $3 } }
        else if (NF == 2) { print $1*60 + $2 }
        else { print $1 }
    }'
}

print_walltime_line() {   # <label> <per-task> <estimate> <budget> <wall>
    local label="$1" per_task="$2" estimate="$3" budget="$4" wall="$5"
    local percent=$(( 100 * estimate / budget ))
    printf "  %-6s %4s per task   est %dh%02dm of %-9s (%d%% of budget)" \
        "$label" "$per_task" $(( estimate / 3600 )) $(( (estimate % 3600) / 60 )) "$wall" "$percent"
    if [ "$percent" -ge 100 ]; then
        printf "   <-- WILL NOT FINISH\n"
    elif [ "$percent" -ge 70 ]; then
        printf "   <-- tight\n"
    else
        printf "\n"
    fi
}

walltime_preflight() {
    # TRAINING: a task runs its slice through GNU parallel with --jobs train_jobs on
    # its card, so its points go in concurrent waves of train_jobs. A wave costs
    # n_epochs x (one epoch of a wave, measured) x (gradient updates per epoch / 500,
    # the measured count) -- batch_run.jl's estimate. An enriched or bethe arm sets
    # the cost of every wave: the interleaved split mixes the arms within a task, and
    # the slowest point of a wave sets its length.
    local seconds_per_epoch="$SECONDS_PER_EPOCH_TANH_WAVE" wave_kind="tanh" check_node_arm
    for check_node_arm in $CHECK_NODE_ARMS; do
        case "${check_node_arm%%:*}" in
            enriched|bethe) seconds_per_epoch="$SECONDS_PER_EPOCH_ENRICHED_WAVE"; wave_kind="enriched" ;;
        esac
    done
    TRAIN_WAVE_SECONDS="$(awk -v e="$N_EPOCHS" -v s="$seconds_per_epoch" -v u="$N_UPDATES_PER_EPOCH" \
        'BEGIN { w = e * s * u / 500; printf "%d", (w == int(w)) ? w : int(w) + 1 }')"
    local train_points_per_task=$(( (N_POINTS + TRAIN_ARRAY - 1) / TRAIN_ARRAY ))
    local train_waves=$(( (train_points_per_task + TRAIN_JOBS - 1) / TRAIN_JOBS ))
    local train_estimate=$(( JOB_STARTUP_SECONDS + train_waves * TRAIN_WAVE_SECONDS ))
    local train_budget
    train_budget="$(walltime_to_seconds "$TRAIN_WALL")"
    # TESTING: strictly sequential within a task (test_jobs = 1, one process per card).
    local test_points_per_task=$(( (N_TEST_POINTS + TEST_ARRAY - 1) / TEST_ARRAY ))
    local test_estimate=$(( JOB_STARTUP_SECONDS + test_points_per_task * SECONDS_PER_TEST ))
    local test_budget
    test_budget="$(walltime_to_seconds "$TEST_WALL")"

    local updates_note=""
    if [ "$N_UPDATES_PER_EPOCH" -ne 500 ]; then
        updates_note=" x $N_UPDATES_PER_EPOCH/500 updates"
    fi
    echo
    echo "[hp_sweep] walltime preflight ($CLUSTER_NAME: ${JOB_STARTUP_SECONDS}s startup, ${SECONDS_PER_TEST}s per test;"
    echo "           a training wave of ${TRAIN_JOBS} = ${N_EPOCHS} epochs x ${seconds_per_epoch}s (${wave_kind})${updates_note} = ${TRAIN_WAVE_SECONDS}s)"
    print_walltime_line "train" "$train_points_per_task" "$train_estimate" "$train_budget" "$TRAIN_WALL"
    print_walltime_line "test"  "$test_points_per_task"  "$test_estimate"  "$test_budget"  "$TEST_WALL"

    local overrun=0 waves_that_fit minimum_tasks
    if [ "$train_estimate" -ge "$train_budget" ]; then
        waves_that_fit=$(( (train_budget - JOB_STARTUP_SECONDS - 1) / TRAIN_WAVE_SECONDS ))
        if [ "$train_budget" -gt "$JOB_STARTUP_SECONDS" ] && [ "$waves_that_fit" -ge 1 ]; then
            minimum_tasks=$(( (N_POINTS + waves_that_fit * TRAIN_JOBS - 1) / (waves_that_fit * TRAIN_JOBS) ))
            echo "  training cannot finish: raise train_wall_time, or train_array_tasks to ${minimum_tasks}+." >&2
        else
            echo "  training cannot finish: one wave alone needs $(( JOB_STARTUP_SECONDS + TRAIN_WAVE_SECONDS ))s; raise train_wall_time." >&2
        fi
        overrun=1
    fi
    local tests_that_fit
    if [ "$test_estimate" -ge "$test_budget" ]; then
        tests_that_fit=$(( (test_budget - JOB_STARTUP_SECONDS - 1) / SECONDS_PER_TEST ))
        if [ "$test_budget" -gt "$JOB_STARTUP_SECONDS" ] && [ "$tests_that_fit" -ge 1 ]; then
            minimum_tasks=$(( (N_TEST_POINTS + tests_that_fit - 1) / tests_that_fit ))
            echo "  testing cannot finish: raise test_wall_time, or test_array_tasks to ${minimum_tasks}+." >&2
        else
            echo "  testing cannot finish: one test alone needs $(( JOB_STARTUP_SECONDS + SECONDS_PER_TEST ))s; raise test_wall_time." >&2
        fi
        overrun=1
    fi
    if [ "$overrun" -eq 1 ]; then
        echo "  Nothing was written. Adjust $SWEEP_FILE and re-run." >&2
        exit 1
    fi
}

# --------------------------------------------------------- depot preflight ---
# Is the depot the jobs will use precompiled for THIS source? If the package
# changed since its cache was written, the jobs compile it on the compute nodes:
# the lock owner twice (before and after the CUDA_Runtime_jll rebuild), and every
# point the 1800 s wait releases sits at `using CorrelatedBPDecoderWithCER` waiting
# for the same cache. On 2026-10-22 (hp_2026-10-22_16-25-01) the owner had not
# finished an hour in, all 8 points were still at that line, and nothing was
# trained. Judged by modification times (a pull stamps the files it changes), so it
# warns rather than refuses. The depot is the one written into the jobs (the first
# entry of JOB_DEPOT_PATH, see read_cluster_profile); with none (a laptop) there is
# nothing to check. A depot with no cache at all -- a new cluster before
# misc/precompile_depot.sh -- is warned about too: offline, the jobs cannot even
# install the packages.
depot_preflight() {
    local job_depot="${JOB_DEPOT_PATH%%:*}"
    local job_julia_minor
    job_julia_minor="$(echo "$JULIA_MODULE" | sed -nE 's|^julia/([0-9]+\.[0-9]+).*$|\1|p')"
    if [ -z "$job_depot" ] || [ -z "$job_julia_minor" ]; then
        return 0
    fi
    local project_root job_manifest newest_source newest_cache
    project_root="$(cd "$SCRIPT_DIR/.." && pwd)"
    # The manifest that Julia picks: its own minor version's, else the plain one.
    job_manifest="$project_root/Manifest-v$job_julia_minor.toml"
    if [ ! -f "$job_manifest" ]; then
        job_manifest="$project_root/Manifest.toml"
    fi
    newest_source="$(ls -t "$project_root"/src/*.jl "$project_root/Project.toml" "$job_manifest" 2>/dev/null | head -1)"
    newest_cache="$(ls -t "$job_depot/compiled/v$job_julia_minor/CorrelatedBPDecoderWithCER"/*.ji 2>/dev/null | head -1)"
    if [ -z "$newest_cache" ] || { [ -n "$newest_source" ] && [ "$newest_source" -nt "$newest_cache" ]; }; then
        echo >&2
        echo "[hp_sweep] WARNING: the depot is not precompiled for this source." >&2
        if [ -z "$newest_cache" ]; then
            echo "           No cache of CorrelatedBPDecoderWithCER in $job_depot/compiled/v$job_julia_minor." >&2
        else
            echo "           ${newest_source#"$project_root"/} is newer than the package's newest cache," >&2
            echo "           $newest_cache." >&2
        fi
        echo "           The jobs would compile it on the compute nodes, which on 2026-10-22 took" >&2
        echo "           longer than the whole job. On ${SUBMIT_HOST:-this login node}, from expts/, first:" >&2
        echo "               module load ${STDENV_MODULE:+$STDENV_MODULE }$JULIA_MODULE $CUDA_MODULE" >&2
        echo "               bash misc/precompile_depot.sh" >&2
        echo >&2
    fi
}

# --------------------------------------------------------- the job scripts ---
# SELF-CHECK OF THIS FILE. The SLURM scripts below are built from UNQUOTED
# heredocs, so bash expands their contents at generation time -- inside comments
# too. A bare dollar-digit is a positional parameter (unbound under `set -u`) and a
# backtick is command substitution. Both have silently broken generation before:
# once a backtick in a comment, once a dollar-nine in a comment. Anything meant
# literally in those heredocs must be backslash-escaped.
check_job_heredocs() {
    local this_file="$SCRIPT_DIR/hp_sweep/cluster.sh" hazards
    hazards="$(awk '
        /<<[[:space:]]*EOF[[:space:]]*$/ { inside_heredoc = 1; next }
        /^EOF$/                          { inside_heredoc = 0; next }
        inside_heredoc && ($0 ~ /(^|[^\\])\$[0-9]/ || $0 ~ /(^|[^\\])`/) {
            print "    line " NR ": " $0
        }
    ' "$this_file")"
    if [ -n "$hazards" ]; then
        echo "unescaped dollar-digit or backtick inside a job-script heredoc of $this_file:" >&2
        echo "$hazards" >&2
        echo "  Escape it (\\\$9, \\\`) or reword. Left as-is, bash expands it while" >&2
        echo "  writing the job script and generation fails or silently mangles it." >&2
        exit 1
    fi
}

# Both jobs run on GPU nodes and start the same way: modules, the depot, and a
# CUDA_Runtime_jll compiled where a driver is visible, behind one lock. Written
# ONCE, to a file both scripts include verbatim, so the two cannot drift. It is
# generated like the scripts (unquoted heredoc: generation-time values expand,
# escaped ones are left for the job) and uses PHASE and TASK, which each script
# assigns before including it: under set -u an unassigned one is a runtime error.
# Sets GPU_PREAMBLE_FILE (a temporary file; write_job_scripts removes it).
write_gpu_preamble() {
    GPU_PREAMBLE_FILE="$(mktemp)"
    # The modules: the standard environment first where a cluster does not load it
    # at login (Trillium), then Julia and CUDA.
    {
        if [ -n "$STDENV_MODULE" ]; then
            echo "module load $STDENV_MODULE"
        fi
        echo "module load $JULIA_MODULE"
        echo "module load $CUDA_MODULE"
    } > "$GPU_PREAMBLE_FILE"
    # The depot: the one decided at generation (read_cluster_profile), written in.
    if [ -n "$JOB_DEPOT_PATH" ]; then
        cat >> "$GPU_PREAMBLE_FILE" <<EOF
# DEPOT: the one misc/precompile_depot.sh warmed, as the shell that generated this
# job saw it (JULIA_DEPOT_PATH, else \$SCRATCH/.julia then Julia's bundled depots).
# Written in, not inherited: a job started with a fresh environment would otherwise
# fall back to another depot and recompile everything on the compute node.
export JULIA_DEPOT_PATH="$JOB_DEPOT_PATH"
EOF
    else
        cat >> "$GPU_PREAMBLE_FILE" <<EOF
# DEPOT. Falling back to \$HOME here is how a depot gets silently corrupted: on
# Nibi /home is a ~50GB quota and a CUDA+Enzyme depot will exhaust it mid-extract,
# leaving packages with some source files and not others (that is what produced
# "Adapt/src/wrappers.jl: No such file or directory"). Fail loudly instead.
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "\${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        echo "  Refusing to default the Julia depot to \$HOME: on this cluster /home" >&2
        echo "  is small and a partial extraction there corrupts packages silently." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="\${SCRATCH}/.julia:"
fi
EOF
    fi
    cat >> "$GPU_PREAMBLE_FILE" <<EOF
echo "[\$PHASE task \$TASK] depot: \$JULIA_DEPOT_PATH"
# GNU parallel runs the points; checked before the depot work, which takes minutes.
if ! command -v parallel >/dev/null 2>&1; then
    echo "ERROR: GNU parallel is not on PATH on this node." >&2
    exit 1
fi
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
# Both jobs use the card: training runs the hand-written backward pass on it
# (manual_backprop = true in every point's TOML), testing the forward pass.
export GPU_BACKEND=cuda USE_GPU=1
# CUDA_Runtime_jll bakes in whether a driver was visible AT PRECOMPILE TIME. The
# depot is warmed on a LOGIN node (misc/precompile_depot.sh), which has no driver,
# so its cache says "no CUDA runtime found", and a later precompile finds
# everything up to date and keeps it. So the JLL must be rebuilt on a GPU node,
# where the driver IS present, before either job uses CUDA.
#
# BUT ONLY ONE TASK MAY DO IT. On 2026-10-01 all 8 array tasks ran this block at
# once against the one Lustre depot. Five waited for a sibling ("CUDA Being
# precompiled by another machine (hostname: ng30707 ...)") and recovered; three
# read a half-written .ji and died with
#     ERROR: ArgumentError: No value arguments present
#     _include_from_serialized ... loading.jl:1288
# at the CUDA gate below, losing 105 of 280 tests. The gate did its job — the
# alternative was 105 silent garbage results — but the race is the bug.
#
# One task wins the mkdir (atomic enough on Lustre for this), rebuilds, and
# touches a sentinel. The others wait for it. The lock name carries the sweep
# timestamp and is SHARED by the two jobs of a generation: the training job,
# which runs first, rebuilds, and the test job chained after it finds the
# sentinel and goes straight to the gate. A NEW generation is never blocked by a
# stale lock; re-running the SAME generation reuses the already-warm depot, which
# is what we want.
REBUILD_LOCK="$CLUSTER_DIR/hp_${TS}_cuda_rebuild.lock"
REBUILD_DONE="$CLUSTER_DIR/hp_${TS}_cuda_rebuild.done"
REBUILD_WAIT_SECONDS=1800
if mkdir "\$REBUILD_LOCK" 2>/dev/null; then
    # Every step is stamped. On 2026-10-22 the owner's .out ended at its first line
    # and the job was cancelled an hour later with nothing saying which step held it.
    echo "[\$PHASE task \$TASK] rebuilding CUDA_Runtime_jll (this task owns the depot lock): \$(date)"
    echo "[\$PHASE task \$TASK]   \$(date +%T) Pkg.instantiate()"
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate()'; then
        exit 1
    fi
    echo "[\$PHASE task \$TASK]   \$(date +%T) compilecache(CUDA_Runtime_jll)"
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e '
        pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
        Base.compilecache(pkg)'; then
        exit 1
    fi
    echo "[\$PHASE task \$TASK]   \$(date +%T) Pkg.precompile()"
    if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.precompile()'; then
        exit 1
    fi
    touch "\$REBUILD_DONE"
    echo "[\$PHASE task \$TASK] depot rebuild done; siblings released: \$(date)"
else
    echo "[\$PHASE task \$TASK] another task owns the depot lock; waiting for the rebuild: \$(date)"
    WAITED=0
    while [ ! -f "\$REBUILD_DONE" ] && [ "\$WAITED" -lt "\$REBUILD_WAIT_SECONDS" ]; do
        sleep 15
        WAITED=\$(( WAITED + 15 ))
    done
    if [ -f "\$REBUILD_DONE" ]; then
        echo "[\$PHASE task \$TASK] rebuild signalled after \${WAITED}s"
    else
        # The owner died before signalling. Proceeding is still the right move:
        # the depot may well be fine, and the gate below is what decides.
        echo "[\$PHASE task \$TASK] WARNING: no rebuild sentinel after \${WAITED}s; proceeding to the gate anyway" >&2
    fi
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

# Hard gate. Without it the job proceeds and every point dies one by one at its
# first device array, each burning its own startup, and the stage-out returns
# nothing. functional(true) throws with the REASON (no driver, no runtime
# artifact, ...) where functional() only returns false, so a failure explains
# itself in the .err file.
cuda_is_functional() {
    julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; CUDA.functional(true)'
}
# The check loads CUDA.jl, which costs most of a minute, so it runs a second
# time only after a failure.
if ! cuda_is_functional; then
    # One retry. The 2026-10-01 failures were torn reads of a .ji another node
    # was still writing; by now the owner has finished, and a fresh process
    # usually loads it cleanly. If it fails twice the depot really is bad.
    echo "[\$PHASE task \$TASK] CUDA check failed once; re-checking in 60s" >&2
    sleep 60
    if ! cuda_is_functional; then
        echo "ERROR: CUDA is not functional on this node after forcing a JLL rebuild;" >&2
        echo "  the reason is the error CUDA.functional(true) printed above." >&2
        exit 1
    fi
fi
echo "[\$PHASE task \$TASK] CUDA functional: \$(date +%T)"
EOF
}

# The #SBATCH header of a job script: one array task = one card. A cluster that
# schedules cores and memory with each card (Trillium: a quarter node, 24 cores and
# 188 GiB, memory requests ignored) gets no --ntasks/--cpus-per-task/--mem-per-gpu
# lines (request_cpus_and_memory = false); an empty account_gpu leaves the account
# to the scheduler.
sbatch_header() {   # <job name> <output stem, before .out/.err> <array tasks> <cpus per task> <walltime>
    echo "#!/bin/bash"
    if [ -n "$ACCOUNT_GPU" ]; then
        echo "#SBATCH --account=$ACCOUNT_GPU"
    fi
    echo "#SBATCH --job-name=$1"
    echo "#SBATCH --output=$2.out"
    echo "#SBATCH --error=$2.err"
    echo "#SBATCH --array=0-$(( $3 - 1 ))"
    echo "$GPU_SBATCH_LINE"
    echo "#SBATCH --nodes=1"
    if [ "$REQUEST_CPUS_AND_MEMORY" = "true" ]; then
        echo "#SBATCH --ntasks=1"
        echo "#SBATCH --cpus-per-task=$4"
        echo "#SBATCH --mem-per-gpu=$MEM_PER_GPU"
    fi
    echo "#SBATCH --time=$5"
    echo "#SBATCH --signal=B:TERM@600"
    echo "#SBATCH --mail-type=ALL"
    echo "#SBATCH --mail-user=$EMAIL"
}

# ------------------------------------------------------------ SLURM: train ---
# The training job: one array task per slice of TRAIN_CMDS (interleaved), each on
# one card with train_jobs trainings sharing it. Writes SLURM_TRAIN.
write_train_job() {
    sbatch_header "hptrain_$TS" "$CLUSTER_DIR/hp_sweep_train_${TS}_task%a" "$TRAIN_ARRAY" "$TRAIN_CPUS" "$TRAIN_WALL" > "$SLURM_TRAIN"
    cat >> "$SLURM_TRAIN" <<EOF
set -uo pipefail
# NOTE: no backticks in this heredoc -- it is unquoted (<<EOF), so backticks are
# command substitution and bash would try to RUN whatever they enclose.
#
# PHASE and TASK come FIRST: the shared preamble below echoes both, and this
# script runs under \`set -u\`. On 2026-10-05 TASK was assigned below the block
# that echoed it, and all 8 test tasks died in under a second with
#     slurm_script: line 75: TASK: unbound variable
# \`bash -n\` does not catch this -- an unbound variable under \`set -u\` is a
# RUNTIME error -- which is why generated scripts are checked by executing them
# with stub binaries.
PHASE=train
TASK=\${SLURM_ARRAY_TASK_ID:-0}
cd \$SLURM_SUBMIT_DIR
EOF
    cat "$GPU_PREAMBLE_FILE" >> "$SLURM_TRAIN"
    cat >> "$SLURM_TRAIN" <<EOF

LOCAL="\$SLURM_TMPDIR/$CODENAME"
# Stage in only what this job's points read, as listed in
# $(basename "$TRAIN_STAGE_LIST"). An entry not on disk is left out rather than failing tar.
STAGE_LIST="\$SLURM_TMPDIR/stage_in.txt"
while IFS= read -r staged_path; do
    if [ -e "$WORKDIR/\$staged_path" ]; then
        echo "\$staged_path"
    fi
done < "$TRAIN_STAGE_LIST" > "\$STAGE_LIST"
tar -chf - -C "$WORKDIR" -T "\$STAGE_LIST" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}"
echo "[train task \$TASK] staged in \$(wc -l < "\$STAGE_LIST") path(s), \$(du -sh "\$LOCAL" | cut -f1): \$(date +%T)"
# Interleaved slice: task t runs lines t+1, t+1+K, ... of the full point list.
# Interleaved rather than contiguous so a slow region of the grid is spread over
# all tasks instead of landing entirely on one.
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" \\
    | awk -v k=$TRAIN_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/train.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/train.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[train task \$TASK] no points in this slice ($TRAIN_ARRAY tasks > $N_POINTS points); nothing to do."
    exit 0
fi

stage_out() {
    tar -cf - --exclude='hyperparams_hp_*.toml' -C "\$LOCAL" models logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
# See the matching comment in the test job: bash defers a trap until the current
# FOREGROUND command returns, so \`parallel\` is backgrounded and we block in
# \`wait\` -- otherwise the TERM that --signal=B:TERM@600 promises us arrives while
# bash is stuck on parallel, never runs, and the wall takes the whole slice.
# A training point that is killed mid-epoch leaves no weights file, so what this
# saves is the points that COMPLETED: they will not be retrained on resubmission.
on_walltime_signal() {
    trap - EXIT   # we stage out here; don't let the EXIT trap copy it all twice
    echo "[train task \$TASK] TERM (600s before the wall): stopping and staging out completed points" >&2
    if [ -n "\${PARALLEL_PID:-}" ]; then
        kill -TERM "\$PARALLEL_PID" 2>/dev/null
        wait "\$PARALLEL_PID" 2>/dev/null
    fi
    N_DONE=0
    if [ -f "\${JOBLOG:-}" ]; then
        N_DONE=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
    fi
    echo "[train task \$TASK] staging out \$N_DONE completed point(s) of \$N_TASK" >&2
    stage_out
    # Exit 1, NOT 0. The completed points are already staged out above, so nothing
    # is lost -- but a wall-killed task left a partial model set, and reporting
    # success would let an afterok dependency start testing against it.
    exit 1
}
trap on_walltime_signal TERM
trap stage_out EXIT

echo "[train task \$TASK/$TRAIN_ARRAY] \$N_TASK of $N_POINTS point(s), $TRAIN_JOBS at a time on \${SLURM_GPUS_ON_NODE:-1} GPU(s): \$(date)"
# --joblog records seq / exit status / command per point in ONE readable file.
# The --results directories are named after the full command with / = " escaped
# to +z +e +22, so they cannot be cat'd without quoting; the joblog is the index.
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_train_task\${TASK}"
# The trainings share the task's card(s): slot {%} goes to card
# ((slot - 1) mod n_cards) + 1, so with one card all $TRAIN_JOBS use it. Each is one
# julia process with its own CUDA context, well under 1 GB of VRAM at a batch of
# 20. (Testing never shares a card; see the test job.)
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
parallel --jobs $TRAIN_JOBS --joblog "\$JOBLOG" --results "\$RESULTS_ROOT" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/train.txt" &
PARALLEL_PID=\$!
wait "\$PARALLEL_PID"
PARALLEL_STATUS=\$?
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    # No bare dollar-digit in this comment: see the self-check above.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
N_OK=0
if [ -f "\$JOBLOG" ]; then
    N_OK=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
fi
N_FAILED=\$(( N_TASK - N_OK ))
echo "[train task \$TASK] \$N_OK/\$N_TASK point(s) exited 0"
# When points fail, put a REAL error message in this file. The per-job stderr
# lives under --results in directories named after the whole command (with / = "
# escaped to +z +e +22), which cannot be read without quoting -- so print the
# first non-empty one here rather than making the reader go and find it.
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[train task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[train task \$TASK] ---- end ----"
fi
echo "[train task \$TASK] done: \$(date)"

# EXIT STATUS IS THE CONTRACT with \`sbatch --dependency=afterok\`. This used to end
# on an echo, so a task reported SUCCESS even when every one of its points had
# crashed -- and afterok on that is decorative. Fail loudly instead, and let the
# dependency kill the test job.
#
# afterok on a job ARRAY is all-or-nothing: it is satisfied only when EVERY task
# exits 0, so one bad point anywhere stops all 160 tests. That is the intent --
# testing a partial model set produces results that silently describe the wrong
# experiment. Note a NaN-ROLLED-BACK epoch is not a failure: the point still
# exits 0 and still writes weights, so rollbacks do not block testing.
if [ "\$N_FAILED" -gt 0 ]; then
    echo "[train task \$TASK] \$N_FAILED of \$N_TASK point(s) FAILED; exiting 1 so afterok blocks testing" >&2
    exit 1
fi
if [ "\$PARALLEL_STATUS" -ne 0 ]; then
    echo "[train task \$TASK] parallel itself exited \$PARALLEL_STATUS; exiting 1" >&2
    exit 1
fi
exit 0
EOF
    chmod +x "$SLURM_TRAIN"
}

# ------------------------------------------------------------- SLURM: test ---
# The test job: one array task per slice of TEST_CMDS (interleaved), one test at a
# time on its card, after checking every model of the sweep is staged in. Writes
# SLURM_TEST.
write_test_job() {
    sbatch_header "hptest_$TS" "$CLUSTER_DIR/hp_sweep_test_${TS}_task%a" "$TEST_ARRAY" "$TEST_CPUS" "$TEST_WALL" > "$SLURM_TEST"
    cat >> "$SLURM_TEST" <<EOF
set -uo pipefail
# NOTE: no backticks in this heredoc -- it is unquoted (<<EOF), so backticks are
# command substitution and bash would try to RUN whatever they enclose.
#
# PHASE and TASK come FIRST: the shared preamble below echoes both, and this
# script runs under \`set -u\`. On 2026-10-05 TASK was assigned below the block
# that echoed it, and all 8 test tasks died in under a second with
#     slurm_script: line 75: TASK: unbound variable
# \`bash -n\` does not catch this -- an unbound variable under \`set -u\` is a
# RUNTIME error -- which is why generated scripts are checked by executing them
# with stub binaries.
PHASE=test
TASK=\${SLURM_ARRAY_TASK_ID:-0}
cd \$SLURM_SUBMIT_DIR
EOF
    cat "$GPU_PREAMBLE_FILE" >> "$SLURM_TEST"
    cat >> "$SLURM_TEST" <<EOF

LOCAL="\$SLURM_TMPDIR/$CODENAME"
# Stage in only what this job's points read, as listed in
# $(basename "$TEST_STAGE_LIST"). An entry not on disk is left out rather than failing
# tar; a trained model left out that way is named by the model check below.
STAGE_LIST="\$SLURM_TMPDIR/stage_in.txt"
while IFS= read -r staged_path; do
    if [ -e "$WORKDIR/\$staged_path" ]; then
        echo "\$staged_path"
    fi
done < "$TEST_STAGE_LIST" > "\$STAGE_LIST"
tar -chf - -C "$WORKDIR" -T "\$STAGE_LIST" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
echo "[test task \$TASK] staged in \$(wc -l < "\$STAGE_LIST") path(s), \$(du -sh "\$LOCAL" | cut -f1): \$(date +%T)"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS" \\
    | awk -v k=$TEST_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/test.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/test.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[test task \$TASK] no points in this slice ($TEST_ARRAY tasks > $N_TEST_POINTS points); nothing to do."
    exit 0
fi

# neural_bp_experiments.jl SKIPS testing when the results file already exists and
# reports the old numbers as if fresh. results/ is no longer staged in, so this
# finds nothing; it stays as a guard should the stage-in list ever include it.
# Safe to clear ALL of them even under a job array: this deletes only the
# node-local staged copy, and stage_out untars this task's files INTO the shared
# directory without removing anything already there. So a sibling's results that
# this task wipes locally still survive in $WORKDIR.
rm -f "\$LOCAL"/results/simulation_results_*_seed_*.csv

# The generator wrote retrain = true (the training job needs it); flip it so
# this job loads the trained weights rather than training them again.
for f in "\$LOCAL"/models/hyperparams_hp_*.toml; do
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "\$f" > "\$f.tmp"
    mv "\$f.tmp" "\$f"
done
# Second line of defence behind \`--dependency=afterok --kill-on-invalid-dep=yes\`.
# The dependency catches a training job that FAILED; this catches the cases it
# cannot see -- training never submitted, a stage-out that did not land, someone
# resubmitting the test job on its own, or a cleanup.py between the two. A test
# whose weights file is missing does not fail on its own: under retrain = false
# neural_bp_experiments.jl TRAINS a model of its own and reports that.
# Checked by NAME, one line of $EXPECTED_MODELS per training point.
# Counting *.json is not enough: a run directory earlier sweeps used holds their
# models too (72q_BB_cycles_1_trainable_alpha held 20 on 2026-10-22), so the count
# passed with every model of THIS sweep missing.
N_EXPECTED=0
N_MISSING=0
while IFS= read -r expected_model; do
    N_EXPECTED=\$(( N_EXPECTED + 1 ))
    if [ ! -f "\$LOCAL/models/\$expected_model" ]; then
        N_MISSING=\$(( N_MISSING + 1 ))
        if [ "\$N_MISSING" -le 5 ]; then
            echo "  missing: \$expected_model" >&2
        fi
    fi
done < "$EXPECTED_MODELS"
echo "[test task \$TASK/$TEST_ARRAY] \$(( N_EXPECTED - N_MISSING )) of \$N_EXPECTED trained model(s) of this sweep staged in"
if [ "\$N_EXPECTED" -ne $N_POINTS ] || [ "\$N_MISSING" -gt 0 ]; then
    echo "ERROR: \$N_MISSING of \$N_EXPECTED trained model(s) missing (expecting $N_POINTS); refusing to test a partial set." >&2
    echo "  Check that the training job finished and staged out, then resubmit." >&2
    exit 1
fi

stage_out() {
    tar -cf - --exclude='hyperparams_hp_*.toml' -C "\$LOCAL" results logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
# --signal=B:TERM@600 above gives us ten minutes' notice of the wall. Bash runs a
# trap only between foreground commands, so with \`parallel\` in the FOREGROUND the
# handler would sit queued behind it and SLURM's hard kill at the wall would take
# the job with nothing staged out -- every completed point in this slice lost.
# So \`parallel\` runs in the BACKGROUND and we block in \`wait\`, which a trapped
# signal does interrupt. On TERM: stop parallel, let the in-flight point die, and
# copy back the points that did finish.
on_walltime_signal() {
    trap - EXIT   # we stage out here; don't let the EXIT trap copy it all twice
    echo "[test task \$TASK] TERM (600s before the wall): stopping and staging out partial results" >&2
    if [ -n "\${PARALLEL_PID:-}" ]; then
        kill -TERM "\$PARALLEL_PID" 2>/dev/null
        wait "\$PARALLEL_PID" 2>/dev/null
    fi
    N_DONE=0
    if [ -f "\${JOBLOG:-}" ]; then
        N_DONE=\$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)
    fi
    echo "[test task \$TASK] staging out \$N_DONE completed point(s) of \$N_TASK" >&2
    stage_out
    exit 0
}
trap on_walltime_signal TERM
trap stage_out EXIT

export GPU_MEMORY=${GPU_MEMORY_MB}M
echo "[test task \$TASK] \$N_TASK of $N_TEST_POINTS point(s), $TEST_JOBS at a time on \${SLURM_GPUS_ON_NODE:-1} GPU(s): \$(date)"
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
JOBLOG="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/hp_${TS}_test_task\${TASK}"
parallel --jobs $TEST_JOBS --joblog "\$JOBLOG" --results "\$RESULTS_ROOT" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/test.txt" &
PARALLEL_PID=\$!
wait "\$PARALLEL_PID"
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    # No bare dollar-digit in this comment: see the self-check above.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
echo "[test task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
# When points fail, put a REAL error message in this file. The per-job stderr
# lives under --results in directories named after the whole command (with / = "
# escaped to +z +e +22), which cannot be read without quoting -- so print the
# first non-empty one here rather than making the reader go and find it.
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[test task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[test task \$TASK] ---- end ----"
fi
echo "[test task \$TASK] done: \$(date)"
EOF
    chmod +x "$SLURM_TEST"
}

# The two job scripts, after checking this file's heredocs.
write_job_scripts() {
    check_job_heredocs
    write_gpu_preamble
    write_train_job
    write_test_job
    rm -f "$GPU_PREAMBLE_FILE"
}

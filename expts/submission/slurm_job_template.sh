#!/bin/bash
{{SBATCH_LINES}}
# ============================================================================
# {{JOB_NAME}} -- written by expts/submission/slurm.jl from
# expts/submission/slurm_job_template.sh. Change the template, not this file.
#
# Each array task takes a contiguous slice of the commands file and runs it
# through GNU parallel on a copy of the inputs those commands read, in
# $SLURM_TMPDIR. Submit from expts/: every path below is relative to it.
#
# The template is plain bash with double-brace placeholders (the upper-case names
# in the block below) that slurm.jl replaces literally, so it can be filled in by
# any tool and executed with stub julia / parallel / module binaries to test it.
# `bash -n` is not a test: an unset
# variable under `set -u` is a RUNTIME error (that is how all 8 tasks of a test
# job died in under a second on 2026-10-05).
# ============================================================================
set -uo pipefail

# --- filled in by slurm.jl --------------------------------------------------
PHASE="{{PHASE}}"                          # train | test
USE_GPU_FOR_JOB={{USE_GPU}}                # 1: every process runs on a GPU (CUDA); 0: CPU only
CODENAME="{{CODENAME}}"
DATA_ROOT="{{DATA_ROOT}}"                  # the commands' --workdir, relative to expts/
COMMANDS_FILE="{{COMMANDS_FILE}}"
COMMANDS_PER_TASK={{COMMANDS_PER_TASK}}
JOBS_PER_TASK={{JOBS_PER_TASK}}            # julia processes at a time in one task
THREADS_PER_JOB={{THREADS_PER_JOB}}
GPU_MEMORY_MB="{{GPU_MEMORY_MB}}"          # test jobs on a GPU: VRAM budget of the batch sizer
JULIA_MODULE="{{JULIA_MODULE}}"
CUDA_MODULE="{{CUDA_MODULE}}"
MODELS_FILE="{{MODELS_FILE}}"              # test jobs: line n = the model command n loads; empty = no check
DEPOT_LOCK="{{DEPOT_LOCK}}"                # shared with the training job a test job is chained on
DEPOT_READY="{{DEPOT_READY}}"
JOB_NAME="{{JOB_NAME}}"
# ---------------------------------------------------------------------------

# TASK first: everything below echoes it, and this script runs under `set -u`.
TASK=${SLURM_ARRAY_TASK_ID:-0}
cd "$SLURM_SUBMIT_DIR"
START_TIME_SEC=$(date +%s)
echo "[$PHASE task $TASK] started $(date) on $(hostname)"

TARGET_DIR="$DATA_ROOT/$CODENAME"
COMMANDS_STEM=$(basename "$COMMANDS_FILE" .txt)
CHUNK_NAME="${COMMANDS_STEM}_chunk_${TASK}.txt"
SHARED_CHUNK="$TARGET_DIR/cluster/$CHUNK_NAME"
FIRST_LINE=$(( TASK * COMMANDS_PER_TASK + 1 ))
LAST_LINE=$(( FIRST_LINE + COMMANDS_PER_TASK - 1 ))
sed -n "${FIRST_LINE},${LAST_LINE}p" "$COMMANDS_FILE" > "$SHARED_CHUNK"
N_TASK_COMMANDS=$(grep -c . "$SHARED_CHUNK")
if [ "$N_TASK_COMMANDS" -eq 0 ]; then
    echo "[$PHASE task $TASK] lines ${FIRST_LINE}-${LAST_LINE} of $COMMANDS_FILE are empty; nothing to do."
    exit 0
fi
echo "[$PHASE task $TASK] $N_TASK_COMMANDS command(s), $JOBS_PER_TASK at a time"

module load "$JULIA_MODULE"
if [ "$USE_GPU_FOR_JOB" = 1 ]; then
    module load "$CUDA_MODULE"
fi

# DEPOT. Falling back to $HOME is how a depot gets silently corrupted: /home is a
# small quota on these clusters and a CUDA + Enzyme depot can exhaust it
# mid-extract, leaving packages with some source files and not others.
if [ -z "${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        echo "  Refusing to default the Julia depot to \$HOME (small quota, silent corruption)." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="${SCRATCH}/.julia"
fi
echo "[$PHASE task $TASK] depot: $JULIA_DEPOT_PATH"

# One thread per process unless this is a test job holding a whole card: GNU
# parallel runs JOBS_PER_TASK processes and they must not oversubscribe the cores.
export JULIA_NUM_THREADS="$THREADS_PER_JOB" OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
       OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
       JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_OFFLINE=true
if [ "$USE_GPU_FOR_JOB" = 1 ]; then
    # Training runs the hand-written backward pass on the card (manual_backprop =
    # true, checked at submit time); testing runs the forward pass.
    export USE_GPU=1 GPU_BACKEND=cuda
    if [ -n "$GPU_MEMORY_MB" ]; then
        # The prediction batch is sized to THIS, the card's VRAM. Without it the
        # sizer falls back to SLURM_MEM_PER_GPU, which is host RAM.
        export GPU_MEMORY="${GPU_MEMORY_MB}M"
    fi
else
    export USE_GPU=0
fi

# DEPOT PREPARATION, by ONE task. Compute nodes have no internet, so the depot
# must already be warm (bash misc/precompile_depot.sh on a login node). What is
# left to do here is the GPU part: CUDA_Runtime_jll records whether a driver was
# visible when it was precompiled, the login node has none, so the JLL is rebuilt
# on a GPU node before anything loads CUDA. If every array task does that at once
# against the one shared depot, the losers read a half-written .ji and die with
# `ArgumentError: No value arguments present` (3 of 8 tasks on 2026-10-01). One
# task wins the mkdir, rebuilds and touches the sentinel; the others wait for it.
if mkdir "$DEPOT_LOCK" 2>/dev/null; then
    echo "[$PHASE task $TASK] preparing the depot (this task holds $DEPOT_LOCK)"
    if ! julia --project="$SLURM_SUBMIT_DIR/.." -e 'using Pkg; Pkg.instantiate()'; then
        echo "ERROR: Pkg.instantiate() failed offline. Warm the depot on a login node:" >&2
        echo "  cd expts && bash misc/precompile_depot.sh" >&2
        exit 1
    fi
    if [ "$USE_GPU_FOR_JOB" = 1 ]; then
        if ! julia --project="$SLURM_SUBMIT_DIR/.." -e '
            pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
            Base.compilecache(pkg)'; then
            exit 1
        fi
    fi
    if ! julia --project="$SLURM_SUBMIT_DIR/.." -e 'using Pkg; Pkg.precompile()'; then
        exit 1
    fi
    touch "$DEPOT_READY"
    echo "[$PHASE task $TASK] depot ready; other tasks released"
else
    echo "[$PHASE task $TASK] another task is preparing the depot; waiting for it"
    WAITED=0
    while [ ! -f "$DEPOT_READY" ] && [ "$WAITED" -lt 1800 ]; do
        sleep 15
        WAITED=$(( WAITED + 15 ))
    done
    if [ -f "$DEPOT_READY" ]; then
        echo "[$PHASE task $TASK] depot ready after ${WAITED}s"
    else
        # The owner died before signalling. Going on is still right: the depot
        # may well be fine, and the check below (GPU jobs) decides.
        echo "[$PHASE task $TASK] WARNING: no depot sentinel after ${WAITED}s; continuing" >&2
    fi
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

# CUDA GATE (GPU jobs). Without it every command dies at its first device array,
# each after its own startup, and the stage-out brings back nothing.
# functional(true) throws with the reason; functional() only returns false.
cuda_is_functional() {
    julia --project="$SLURM_SUBMIT_DIR/.." -e 'using CUDA; CUDA.functional(true)'
}
if [ "$USE_GPU_FOR_JOB" = 1 ]; then
    if ! cuda_is_functional; then
        # One retry: a torn read of a .ji another node was still writing usually
        # loads cleanly in a fresh process a minute later.
        echo "[$PHASE task $TASK] CUDA check failed once; re-checking in 60s" >&2
        sleep 60
        if ! cuda_is_functional; then
            echo "ERROR: CUDA is not functional on this node; the reason is printed above." >&2
            exit 1
        fi
    fi
    echo "[$PHASE task $TASK] CUDA functional."
fi

# STAGE IN only what this task's commands read: code/, models/ (hyperparameter
# files and trained weights), correlated_weights/ (small), and the training and
# test files the chunk names. A dataset directory holds every (p, sigma, sample)
# file -- tens of GB -- and one task reads a handful of them. results/ is NOT
# staged: neural_bp_experiments.jl skips a test whose results file exists and
# reports the old numbers as new, so each test job starts from an empty one.
LOCAL_WORK_DIR="$SLURM_TMPDIR/$CODENAME"
mkdir -p "$LOCAL_WORK_DIR/results" "$LOCAL_WORK_DIR/models" "$LOCAL_WORK_DIR/logs" \
         "$LOCAL_WORK_DIR/cluster/logs"

# STAGE OUT only the files this task WROTE (newer than the marker touched after
# the stage-in). Copying whole directories back would put this task's stale
# copies of models/ over weights another job wrote in the meantime — a test job
# still running when the next training job finishes, for instance.
STAGE_MARKER="$SLURM_TMPDIR/stage_in_done_task${TASK}"
STAGE_OUT_DONE=0
stage_out() {
    if [ "$STAGE_OUT_DONE" = 1 ]; then
        return 0
    fi
    STAGE_OUT_DONE=1
    if [ ! -e "$STAGE_MARKER" ]; then
        echo "[$PHASE task $TASK] stage-out: nothing ran, nothing to copy back"
        return 0
    fi
    local stage_out_list="$SLURM_TMPDIR/stage_out_task${TASK}.txt"
    (cd "$LOCAL_WORK_DIR" && find results models logs cluster/logs -type f -newer "$STAGE_MARKER" 2>/dev/null) \
        > "$stage_out_list"
    echo "[$PHASE task $TASK] stage-out to $TARGET_DIR: $(wc -l < "$stage_out_list") new file(s)"
    if [ -s "$stage_out_list" ]; then
        tar -cf - -C "$LOCAL_WORK_DIR" -T "$stage_out_list" | tar -xf - -C "$TARGET_DIR"
    fi
}
# Bash runs a trap only between FOREGROUND commands, so parallel is started in the
# background and waited on below: the TERM that --signal=B:TERM promises then
# interrupts the wait at once and the finished commands' outputs are saved.
on_walltime_signal() {
    trap - EXIT
    echo "[$PHASE task $TASK] TERM before the wall: stopping and staging out what finished" >&2
    if [ -n "${PARALLEL_PID:-}" ]; then
        kill -TERM "$PARALLEL_PID" 2>/dev/null
        wait "$PARALLEL_PID" 2>/dev/null
    fi
    stage_out
    # Exit 1: the task did not finish its slice.
    exit 1
}
trap on_walltime_signal TERM
trap stage_out EXIT

STAGE_LIST="$SLURM_TMPDIR/stage_in_task${TASK}.txt"
{
    echo code
    echo models
    echo correlated_weights
    grep -oE -- '--train [^ ]+' "$SHARED_CHUNK" | awk '{print "training_data/" $2}'
    grep -oE -- '--test [^ ]+' "$SHARED_CHUNK" | awk '{print "testing_data/" $2}'
} | sort -u | while IFS= read -r relative_path; do
    # A test job names the training file too (it identifies the model) but never
    # reads it, so a path that is not there is left out rather than an error.
    if [ -e "$TARGET_DIR/$relative_path" ]; then
        echo "$relative_path"
    fi
done > "$STAGE_LIST"
STAGE_IN_START=$(date +%s)
if ! tar -chf - -C "$TARGET_DIR" -T "$STAGE_LIST" | tar -xf - -C "$LOCAL_WORK_DIR"; then
    echo "ERROR: stage-in from $TARGET_DIR failed" >&2
    exit 1
fi
echo "[$PHASE task $TASK] staged $(wc -l < "$STAGE_LIST") path(s) in $(( $(date +%s) - STAGE_IN_START ))s"

# The commands name --workdir as seen from expts/; point them at the local copy.
LOCAL_CHUNK="$LOCAL_WORK_DIR/cluster/$CHUNK_NAME"
sed "s|--workdir [^ ]*|--workdir $SLURM_TMPDIR|g" "$SHARED_CHUNK" > "$LOCAL_CHUNK"

# MODEL CHECK (test jobs). A test whose trained model is missing does not fail:
# with retrain = false and no weights file the experiment trains a fresh model
# under the test job's wall clock and reports it as the trained one. So every
# command's model (the same line of MODELS_FILE) must have been staged in; a
# command whose model is not here is not run, and the task exits 1 at the end.
# Behind --dependency=afterok this catches a stage-out that did not land, or a
# cleanup between the two jobs.
RUN_CHUNK="$LOCAL_CHUNK"
N_MISSING_MODELS=0
if [ -n "$MODELS_FILE" ]; then
    TASK_MODELS="$SLURM_TMPDIR/models_task${TASK}.txt"
    sed -n "${FIRST_LINE},${LAST_LINE}p" "$MODELS_FILE" > "$TASK_MODELS"
    if [ "$(grep -c . "$TASK_MODELS")" -ne "$N_TASK_COMMANDS" ]; then
        echo "ERROR: lines ${FIRST_LINE}-${LAST_LINE} of $MODELS_FILE do not match $COMMANDS_FILE" >&2
        exit 1
    fi
    RUN_CHUNK="$LOCAL_WORK_DIR/cluster/${COMMANDS_STEM}_runnable_${TASK}.txt"
    : > "$RUN_CHUNK"
    while IFS= read -r command_line <&3 && IFS= read -r model_name <&4; do
        if [ -f "$LOCAL_WORK_DIR/models/$model_name" ]; then
            printf '%s\n' "$command_line" >> "$RUN_CHUNK"
        else
            echo "[$PHASE task $TASK] model missing, test NOT run: models/$model_name" >&2
            N_MISSING_MODELS=$(( N_MISSING_MODELS + 1 ))
        fi
    done 3< "$LOCAL_CHUNK" 4< "$TASK_MODELS"
    echo "[$PHASE task $TASK] models: $(( N_TASK_COMMANDS - N_MISSING_MODELS ))/$N_TASK_COMMANDS present"
fi

# Everything modified after this is the task's own output (see stage_out). The
# staged inputs keep their original times (tar preserves them), so none is "newer".
touch "$STAGE_MARKER"

JOBLOG="$LOCAL_WORK_DIR/cluster/logs/${JOB_NAME}_task${TASK}.joblog"
RESULTS_ROOT="$LOCAL_WORK_DIR/cluster/logs/${JOB_NAME}_task${TASK}"
echo "[$PHASE task $TASK] running: $(date)"
if [ "$USE_GPU_FOR_JOB" = 1 ]; then
    # Slot {%} runs on card ((slot - 1) mod n_cards) + 1 of SLURM's list, which
    # holds indices for whole cards and UUIDs for MIG slices alike. Several
    # trainings share a card this way; a test job has one process per card.
    export SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    GPU_SLOT_COUNT=$(echo "$SLURM_CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    export GPU_SLOT_COUNT
    parallel --jobs "$JOBS_PER_TASK" --joblog "$JOBLOG" --results "$RESULTS_ROOT" \
        'card=$(( ({%} - 1) % GPU_SLOT_COUNT + 1 )); export CUDA_VISIBLE_DEVICES=$(echo "$SLURM_CUDA_VISIBLE_DEVICES" | cut -d, -f$card); bash -c {}' \
        < "$RUN_CHUNK" &
else
    parallel --jobs "$JOBS_PER_TASK" --joblog "$JOBLOG" --results "$RESULTS_ROOT" < "$RUN_CHUNK" &
fi
PARALLEL_PID=$!
wait "$PARALLEL_PID"
PARALLEL_STATUS=$?

N_OK=0
if [ -f "$JOBLOG" ]; then
    # The Command column contains spaces: print from where it starts.
    awk 'NR>1 && $7 != 0 {print "  FAILED (exit " $7 "): " substr($0, index($0, $9))}' "$JOBLOG"
    N_OK=$(awk 'NR>1 && $7 == 0' "$JOBLOG" | wc -l)
fi
N_FAILED=$(( N_TASK_COMMANDS - N_OK ))
if [ "$N_MISSING_MODELS" -gt 0 ]; then
    echo "[$PHASE task $TASK] $N_OK/$N_TASK_COMMANDS command(s) exited 0; $N_MISSING_MODELS not run (model missing)"
else
    echo "[$PHASE task $TASK] $N_OK/$N_TASK_COMMANDS command(s) exited 0"
fi
if [ "$N_FAILED" -gt 0 ]; then
    # Per-command stderr sits under --results in directories named after the whole
    # command, which are awkward to open; show the end of the first non-empty one.
    FIRST_STDERR=$(find "$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
    if [ -n "$FIRST_STDERR" ]; then
        echo "[$PHASE task $TASK] ---- end of the first non-empty stderr ----"
        tail -25 "$FIRST_STDERR" | sed 's/^/    /'
        echo "[$PHASE task $TASK] ----"
    fi
fi

DURATION_SEC=$(( $(date +%s) - START_TIME_SEC ))
echo "[$PHASE task $TASK] finished $(date): $(( DURATION_SEC / 3600 ))h $(( (DURATION_SEC % 3600) / 60 ))m $(( DURATION_SEC % 60 ))s"

# A non-zero exit when any command failed, so the failure is visible in sacct and
# a job chained with --dependency=afterok does not start on a partial set.
if [ "$N_FAILED" -gt 0 ] || [ "$PARALLEL_STATUS" -ne 0 ]; then
    echo "[$PHASE task $TASK] $N_FAILED command(s) failed (parallel exit $PARALLEL_STATUS); exiting 1" >&2
    exit 1
fi
exit 0

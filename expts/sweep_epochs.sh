#!/usr/bin/env bash
# Sweep training epochs for the Neural BP experiment, running multiple
# (n_epochs, rep) combinations in parallel via GNU parallel.
#
# Usage:
#   ./sweep_epochs.sh                       # default: epochs 10 20 30 40 50, 1 rep each
#   ./sweep_epochs.sh 10 20 30              # custom epoch list
#   N_REPS=3 ./sweep_epochs.sh              # 3 reps per epoch count
#   PARALLEL_JOBS=8 N_REPS=3 ./sweep_epochs.sh 10 20 30 40 50
#
# Each run produces:
#   <out>/epochs_<N>_rep_<R>.log    full stdout/stderr from the Julia run
#   <out>/epochs_<N>_rep_<R>.json   the single result-line JSON, isolated
#   <out>/per_run_summaries/<id>.tsv   a single summary row per run
#
# After all runs complete, per-run rows are merged into <out>/summary.tsv.
#
# Notes on parallelism:
#   - PARALLEL_JOBS controls how many Julia processes run concurrently.
#     Each Julia process uses BLAS threads on its own, so setting
#     PARALLEL_JOBS too high will oversubscribe the CPUs.
#   - On a 32-CPU box, PARALLEL_JOBS=4 with OPENBLAS_NUM_THREADS=8 is
#     a reasonable starting point. Override BLAS threads via the
#     OPENBLAS_NUM_THREADS env var (set automatically below if unset).

set -u

# ---- preflight ----
if ! command -v parallel >/dev/null 2>&1; then
    echo "Error: GNU parallel is required but not on PATH." >&2
    echo "  macOS: brew install parallel" >&2
    echo "  Linux: apt install parallel  /  dnf install parallel" >&2
    exit 1
fi

# ---- paths and fixed experiment args ----
PROJECT_ROOT="/Users/pavi/Documents/IQC/CorrelatedBPDecoderWithCER"
EXPTS_DIR="$PROJECT_ROOT/expts"
DATA_DIR="$PROJECT_ROOT/data"
PROJECT_DIR="$PROJECT_ROOT"  # for `julia --project`
CODENAME="7q_Hamm_code_data_q_mean_0.1_std_0.2"
N_HIDDEN_LAYERS=100
BATCH_SIZE=8
CORR_FILE="correlated_weights_p_0.006_q_0.1_s_1.txt"
TRAIN_FILE="train_ballistic_p_0.006_q_0.1_s_1.txt"
TEST_FILE="test_ballistic_p_0.006_q_0.1_s_1.txt"

# ---- sweep config ----
if [ "$#" -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=(10 20 30 40 50)
fi
N_REPS="${N_REPS:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"

# Default BLAS threads per process so we don't oversubscribe a 32-CPU box.
# 4 jobs * 8 threads = 32. Override by exporting OPENBLAS_NUM_THREADS yourself.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$OPENBLAS_NUM_THREADS}"

# ---- output dir (timestamped) ----
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$DATA_DIR/sweep_results/$TS"
mkdir -p "$OUT_DIR/per_run_summaries"
SUMMARY="$OUT_DIR/summary.tsv"
printf 'n_epochs\trep\tnum_failures\taverage_LER\tstd_LER\truntime_s\tlog_file\n' > "$SUMMARY"

echo "==== Sweep config ===="
echo "  epochs:        ${EPOCHS[*]}"
echo "  reps per:      $N_REPS"
echo "  parallel jobs: $PARALLEL_JOBS"
echo "  BLAS threads:  $OPENBLAS_NUM_THREADS"
echo "  output dir:    $OUT_DIR"
echo ""

# ---- function executed by each parallel job ----
run_one() {
    local n_epochs="$1"
    local rep="$2"
    local run_id="epochs_${n_epochs}_rep_${rep}"
    local log_file="$OUT_DIR/${run_id}.log"
    local json_file="$OUT_DIR/${run_id}.json"
    local row_file="$OUT_DIR/per_run_summaries/${run_id}.tsv"

    cd "$EXPTS_DIR" || return 1

    julia --project="$PROJECT_DIR" neural_bp_experiments.jl \
        --codename "$CODENAME" \
        --n_hidden_layers "$N_HIDDEN_LAYERS" \
        --n_epochs "$n_epochs" \
        --batch_size "$BATCH_SIZE" \
        --correlation_strengths_file "$CORR_FILE" \
        --train "$TRAIN_FILE" \
        --test "$TEST_FILE" \
        --retrain true > "$log_file" 2>&1

    grep -E '"num_failures"' "$log_file" | tail -1 > "$json_file"

    if [ ! -s "$json_file" ]; then
        printf '%s\t%s\tNA\tNA\tNA\tNA\t%s\n' "$n_epochs" "$rep" "$log_file" > "$row_file"
        return 0
    fi

    local parsed
    parsed=$(python3 - "$json_file" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.loads(f.read().strip())
print("\t".join(str(d.get(k, "NA")) for k in
    ("num_failures", "average_logical_error_rate", "std_logical_error_rate", "runtime")))
PYEOF
)
    printf '%s\t%s\t%s\t%s\n' "$n_epochs" "$rep" "$parsed" "$log_file" > "$row_file"
}
export -f run_one
# Variables the function needs in each parallel subshell:
export EXPTS_DIR PROJECT_DIR CODENAME N_HIDDEN_LAYERS BATCH_SIZE \
       CORR_FILE TRAIN_FILE TEST_FILE OUT_DIR

# ---- launch in parallel: Cartesian product of EPOCHS × reps ----
parallel \
    --jobs "$PARALLEL_JOBS" \
    --joblog "$OUT_DIR/parallel.log" \
    --bar \
    --line-buffer \
    run_one ::: "${EPOCHS[@]}" ::: $(seq 1 "$N_REPS")

# ---- merge per-run rows into the summary, sorted by (n_epochs, rep) ----
if compgen -G "$OUT_DIR/per_run_summaries/*.tsv" >/dev/null; then
    cat "$OUT_DIR/per_run_summaries"/*.tsv | sort -k1,1n -k2,2n >> "$SUMMARY"
fi

# ---- show summary ----
echo ""
echo "==== Sweep complete ===="
echo "Output dir: $OUT_DIR"
echo "Summary:    $SUMMARY"
echo "Joblog:     $OUT_DIR/parallel.log"
echo ""
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

# ---- aggregate per epoch count if there are multiple reps ----
if [ "$N_REPS" -gt 1 ]; then
    echo ""
    echo "==== Aggregate per epoch count ===="
    python3 - "$SUMMARY" <<'PYEOF'
import sys, csv
from collections import defaultdict
from statistics import mean, stdev

groups = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f, delimiter='\t'):
        try:
            groups[row['n_epochs']].append(int(row['num_failures']))
        except (ValueError, KeyError):
            pass

print(f"{'n_epochs':>10}  {'mean':>10}  {'std':>10}  {'min':>6}  {'max':>6}  {'n':>3}")
for k in sorted(groups, key=int):
    vals = groups[k]
    s = stdev(vals) if len(vals) > 1 else 0.0
    print(f"{k:>10}  {mean(vals):>10.1f}  {s:>10.1f}  {min(vals):>6}  {max(vals):>6}  {len(vals):>3}")
PYEOF
fi

#!/bin/bash
#SBATCH --account=def-jemerson
#SBATCH --job-name=cer_sweep_test_2026-08-02_23-02-30
#SBATCH --output=./../data/72q_BB_cycles_1/cluster/sweep_test_2026-08-02_23-02-30_%a.out
#SBATCH --error=./../data/72q_BB_cycles_1/cluster/sweep_test_2026-08-02_23-02-30_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10G
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com

# CER sweep — TEST phase. 126 command(s), 1 at a time, 1 x a100_3g.20gb.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep TEST started: $(date)   points: 126   slots: 1   gpus: a100_3g.20gb:1"
echo "========================================="

module load julia/1.12.5
module load cuda

if [ -z "${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="$HOME/.julia"
    fi
fi
echo "[depot] $JULIA_DEPOT_PATH"

export USE_GPU=1
export GPU_BACKEND=cuda
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1

cd $SLURM_SUBMIT_DIR

# Precompile ONCE, ON THE GPU NODE, before fanning out. Doing it here rather
# than on a login node is also what lets CUDA_Runtime_jll see a real driver —
# precompiling CUDA.jl without one bakes in "no CUDA runtime found".
julia --project=$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export JULIA_PKG_PRECOMPILE_AUTO=0

# Report what CUDA sees, so a silent CPU fallback is visible in the log instead
# of costing the whole allocation.
julia --project=$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @info "CUDA functional: $(CUDA.functional())"; CUDA.functional() && @info CUDA.device()' ||     echo "WARNING: CUDA check failed — runs below may fall back to CPU."

LOCAL_WORK_DIR="$SLURM_TMPDIR/72q_BB_cycles_1"
echo "staging 72q_BB_cycles_1 -> $SLURM_TMPDIR"
STAGE_IN_START=$(date +%s)
tar -cf - -C "$(dirname ./../data/72q_BB_cycles_1)" "$(basename ./../data/72q_BB_cycles_1)" | tar -xf - -C "$SLURM_TMPDIR"
echo "[stage-in] done in $(( $(date +%s) - STAGE_IN_START ))s"

COMMANDS_LOCAL="$SLURM_TMPDIR/sweep_commands_test.txt"
sed "s|\$WORKDIR_RUNTIME|$SLURM_TMPDIR|g" "./../data/72q_BB_cycles_1/cluster/sweep_commands_test_2026-08-02_23-02-30.txt" > "$COMMANDS_LOCAL"

LOCAL_LOGS="$LOCAL_WORK_DIR/cluster/logs/sweep_test_2026-08-02_23-02-30"
mkdir -p "$LOCAL_LOGS"

stage_out_done=0
stage_out() {
    [ "$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    echo "[stage-out] $(date '+%F %T')"
    DIRS=()
    for d in results models cluster/logs; do
        [ -d "$LOCAL_WORK_DIR/$d" ] && DIRS+=("$d")
    done
    if [ ${#DIRS[@]} -gt 0 ]; then
        tar -cf - -C "$LOCAL_WORK_DIR" "${DIRS[@]}" | tar -xf - -C "./../data/72q_BB_cycles_1"
        echo "[stage-out] copied: ${DIRS[*]}"
    else
        echo "[stage-out] nothing to copy."
    fi
}
term_handler() { stage_out; exit 0; }
trap term_handler TERM
trap stage_out EXIT

parallel --jobs 1 --results "$LOCAL_LOGS" < "$COMMANDS_LOCAL" &
wait $!

echo "========================================="
echo "sweep TEST finished: $(date)"
echo "========================================="

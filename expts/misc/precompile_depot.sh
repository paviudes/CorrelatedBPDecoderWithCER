#!/usr/bin/env bash
# ============================================================================
# precompile_depot.sh — warm the Julia depot ON THE LOGIN NODE, before sbatch
# ============================================================================
# RUN FROM expts/ , ON A LOGIN NODE, ONCE, after any change to src/ :
#
#     bash misc/precompile_depot.sh
#
# WHY THIS EXISTS
#
# The sweep scripts used to call `Pkg.instantiate(); Pkg.precompile()` inside the
# SLURM job. That was survivable while the array had a single task. It is not
# survivable with `--nodes 2`, for two independent reasons:
#
#   1. COMPUTE NODES HAVE NO INTERNET. `Pkg.instantiate()` contacts the package
#      server whenever anything is missing or the registry looks stale. Offline,
#      that is not an error — it is a TCP connect that blocks until the kernel
#      gives up, repeatedly. The process sits in state `I` burning no CPU.
#      (This is the same failure mode as the CUDA artifact-download timeout.)
#
#   2. EVERY ARRAY TASK SHARES ONE LUSTRE DEPOT. N tasks starting at the same
#      instant all try to precompile into $SCRATCH/.julia at once and serialise
#      on Julia's pidfile lock. Worse, the lock holder's PID lives on a DIFFERENT
#      NODE, so the staleness check cannot validate it and the waiter has no
#      basis to break the lock. With JULIA_NUM_PRECOMPILE_TASKS=1 (set for
#      per-worker memory reasons) the holder is compiling serially — Enzyme alone
#      is many minutes — so the waiter can stall for a large part of the walltime.
#
# Editing anything under src/ invalidates the cache and makes both of these live
# again, which is exactly what happened after the prior_llr_clip change.
#
# Login nodes have internet and no lock contention, so doing it here once is both
# faster and deterministic. The job then runs with JULIA_PKG_OFFLINE=true and
# only VERIFIES the depot, failing in seconds instead of hanging for hours.
# ============================================================================
set -euo pipefail

PROJECT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: this is running inside a SLURM job (job \$SLURM_JOB_ID)." >&2
    echo "       Run it on a LOGIN node — compute nodes have no internet." >&2
    exit 1
fi

if [ -z "${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="$HOME/.julia"
    fi
fi

command -v julia >/dev/null || { echo "no julia on PATH — 'module load julia/1.12.5' first" >&2; exit 1; }

echo "project : $PROJECT"
echo "depot   : $JULIA_DEPOT_PATH"
echo "julia   : $(julia --version)"
echo

# Parallel here: a login node can afford it, and this is the step we want fast.
# (The JOB sets JULIA_NUM_PRECOMPILE_TASKS=1 to bound per-worker memory; that
# constraint does not apply to a one-off warm-up.)
unset JULIA_NUM_PRECOMPILE_TASKS
unset JULIA_PKG_OFFLINE

START=$(date +%s)
julia --project="$PROJECT" -e '
    using Pkg
    Pkg.instantiate()
    Pkg.precompile()
'

# The real acceptance test: can a worker `using` the package with the network
# switched off? This is precisely what the job will do.
echo
echo "verifying the depot is usable OFFLINE (what the compute nodes will see)..."
JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 \
    julia --project="$PROJECT" -e 'using CorrelatedBPDecoderWithCER; println("  load OK")'

echo
echo "depot warm in $(( $(date +%s) - START ))s. Safe to sbatch."

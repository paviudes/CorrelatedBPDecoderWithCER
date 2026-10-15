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

if ! command -v julia >/dev/null; then
    echo "no julia on PATH — 'module load julia/1.12.5' first" >&2
    exit 1
fi

echo "project : $PROJECT"
echo "depot   : $JULIA_DEPOT_PATH"
echo "julia   : $(julia --version)"

# A MANIFEST ONLY WORKS WITH THE JULIA MINOR VERSION THAT RESOLVED IT. The laptop
# runs 1.13 and this cluster 1.12, so each keeps its own: Julia X.Y reads
# Manifest-vX.Y.toml in preference to Manifest.toml. Running 1.12 on the laptop's
# 1.13 Manifest.toml is what failed on 2026-10-15, while precompiling Metal:
#     Package Zstd_jll [3161d3a3-...] is required but does not seem to be installed
# 1.13 made Zstd_jll a standard library, so a 1.13 manifest records it without a
# git-tree-sha1, and 1.12, where it is an ordinary package, has nothing to
# install. Checked here, before Pkg fails somewhere less obvious.
JULIA_MINOR=$(julia --startup-file=no -e 'print(VERSION.major, ".", VERSION.minor)')
MANIFEST="$PROJECT/Manifest-v$JULIA_MINOR.toml"
if [ ! -f "$MANIFEST" ]; then
    MANIFEST="$PROJECT/Manifest.toml"
fi
if [ -f "$MANIFEST" ]; then
    MANIFEST_NAME=$(basename "$MANIFEST")
    MANIFEST_JULIA=$(sed -n 's/^julia_version = "\(.*\)"$/\1/p' "$MANIFEST")
    echo "manifest: $MANIFEST_NAME (resolved by Julia ${MANIFEST_JULIA:-unknown})"
    case "$MANIFEST_JULIA" in
        "$JULIA_MINOR".*)
            ;;
        *)
            echo >&2
            echo "ERROR: Julia $JULIA_MINOR would use $MANIFEST_NAME, which Julia ${MANIFEST_JULIA:-unknown} resolved." >&2
            echo "  Give this Julia a manifest of its own, then rerun this script:" >&2
            echo >&2
            echo "      cd $PROJECT" >&2
            if [ "$MANIFEST_NAME" = "Manifest.toml" ]; then
                echo "      cp Manifest.toml Manifest-v$JULIA_MINOR.toml" >&2
            fi
            echo "      julia --project=. -e 'using Pkg; Pkg.resolve()'" >&2
            echo >&2
            echo "  (Pkg.update() in place of Pkg.resolve() if it reports unsatisfiable requirements.)" >&2
            echo "  Commit Manifest-v$JULIA_MINOR.toml; Julia versions without one keep using Manifest.toml." >&2
            exit 1
            ;;
    esac
fi
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

# Which CUDA toolkit this node selected. The CUDA runtime is an ARTIFACT, chosen
# from the driver Julia can see and downloaded lazily on first load, which on a
# login node happens above, online. (CUDA_Runtime_jll reads only the `version`
# and `local` preferences; the `local_toolkit = true` in LocalPreferences.toml is
# not a key it knows, so the cluster has always used the artifact.) A GPU node is
# offline and can only use an artifact already in the depot, so "none" here
# means the job's CUDA gate would fail.
CUDA_WARNING=0
if [ "$(uname -s)" = "Linux" ]; then
    CUDA_TOOLKIT=$(JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0 julia --project="$PROJECT" -e '
        runtime_jll::Module = Base.require(Base.PkgId(
            Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll"))
        print(runtime_jll.host_platform["cuda"])') || CUDA_TOOLKIT="unknown (the query failed)"
    echo "  CUDA toolkit selected on this node: $CUDA_TOOLKIT"
    if [ "$CUDA_TOOLKIT" = "none" ]; then
        CUDA_WARNING=1
        echo >&2
        echo "WARNING: no CUDA driver is visible on this node, so no CUDA runtime was downloaded," >&2
        echo "  and an offline GPU node cannot fetch the one its own driver selects. Pin the" >&2
        echo "  toolkit (12.9 runs on any CUDA 12 or 13 driver), then rerun this script:" >&2
        echo >&2
        echo "      julia --project=$PROJECT -e 'using CUDA; CUDA.set_runtime_version!(v\"12.9\")'" >&2
        echo >&2
        echo "  That writes version = \"12.9\" under [CUDA_Runtime_jll] and [CUDA_Compiler_jll]" >&2
        echo "  in LocalPreferences.toml, so this node and the GPU nodes use the same toolkit." >&2
    fi
fi

echo
if [ "$CUDA_WARNING" = "1" ]; then
    echo "depot warm in $(( $(date +%s) - START ))s, but NOT safe to sbatch until the CUDA toolkit is pinned (above)."
    exit 1
fi
echo "depot warm in $(( $(date +%s) - START ))s. Safe to sbatch."

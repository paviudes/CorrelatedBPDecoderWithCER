#!/usr/bin/env bash
# ============================================================================
# rerun_failed_tests.sh — TEMPORARY. Re-run the two test points that OOMed in
#                         the 2026-08-22 sweep. Delete once they are collected.
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/rerun_failed_tests.sh            # run them
#     bash misc/rerun_failed_tests.sh --dry_run  # print the commands only
#
# WHY ONLY TWO POINTS, AND WHY TEST-ONLY.
#
# Phase 1 of that job trained all 20 points successfully — all 20 weights JSONs
# are on disk. Phase 2 ran 8 test processes across 2 GPUs, 4 per card, and two of
# them died at cuDevicePrimaryCtxRetain:
#
#     p_0.0005_sig_0.001_s_1   lam0p3
#     p_0.0005_sig_0.001_s_2   lam1p5
#
# The cause was a wrong assumption in the sweep script (since reverted): that
# GPU_MEMORY bounds a process's GPU footprint. It does not — it is an input to
# `compute_optimal_batch_size_for` and sizes the prediction batch only. The CUDA
# context, the densified check matrix and the per-layer state sit on top.
#
# So this only needs the forward pass repeated, ONE PROCESS AT A TIME with the
# whole card. That is ~4-5 minutes per point on a GPU.
#
# WHY IT REGENERATES THE HYPERPARAMETER FILES.
#
# The sweep's stage-out carries `--exclude='hyperparams_cw_*.toml'`, so the
# generated TOMLs only ever existed on the compute node. They are reproduced here
# byte-identically EXCEPT `retrain = false`, so the run loads the existing weights
# instead of training again. If a value here drifts from what the sweep writes,
# the loaded model no longer matches the configuration it is recorded under —
# hence the weights-file check below, which fails loudly rather than retraining.
# ============================================================================
set -euo pipefail

WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_spread_comparison"
BASE_HP="hyperparams_epochs_5_corrs.toml"
NLAYERS=90
SEED=1
SPARSITY="0.0"
GATE_TAU="0.5"
SINGLE_QUBIT_RESCALE="0.1"
HEAP_HINT="4G"
# 85% of a whole A100-40GB. Override for a smaller card: --gpu_memory 8704M
GPU_MEMORY="34816M"
DRY_RUN=0

# <dataset key>:<lambda>
POINTS="p_0.0005_sig_0.001_s_1:0.3 p_0.0005_sig_0.001_s_2:1.5"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry_run)    DRY_RUN=1; shift;;
        --points)     POINTS="$2"; shift 2;;
        --gpu_memory) GPU_MEMORY="$2"; shift 2;;
        --codename)   CODENAME="$2"; shift 2;;
        --base_hp)    BASE_HP="$2"; shift 2;;
        -h|--help)    sed -n '2,36p' "$0"; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

MODELS_DIR="$WORKDIR/$CODENAME/models"
RESULTS_DIR="$WORKDIR/$CODENAME/results"

if [ ! -d "$MODELS_DIR" ]; then
    echo "no models dir: $MODELS_DIR — run this from expts/" >&2
    exit 1
fi
if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
    echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
    exit 1
fi
mkdir -p "$WORKDIR/$CODENAME/logs"

tag_of() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# Training is CPU-only here regardless; this is a pure forward pass, so a GPU is
# a large speedup but not a requirement. Fall back rather than fail.
USE_GPU=1
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    USE_GPU=0
    echo "  NOTE: no usable GPU visible — falling back to USE_GPU=0."
    echo "  A 10^6-sample forward pass on CPU takes appreciably longer than the"
    echo "  ~4-5 min it takes on an A100. If you are on a login node, stop and run"
    echo "  this inside salloc instead."
    echo
fi

echo "rerun of $(echo $POINTS | wc -w) failed test point(s)"
echo "  codename   : $CODENAME"
echo "  base       : $MODELS_DIR/$BASE_HP"
echo "  USE_GPU    : $USE_GPU   GPU_MEMORY=$GPU_MEMORY   (one process at a time)"
echo

n_missing=0
for point in $POINTS; do
    key="${point%%:*}"
    lambda="${point##*:}"
    lam_tag="_lam$(tag_of "$lambda")"
    run_tag="_cwcer_gated_sp$(tag_of "$SPARSITY")${lam_tag}"
    weights="$MODELS_DIR/neuralbp_weights_nlayers_${NLAYERS}_epochs_5_trained_using_train_${key}${run_tag}_seed_${SEED}.json"
    if [ ! -f "$weights" ]; then
        echo "  MISSING WEIGHTS for ${key} lambda=${lambda}:" >&2
        echo "    $(basename "$weights")" >&2
        n_missing=$((n_missing + 1))
    fi
done
if [ "$n_missing" -gt 0 ]; then
    echo >&2
    echo "  $n_missing point(s) have no trained model. This script is TEST-ONLY and" >&2
    echo "  deliberately will not retrain: a model trained here would not share the" >&2
    echo "  cluster run's environment. Re-run the full sweep for those points." >&2
    exit 1
fi

for point in $POINTS; do
    key="${point%%:*}"
    lambda="${point##*:}"
    lam_tag="_lam$(tag_of "$lambda")"
    run_tag="_cwcer_gated_sp$(tag_of "$SPARSITY")${lam_tag}"
    hp="hyperparams_rerun_cer_gated_sp$(tag_of "$SPARSITY")${lam_tag}_$(tag_of "$key")_seed${SEED}.toml"

    # Same keys the sweep strips, so nothing is defined twice.
    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|require_correlations|correlation_weight)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp"
    cat >> "$MODELS_DIR/$hp" <<EOF

# ---- injected by rerun_failed_tests.sh ----
# retrain = false: the weights already exist from the cluster run's phase 1. This
# script must never train, or the recorded model would come from a different
# environment than the 18 points it is being compared against.
retrain = false
run_tag = "${run_tag}"
use_CER = true
seed = ${SEED}
sparsity_importance = "${SPARSITY},${SPARSITY},0.8,up"
syndrome_gate_threshold = ${GATE_TAU}
single_qubit_rescale = ${SINGLE_QUBIT_RESCALE}
require_correlations = true
correlation_weight = "${lambda},${lambda},0.7,up"
EOF

    cmd="julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl \
--workdir $WORKDIR --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${key}.txt --quiet true --diagnose true \
--train train_${key}.txt --test test_${key}.txt"

    echo "  ---- ${key}  lambda = ${lambda} ----"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    $cmd"
        echo
        continue
    fi
    started=$(date +%s)
    USE_GPU=$USE_GPU GPU_MEMORY=$GPU_MEMORY JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        bash -c "$cmd" || {
            echo "    FAILED — see the output above. The other point(s) still run." >&2
            continue
        }
    echo "    done in $(( ($(date +%s) - started) / 60 )) min"
    echo
done

if [ "$DRY_RUN" = "1" ]; then
    exit 0
fi

echo "expected result files:"
for point in $POINTS; do
    key="${point%%:*}"; lambda="${point##*:}"
    run_tag="_cwcer_gated_sp$(tag_of "$SPARSITY")_lam$(tag_of "$lambda")"
    f="$RESULTS_DIR/simulation_results_test_${key}_nlayers_${NLAYERS}_epochs_5_trained_using_train_${key}${run_tag}_seed_${SEED}.csv"
    if [ -f "$f" ]; then
        printf "  OK      %s\n" "$(basename "$f")"
    else
        printf "  MISSING %s\n" "$(basename "$f")"
    fi
done
echo
echo "then re-collect:  bash misc/sweep_correlation_weight.sh --collect"
echo "and delete this script — it hard-codes one job's failures."

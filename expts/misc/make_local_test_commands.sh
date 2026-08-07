#!/usr/bin/env bash
# ============================================================================
# make_local_test_commands.sh — build a local TEST command list from whatever
#                               trained models are actually on disk
# ============================================================================
# Scans <workdir>/<codename>/models/ for neuralbp_weights_*.json, reconstructs
# each model's run_tag and hyperparameters file, and writes one julia test
# command per model. Nothing about the sweep grid is assumed — if you trained
# 126 ladder points, you get 126 commands; if you trained 16, you get 16.
#
# For every model it also:
#   * flips `retrain = false` in that model's hyperparameters TOML (the TRAIN
#     phase leaves it `true`, and running a test with retrain = true would
#     RETRAIN the model on your laptop instead of loading it), and
#   * checks the test-errors file and CER file exist, skipping (with a note)
#     any model whose inputs are missing locally.
#
# USAGE (from expts/):
#     bash misc/make_local_test_commands.sh
#     bash misc/make_local_test_commands.sh --out misc/local_test_commands.txt
# then:
#     bash misc/local_test_commands.txt
#
# The emitted file exports USE_GPU=1 (Metal on a Mac) and passes --quiet false.
# USE_GPU=1 is not optional: the CPU path in predict.jl is UNBATCHED and would
# try to allocate an (n_bits x n_samples x n_layers) tensor — tens of TB at
# 10^6 test samples.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_sweep_common.sh"

WORKDIR="./../data"
CODENAME="72q_BB_cycles_1"
OUT="misc/local_test_commands.txt"
SKIP_EXISTING=0                # 1 => omit points whose results CSV already exists

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)       WORKDIR="$2";  shift 2;;
        --codename)      CODENAME="$2"; shift 2;;
        --out)           OUT="$2";      shift 2;;
        --skip_existing) SKIP_EXISTING=1; shift;;
        -h|--help)       sed -n '2,28p' "$0"; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

MODELS_DIR="$WORKDIR/$CODENAME/models"
TEST_DIR="$WORKDIR/$CODENAME/testing_data"
CER_DIR="$WORKDIR/$CODENAME/correlated_weights"
RESULTS_DIR="$WORKDIR/$CODENAME/results"

[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR (run from expts/)" >&2; exit 1; }

shopt -s nullglob
weights=( "$MODELS_DIR"/neuralbp_weights_*.json )
shopt -u nullglob
[ "${#weights[@]}" -gt 0 ] || { echo "no neuralbp_weights_*.json in $MODELS_DIR" >&2; exit 1; }

n_emitted=0
n_baseline=0
n_no_toml=0
n_no_inputs=0
n_done=0
missing_tomls=""

tmp_body="$(mktemp)"
trap 'rm -f "$tmp_body"' EXIT

for weights_path in "${weights[@]}"; do
    base="${weights_path##*/}"; base="${base%.json}"

    # neuralbp_weights_nlayers_<L>_epochs_<E>_trained_using_train_p_<P>_s_<S>[_no_cer][<run_tag>]
    rest="${base#neuralbp_weights_nlayers_}"
    n_layers="${rest%%_epochs_*}"
    rest="${rest#*_epochs_}"
    rest="${rest#*_trained_using_}"          # train_p_<P>_s_<S>[_no_cer][run_tag]

    train_body="${rest#train_p_}"
    train_p="${train_body%%_s_*}"
    after_seed="${train_body#*_s_}"          # <S>[_no_cer][run_tag]
    seed="${after_seed%%[!0-9]*}"
    remainder="${after_seed#"$seed"}"        # [_no_cer][run_tag]

    if [ "${remainder#_no_cer}" != "$remainder" ]; then
        use_cer=false
        run_tag="${remainder#_no_cer}"
    else
        use_cer=true
        run_tag="$remainder"
    fi

    # No run_tag => a baseline model from an earlier campaign, not a sweep point.
    if [ -z "$run_tag" ]; then
        n_baseline=$((n_baseline + 1))
        continue
    fi

    hp_name=$(sweep_hp_name "$run_tag" "$use_cer")
    hp_path="$MODELS_DIR/$hp_name"
    if [ ! -f "$hp_path" ]; then
        n_no_toml=$((n_no_toml + 1))
        missing_tomls="$missing_tomls\n    $hp_name  (for $base)"
        continue
    fi

    test_file="test_p_${train_p}_s_${seed}.txt"
    cer_file="correlated_weights_p_${train_p}_s_${seed}.txt"
    if [ ! -f "$TEST_DIR/$test_file" ] || [ ! -f "$CER_DIR/$cer_file" ]; then
        n_no_inputs=$((n_no_inputs + 1))
        continue
    fi

    if [ "$SKIP_EXISTING" = "1" ]; then
        n_epochs=$(sweep_toml_get "$hp_path" n_epochs)
        cer_tag=$(sweep_cer_tag_for "$use_cer")
        existing="$RESULTS_DIR/simulation_results_test_p_${train_p}_s_${seed}_nlayers_${n_layers}_epochs_${n_epochs}_trained_using_train_p_${train_p}_s_${seed}${cer_tag}${run_tag}.csv"
        if [ -f "$existing" ]; then
            n_done=$((n_done + 1))
            continue
        fi
    fi

    # The TRAIN phase leaves retrain = true; testing must LOAD, not retrain.
    sweep_disable_retrain "$hp_path"

    printf 'julia --project="./../" neural_bp_experiments.jl --workdir %s --codename %s --n_hidden_layers %s --hyperparams %s --cer_data %s --quiet false --train train_p_%s_s_%s.txt --test %s\n' \
        "$WORKDIR" "$CODENAME" "$n_layers" "$hp_name" "$cer_file" "$train_p" "$seed" "$test_file" >> "$tmp_body"
    n_emitted=$((n_emitted + 1))
done

[ "$n_emitted" -gt 0 ] || { echo "ERROR: no testable models found (see counts above)." >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
{
    echo "# ==========================================================================="
    echo "# local_test_commands.txt — generated $(date '+%F %T') by make_local_test_commands.sh"
    echo "# ==========================================================================="
    echo "# $n_emitted test command(s), one per trained sweep model found in"
    echo "#   $MODELS_DIR"
    echo "#"
    echo "# RUN FROM expts/ :   bash $OUT"
    echo "# Sequential; one failure does not stop the rest. --quiet false shows progress."
    echo "#"
    echo "# Every referenced hyperparameters TOML has already had retrain flipped to"
    echo "# false, so each run LOADS its trained model instead of retraining."
    echo "# ==========================================================================="
    echo
    echo "# USE_GPU=1 is REQUIRED: the CPU path in predict.jl is unbatched and would"
    echo "# allocate an (n_bits x n_samples x n_layers) tensor — tens of TB at 10^6"
    echo "# test samples. 1 selects the batched Metal path on a Mac."
    echo 'export USE_GPU="1"'
    echo
    sort "$tmp_body"
} > "$OUT"

echo "[local-test] wrote $n_emitted command(s) -> $OUT"
[ "$n_baseline"   -gt 0 ] && echo "  skipped $n_baseline baseline model(s) (no run_tag — earlier campaign)"
[ "$n_done"       -gt 0 ] && echo "  skipped $n_done model(s) already having a results CSV (--skip_existing)"
[ "$n_no_inputs"  -gt 0 ] && echo "  skipped $n_no_inputs model(s) missing their test/CER input file locally"
if [ "$n_no_toml" -gt 0 ]; then
    echo "  WARNING: $n_no_toml model(s) have no matching hyperparameters TOML — copy them"
    echo "           from the cluster's models/ folder:" >&2
    printf "$missing_tomls\n" | head -10 >&2
fi
echo
echo "run with:  bash $OUT"

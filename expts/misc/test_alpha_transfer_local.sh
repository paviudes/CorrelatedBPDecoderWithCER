#!/usr/bin/env bash
# test_alpha_transfer_local.sh -- test a run's trained ENRICHED models on one test
# set with α carried to that set's prior scale (coupling_scale_transfer =
# "alpha_star"), locally, one model per seed, then print the result beside the
# learned-α test of the same models.
#
#   cd expts
#   bash misc/test_alpha_transfer_local.sh                       # p_0.0005_sig_0.0_s_1, seeds 1-5
#   bash misc/test_alpha_transfer_local.sh --key p_0.0015_sig_0.0015_s_1 --seeds "1 3"
#
# Nothing is trained: the TOML it writes says retrain = false, and the script
# refuses to start unless every model is on disk (a missing weights file would
# otherwise make neural_bp_experiments.jl TRAIN a model of its own). The models
# stay as they are; the results go to
#   results/simulation_results_test_<key>_..._<run tag>_alphastar_seed_<n>.csv
# (plus the per_sample_failures_ / layer_profile_ files), so the learned-α results
# of the same models are never overwritten or mistaken for these.
#
# Options
#   --codename <dir>         run directory under data/        (72q_BB_cycles_1_sweep)
#   --key <key>              test set, p_<p>_sig_<sigma>_s_<n> (p_0.0005_sig_0.0_s_1)
#   --train-key <key>        what the models were trained on  (p_0.0015_sig_0.003_s_1)
#   --seeds "<n> ..."        which models                     ("1 2 3 4 5")
#   --run-tag <tag>          the models' run tag              (_cer_enr)
#   --hyperparams <file>     base TOML the models were trained with
#                            (sweeps/hyperparams_cer_enriched.toml)
#   --testing-source <dir>   where to link test_<key>.txt from when the run's own
#                            testing_data/ lacks it (../data/72q_BB_cycles_1_trainable_alpha/testing_data)
#   --gpu-memory <size>      GPU_MEMORY for the test batch    (6G)
#   --cpu                    test on the CPU (USE_GPU=0) instead of Metal
#
# One test is ~1e6 samples through 90 enriched layers; the first point's time,
# printed when it finishes, is the number to plan the rest by.
set -euo pipefail

CODENAME="72q_BB_cycles_1_sweep"
KEY="p_0.0005_sig_0.0_s_1"
TRAIN_KEY="p_0.0015_sig_0.003_s_1"
SEEDS="1 2 3 4 5"
RUN_TAG="_cer_enr"
BASE_TOML="sweeps/hyperparams_cer_enriched.toml"
TESTING_SOURCE="../data/72q_BB_cycles_1_trainable_alpha/testing_data"
GPU_MEM="6G"
USE_GPU_VALUE="1"
N_LAYERS="90"

while [ $# -gt 0 ]; do
    case "$1" in
        --codename) CODENAME="${2:?--codename needs a value}"; shift ;;
        --key) KEY="${2:?--key needs a value}"; shift ;;
        --train-key) TRAIN_KEY="${2:?--train-key needs a value}"; shift ;;
        --seeds) SEEDS="${2:?--seeds needs a value}"; shift ;;
        --run-tag) RUN_TAG="${2:?--run-tag needs a value}"; shift ;;
        --hyperparams) BASE_TOML="${2:?--hyperparams needs a value}"; shift ;;
        --testing-source) TESTING_SOURCE="${2:?--testing-source needs a value}"; shift ;;
        --gpu-memory) GPU_MEM="${2:?--gpu-memory needs a value}"; shift ;;
        --cpu) USE_GPU_VALUE="0" ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "test_alpha_transfer_local: unknown option $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

EXPTS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXPTS_DIR"
DATA="../data/$CODENAME"

refuse() {
    echo "test_alpha_transfer_local: $1" >&2
    exit 1
}

if [ ! -d "$DATA" ]; then refuse "no run directory $DATA"; fi
if [ ! -f "$BASE_TOML" ]; then refuse "no base hyperparameters file $BASE_TOML"; fi
if ! command -v julia > /dev/null 2>&1; then refuse "julia is not on PATH"; fi

# The weights name carries n_epochs, so take it from the same TOML the models were
# trained with.
N_EPOCHS="$(sed -nE 's/^[[:space:]]*n_epochs[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' "$BASE_TOML" | head -n 1)"
if [ -z "$N_EPOCHS" ]; then refuse "$BASE_TOML has no n_epochs"; fi
if ! grep -qE '^[[:space:]]*check_node[[:space:]]*=[[:space:]]*"enriched"' "$BASE_TOML"; then
    refuse "$BASE_TOML is not an enriched-node TOML (check_node = \"enriched\"); there is no α to carry"
fi
STEM="nlayers_${N_LAYERS}_epochs_${N_EPOCHS}_trained_using_train_${TRAIN_KEY}${RUN_TAG}"

# 1. Every model must be on disk -- checked BEFORE anything runs.
MISSING=0
for SEED in $SEEDS; do
    MODEL="$DATA/models/neuralbp_weights_${STEM}_seed_${SEED}.json"
    if [ ! -f "$MODEL" ]; then
        echo "  missing model: $MODEL" >&2
        MISSING=$((MISSING + 1))
    fi
done
if [ "$MISSING" -ne 0 ]; then refuse "$MISSING model(s) missing; nothing was run"; fi

# 2. The CER files of the test key (priors, couplings, α*) and of the training key (α*).
for CER_KEY in "$KEY" "$TRAIN_KEY"; do
    CER_FILE="$DATA/correlated_weights/correlated_weights_${CER_KEY}.txt"
    if [ ! -f "$CER_FILE" ]; then refuse "no CER file $CER_FILE"; fi
done

# 3. The test file: the run's own, or linked from --testing-source.
TEST_FILE="$DATA/testing_data/test_${KEY}.txt"
if [ ! -e "$TEST_FILE" ]; then
    SOURCE_FILE="$TESTING_SOURCE/test_${KEY}.txt"
    if [ ! -f "$SOURCE_FILE" ]; then
        refuse "no $TEST_FILE, and none in --testing-source $TESTING_SOURCE to link"
    fi
    mkdir -p "$DATA/testing_data"
    SOURCE_ABSOLUTE="$(cd "$(dirname "$SOURCE_FILE")" && pwd)/$(basename "$SOURCE_FILE")"
    ln -s "$SOURCE_ABSOLUTE" "$TEST_FILE"
    echo "  linked $TEST_FILE -> $SOURCE_ABSOLUTE"
fi

# 4. Are they the samples the learned-α test saw? Its per-sample failure file lists
#    (sample index, error weight); every one must match the test file's column.
python3 - "$TEST_FILE" "$DATA/results" "$KEY" "$STEM" $SEEDS <<'PYTHON' || refuse "the test file's samples differ from the ones the learned-α test decoded"
import csv, os, sys
test_file, results_dir, key, stem = sys.argv[1:5]
seeds = sys.argv[5:]
failure_files = [os.path.join(results_dir, f"per_sample_failures_test_{key}_{stem}_seed_{seed}.csv") for seed in seeds]
failure_files = [path for path in failure_files if os.path.isfile(path)]
if not failure_files:
    print("  no learned-α per-sample file for this key: sample check skipped")
    sys.exit(0)
expected = {}
for path in failure_files:
    with open(path) as handle:
        for row in csv.DictReader(handle):
            expected[int(row["sample_index"])] = int(row["error_weight"])
weights = dict.fromkeys(expected, 0)
with open(test_file) as handle:
    for line in handle:
        for sample_index in weights:
            if line[2 * (sample_index - 1)] == "1":   # columns are samples, 1-based
                weights[sample_index] += 1
mismatched = [index for index in expected if weights[index] != expected[index]]
if mismatched:
    print(f"  {len(mismatched)} of {len(expected)} listed failures have another error weight in {test_file}")
    sys.exit(1)
print(f"  samples check: all {len(expected)} learned-α failures have the same error weight in the test file")
PYTHON

# 5. The test TOML: the base with retrain = false, the arm's run tag, and the transfer.
#    Under models/submissions/ (neural_bp_experiments.jl reads <run>/models/<name>;
#    cleanup.py clears submissions/).
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TOML_REL="submissions/local_alphastar_${TIMESTAMP}/hyperparams${RUN_TAG}_alphastar_test.toml"
TOML_PATH="$DATA/models/$TOML_REL"
mkdir -p "$(dirname "$TOML_PATH")"
{
    grep -vE '^[[:space:]]*(retrain|use_CER|run_tag|seed|coupling_scale_transfer)[[:space:]]*=' "$BASE_TOML"
    echo ""
    echo "# Written by misc/test_alpha_transfer_local.sh: test the trained models, alpha carried to the test key."
    echo "retrain = false"
    echo "use_CER = true"
    echo "run_tag = \"$RUN_TAG\""
    echo "coupling_scale_transfer = \"alpha_star\""
} > "$TOML_PATH"
echo "  test TOML -> $TOML_PATH"

# 6. One test per model, in sequence, output live.
echo
echo "[alpha transfer] $KEY, models trained on $TRAIN_KEY, run tag $RUN_TAG, seeds: $SEEDS"
if [ "$USE_GPU_VALUE" = "1" ]; then
    echo "  device: Metal (USE_GPU=1, GPU_MEMORY=$GPU_MEM)"
else
    echo "  device: CPU (USE_GPU=0)"
fi
N_FAILED=0
for SEED in $SEEDS; do
    RESULTS_FILE="$DATA/results/simulation_results_test_${KEY}_${STEM}_alphastar_seed_${SEED}.csv"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  seed $SEED: $(basename "$RESULTS_FILE") exists -- kept (delete it to re-run)"
        continue
    fi
    echo
    echo "----- seed $SEED"
    STARTED=$(date +%s)
    STATUS=0
    USE_GPU="$USE_GPU_VALUE" GPU_MEMORY="$GPU_MEM" JULIA_NUM_THREADS=1 \
        julia --project="./../" --heap-size-hint=4G neural_bp_experiments.jl \
        --workdir ./../data --codename "$CODENAME" --n_hidden_layers "$N_LAYERS" \
        --hyperparams "$TOML_REL" --cer_data "correlated_weights_${KEY}.txt" \
        --quiet true --seed "$SEED" --diagnose true \
        --train "train_${TRAIN_KEY}.txt" --test "test_${KEY}.txt" || STATUS=$?
    echo "      seed $SEED: exit $STATUS, $(( $(date +%s) - STARTED )) s"
    if [ "$STATUS" -ne 0 ]; then N_FAILED=$((N_FAILED + 1)); fi
done

# 7. Side by side with the learned-α tests of the same models.
echo
python3 - "$DATA/results" "$KEY" "$STEM" $SEEDS <<'PYTHON'
import csv, os, sys
results_dir, key, stem = sys.argv[1:4]
seeds = sys.argv[4:]
def first_row(path):
    """The one row of a results CSV, or None when the file is not there."""
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return next(csv.DictReader(handle))
def failures_text(result):
    if result is None:
        return "-"
    return f"{result['num_failures']} ({result['num_coset_failures']} | {result['num_convergence_failures']})"
def alpha_text(result, column):
    if result is None or not result.get(column):
        return "-"
    return f"{float(result[column]):.4f}"
print(f"{key}: failures per 1e6 samples (coset | convergence)")
print(f"{'seed':>4}  {'alpha learned':>13}  {'alpha used':>10}  {'learned alpha':>22}  {'alpha_star':>22}")
for seed in seeds:
    learned = first_row(os.path.join(results_dir, f"simulation_results_test_{key}_{stem}_seed_{seed}.csv"))
    transferred = first_row(os.path.join(results_dir, f"simulation_results_test_{key}_{stem}_alphastar_seed_{seed}.csv"))
    alpha_learned = alpha_text(transferred, "learned_coupling_scale")
    if alpha_learned == "-":
        alpha_learned = alpha_text(learned, "coupling_scale")
    alpha_used = alpha_text(transferred, "coupling_scale")
    print(f"{seed:>4}  {alpha_learned:>13}  {alpha_used:>10}  {failures_text(learned):>22}  {failures_text(transferred):>22}")
PYTHON

if [ "$N_FAILED" -ne 0 ]; then
    echo "  $N_FAILED test(s) exited non-zero" >&2
    exit 1
fi

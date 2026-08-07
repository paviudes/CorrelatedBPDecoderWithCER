#!/usr/bin/env bash
# ============================================================================
# test_seed_reproducibility.sh — acceptance tests for the `seed` hyperparameter
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/test_seed_reproducibility.sh
#
# Four tests, matching the specification:
#
#   1. REPRODUCIBILITY  same TOML + seed = 1, twice
#                       -> byte-identical weights JSON, identical num_failures
#   2. SEEDS DIFFER     seed = 1 vs seed = 2
#                       -> different weights, both files present, no collision
#   3. NO-SEED PATH     seed absent -> filenames exactly as before, and two runs
#                       still differ from each other (old behaviour preserved,
#                       not accidentally frozen)
#   4. CLI OVERRIDE     --seed 3 beats a TOML that sets seed = 1
#
# Deliberately tiny: 2 layers, 2 epochs, 5 updates/epoch, batch 10. This tests
# DETERMINISM, not decoder quality, and a small model exercises every RNG draw
# just as well as a big one while finishing in seconds.
#
# Everything is written under a scratch codename and deleted at the end, so no
# real result or model file is touched.
# ============================================================================
set -uo pipefail

WORKDIR="./../data"
SOURCE_CODENAME="${SOURCE_CODENAME:-72q_BB_cycles_1}"
CODENAME="seedtest_$$"
SCRATCH="$WORKDIR/$CODENAME"
NLAYERS=2
TRAIN_FILE="train_p_0.0005_s_1.txt"
TEST_FILE="test_p_0.0005_s_1.txt"
N_TEST_SAMPLES=2000          # subset of the test file: keeps the run quick

pass_count=0
fail_count=0
report() {
    local outcome="$1" name="$2" detail="${3:-}"
    if [ "$outcome" = "PASS" ]; then
        pass_count=$((pass_count + 1)); printf '  \033[32mPASS\033[0m  %s\n' "$name"
    else
        fail_count=$((fail_count + 1)); printf '  \033[31mFAIL\033[0m  %s\n' "$name"
    fi
    [ -n "$detail" ] && printf '        %s\n' "$detail"
    return 0
}

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# ------------------------------------------------------------------ set-up ---
[ -d "$WORKDIR/$SOURCE_CODENAME" ] || { echo "no $WORKDIR/$SOURCE_CODENAME — run from expts/" >&2; exit 1; }

mkdir -p "$SCRATCH"/{code,models,training_data,testing_data,results,logs,correlated_weights}
cp "$WORKDIR/$SOURCE_CODENAME/code/HZ.txt" "$WORKDIR/$SOURCE_CODENAME/code/LZ.txt" "$SCRATCH/code/"
# Small slices of the data: same 72-row layout, fewer sample columns.
cut -d' ' -f1-20000 "$WORKDIR/$SOURCE_CODENAME/training_data/$TRAIN_FILE" > "$SCRATCH/training_data/$TRAIN_FILE"
cut -d' ' -f1-$N_TEST_SAMPLES "$WORKDIR/$SOURCE_CODENAME/testing_data/$TEST_FILE" > "$SCRATCH/testing_data/$TEST_FILE"

write_toml() {   # write_toml <file> [seed]
    local target="$1" seed_value="${2:-}"
    cat > "$SCRATCH/models/$target" <<EOF
retrain = true
learning_rate = 0.01
max_grad_norm = 2.0
weight_decay = 0.0001
nanskip = 5
adam_eps = 0.0001
batch_size = 10
n_epochs = 2
warmup_layers = 1
initial_conditions_scale = 0.1
online_training = true
n_gradient_updates_per_epoch = 5
use_CER = false
loss_layer_temperature = "2e-1,1e0,0.7,down"
correlation_weight = "1e-2,1,0.7,up"
llr_certainty_importance = "1e-3,1e-2,0.7,up"
sparsity_importance = "0,5e-1,0.8,up"
EOF
    if [ -n "$seed_value" ]; then
        echo "seed = $seed_value" >> "$SCRATCH/models/$target"
    fi
}

run_training() {   # run_training <toml> [extra flags...]
    local toml="$1"; shift
    julia --project="./../" neural_bp_experiments.jl \
        --workdir "$WORKDIR" --codename "$CODENAME" --n_hidden_layers "$NLAYERS" \
        --hyperparams "$toml" --quiet true \
        --train "$TRAIN_FILE" --test "$TEST_FILE" "$@" > /dev/null 2>&1
    return $?
}

weights_of() { ls "$SCRATCH/models"/neuralbp_weights_*"$1".json 2>/dev/null | head -1; }
failures_of() {  # failures_of <results csv>
    python3 -c "
import csv,sys
with open('$1') as f:
    row = next(csv.DictReader(f))
print(row['num_failures'])
" 2>/dev/null
}

echo "seed acceptance tests   (scratch codename: $CODENAME)"
echo "======================================================================"

# --- 1. same seed twice -> identical -----------------------------------------
write_toml "hp_seed1.toml" 1
run_training "hp_seed1.toml"
w1=$(weights_of "_seed_1"); r1=$(ls "$SCRATCH/results"/*_seed_1.csv 2>/dev/null | head -1)
cp "$w1" "$SCRATCH/first_seed1.json" 2>/dev/null; f1=$(failures_of "$r1")
rm -f "$w1" "$r1"
run_training "hp_seed1.toml"
w1b=$(weights_of "_seed_1"); r1b=$(ls "$SCRATCH/results"/*_seed_1.csv 2>/dev/null | head -1)
f1b=$(failures_of "$r1b")

if [ -z "${w1:-}" ] || [ -z "${w1b:-}" ]; then
    report FAIL "1. reproducibility" "weights file missing — did training run? (drop >/dev/null in run_training to see why)"
elif cmp -s "$SCRATCH/first_seed1.json" "$w1b" && [ "$f1" = "$f1b" ] && [ -n "$f1" ]; then
    report PASS "1. reproducibility: same seed -> identical weights and num_failures" "num_failures = $f1 both runs"
else
    report FAIL "1. reproducibility" "weights identical: $(cmp -s "$SCRATCH/first_seed1.json" "$w1b" && echo yes || echo NO); failures $f1 vs $f1b"
fi

# --- 2. different seeds -> different, and coexisting --------------------------
write_toml "hp_seed2.toml" 2
run_training "hp_seed2.toml"
w2=$(weights_of "_seed_2")
if [ -n "${w2:-}" ] && [ -n "${w1b:-}" ] && ! cmp -s "$w1b" "$w2" && [ -f "$w1b" ] && [ -f "$w2" ]; then
    report PASS "2. seeds differ: different weights, both filenames coexist" "$(basename "$w1b")  /  $(basename "$w2")"
else
    report FAIL "2. seeds differ" "seed1=$(basename "${w1b:-none}") seed2=$(basename "${w2:-none}")"
fi

# --- 3. no seed -> old filenames, still non-deterministic ---------------------
write_toml "hp_noseed.toml"
run_training "hp_noseed.toml"
w3=$(ls "$SCRATCH/models"/neuralbp_weights_*.json 2>/dev/null | grep -v "_seed_" | head -1)
cp "$w3" "$SCRATCH/first_noseed.json" 2>/dev/null
rm -f "$w3"
run_training "hp_noseed.toml"
w3b=$(ls "$SCRATCH/models"/neuralbp_weights_*.json 2>/dev/null | grep -v "_seed_" | head -1)

expected_noseed="neuralbp_weights_nlayers_${NLAYERS}_epochs_2_trained_using_${TRAIN_FILE%.txt}_no_cer.json"
if [ -z "${w3b:-}" ]; then
    report FAIL "3. no-seed path" "no unseeded weights file produced"
elif [ "$(basename "$w3b")" != "$expected_noseed" ]; then
    report FAIL "3. no-seed filename unchanged" "got $(basename "$w3b"), expected $expected_noseed"
elif cmp -s "$SCRATCH/first_noseed.json" "$w3b"; then
    report FAIL "3. no-seed still non-deterministic" "two unseeded runs produced IDENTICAL weights — the RNG got frozen"
else
    report PASS "3. no-seed: filename unchanged AND runs still differ" "$expected_noseed"
fi

# --- 4. CLI overrides the TOML -----------------------------------------------
run_training "hp_seed1.toml" --seed 3
w4=$(weights_of "_seed_3")
if [ -n "${w4:-}" ]; then
    report PASS "4. --seed 3 overrides TOML seed = 1" "$(basename "$w4")"
else
    report FAIL "4. CLI override" "no _seed_3 file; got: $(ls "$SCRATCH/models"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
fi

# --- 5. seed recorded INSIDE the weights JSON --------------------------------
json_seed=$(python3 -c "
import json,sys
try:
    d=json.load(open('${w1b:-/nonexistent}'))
    print(d.get('seed','ABSENT'))
except Exception as e:
    print('ERROR')
" 2>/dev/null)
if [ "$json_seed" = "1" ]; then
    report PASS "5. weights JSON records seed = 1"
else
    report FAIL "5. weights JSON seed" "got '$json_seed', expected 1"
fi

# --- 6. seed recorded as a column in the results CSV -------------------------
csv_seed=$(python3 -c "
import csv
try:
    row=next(csv.DictReader(open('${r1b:-/nonexistent}')))
    print(row.get('seed','ABSENT'))
except Exception:
    print('ERROR')
" 2>/dev/null)
if [ "$csv_seed" = "1" ]; then
    report PASS "6. results CSV has a seed column = 1"
else
    report FAIL "6. results CSV seed column" "got '$csv_seed', expected 1"
fi

# --- 7. unseeded outputs carry NO seed key/column (schema unchanged) ---------
noseed_results=$(ls "$SCRATCH/results"/*.csv 2>/dev/null | grep -v "_seed_" | head -1)
json_absent=$(python3 -c "
import json; print('seed' in json.load(open('${w3b:-/nonexistent}')))" 2>/dev/null)
csv_absent=$(python3 -c "
import csv; print('seed' in next(csv.DictReader(open('${noseed_results:-/nonexistent}'))))" 2>/dev/null)
if [ "$json_absent" = "False" ] && [ "$csv_absent" = "False" ]; then
    report PASS "7. unseeded run: no seed key in JSON, no seed column in CSV"
else
    report FAIL "7. unseeded schema unchanged" "json has seed: $json_absent, csv has seed: $csv_absent"
fi

echo "======================================================================"
echo "  $pass_count passed, $fail_count failed"
exit $((fail_count > 0))

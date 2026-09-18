#!/usr/bin/env bash
# ============================================================================
# sweep_transfer.sh — train on the hardest dataset, one model per sample index,
# then score each model down its own rate ladder plus a cross-index probe.
#
# Run from `expts/`:
#     bash misc/sweep_transfer.sh --pilot      one arm, one device, one test set
#     bash misc/sweep_transfer.sh --dry-run    print every command, run nothing
#     bash misc/sweep_transfer.sh --train-only just the training half
#     bash misc/sweep_transfer.sh --tests-only reuse existing weights
#     bash misc/sweep_transfer.sh              all of it
#
#     12 training runs (3 sample indices x 4 arms), CPU, parallelisable
#     64 test runs     (16 model/test-set pairs x 4 arms), GPU, sequential
#
# Worker count and heap hint are derived from the machine (performance cores
# and physical RAM) unless pinned:
#     TRAIN_JOBS=4 HEAP_HINT=6G bash misc/sweep_transfer.sh
#
# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------
# MATCHED (14 pairs). Each model is scored only on test sets carrying its own
# sample index, walking down from the training rate p = 0.0015 to p = 0.0005.
# Three indices give n = 3 replicates at every rate rung that exists for all
# three, which is what the CER-vs-no-CER verdict needs.
#
# PROBE (2 pairs). The s_1 model is additionally scored on the s_2 and s_3 test
# sets AT THE TRAINING RATE, so it differs from its own matched in-distribution
# run in exactly one respect: which qubits are hot.
#
# The probe exists because the sample index looks like it seeds the per-CNOT
# draw rather than labelling an independent repeat. Fitting the 72-qubit rate
# vectors against correlated_weights_p_0.0015_sig_0.0015_s_1:
#
#     p=0.0005 sig=0.0005 s_1   a = 2.876   R^2 = 0.9935   (a ~ the p ratio, 3)
#     p=0.0007 sig=0.001  s_1   a = 1.653   R^2 = 0.9770
#     p=0.0005 sig=0.001  s_1   a = 1.775   R^2 = 0.9399
#     p=0.0005 sig=0.001  s_2   a = 0.353   R^2 = 0.0445
#
# i.e. within one index the rate vector is a rescaling of a single fixed
# pattern, and across indices it is unrelated. If that is real, the probe rows
# degrade against the matched ones and every future experiment needs one model
# per index. If the probe rows hold up, the pattern does not matter and one
# training run would have been enough. Either answer is worth the 8 runs.
#
# `sigma = 0.0` files are the exception (R^2 = 0.23 even within s_1): flattening
# the spread destroys the pattern, so read those two rows loosely.
#
# ---------------------------------------------------------------------------
# Why nothing collides
# ---------------------------------------------------------------------------
# The weights filename is keyed on the TRAINING source only:
#     models/neuralbp_weights_nlayers_<L>_epochs_<E>_
#            trained_using_<train><cer_tag><run_tag><seed_tag>.json
# while the results filename carries BOTH sources:
#     results/simulation_results_<test>_nlayers_<L>_epochs_<E>_
#             trained_using_<train><cer_tag><run_tag><seed_tag>.csv
#
# So a test step holds `--train` at the file that produced the weights (which
# `retrain = false` then re-loads) while `--test` and `--cer_data` move together
# to the set being scored. The decoder always gets the priors belonging to the
# data in front of it; only the weights come from elsewhere.
#
# ---------------------------------------------------------------------------
# Watch the first two epochs
# ---------------------------------------------------------------------------
# On the old dataset ~87% of gradient updates had every scored layer at exactly
# zero loss, so `learning_rate = 0.01` was tuned under starvation. This dataset
# carries real signal in roughly 50-70% of batches — a 4-5x larger effective
# gradient — and the enriched arms were ALREADY diverging in epochs 4-5 before
# that change. Run --pilot first and read `base_loss` out of
#     data/<codename>/logs/debugging_<train>_hp<tag>_seed_<n>_individual_losses.csv
# If it climbs across epochs rather than falling, lower LEARNING_RATE.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ------------------------------------------------------------------ config ---
CODENAME="72q_BB_cycles_1_soft_constraints"
WORKDIR="./../data"
DATA_DIR="$REPO_ROOT/data/$CODENAME"

# Everything is trained at this rate; only the sample index varies.
TRAIN_RATE_KEY="p_0.0015_sig_0.0015"
TRAIN_DEVICES=("s_1" "s_2" "s_3")

N_LAYERS=90
N_EPOCHS=5
LEARNING_RATE="${LEARNING_RATE:-0.01}"
SEED="${SEED:-1}"
GPU_MEMORY="${GPU_MEMORY:-}"

# TRAIN_JOBS and HEAP_HINT are both derived below from the machine unless the
# caller pins them. See the "local resources" block.
TRAIN_JOBS="${TRAIN_JOBS:-}"
HEAP_HINT="${HEAP_HINT:-}"

# The (model, test set) pairs, in run order. Each test key doubles as the
# correlated_weights suffix: test_<X>.txt pairs with
# correlated_weights_<X>.txt, one to one.
#
# Ordered so the argument settles early: the three in-distribution anchors
# first, then the two probe rows that are directly comparable to them, then the
# rate ladder grouped by rung so n = 3 arrives together.
PAIR_DEVICE=()
PAIR_TEST_KEY=()
PAIR_KIND=()
add_pair() {   # <train-device> <test-key> <matched|probe>
    PAIR_DEVICE+=("$1")
    PAIR_TEST_KEY+=("$2")
    PAIR_KIND+=("$3")
}

# 1. in-distribution anchors, one per index
add_pair "s_1" "p_0.0015_sig_0.0015_s_1" "matched"
add_pair "s_2" "p_0.0015_sig_0.0015_s_2" "matched"
add_pair "s_3" "p_0.0015_sig_0.0015_s_3" "matched"
# 2. cross-index probe, same rate, s_1 weights
add_pair "s_1" "p_0.0015_sig_0.0015_s_2" "probe"
add_pair "s_1" "p_0.0015_sig_0.0015_s_3" "probe"
# 3. rate ladder, device-matched, grouped by rung
add_pair "s_1" "p_0.0007_sig_0.001_s_1"  "matched"
add_pair "s_2" "p_0.0007_sig_0.001_s_2"  "matched"
add_pair "s_3" "p_0.0007_sig_0.001_s_3"  "matched"
add_pair "s_1" "p_0.0005_sig_0.001_s_1"  "matched"
add_pair "s_2" "p_0.0005_sig_0.001_s_2"  "matched"
add_pair "s_3" "p_0.0005_sig_0.001_s_3"  "matched"
add_pair "s_1" "p_0.0005_sig_0.0005_s_1" "matched"
add_pair "s_2" "p_0.0005_sig_0.0005_s_2" "matched"
add_pair "s_3" "p_0.0005_sig_0.0005_s_3" "matched"
# 4. the two zero-spread sets, which exist only at s_1
add_pair "s_1" "p_0.0015_sig_0.0_s_1"    "matched"
add_pair "s_1" "p_0.0005_sig_0.0_s_1"    "matched"

# Arms, as parallel indexed arrays. Associative arrays are avoided on purpose:
# macOS still ships bash 3.2, which does not have them.
#
# `run_tag` enters both the weights and the results filename. The two tanh arms
# may share one because `cer_tag` (`_no_cer` vs empty) already separates them;
# the enriched arms MUST carry their own or they would load the tanh weights.
ARM_NAMES=(        "nocer"   "cer"     "cer_enr_fixed"      "cer_enr_learned"   )
ARM_USE_CER=(      "false"   "true"    "true"               "true"              )
ARM_RUN_TAGS=(     "_xfer"   "_xfer"   "_xfer_cnenr0p42F"   "_xfer_cnenr0p42L"  )
ARM_CHECK_NODES=(  "tanh"    "tanh"    "enriched"           "enriched"          )
ARM_ALPHAS=(       "1.0"     "1.0"     "0.42"               "0.42"              )
ARM_ALPHA_LEARN=(  "false"   "false"   "false"              "true"              )

# -------------------------------------------------------------- arguments ---
DRY_RUN=0
DO_TRAIN=1
DO_TESTS=1
PILOT=0
for argument in "$@"; do
    case "$argument" in
        --dry-run|-n)  DRY_RUN=1 ;;
        --train-only)  DO_TESTS=0 ;;
        --tests-only)  DO_TRAIN=0 ;;
        --pilot)       PILOT=1 ;;
        --help|-h)     awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $argument" >&2
            echo "usage: bash misc/sweep_transfer.sh [--pilot] [--dry-run] [--train-only] [--tests-only]" >&2
            exit 2
            ;;
    esac
done

# The pilot is the cheapest thing that can say the run is healthy: the arm most
# likely to diverge, on one index, scored on the set it was trained for.
if [ "$PILOT" -eq 1 ]; then
    ARM_NAMES=(       "cer_enr_fixed"    )
    ARM_USE_CER=(     "true"             )
    ARM_RUN_TAGS=(    "_xfer_cnenr0p42F" )
    ARM_CHECK_NODES=( "enriched"         )
    ARM_ALPHAS=(      "0.42"             )
    ARM_ALPHA_LEARN=( "false"            )
    TRAIN_DEVICES=(   "s_1"              )
    PAIR_DEVICE=(     "s_1"                      )
    PAIR_TEST_KEY=(   "p_0.0015_sig_0.0015_s_1"  )
    PAIR_KIND=(       "matched"                  )
    echo "  [--pilot] one arm (cer_enr_fixed), one index (s_1), in-distribution only."
fi

N_ARMS=${#ARM_NAMES[@]}
N_DEVICES=${#TRAIN_DEVICES[@]}
N_PAIRS=${#PAIR_DEVICE[@]}
N_TRAIN_RUNS=$(( N_DEVICES * N_ARMS ))
N_TEST_RUNS=$(( N_PAIRS * N_ARMS ))

# -------------------------------------------------------- local resources ---
# Two facts about this codebase decide how many trainings fit on one machine,
# and neither of them is the core count:
#
#   1. src/CorrelatedBPDecoderWithCER.jl:21 runs
#          BLAS.set_num_threads(Threads.nthreads())
#      at package load. So every worker grabs as many BLAS threads as it has
#      Julia threads. If JULIA_NUM_THREADS is `auto` in your shell profile,
#      six workers on eight cores ask for 48 threads and spend their time
#      fighting each other. The training phase below pins it to 1.
#
#   2. src/train.jl:632 loads the training set as
#          convert.(Bool, readdlm(training_errors_file, Int))
#      readdlm builds a 72 x 1_000_000 Matrix{Int64} first: 576 MB, before the
#      Bool copy and before the syndrome product on line 634 allocates two more
#      36 x 1_000_000 intermediates. Peak is roughly 1 GB per worker during
#      load, settling to a couple of hundred MB. That transient is what decides
#      the job count, and it is why --heap-size-hint matters: Julia's GC sizes
#      its heap against TOTAL physical memory, so N unhinted workers each drift
#      toward their own high-water mark and the machine starts swapping.
detect_total_memory_gb() {
    local memory_bytes=""
    if command -v sysctl > /dev/null 2>&1; then
        memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    fi
    if [ -z "$memory_bytes" ] && [ -r /proc/meminfo ]; then
        local memory_kb
        memory_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
        memory_bytes=$(( memory_kb * 1024 ))
    fi
    if [ -z "$memory_bytes" ]; then
        echo 0
    else
        echo $(( memory_bytes / 1024 / 1024 / 1024 ))
    fi
}

detect_fast_cores() {
    # Apple Silicon splits into performance (perflevel0) and efficiency cores.
    # A compute-bound Julia worker parked on an efficiency core runs several
    # times slower, so the performance count is the number that matters.
    local fast_cores=""
    if command -v sysctl > /dev/null 2>&1; then
        fast_cores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
        if [ -z "$fast_cores" ]; then
            fast_cores="$(sysctl -n hw.physicalcpu 2>/dev/null || true)"
        fi
    fi
    if [ -z "$fast_cores" ] && command -v nproc > /dev/null 2>&1; then
        fast_cores="$(nproc)"
    fi
    if [ -z "$fast_cores" ]; then
        echo 2
    else
        echo "$fast_cores"
    fi
}

TOTAL_MEMORY_GB="$(detect_total_memory_gb)"
FAST_CORES="$(detect_fast_cores)"

# One worker per performance core. More than that and the extra workers land on
# efficiency cores, where they hold a slot without finishing any sooner.
if [ -z "$TRAIN_JOBS" ]; then
    TRAIN_JOBS="$FAST_CORES"
    if [ "$TRAIN_JOBS" -lt 1 ]; then
        TRAIN_JOBS=1
    fi
fi
if [ "$TRAIN_JOBS" -gt "$N_TRAIN_RUNS" ]; then
    TRAIN_JOBS="$N_TRAIN_RUNS"
fi

# Leave a quarter of RAM to macOS, the page cache and whatever else is open,
# then split the rest evenly. Floor of 2G: below that the GC thrashes.
if [ -z "$HEAP_HINT" ]; then
    heap_gb=2
    if [ "$TOTAL_MEMORY_GB" -gt 0 ]; then
        heap_gb=$(( TOTAL_MEMORY_GB * 3 / 4 / TRAIN_JOBS ))
        if [ "$heap_gb" -lt 2 ]; then
            heap_gb=2
        fi
    fi
    HEAP_HINT="${heap_gb}G"
fi

# ------------------------------------------------------------- preflight ----
# Every input is checked before anything runs, so a typo fails in a second
# rather than eight hours in.
echo
echo "  Transfer sweep"
echo "    data        $DATA_DIR"
echo "    train rate  $TRAIN_RATE_KEY   indices: ${TRAIN_DEVICES[*]}"
echo "    arms        $N_ARMS"
echo "    training    $N_TRAIN_RUNS run(s)   ($N_DEVICES index x $N_ARMS arm)"
echo "    testing     $N_TEST_RUNS run(s)   ($N_PAIRS pair x $N_ARMS arm)"
echo
echo "    machine     ${TOTAL_MEMORY_GB} GB RAM, ${FAST_CORES} performance core(s)"
echo "    workers     $TRAIN_JOBS at a time, --heap-size-hint=$HEAP_HINT each"
echo "                (worst case $(( TRAIN_JOBS * ${HEAP_HINT%G} )) GB of heap hint against ${TOTAL_MEMORY_GB} GB)"
echo "                JULIA_NUM_THREADS pinned to 1 for training; see the note above"

# The hint is a ceiling the GC aims at, not an allocation, so exceeding RAM is
# not fatal — but it means the workers are permitted, collectively, to outgrow
# the machine before the GC gets serious, and that ends in swap.
TOTAL_HEAP_HINT_GB=$(( TRAIN_JOBS * ${HEAP_HINT%G} ))
if [ "$TOTAL_MEMORY_GB" -gt 0 ] && [ "$TOTAL_HEAP_HINT_GB" -gt "$TOTAL_MEMORY_GB" ]; then
    echo
    echo "  WARNING  $TRAIN_JOBS workers x $HEAP_HINT = ${TOTAL_HEAP_HINT_GB} GB of heap hint on a"
    echo "           ${TOTAL_MEMORY_GB} GB machine. Lower TRAIN_JOBS, or pin a smaller HEAP_HINT."
fi
echo

MISSING_FILES=0
check_file() {   # <path> <description>
    if [ ! -f "$1" ]; then
        echo "  MISSING  $2: $1" >&2
        MISSING_FILES=$(( MISSING_FILES + 1 ))
    fi
}
check_file "$DATA_DIR/code/HZ.txt" "parity check matrix"
for train_device in "${TRAIN_DEVICES[@]}"; do
    check_file "$DATA_DIR/training_data/train_${TRAIN_RATE_KEY}_${train_device}.txt" "training errors"
    check_file "$DATA_DIR/correlated_weights/correlated_weights_${TRAIN_RATE_KEY}_${train_device}.txt" "training CER data"
done
pair_index=0
while [ "$pair_index" -lt "$N_PAIRS" ]; do
    test_key="${PAIR_TEST_KEY[$pair_index]}"
    check_file "$DATA_DIR/testing_data/test_${test_key}.txt" "test errors"
    check_file "$DATA_DIR/correlated_weights/correlated_weights_${test_key}.txt" "test CER data"
    pair_index=$(( pair_index + 1 ))
done
if [ "$MISSING_FILES" -gt 0 ]; then
    echo >&2
    echo "  $MISSING_FILES file(s) missing. Nothing was run." >&2
    exit 1
fi
echo "  Preflight OK — all inputs present."

# ------------------------------------------------ hyperparameter TOMLs ------
# One per arm; the three indices are separated by `trained_using_<source>` in
# the filenames, so they need no TOML of their own. Written fresh on every
# invocation so the file on disk always matches the config block above, and
# carrying only the keys the current loss reads — the L2/L3/sparsity keys were
# removed on 2026-09-16 and are ignored even when present.
write_arm_toml() {   # <arm-index>
    local arm_index="$1"
    local arm_name="${ARM_NAMES[$arm_index]}"
    local toml_path="$DATA_DIR/models/hyperparams_xfer_${arm_name}.toml"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would write  models/hyperparams_xfer_${arm_name}.toml"
        return 0
    fi

    cat > "$toml_path" <<TOML
# Generated by misc/sweep_transfer.sh — edits here are overwritten on the next run.
# Arm: ${arm_name}

retrain = false

learning_rate = ${LEARNING_RATE}
max_grad_norm = 2.0
weight_decay = 0.0001
adam_eps = 0.0001
nanskip = 5

batch_size = 20
n_epochs = ${N_EPOCHS}
online_training = true
n_gradient_updates_per_epoch = 500
initial_conditions_scale = 0.1

# Layers dropped from scoring, and the softmin temperature over the rest.
# These are the only two knobs the loss still has.
warmup_layers = 10
loss_layer_temperature = "2e-1,1e0,0.7,down"

seed = ${SEED}

# Inference temperature on the single-qubit priors: the median rate is mapped
# to this value, couplings untouched. It is what makes a p = 0.0015 model
# comparable on p = 0.0005 data — the absolute rate is normalised away at
# input, so only the PATTERN and the relative spread survive. That is also why
# the cross-index probe isolates the pattern and nothing else.
single_qubit_rescale = 0.1

use_CER = ${ARM_USE_CER[$arm_index]}
run_tag = "${ARM_RUN_TAGS[$arm_index]}"

check_node = "${ARM_CHECK_NODES[$arm_index]}"
coupling_scale_init = ${ARM_ALPHAS[$arm_index]}
coupling_scale_learnable = ${ARM_ALPHA_LEARN[$arm_index]}
TOML

    if [ -n "$GPU_MEMORY" ]; then
        echo "gpu_memory = \"${GPU_MEMORY}\"" >> "$toml_path"
    fi
}

echo "  Writing $N_ARMS hyperparameter TOML(s) into models/ ..."
arm_index=0
while [ "$arm_index" -lt "$N_ARMS" ]; do
    write_arm_toml "$arm_index"
    arm_index=$(( arm_index + 1 ))
done

# ------------------------------------------------------------- run helper ---
# Failures are recorded in a FILE, not a shell array: the training half runs its
# jobs in the background, and an array appended to inside a subshell is lost
# when that subshell exits.
FAILURE_LOG="$(mktemp)"
trap 'rm -f "$FAILURE_LOG"' EXIT

run_julia() {   # <label> <julia-args...>
    local label="$1"
    shift
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [$label]"
        echo "      julia $*"
        return 0
    fi
    echo
    echo "  ---- $label ----"
    # A failed step must not abort the sweep: 63 good runs are worth more than
    # stopping on the 12th. Every failure is collected and reported at the end.
    if julia "$@"; then
        return 0
    else
        echo "  FAILED: $label" >&2
        echo "$label" >> "$FAILURE_LOG"
        return 0
    fi
}

# ---------------------------------------------------------------- training ---
# Enzyme differentiates on the CPU only, so USE_GPU stays off here whatever the
# testing half does.
train_one() {   # <train-device> <arm-index>
    local train_device="$1"
    local arm_index="$2"
    local arm_name="${ARM_NAMES[$arm_index]}"
    run_julia "train ${arm_name} on ${TRAIN_RATE_KEY}_${train_device}" \
        --project="./../" --heap-size-hint="$HEAP_HINT" \
        neural_bp_experiments.jl \
        --workdir "$WORKDIR" \
        --codename "$CODENAME" \
        --n_hidden_layers "$N_LAYERS" \
        --hyperparams "hyperparams_xfer_${arm_name}.toml" \
        --cer_data "correlated_weights_${TRAIN_RATE_KEY}_${train_device}.txt" \
        --train "train_${TRAIN_RATE_KEY}_${train_device}.txt" \
        --quiet true \
        --diagnose true
}

if [ "$DO_TRAIN" -eq 1 ]; then
    echo
    echo "=============================================================="
    echo "  TRAINING — $N_TRAIN_RUNS run(s), CPU, $TRAIN_JOBS at a time"
    echo "=============================================================="
    export USE_GPU="0"
    # See note 1 in the local-resources block: BLAS threads follow Julia
    # threads, so an unpinned worker would grab a whole machine's worth.
    export JULIA_NUM_THREADS="1"
    export OPENBLAS_NUM_THREADS="1"

    # Flatten (index, arm) into one list.
    JOB_DEVICES=()
    JOB_ARMS=()
    for train_device in "${TRAIN_DEVICES[@]}"; do
        arm_index=0
        while [ "$arm_index" -lt "$N_ARMS" ]; do
            JOB_DEVICES+=("$train_device")
            JOB_ARMS+=("$arm_index")
            arm_index=$(( arm_index + 1 ))
        done
    done

    # A polling pool, not chunked `wait`: chunking would idle every finished
    # worker until the slowest in its chunk returned, and these runs do not
    # take equal time — the enriched arms carry a 64-state sum per check that
    # the tanh arms do not. `wait -n` would be the clean way to do this and
    # bash 3.2, which macOS still ships, does not have it.
    wait_for_free_slot() {   # <max-concurrent>
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$1" ]; do
            sleep 2
        done
    }

    job_index=0
    while [ "$job_index" -lt "$N_TRAIN_RUNS" ]; do
        echo "  [train $(( job_index + 1 ))/$N_TRAIN_RUNS]"
        if [ "$TRAIN_JOBS" -le 1 ]; then
            train_one "${JOB_DEVICES[$job_index]}" "${JOB_ARMS[$job_index]}"
        else
            wait_for_free_slot "$TRAIN_JOBS"
            train_one "${JOB_DEVICES[$job_index]}" "${JOB_ARMS[$job_index]}" &
        fi
        job_index=$(( job_index + 1 ))
    done
    wait
    echo
    echo "  Training done."
else
    echo
    echo "  [--tests-only] skipping training; the weights on disk will be reused."
fi

# ----------------------------------------------------------------- testing ---
# One forward pass per (pair, arm). `--train` names the file that produced the
# weights; `--test` and `--cer_data` move together so the decoder is always
# handed the priors belonging to the data it is scoring.
test_one() {   # <train-device> <test-key> <kind> <arm-index>
    local train_device="$1"
    local test_key="$2"
    local kind="$3"
    local arm_index="$4"
    local arm_name="${ARM_NAMES[$arm_index]}"
    run_julia "test ${arm_name}: ${train_device} model -> ${test_key} [${kind}]" \
        --project="./../" --heap-size-hint="$HEAP_HINT" \
        neural_bp_experiments.jl \
        --workdir "$WORKDIR" \
        --codename "$CODENAME" \
        --n_hidden_layers "$N_LAYERS" \
        --hyperparams "hyperparams_xfer_${arm_name}.toml" \
        --cer_data "correlated_weights_${test_key}.txt" \
        --train "train_${TRAIN_RATE_KEY}_${train_device}.txt" \
        --test "test_${test_key}.txt" \
        --quiet true \
        --diagnose true
}

if [ "$DO_TESTS" -eq 1 ]; then
    echo
    echo "=============================================================="
    echo "  TESTING — $N_TEST_RUNS run(s), sequential, GPU"
    echo "=============================================================="
    export USE_GPU="1"
    # Sequential from here, so the single worker may have the whole machine
    # back. Undo the pin the training phase set.
    export JULIA_NUM_THREADS="$FAST_CORES"
    export OPENBLAS_NUM_THREADS="$FAST_CORES"

    step_number=0
    pair_index=0
    while [ "$pair_index" -lt "$N_PAIRS" ]; do
        arm_index=0
        while [ "$arm_index" -lt "$N_ARMS" ]; do
            step_number=$(( step_number + 1 ))
            echo
            echo "  [$step_number/$N_TEST_RUNS]"
            test_one "${PAIR_DEVICE[$pair_index]}" "${PAIR_TEST_KEY[$pair_index]}" \
                     "${PAIR_KIND[$pair_index]}" "$arm_index"
            arm_index=$(( arm_index + 1 ))
        done
        pair_index=$(( pair_index + 1 ))
    done
else
    echo
    echo "  [--train-only] skipping the testing half."
fi

# ----------------------------------------------------------------- summary ---
echo
echo "=============================================================="
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  --dry-run: nothing was executed."
    exit 0
fi
FAILURE_COUNT=$(wc -l < "$FAILURE_LOG" | tr -d ' ')
if [ "$FAILURE_COUNT" -eq 0 ]; then
    echo "  All steps completed."
else
    echo "  $FAILURE_COUNT step(s) FAILED:"
    sed 's/^/    - /' "$FAILURE_LOG"
fi
echo
echo "  Results   $DATA_DIR/results/"
echo "            simulation_results_test_<set>_nlayers_${N_LAYERS}_epochs_${N_EPOCHS}_"
echo "            trained_using_train_${TRAIN_RATE_KEY}_<index><cer_tag><run_tag>_seed_${SEED}.csv"
echo
echo "  The probe rows are the two results whose test index does NOT match the"
echo "  trained_using index. Compare them against the s_2 and s_3 matched rows"
echo "  at the same rate: same data, same priors, different weights."
echo
echo "  Training  $DATA_DIR/logs/"
echo "            debugging_*_individual_losses.csv  <- check base_loss per epoch"
echo "=============================================================="

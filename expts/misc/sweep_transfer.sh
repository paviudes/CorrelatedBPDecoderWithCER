#!/usr/bin/env bash
# sweep_transfer.sh — train on the hardest dataset, test down the rate ladder.
#
# Same shape as sweep_hyperparams.sh: writes a settings TOML, opens it in
# $EDITOR, then generates
#     <codename>/cluster/xf_sweep_train_<ts>.txt   one julia command per model
#     <codename>/cluster/xf_sweep_test_<ts>.txt    one per (model, test set)
#     <codename>/cluster/xf_sweep_train_<ts>.sh    CPU job, GNU parallel
#     <codename>/cluster/xf_sweep_test_<ts>.sh     GPU job
#     <codename>/models/hyperparams_xf_*.toml      one per model
#
# Run from `expts/`, exactly like sweep_hyperparams.sh:
#   bash misc/sweep_transfer.sh              edit settings, then generate
#   bash misc/sweep_transfer.sh --no-edit    use the defaults as written
#   bash misc/sweep_transfer.sh --local      also emit a 1-point local smoke command
#
# It prints the two sbatch lines at the end. Nothing is submitted for you.
#
# WHAT IS DIFFERENT FROM sweep_hyperparams.sh
# -------------------------------------------
# There, training and testing are 1:1 — a point trains on a dataset and is
# tested on the matching test set. Here they are MANY-TO-MANY: each trained
# model is scored against several test sets, so the two command files have
# different lengths (60 and 320 at the defaults below) and the test job's array
# is sized against the larger one.
#
# Every test command holds `--train` at the file that produced the weights,
# which is what `retrain = false` re-loads, while `--test` and `--cer_data`
# move together to the set being scored. So the decoder is always handed the
# priors belonging to the data in front of it and only the WEIGHTS come from
# elsewhere. Nothing collides because the results filename carries both sources:
#     simulation_results_<test>_nlayers_<L>_epochs_<E>_
#             trained_using_<train><cer_tag><run_tag><seed_tag>.csv
#
# TWO GROUPS OF TEST SETS
# -----------------------
#   MATCHED   model and test set share an _s_<n> suffix. Walks the rate ladder
#             from the training rate down, with n = len(seeds) replicates at
#             every rung that exists for all three indices.
#   PROBE     the probe_source model scored on OTHER indices at the training
#             rate, so it differs from its own matched run in exactly one
#             respect: which qubits are hot.
#
# The probe exists because the sample index looks like it seeds the per-CNOT
# draw rather than labelling an independent repeat. Fitting the 72-qubit rate
# vectors against correlated_weights_p_0.0015_sig_0.0015_s_1:
#     p=0.0005 sig=0.0005 s_1   a = 2.876   R^2 = 0.9935   (a ~ the p ratio, 3)
#     p=0.0007 sig=0.001  s_1   a = 1.653   R^2 = 0.9770
#     p=0.0005 sig=0.001  s_2   a = 0.353   R^2 = 0.0445
# i.e. within one index the rate vector is a rescaling of a single fixed
# pattern, and across indices it is unrelated. If that carries into the decoder,
# probe rows degrade against matched ones and every future experiment needs one
# model per index. If they hold up, one training run would have been enough.
#
# WHY TRAIN AT p = 0.0015 sig = 0.0015
# ------------------------------------
# The old training set (p = 0.0005 sig = 0.001) is 79.9% identity errors, and
# BP solves nearly everything below weight ~5 unaided: 87% of gradient updates
# had every scored layer at exactly zero base loss, making the total the softmin
# floor -T*log(80) = -4.382, a constant with no gradient. The new set is 61.0%
# identity with mean weight 0.91, which puts real signal in roughly 50-70% of
# batches. Nothing is filtered out — the weight spectrum still runs from 0 up,
# so the decoder stays calibrated on identity and low-weight errors.
#
# That is a 4-5x larger effective gradient than `learning_rate` was ever tuned
# against, and the enriched arms were ALREADY diverging in epochs 4-5 on the old
# data (base loss 0.011 -> 4.21, 469/2500 batches NaN-skipped). Run --local
# first, or at least read base_loss per epoch out of
#     <codename>/logs/debugging_<train>_xf<tag>_seed_<n>_individual_losses.csv
# before spending a GPU allocation on 320 tests.
set -eu

NO_EDIT=0
LOCAL=0
SMOKE_N=5000
for arg in "$@"; do
    case "$arg" in
        --no-edit) NO_EDIT=1 ;;
        --local)   LOCAL=1 ;;
        --local=*) LOCAL=1; SMOKE_N="${arg#*=}" ;;
        --help|-h) awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# This file lives in expts/misc/ but every path it emits is relative to expts/:
# `--project="./../"` and `workdir = "./../data"` both assume that, and so does
# sbatch's $SLURM_SUBMIT_DIR. So EXPTS_DIR, not SCRIPT_DIR, is what the
# generated commands are anchored to, and the settings file goes in the same
# expts/scripts/ that sweep_hyperparams.sh uses.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPTS_DIR="$EXPTS_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"

# Run from anywhere else and `./../data` points at nothing, which would surface
# much later as a missing base-hyperparameters file. Say so now instead.
if [ "$(pwd -P)" != "$EXPTS_DIR" ]; then
    echo "[xf_sweep] run this from $EXPTS_DIR:" >&2
    echo "    cd $EXPTS_DIR && bash misc/$(basename "$0") $*" >&2
    echo "  Every generated path is relative to expts/ (--project=\"./../\"," >&2
    echo "  workdir=\"./../data\"), so a different working directory silently" >&2
    echo "  misresolves all of them." >&2
    exit 2
fi
TS="$(date +%Y-%m-%d_%H-%M-%S)"
SETTINGS_FILE="$SCRIPTS_DIR/xf_sweep_settings_${TS}.toml"

cat > "$SETTINGS_FILE" <<'EOF'
workdir          = "./../data"
codename         = "72q_BB_cycles_1_soft_constraints"

# KEEP EACH ARRAY ON ONE LINE: the reader below is grep | head -1, so a wrapped
# array silently loses everything after the first line.

# TRAINING datasets. One model per entry x seed x arm. The _s_<n> suffix is what
# pairs a model with its matched test sets.
train_datasets   = ["p_0.0015_sig_0.0015_s_1", "p_0.0015_sig_0.0015_s_2", "p_0.0015_sig_0.0015_s_3"]

# TEST sets, run in this order. Each is scored by the trained model sharing its
# _s_<n> suffix; a set whose index has no trained model is skipped with a
# warning rather than silently dropped. Ordered so the in-distribution anchors
# come first and each rate rung arrives with all its replicates together.
test_keys        = ["p_0.0015_sig_0.0015_s_1", "p_0.0015_sig_0.0015_s_2", "p_0.0015_sig_0.0015_s_3", "p_0.0007_sig_0.001_s_1", "p_0.0007_sig_0.001_s_2", "p_0.0007_sig_0.001_s_3", "p_0.0005_sig_0.001_s_1", "p_0.0005_sig_0.001_s_2", "p_0.0005_sig_0.001_s_3", "p_0.0005_sig_0.0005_s_1", "p_0.0005_sig_0.0005_s_2", "p_0.0005_sig_0.0005_s_3", "p_0.0015_sig_0.0_s_1", "p_0.0005_sig_0.0_s_1"]

# CROSS-INDEX PROBE. `probe_source` must be one of train_datasets; its model is
# additionally scored on each `probe_test_keys` entry. Keep the rate equal to
# the training rate so the ONLY thing that changes is the per-qubit pattern.
# Set probe_test_keys = [] to skip the probe entirely.
probe_source     = "p_0.0015_sig_0.0015_s_1"
probe_test_keys  = ["p_0.0015_sig_0.0015_s_2", "p_0.0015_sig_0.0015_s_3"]

base_hyperparams = "hyperparams_epochs_5_corrs.toml"
n_hidden_layers  = 90
# Network seeds, on every cell. Seed variance has been the binding error bar
# throughout (sd 213 on a mean of 547 failures, i.e. +-39%, where Poisson
# counting noise on 547 is only +-4.3%), so nothing is read off fewer than five.
# Each seed multiplies BOTH the training and the testing point count.
seeds            = [1, 2, 3, 4, 5]

# --- arms -------------------------------------------------------------------
#   no-CER    flat p = 0.1 priors, standard check node        (the baseline)
#   CER       CER single-qubit priors, check node per `check_node_arms`
# The no-CER arm ignores correlated_weights entirely, so its test rows differ
# only in the test set; do not read variation across them as a priors effect.
include_nocer    = true
single_qubit_rescale = 0.1

# "tanh" is the standard check-to-variable rule (untagged). "enriched:<a>:<fixed
# |learn>" puts the CER couplings inside each check factor with a as the scale
# on J. Tag _cnenr<a>F / _cnenr<a>L. alpha = 0.42 = LLR_rescaled / LLR_raw is
# the temperature-consistent value and the classical optimum.
check_node_arms  = ["tanh", "enriched:0.42:fixed", "enriched:0.42:learn"]

# --- cluster ----------------------------------------------------------------
# ACTIVE: NARVAL. See sweep_hyperparams.sh for the Nibi profile values.
cluster_name     = "narval"
account_cpu      = "def-jemerson"
account_gpu      = "def-jemerson_gpu"
email            = "pavithran.sridhar@gmail.com"
julia_module     = "julia/1.12.5"
cuda_module      = "cuda"
heap_size_hint   = "4G"

# 60 training points / 5 tasks = 12 per task, one 54-core wave.
train_array_tasks = 5
train_cpus       = 54
train_mem_per_cpu = "6G"
cpu_node_memory_mb = 510000
train_wall_time  = "4:00:00"

# 320 test points is 5x the usual sweep, so the array is wider to match:
# 320 / 16 = 20 per task x ~4 min = ~80 min, inside a 4h wall with room over.
# ONE CARD PER TASK, ONE PROCESS ON IT. Never two on an unpartitioned card:
# the real footprint is ~1.5x nominal GPU_MEMORY and they die stochastically at
# OOM. Scale with test_array_tasks, never with test_jobs.
test_array_tasks = 16
gpu_type         = "a100"
gpu_request_style = "auto"
n_gpus_per_node  = 1
test_jobs        = 1
mem_per_gpu      = "32G"
vram_per_gpu     = ""
test_cpus        = 12
test_wall_time   = "4:00:00"
EOF

echo "[xf_sweep] wrote defaults to: $SETTINGS_FILE"

open_editor() {
    local editor_cmd=""
    if [ -n "${EDITOR:-}" ]; then
        editor_cmd="$EDITOR"
    else
        for cand in nano vim vi; do
            if command -v "$cand" >/dev/null 2>&1; then editor_cmd="$cand"; break; fi
        done
    fi
    if [ -z "$editor_cmd" ]; then
        echo "[xf_sweep] no editor found (set \$EDITOR, or install nano/vim/vi)." >&2
        return 1
    fi
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        echo "[xf_sweep] no interactive terminal — skipping editor." >&2
        return 1
    fi
    "$editor_cmd" "$SETTINGS_FILE"
}

if [ "$NO_EDIT" -eq 0 ]; then
    if ! open_editor; then
        echo "[xf_sweep] edit $SETTINGS_FILE by hand, then re-run with --no-edit."
    fi
fi

# ---------------------------------------------------------------- settings ---
# Strip the inline "# ..." comment BEFORE unquoting, or it lands in filenames.
get()  { grep -E "^[[:space:]]*$1[[:space:]]*=" "$SETTINGS_FILE" | head -1 |
         sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/^"//; s/"$//; s/[[:space:]]*$//'; }
list() { get "$1" | tr -d '[]"' | tr ',' ' '; }

WORKDIR=$(get workdir);              CODENAME=$(get codename)
TRAIN_DATASETS=$(list train_datasets)
TEST_KEYS=$(list test_keys)
PROBE_SOURCE=$(get probe_source);    PROBE_TEST_KEYS=$(list probe_test_keys)
BASE_HP=$(get base_hyperparams);     NLAYERS=$(get n_hidden_layers)
SEEDS=$(list seeds)
INCLUDE_NOCER=$(get include_nocer)
RESCALE=$(get single_qubit_rescale)
CHECK_NODE_ARMS=$(list check_node_arms)
if [ -z "$CHECK_NODE_ARMS" ]; then
    CHECK_NODE_ARMS="tanh"
fi
ACCOUNT_CPU=$(get account_cpu);      ACCOUNT_GPU=$(get account_gpu)
EMAIL=$(get email);                  JULIA_MODULE=$(get julia_module)
CUDA_MODULE=$(get cuda_module);      HEAP=$(get heap_size_hint)
TRAIN_CPUS=$(get train_cpus);        TRAIN_MEM=$(get train_mem_per_cpu)
TRAIN_ARRAY=$(get train_array_tasks); TEST_ARRAY=$(get test_array_tasks)
TRAIN_WALL=$(get train_wall_time)
GPU_TYPE=$(get gpu_type);            N_GPUS=$(get n_gpus_per_node)
TEST_JOBS=$(get test_jobs);          MEM_PER_GPU=$(get mem_per_gpu)
VRAM_PER_GPU=$(get vram_per_gpu)
TEST_CPUS=$(get test_cpus);          TEST_WALL=$(get test_wall_time)
CLUSTER_NAME=$(get cluster_name);    CPU_NODE_MEM_MB=$(get cpu_node_memory_mb)
GPU_REQUEST_STYLE=$(get gpu_request_style)

# VRAM per card, for the prediction batch sizer. This is NOT --mem-per-gpu,
# which is host RAM.
if [ -z "$VRAM_PER_GPU" ]; then
    case "$GPU_TYPE" in
        h100_1g.10gb) VRAM_PER_GPU=10 ;;
        h100_2g.20gb) VRAM_PER_GPU=20 ;;
        h100_3g.40gb) VRAM_PER_GPU=40 ;;
        h100_80gb)    VRAM_PER_GPU=80 ;;
        a100_1g.5gb)  VRAM_PER_GPU=5  ;;
        a100_2g.10gb) VRAM_PER_GPU=10 ;;
        a100_3g.20gb) VRAM_PER_GPU=20 ;;
        a100)         VRAM_PER_GPU=40 ;;
        h100)         VRAM_PER_GPU=80 ;;
        v100*)        VRAM_PER_GPU=32 ;;
        *) echo "unknown gpu_type '$GPU_TYPE': set vram_per_gpu explicitly." >&2; exit 1 ;;
    esac
fi
# PREFLIGHT: --mem-per-cpu is POOLED, so cpus_per_task x mem_per_cpu must fit
# the node. Getting this wrong is a rejected submission after the queue wait.
TRAIN_MEM_MB=$(echo "$TRAIN_MEM" | awk '{u=toupper($0); v=u; gsub(/[^0-9.]/,"",v);
    if (u ~ /G/) printf "%d", v*1024; else printf "%d", v}')
TRAIN_TOTAL_MB=$(( TRAIN_CPUS * TRAIN_MEM_MB ))
if [ "$TRAIN_TOTAL_MB" -gt "$CPU_NODE_MEM_MB" ]; then
    echo "train_cpus x train_mem_per_cpu = ${TRAIN_CPUS} x ${TRAIN_MEM} = ${TRAIN_TOTAL_MB}M" >&2
    echo "  exceeds the ${CLUSTER_NAME} CPU node's ${CPU_NODE_MEM_MB}M. --mem-per-cpu is POOLED." >&2
    echo "  Lower train_cpus to $(( CPU_NODE_MEM_MB / TRAIN_MEM_MB )) or reduce train_mem_per_cpu." >&2
    exit 1
fi

GPU_SBATCH_LINE=""
case "$GPU_REQUEST_STYLE" in
    gpus-per-node) GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
    gpus)          GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
    auto)
        case "$GPU_TYPE" in
            *g.*gb) GPU_SBATCH_LINE="#SBATCH --gpus=${GPU_TYPE}:${N_GPUS}" ;;
            *)      GPU_SBATCH_LINE="#SBATCH --gpus-per-node=${GPU_TYPE}:${N_GPUS}" ;;
        esac ;;
    *) echo "unknown gpu_request_style '$GPU_REQUEST_STYLE': use auto, gpus or gpus-per-node." >&2; exit 1 ;;
esac

JOBS_PER_GPU=$(( TEST_JOBS / N_GPUS ))
if [ "$JOBS_PER_GPU" -lt 1 ]; then
    JOBS_PER_GPU=1
fi
GPU_MEMORY_MB=$(( VRAM_PER_GPU * 1024 * 85 / (100 * JOBS_PER_GPU) ))

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
    echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
    echo "  (resolved from expts/; check codename and base_hyperparams in" >&2
    echo "   $SETTINGS_FILE)" >&2
    exit 1
fi
mkdir -p "$CLUSTER_DIR" "$MODELS_DIR" "$WORKDIR/$CODENAME/results" "$WORKDIR/$CODENAME/logs"

# PREFLIGHT: every dataset must have its input files HERE, before the job tars
# this directory. A training key needs all three; a test key needs the two the
# test step reads. Without this check a missing input is discovered only after
# the queue wait, as N identical "exit 1" lines with no message.
MISSING_INPUTS=""
for key in $TRAIN_DATASETS; do
    for required in "training_data/train_${key}.txt" \
                    "testing_data/test_${key}.txt" \
                    "correlated_weights/correlated_weights_${key}.txt"; do
        if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
            MISSING_INPUTS="${MISSING_INPUTS}\n    $required"
        fi
    done
done
for key in $TEST_KEYS $PROBE_TEST_KEYS; do
    for required in "testing_data/test_${key}.txt" \
                    "correlated_weights/correlated_weights_${key}.txt"; do
        if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
            MISSING_INPUTS="${MISSING_INPUTS}\n    $required"
        fi
    done
done
if [ -n "$MISSING_INPUTS" ]; then
    echo "missing dataset input(s) under $WORKDIR/$CODENAME:" >&2
    printf "%b\n" "$MISSING_INPUTS" | sort -u >&2
    echo "  The sweep stages this directory as it stands, so these must exist" >&2
    echo "  before submitting." >&2
    exit 1
fi

# PROBE SANITY. A probe_source that is not a trained dataset would emit test
# commands pointing at weights nothing ever writes, and every one of them would
# fail after the queue wait.
if [ -n "$PROBE_TEST_KEYS" ]; then
    PROBE_SOURCE_IS_TRAINED=0
    for key in $TRAIN_DATASETS; do
        if [ "$key" = "$PROBE_SOURCE" ]; then
            PROBE_SOURCE_IS_TRAINED=1
        fi
    done
    if [ "$PROBE_SOURCE_IS_TRAINED" -eq 0 ]; then
        echo "probe_source '$PROBE_SOURCE' is not in train_datasets." >&2
        echo "  Its weights would never be written, so every probe test would fail." >&2
        exit 1
    fi
fi

TRAIN_CMDS="$CLUSTER_DIR/xf_sweep_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/xf_sweep_test_${TS}.txt"
PAIR_MANIFEST="$CLUSTER_DIR/xf_sweep_pairs_${TS}.txt"
SLURM_TRAIN="$CLUSTER_DIR/xf_sweep_train_${TS}.sh"
SLURM_TEST="$CLUSTER_DIR/xf_sweep_test_${TS}.sh"
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"; : > "$PAIR_MANIFEST"

tag_of()   { echo "$1" | tr '.' 'p' | tr -d '-'; }
# Sample index of a dataset key: the trailing _s_<n>. This is what pairs a model
# with its matched test sets, so it has to be exact — a key without the suffix
# returns itself, which will never match anything, and that is the intended
# failure mode rather than a silent mispairing.
index_of() { echo "s_${1##*_s_}"; }

# ------------------------------------------------------- arm decomposition ---
# Sets ARM, REQUIRE, CHECK_NODE, COUPLING_INIT, COUPLING_LEARN, CHECK_NODE_TAG,
# RUN_TAG from a use_cer flag and a check-node spec. Shared by the TOML writer
# and both command emitters so the three can never disagree about a tag.
decompose_arm() {   # <use_cer> <check_node_spec>
    local use_cer="$1" spec="$2"
    ARM="cer"; REQUIRE="true"
    if [ "$use_cer" = "false" ]; then
        ARM="nocer"
        REQUIRE="false"
    fi
    CHECK_NODE="${spec%%:*}"
    COUPLING_INIT="1.0"
    COUPLING_LEARN="false"
    CHECK_NODE_TAG=""
    if [ "$CHECK_NODE" = "enriched" ]; then
        if [ "$use_cer" = "false" ]; then
            echo "decompose_arm: an enriched check node needs couplings; refusing it on the no-CER arm." >&2
            exit 1
        fi
        local rest="${spec#*:}"
        COUPLING_INIT="${rest%%:*}"
        local learn_spec="${rest#*:}"
        if [ "$learn_spec" = "learn" ]; then
            COUPLING_LEARN="true"
            CHECK_NODE_TAG="_cnenr$(tag_of "$COUPLING_INIT")L"
        elif [ "$learn_spec" = "fixed" ]; then
            CHECK_NODE_TAG="_cnenr$(tag_of "$COUPLING_INIT")F"
        else
            echo "decompose_arm: check node spec '$spec' must end in :fixed or :learn." >&2
            exit 1
        fi
    elif [ "$CHECK_NODE" != "tanh" ]; then
        echo "decompose_arm: unknown check node '$CHECK_NODE' (tanh or enriched:<alpha>:<fixed|learn>)." >&2
        exit 1
    fi
    # `_xf` rather than `_hp` so these results never land in the old sweep's
    # collector glob, and the old ones never land in this one's.
    RUN_TAG="_xf${ARM}${CHECK_NODE_TAG}"
    # The TOML is keyed on the TRAINING dataset, because that is what determines
    # the weights. Every test of a model reuses its training TOML unchanged.
    HP_NAME="hyperparams_xf_${ARM}${CHECK_NODE_TAG}_$(tag_of "$TRAIN_KEY")_seed${SEED_N}.toml"
}

# ------------------------------------------------------------ emit points ---
emit_training_point() {   # <train_key> <seed> <use_cer> <check_node_spec>
    TRAIN_KEY="$1"; SEED_N="$2"
    decompose_arm "$3" "${4:-tanh}"

    # Start from the base TOML minus every key this generator sets itself, so a
    # stale value in the base can never override a swept one. The removed loss
    # terms' keys are stripped too: ignored by the code now, but a generated
    # file should not carry dead settings.
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$HP_NAME"
    {
        echo ""
        echo "# generated by sweep_transfer.sh $TS"
        echo "# trained on ${TRAIN_KEY}; reused unchanged by every test of this model."
        echo "retrain = true"
        echo "run_tag = \"${RUN_TAG}\""
        echo "use_CER = $3"
        echo "seed = ${SEED_N}"
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = ${REQUIRE}"
        echo "check_node = \"${CHECK_NODE}\""
        echo "coupling_scale_init = ${COUPLING_INIT}"
        echo "coupling_scale_learnable = ${COUPLING_LEARN}"
    } >> "$MODELS_DIR/$HP_NAME"

    echo "julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $HP_NAME --cer_data correlated_weights_${TRAIN_KEY}.txt --quiet true \
--isdebug true --train train_${TRAIN_KEY}.txt" >> "$TRAIN_CMDS"
}

emit_test_point() {   # <train_key> <test_key> <seed> <use_cer> <check_node_spec> <kind>
    TRAIN_KEY="$1"; SEED_N="$3"
    local test_key="$2" kind="$6"
    decompose_arm "$4" "${5:-tanh}"

    # --train names the file that produced the weights. --test and --cer_data
    # move together to the set being scored.
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $HP_NAME --cer_data correlated_weights_${test_key}.txt --quiet true \
--diagnose true --train train_${TRAIN_KEY}.txt --test test_${test_key}.txt" >> "$TEST_CMDS"
    echo "${kind}	${TRAIN_KEY}	${test_key}	${ARM}${CHECK_NODE_TAG}	seed${SEED_N}" >> "$PAIR_MANIFEST"
}

# One model per (training dataset, seed, arm).
for train_key in $TRAIN_DATASETS; do
    for seed in $SEEDS; do
        for cn in $CHECK_NODE_ARMS; do
            emit_training_point "$train_key" "$seed" true "$cn"
        done
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_training_point "$train_key" "$seed" false "tanh"
        fi
    done
done

# MATCHED: every test set scored by the model sharing its sample index.
UNMATCHED_TEST_KEYS=""
for test_key in $TEST_KEYS; do
    test_index=$(index_of "$test_key")
    matched_train_key=""
    for train_key in $TRAIN_DATASETS; do
        if [ "$(index_of "$train_key")" = "$test_index" ]; then
            matched_train_key="$train_key"
        fi
    done
    if [ -z "$matched_train_key" ]; then
        UNMATCHED_TEST_KEYS="${UNMATCHED_TEST_KEYS} ${test_key}"
        continue
    fi
    for seed in $SEEDS; do
        for cn in $CHECK_NODE_ARMS; do
            emit_test_point "$matched_train_key" "$test_key" "$seed" true "$cn" "matched"
        done
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_test_point "$matched_train_key" "$test_key" "$seed" false "tanh" "matched"
        fi
    done
done

# PROBE: the probe_source model on other indices, same rate.
for test_key in $PROBE_TEST_KEYS; do
    for seed in $SEEDS; do
        for cn in $CHECK_NODE_ARMS; do
            emit_test_point "$PROBE_SOURCE" "$test_key" "$seed" true "$cn" "probe"
        done
        if [ "$INCLUDE_NOCER" = "true" ]; then
            emit_test_point "$PROBE_SOURCE" "$test_key" "$seed" false "tanh" "probe"
        fi
    done
done

if [ -n "$UNMATCHED_TEST_KEYS" ]; then
    echo "[xf_sweep] WARNING: no trained model shares an index with:${UNMATCHED_TEST_KEYS}"
    echo "           Those test sets were skipped. Add the matching training dataset,"
    echo "           or drop them from test_keys."
fi

# Two identical commands are not merely wasted cores: both write the SAME
# results file, concurrently, from different array tasks. Deduplicate, keeping
# first-appearance order so the interleaved array split stays balanced.
N_RAW_TRAIN=$(wc -l < "$TRAIN_CMDS")
N_RAW_TEST=$(wc -l < "$TEST_CMDS")
for cmd_file in "$TRAIN_CMDS" "$TEST_CMDS"; do
    awk '!seen[$0]++' "$cmd_file" > "$cmd_file.tmp"
    mv "$cmd_file.tmp" "$cmd_file"
done
N_TRAIN_POINTS=$(wc -l < "$TRAIN_CMDS")
N_TEST_POINTS=$(wc -l < "$TEST_CMDS")
N_DUPLICATES=$(( (N_RAW_TRAIN - N_TRAIN_POINTS) + (N_RAW_TEST - N_TEST_POINTS) ))
if [ "$N_DUPLICATES" -gt 0 ]; then
    echo "[xf_sweep] removed $N_DUPLICATES duplicate point(s) — probe_test_keys overlapping a matched set."
fi
if [ "$N_TEST_POINTS" -eq 0 ]; then
    echo "no test points were emitted; check test_keys against train_datasets." >&2
    exit 1
fi

# SELF-CHECK OF THIS FILE. The two SLURM scripts below are built from UNQUOTED
# heredocs, so bash expands their contents at generation time -- inside comments
# too. A bare dollar-digit is a positional parameter (unbound under `set -u`)
# and a backtick is command substitution. Both have silently broken generation
# before. Anything meant literally in those heredocs must be backslash-escaped.
HEREDOC_HAZARDS=$(awk '
    /<<[[:space:]]*EOF[[:space:]]*$/ { inside_heredoc = 1; next }
    /^EOF$/                          { inside_heredoc = 0; next }
    inside_heredoc && ($0 ~ /(^|[^\\])\$[0-9]/ || $0 ~ /(^|[^\\])`/) {
        print "    line " NR ": " $0
    }
' "$0")
if [ -n "$HEREDOC_HAZARDS" ]; then
    echo "unescaped dollar-digit or backtick inside a generated-script heredoc:" >&2
    echo "$HEREDOC_HAZARDS" >&2
    echo "  Escape it (\\\$9, \\\`) or reword." >&2
    exit 1
fi

# ------------------------------------------------------------ SLURM: train ---
cat > "$SLURM_TRAIN" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT_CPU
#SBATCH --job-name=xftrain_$TS
#SBATCH --output=$CLUSTER_DIR/xf_sweep_train_${TS}_task%a.out
#SBATCH --error=$CLUSTER_DIR/xf_sweep_train_${TS}_task%a.err
#SBATCH --array=0-$((TRAIN_ARRAY - 1))
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$TRAIN_CPUS
#SBATCH --mem-per-cpu=$TRAIN_MEM
#SBATCH --time=$TRAIN_WALL
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL
set -uo pipefail
module load $JULIA_MODULE
# CUDA module even though training is CPU-only. LocalPreferences.toml pins
# CUDA_Runtime_jll local_toolkit = true, so CUDA.jl discovers a SYSTEM toolkit
# rather than downloading an artifact; with no toolkit on PATH its precompile
# fails and the Pkg.precompile() below exits non-zero. USE_GPU=0 still keeps
# training on the CPU, which is what Enzyme requires.
# NOTE: no backticks in this heredoc -- it is unquoted, so backticks would run.
module load $CUDA_MODULE
# DEPOT. Falling back to \$HOME is how a depot gets silently corrupted: /home is
# a small quota and a CUDA+Enzyme depot will exhaust it mid-extract, leaving
# packages with some source files and not others. Fail loudly instead.
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "\${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="\${SCRATCH}/.julia"
fi
echo "[\${SLURM_JOB_NAME:-job}] depot: \$JULIA_DEPOT_PATH"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export USE_GPU=0
cd \$SLURM_SUBMIT_DIR
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

TASK=\${SLURM_ARRAY_TASK_ID:-0}
LOCAL="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "$WORKDIR" "$CODENAME" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/xf_${TS}_train_task\${TASK}"
# Interleaved slice: task t runs lines t+1, t+1+K, ... of the full point list,
# so a slow region of the grid is spread over all tasks instead of landing on
# one. The enriched arms are the slow ones here.
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" \\
    | awk -v k=$TRAIN_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/train.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/train.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[train task \$TASK] no points in this slice ($TRAIN_ARRAY tasks > $N_TRAIN_POINTS points); nothing to do."
    exit 0
fi

stage_out() {
    tar -cf - --exclude='hyperparams_xf_*.toml' -C "\$LOCAL" models logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

echo "[train task \$TASK/$TRAIN_ARRAY] \$N_TASK of $N_TRAIN_POINTS model(s), \$SLURM_CPUS_PER_TASK at a time: \$(date)"
JOBLOG="\$LOCAL/cluster/logs/xf_${TS}_train_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/xf_${TS}_train_task\${TASK}"
parallel --jobs \$SLURM_CPUS_PER_TASK --joblog "\$JOBLOG" \\
    --results "\$RESULTS_ROOT" < "\$SLURM_TMPDIR/train.txt"
if [ -f "\$JOBLOG" ]; then
    # Print the Command column in full. It contains spaces, so awk field nine on
    # its own yields only the first token -- which is how a whole failed sweep
    # once reported itself as "FAILED (exit 1): julia" and nothing else.
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
echo "[train task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK model(s) exited 0"
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[train task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[train task \$TASK] ---- end ----"
fi
echo "[train task \$TASK] done: \$(date)"
EOF
chmod +x "$SLURM_TRAIN"

# ------------------------------------------------------------- SLURM: test ---
cat > "$SLURM_TEST" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT_GPU
#SBATCH --job-name=xftest_$TS
#SBATCH --output=$CLUSTER_DIR/xf_sweep_test_${TS}_task%a.out
#SBATCH --error=$CLUSTER_DIR/xf_sweep_test_${TS}_task%a.err
#SBATCH --array=0-$((TEST_ARRAY - 1))
${GPU_SBATCH_LINE}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$TEST_CPUS
#SBATCH --mem-per-gpu=$MEM_PER_GPU
#SBATCH --time=$TEST_WALL
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL
set -uo pipefail
module load $JULIA_MODULE
module load $CUDA_MODULE
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -z "\${SCRATCH:-}" ]; then
        echo "ERROR: neither JULIA_DEPOT_PATH nor SCRATCH is set." >&2
        exit 1
    fi
    export JULIA_DEPOT_PATH="\${SCRATCH}/.julia"
fi
echo "[\${SLURM_JOB_NAME:-job}] depot: \$JULIA_DEPOT_PATH"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_OFFLINE=true
export GPU_BACKEND=cuda USE_GPU=1
cd \$SLURM_SUBMIT_DIR
# CUDA_Runtime_jll bakes in whether a driver was visible AT PRECOMPILE TIME. The
# CPU training job has no driver, so its Pkg.precompile() poisons the shared
# depot with "no CUDA runtime found"; this job's precompile then finds everything
# up to date and leaves the bad cache in place. Force a rebuild of that one JLL
# here, where the driver IS present, in its own process.
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate()'; then
    exit 1
fi
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e '
    pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
    Base.compilecache(pkg)'; then
    exit 1
fi
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.precompile()'; then
    exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

# Hard gate. Without it the job proceeds and all $N_TEST_POINTS tests die one by
# one at _to_dense_gpu, each burning its own startup, and stage-out returns nothing.
if ! julia --project=\$SLURM_SUBMIT_DIR/.. -e '
    using CUDA
    if !CUDA.functional()
        exit(1)
    end'; then
    echo "ERROR: CUDA is not functional on this node after forcing a JLL rebuild." >&2
    echo "  Check that 'module load $CUDA_MODULE' succeeded and that" >&2
    echo "  LocalPreferences.toml still has [CUDA_Runtime_jll] local_toolkit = true." >&2
    exit 1
fi
echo "[test] CUDA functional."

TASK=\${SLURM_ARRAY_TASK_ID:-0}
LOCAL="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "$WORKDIR" "$CODENAME" | tar -xf - -C "\$SLURM_TMPDIR"
mkdir -p "\$LOCAL"/{models,results,logs} "\$LOCAL/cluster/logs/xf_${TS}_test_task\${TASK}"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS" \\
    | awk -v k=$TEST_ARRAY -v t="\$TASK" '(NR - 1) % k == t' > "\$SLURM_TMPDIR/test.txt"
N_TASK=\$(wc -l < "\$SLURM_TMPDIR/test.txt")
if [ "\$N_TASK" -eq 0 ]; then
    echo "[test task \$TASK] no points in this slice ($TEST_ARRAY tasks > $N_TEST_POINTS points); nothing to do."
    exit 0
fi

# neural_bp_experiments.jl SKIPS testing when the results file already exists
# and reports the old numbers as if fresh. The staged-in copy carries the
# previous run's results, so remove this sweep's targets before testing.
# Safe to clear ALL of them even under a job array: this deletes only the
# node-local staged copy, and stage_out untars this task's files INTO the shared
# directory without removing anything already there.
rm -f "\$LOCAL"/results/simulation_results_*_xf*_seed_*.csv

# The generator wrote retrain = true; flip it so this job loads the trained
# weights rather than retraining on a GPU it cannot use for AD. This is also
# what makes many tests share one model: every test command names the TRAINING
# file in --train, so each re-loads the weights that file produced.
for f in "\$LOCAL"/models/hyperparams_xf_*.toml; do
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "\$f" > "\$f.tmp"
    mv "\$f.tmp" "\$f"
done
N_MODELS=\$(ls "\$LOCAL"/models/*.json 2>/dev/null | wc -l)
echo "[test task \$TASK/$TEST_ARRAY] \$N_MODELS trained model(s) staged in; expecting $N_TRAIN_POINTS"
if [ "\$N_MODELS" -lt $N_TRAIN_POINTS ]; then
    echo "[test task \$TASK] WARNING: fewer models than training points. Tests whose"
    echo "                   weights are missing will fail; check the train job first."
fi

stage_out() {
    tar -cf - --exclude='hyperparams_xf_*.toml' -C "\$LOCAL" results logs cluster/logs \\
        2>/dev/null | tar -xf - -C "$WORKDIR/$CODENAME"
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

export GPU_MEMORY=${GPU_MEMORY_MB}M
echo "[test task \$TASK] \$N_TASK of $N_TEST_POINTS point(s), $TEST_JOBS at a time on \${SLURM_GPUS_ON_NODE:-1} GPU(s): \$(date)"
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
JOBLOG="\$LOCAL/cluster/logs/xf_${TS}_test_task\${TASK}.joblog"
RESULTS_ROOT="\$LOCAL/cluster/logs/xf_${TS}_test_task\${TASK}"
parallel --jobs $TEST_JOBS --joblog "\$JOBLOG" --results "\$RESULTS_ROOT" \\
    'card=\$(( ({%} - 1) % \${SLURM_GPUS_ON_NODE:-1} + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$SLURM_TMPDIR/test.txt"
if [ -f "\$JOBLOG" ]; then
    awk 'NR>1 && \$7 != 0 {print "  FAILED (exit " \$7 "): " substr(\$0, index(\$0, \$9))}' "\$JOBLOG"
fi
echo "[test task \$TASK] \$(awk 'NR>1 && \$7 == 0' "\$JOBLOG" | wc -l)/\$N_TASK point(s) exited 0"
FIRST_STDERR=\$(find "\$RESULTS_ROOT" -name stderr -size +0c 2>/dev/null | head -1)
if [ -n "\$FIRST_STDERR" ]; then
    echo "[test task \$TASK] ---- first failing point's stderr ----"
    tail -25 "\$FIRST_STDERR" | sed 's/^/    /'
    echo "[test task \$TASK] ---- end ----"
fi
echo "[test task \$TASK] done: \$(date)"
EOF
chmod +x "$SLURM_TEST"

# ------------------------------------------------------------------ report ---
N_MATCHED=$(awk -F'\t' '$1 == "matched"' "$PAIR_MANIFEST" | wc -l | tr -d ' ')
N_PROBE=$(awk -F'\t' '$1 == "probe"' "$PAIR_MANIFEST" | wc -l | tr -d ' ')
echo
echo "[xf_sweep] $N_TRAIN_POINTS model(s), $N_TEST_POINTS test(s)"
echo "  train on  -> $TRAIN_DATASETS"
echo "  seeds     -> $SEEDS"
echo "  arms      -> $CHECK_NODE_ARMS   (no-CER baseline: $INCLUDE_NOCER)"
echo "  matched   -> $N_MATCHED test(s)"
echo "  probe     -> $N_PROBE test(s): $PROBE_SOURCE weights on $PROBE_TEST_KEYS"
echo "  cluster   -> $CLUSTER_NAME"
echo "  train     -> $ACCOUNT_CPU: array 0-$((TRAIN_ARRAY-1)) ($TRAIN_ARRAY x $TRAIN_CPUS cpu x $TRAIN_MEM = $(( TRAIN_TOTAL_MB / 1024 ))G/node), $TRAIN_WALL"
echo "               ~$(( (N_TRAIN_POINTS + TRAIN_ARRAY - 1) / TRAIN_ARRAY )) model(s) per task"
echo "  test      -> $ACCOUNT_GPU: array 0-$((TEST_ARRAY-1)) ($TEST_ARRAY tasks x ${N_GPUS}x $GPU_TYPE (${VRAM_PER_GPU}G vram), $TEST_CPUS cpu),"
echo "               --mem-per-gpu=$MEM_PER_GPU host ram, GPU_MEMORY=${GPU_MEMORY_MB}M, $TEST_JOBS at a time"
echo "               ~$(( (N_TEST_POINTS + TEST_ARRAY - 1) / TEST_ARRAY )) test(s) per task"
echo "  commands  -> $TRAIN_CMDS"
echo "               $TEST_CMDS"
echo "  pairs     -> $PAIR_MANIFEST   (kind / train / test / arm / seed)"
echo

FIRST_TRAIN_KEY=$(echo $TRAIN_DATASETS | awk '{print $1}')
FIRST_SEED=$(echo $SEEDS | awk '{print $1}')
N_EPOCHS_BASE=$(grep -E '^[[:space:]]*n_epochs' "$MODELS_DIR/$BASE_HP" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
# Must match decompose_arm's tag exactly, or the check reads a path that was
# never written. This is the CER tanh arm of the first dataset and seed.
FIRST_MODEL="neuralbp_weights_nlayers_${NLAYERS}_epochs_${N_EPOCHS_BASE}_trained_using_train_${FIRST_TRAIN_KEY}_xfcer_seed_${FIRST_SEED}.json"
FIRST_LOSSLOG="debugging_train_${FIRST_TRAIN_KEY}_xfcer_cnenr0p42F_seed_${FIRST_SEED}_individual_losses.csv"

echo "submit — TRAIN first (CPU, $ACCOUNT_CPU), then TEST (GPU, $ACCOUNT_GPU):"
echo
echo "  # 1. training"
echo "  sbatch $SLURM_TRAIN"
echo
echo "  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU."
echo "  #    (a) did the weights move at all?"
echo "  julia -e 'using JSON, Statistics; w=JSON.parsefile(\"$MODELS_DIR/$FIRST_MODEL\");"
echo "            println(std(vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"])))'"
echo "  # 0.058 => never trained (every batch NaN-skipped); larger => trained."
echo
echo "  #    (b) did the enriched arm DIVERGE? base loss should fall across epochs,"
echo "  #        not climb. On the old dataset it went 0.011 -> 4.21 by epoch 5."
echo "  awk -F, 'NR>1 && \$3==\"80\"' $WORKDIR/$CODENAME/logs/$FIRST_LOSSLOG | head -1"
echo "  # If it climbs, lower learning_rate in $BASE_HP and re-run this generator."
echo
echo "  # 3. testing"
echo "  sbatch $SLURM_TEST"
echo
echo "  To chain them without the check instead:"
echo "    TRAIN=\$(sbatch --parsable $SLURM_TRAIN)"
echo "    sbatch --dependency=afterok:\$TRAIN $SLURM_TEST"

if [ "$LOCAL" -eq 1 ]; then
    # A smoke test must not read the real datasets. They are 72 x 1e6, and
    # readdlm parses them into a Matrix{Int64} (576 MB final, several times that
    # in intermediates) -- enough to be OOM-killed on a laptop before the first
    # batch. Cut the first N samples into a parallel dataset key so every
    # downstream filename stays consistent.
    SMOKE_KEY="${FIRST_TRAIN_KEY}_smoke${SMOKE_N}"
    DATA="$WORKDIR/$CODENAME"
    echo
    echo "[xf_sweep] building a ${SMOKE_N}-sample smoke dataset: $SMOKE_KEY"
    cut -d' ' -f1-${SMOKE_N} "$DATA/training_data/train_${FIRST_TRAIN_KEY}.txt" > "$DATA/training_data/train_${SMOKE_KEY}.txt"
    cut -d' ' -f1-${SMOKE_N} "$DATA/testing_data/test_${FIRST_TRAIN_KEY}.txt"   > "$DATA/testing_data/test_${SMOKE_KEY}.txt"
    cp -f "$DATA/correlated_weights/correlated_weights_${FIRST_TRAIN_KEY}.txt" \
          "$DATA/correlated_weights/correlated_weights_${SMOKE_KEY}.txt"

    SMOKE_HP="hyperparams_xf_smoke.toml"
    grep -vE '^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|coupling_scale_init|coupling_scale_learnable|n_epochs|n_gradient_updates_per_epoch|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$SMOKE_HP"
    {
        echo ""
        echo "# smoke test: 1 epoch, 20 updates — enough to exercise every code path."
        echo "retrain = true"
        echo "run_tag = \"_xfsmoke\""
        echo "use_CER = true"
        echo "seed = 1"
        echo "n_epochs = 1"
        echo "n_gradient_updates_per_epoch = 20"
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = true"
        echo "check_node = \"enriched\""
        echo "coupling_scale_init = 0.42"
        echo "coupling_scale_learnable = true"
    } >> "$MODELS_DIR/$SMOKE_HP"

    SMOKE_CMD="julia --project=\"./../\" neural_bp_experiments.jl --workdir $WORKDIR --codename $CODENAME \
--n_hidden_layers $NLAYERS --hyperparams $SMOKE_HP --cer_data correlated_weights_${SMOKE_KEY}.txt \
--quiet false --isdebug true --train train_${SMOKE_KEY}.txt --test test_${SMOKE_KEY}.txt"

    echo
    echo "local smoke test — enriched check node, learned alpha from 0.42, $SMOKE_N samples, 1 epoch:"
    echo "  cd $EXPTS_DIR"
    echo "  rm -f $DATA/results/simulation_results_*_xfsmoke_seed_1.csv"
    echo "  USE_GPU=0 $SMOKE_CMD"
    echo
    echo "  It should train 20 batches and then test. What to check afterwards:"
    echo "    - the run completes without 'killed'"
    echo "    - $DATA/logs/debugging_train_${SMOKE_KEY}_xfsmoke_seed_1.csv has non-zero rows"
    echo "    - the weights moved:"
    echo "        julia -e 'using JSON, Statistics; w=JSON.parsefile(\"$DATA/models/neuralbp_weights_nlayers_${NLAYERS}_epochs_1_trained_using_train_${SMOKE_KEY}_xfsmoke_seed_1.json\");"
    echo "                  println(\"sd = \", std(vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"])))'"
    echo "      sd ~ 0.058 means it never trained; anything larger means it did."
fi

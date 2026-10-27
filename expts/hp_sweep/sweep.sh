# hp_sweep/sweep.sh -- the SWEEP side: the sweep file in, the points out.
#
# Sourced by sweep_hyperparams.sh; defines functions only. Nothing here knows
# about SLURM -- hp_sweep/cluster.sh turns the lists written here into jobs.
#
# A point is (training key, seed, optimizer arm, check-node arm), plus the no-CER
# baseline when include_nocer = true. For each one, emit_point writes
#     models/hyperparams_hp<run tag>_<key tag>_seed<s>.toml    (into a staging
#                                                             directory first)
#     one line of TRAIN_CMDS        julia ... --train train_<key>.txt
#     one line of TEST_CMDS per     julia ... --train train_<key>.txt --test test_<k>.txt
#       test set k it is scored on
#     one line of EXPECTED_MODELS   the weights file the training writes
#
# WHICH TEST SETS A MODEL IS SCORED ON is decided in ONE place, test_keys_of: with
# test_keys = [] in the sweep file, only the test set of the model's own training
# key; with a list, every set on it. check_test_pairing then reads the written
# test commands back and refuses the sweep if any of them breaks that rule.

# Every key a sweep file may hold (sweeps/template.toml describes each one), and
# the ones it must hold. A list may be present but empty ([]).
SWEEP_KEYS="cluster workdir codename datasets ref_datasets test_keys base_hyperparams seeds include_nocer single_qubit_rescale check_node_arms optimizer_arms loss_layer_selection commit_layer_rule train_array_tasks train_wall_time test_array_tasks test_wall_time"
REQUIRED_SWEEP_KEYS="cluster workdir codename datasets test_keys base_hyperparams seeds include_nocer single_qubit_rescale check_node_arms optimizer_arms train_array_tasks train_wall_time test_array_tasks test_wall_time"
SWEEP_TEMPLATE="sweeps/template.toml"

# The network is unrolled to 90 layers, always (50 was worse on every arm,
# 2026-10-08). Not a setting: the count is in every filename.
NLAYERS=90

# The keys emit_point writes itself, stripped from the base TOML first so a stale
# value there can never override a swept one. The keys of the removed loss terms
# and of the removed layer schedule (coupling_schedule*) are stripped too: the code
# ignores them now (and refuses a "step" schedule), so a generated file should not
# carry them.
GENERATED_KEYS_PATTERN='^[[:space:]]*(retrain|run_tag|use_CER|seed|single_qubit_rescale|require_correlations|check_node|enriched_table|coupling_scale_init|coupling_scale_learnable|coupling_schedule|coupling_schedule_layer_init|coupling_schedule_width_init|coupling_schedule_learnable|manual_backprop|loss_layer_selection|commit_layer_rule|loss_layer_ramp_sharpness|base_loss|training_samples|failure_weight_boundary|warmup_layers|adam_eps|weight_decay|initial_conditions_scale|learning_rate|sparsity_importance|syndrome_gate_threshold|correlation_certainty_threshold|correlation_weight|correlation_importance|certainty_penalty|certainty_hinge_width|certainty_syndrome_gate_threshold|syndrome_gate_mode|syndrome_gate_rate|correlation_form|correlation_agreement_floor|llr_certainty_importance)[[:space:]]*='

sweep_value()       { settings_value "$SWEEP_FILE" "$1"; }
sweep_list()        { settings_list "$SWEEP_FILE" "$1"; }
sweep_quoted_list() { settings_quoted_list "$SWEEP_FILE" "$1"; }

# Refuses a list that names one element twice: two equal points would write the
# same files from two array tasks at once.
refuse_repeated_elements() {   # <key> <elements...>
    local key="$1"
    shift
    # (No elements: return before "$@", which bash 3.2 under set -u may call unbound.)
    if [ "$#" -eq 0 ]; then
        return 0
    fi
    local repeated
    repeated="$(printf '%s\n' "$@" | sort | uniq -d | tr '\n' ' ')"
    if [ -n "$repeated" ]; then
        echo "$key in $SWEEP_FILE names ${repeated}more than once." >&2
        exit 1
    fi
}

# ---------------------------------------------------------- the sweep file ---
# Reads every key of the sweep file into a global, after check_settings_keys has
# refused unknown, misplaced, repeated and missing keys. Sets CLUSTER_NAME,
# CLUSTER_PROFILE (clusters/<name>.toml, for messages) and CLUSTER_FILE (its full
# path, which hp_sweep/cluster.sh reads), WORKDIR, CODENAME, DATASETS,
# REF_DATASETS, TEST_KEYS, BASE_HP, SEEDS, INCLUDE_NOCER, RESCALE,
# LOSS_LAYER_SELECTION, COMMIT_LAYER_RULE, CHECK_NODE_ARMS, OPTIMIZER_ARMS,
# N_OPTIMIZER_ARMS, and the job sizing TRAIN_ARRAY, TRAIN_WALL, TEST_ARRAY, TEST_WALL.
read_sweep_file() {
    CLUSTER_NAME="$(sweep_value cluster)"
    CLUSTER_PROFILE="clusters/${CLUSTER_NAME}.toml"
    CLUSTER_FILE="$SCRIPT_DIR/$CLUSTER_PROFILE"
    # Where a cluster key found in the sweep file belongs, for the message.
    local profile_for_messages="$CLUSTER_FILE"
    if [ -z "$CLUSTER_NAME" ]; then
        profile_for_messages="the cluster profile, clusters/<name>.toml"
    fi
    check_settings_keys "$SWEEP_FILE" "sweep file" "$SWEEP_KEYS" "$REQUIRED_SWEEP_KEYS" \
        "$CLUSTER_KEYS" "$profile_for_messages" "$SWEEP_TEMPLATE"
    if [ -z "$CLUSTER_NAME" ] || [ ! -f "$CLUSTER_FILE" ]; then
        echo "cluster = \"$CLUSTER_NAME\" in $SWEEP_FILE: no profile $CLUSTER_PROFILE." >&2
        echo "  The profiles: $(ls "$SCRIPT_DIR"/clusters/*.toml 2>/dev/null | sed -E 's|.*/||; s|\.toml$||' | tr '\n' ' ')" >&2
        exit 1
    fi

    WORKDIR="$(sweep_value workdir)"
    CODENAME="$(sweep_value codename)"
    DATASETS="$(sweep_list datasets)"
    REF_DATASETS="$(sweep_list ref_datasets)"
    TEST_KEYS="$(sweep_list test_keys)"
    BASE_HP="$(sweep_value base_hyperparams)"
    SEEDS="$(sweep_list seeds)"
    INCLUDE_NOCER="$(sweep_value include_nocer)"
    RESCALE="$(sweep_value single_qubit_rescale)"

    if [ -z "$(echo $DATASETS)" ]; then
        echo "datasets in $SWEEP_FILE is empty: name at least one training key." >&2
        exit 1
    fi
    refuse_repeated_elements datasets $DATASETS $REF_DATASETS
    refuse_repeated_elements test_keys $TEST_KEYS
    if [ -z "$(echo $SEEDS)" ]; then
        echo "seeds in $SWEEP_FILE is empty." >&2
        exit 1
    fi
    local seed
    for seed in $SEEDS; do
        if ! echo "$seed" | grep -qE '^[0-9]+$'; then
            echo "seeds in $SWEEP_FILE: '$seed' is not a non-negative integer." >&2
            exit 1
        fi
    done
    refuse_repeated_elements seeds $SEEDS
    case "$INCLUDE_NOCER" in
        true|false) ;;
        *) echo "include_nocer = '$INCLUDE_NOCER' in $SWEEP_FILE: use true or false." >&2; exit 1 ;;
    esac
    # Every point's TOML states it, and the priors the decoder sees depend on it.
    if ! echo "$RESCALE" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "single_qubit_rescale = '$RESCALE' in $SWEEP_FILE must be a number (0.1 maps the median rate to 0.1)." >&2
        exit 1
    fi

    # The layer pair: the default for an optimizer arm that names neither key; an
    # arm may override either. softmin + first when the sweep file sets neither.
    LOSS_LAYER_SELECTION="$(sweep_value loss_layer_selection)"
    COMMIT_LAYER_RULE="$(sweep_value commit_layer_rule)"
    if [ -z "$LOSS_LAYER_SELECTION" ]; then
        LOSS_LAYER_SELECTION="softmin"
    fi
    if [ -z "$COMMIT_LAYER_RULE" ]; then
        COMMIT_LAYER_RULE="first"
    fi
    case "$LOSS_LAYER_SELECTION" in
        softmin|last|mean|ramp) ;;
        *) echo "unknown loss_layer_selection '$LOSS_LAYER_SELECTION': use softmin, last, mean or ramp." >&2; exit 1 ;;
    esac
    case "$COMMIT_LAYER_RULE" in
        first|last) ;;
        *) echo "unknown commit_layer_rule '$COMMIT_LAYER_RULE': use first or last." >&2; exit 1 ;;
    esac

    CHECK_NODE_ARMS="$(sweep_list check_node_arms)"
    if [ -z "$(echo $CHECK_NODE_ARMS)" ]; then
        CHECK_NODE_ARMS="tanh"
    fi
    # Only the name is checked here; parse_check_node_arm reads the whole spec.
    local check_node_arm
    for check_node_arm in $CHECK_NODE_ARMS; do
        case "${check_node_arm%%:*}" in
            tanh|enriched|bethe) ;;
            *)
                echo "check_node_arms: '$check_node_arm' -- use tanh, enriched:<alpha>:<fixed|learn>" >&2
                echo "  or bethe:<alpha>:<fixed|learn>." >&2
                exit 1 ;;
        esac
    done
    refuse_repeated_elements check_node_arms $CHECK_NODE_ARMS
    OPTIMIZER_ARMS="$(sweep_quoted_list optimizer_arms)"
    if [ -z "$OPTIMIZER_ARMS" ]; then
        OPTIMIZER_ARMS="base"
    fi
    refuse_repeated_elements optimizer_arms $OPTIMIZER_ARMS
    # Whether the optimizer tag enters filenames at all: not for a lone "base" arm
    # (see emit_point), so a sweep that ignores this axis keeps the names it
    # always had.
    N_OPTIMIZER_ARMS="$(echo "$OPTIMIZER_ARMS" | wc -w | tr -d ' ')"

    # How the points are cut into array jobs. Read here because they size THIS
    # sweep; hp_sweep/cluster.sh uses them.
    TRAIN_ARRAY="$(sweep_value train_array_tasks)"
    TEST_ARRAY="$(sweep_value test_array_tasks)"
    TRAIN_WALL="$(sweep_value train_wall_time)"
    TEST_WALL="$(sweep_value test_wall_time)"
    local array_key array_value wall_key wall_value
    for array_key in train_array_tasks test_array_tasks; do
        array_value="$(sweep_value "$array_key")"
        if ! echo "$array_value" | grep -qE '^[1-9][0-9]*$'; then
            echo "$array_key = '$array_value' in $SWEEP_FILE must be a positive integer." >&2
            exit 1
        fi
    done
    for wall_key in train_wall_time test_wall_time; do
        wall_value="$(sweep_value "$wall_key")"
        if ! echo "$wall_value" | grep -qE '^([0-9]+-)?[0-9]+:[0-9][0-9]:[0-9][0-9]$|^[0-9]+:[0-9][0-9]$'; then
            echo "$wall_key = '$wall_value' in $SWEEP_FILE: write it as H:MM:SS (or D-HH:MM:SS)." >&2
            exit 1
        fi
    done
}

# ------------------------------------------------------ the run directory ---
# The run directory, its base TOML, and the folders the jobs stage out into.
# Sets MODELS_DIR and CLUSTER_DIR.
check_run_directory() {
    if [ ! -d "$WORKDIR/$CODENAME" ]; then
        echo "no run directory $WORKDIR/$CODENAME (workdir = \"$WORKDIR\", codename = \"$CODENAME\")." >&2
        echo "  workdir is relative to the directory you run from: run from expts/." >&2
        exit 1
    fi
    MODELS_DIR="$WORKDIR/$CODENAME/models"
    CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
    if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
        echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
        exit 1
    fi
    # The stage-out copies results, models and logs back into these.
    mkdir -p "$CLUSTER_DIR" "$MODELS_DIR" "$WORKDIR/$CODENAME/results" "$WORKDIR/$CODENAME/logs"
}

# The value of <key> in the base TOML.
base_hyperparameter_value() {   # <key>
    settings_value "$MODELS_DIR/$BASE_HP" "$1"
}

# The base TOML's values that the points inherit or that size the jobs. Every
# swept key is written on every point (parse_optimizer_arm starts from these): a
# key that is sometimes stripped and sometimes inherited is how a sweep ends up
# with two different meanings for one filename. Sets N_EPOCHS,
# N_UPDATES_PER_EPOCH and BASE_<KEY> for the eleven sweepable keys.
read_base_hyperparameters() {
    # The training job runs on a GPU node with the hand-written backward pass.
    # Enzyme cannot use the GPU, so without manual_backprop = true every point would
    # run Enzyme on the allocation's few CPU cores (~6600 s a point measured on a
    # CPU node) and the wall would take them all.
    if [ "$(base_hyperparameter_value manual_backprop)" != "true" ]; then
        echo "$MODELS_DIR/$BASE_HP must set manual_backprop = true: the training job runs on" >&2
        echo "  GPU nodes with the hand-written backward pass (Enzyme cannot use the GPU)." >&2
        exit 1
    fi
    # n_epochs is not a swept key, so every point trains for the base TOML's count
    # (5 when the key is absent: the Julia default). It is in every weights name.
    N_EPOCHS="$(base_hyperparameter_value n_epochs)"
    if [ -z "$N_EPOCHS" ]; then
        N_EPOCHS=5
    fi
    if ! echo "$N_EPOCHS" | grep -qE '^[1-9][0-9]*$'; then
        echo "n_epochs = '$N_EPOCHS' in $MODELS_DIR/$BASE_HP must be a positive integer." >&2
        exit 1
    fi
    # Scales the training estimate (the costs were measured at 500).
    N_UPDATES_PER_EPOCH="$(base_hyperparameter_value n_gradient_updates_per_epoch)"
    if [ -z "$N_UPDATES_PER_EPOCH" ]; then
        N_UPDATES_PER_EPOCH=500
    fi
    if ! echo "$N_UPDATES_PER_EPOCH" | grep -qE '^[1-9][0-9]*$'; then
        echo "n_gradient_updates_per_epoch = '$N_UPDATES_PER_EPOCH' in $MODELS_DIR/$BASE_HP must be a positive integer." >&2
        exit 1
    fi
    BASE_LEARNING_RATE="$(base_hyperparameter_value learning_rate)"
    BASE_ADAM_EPS="$(base_hyperparameter_value adam_eps)"
    BASE_WEIGHT_DECAY="$(base_hyperparameter_value weight_decay)"
    BASE_INITIAL_CONDITIONS_SCALE="$(base_hyperparameter_value initial_conditions_scale)"
    BASE_WARMUP_LAYERS="$(base_hyperparameter_value warmup_layers)"
    # k of the "ramp" weighting; read by the Julia side only under ramp. 3.0 is the
    # Julia default (DEFAULT_LOSS_LAYER_RAMP_SHARPNESS), kept in step here so a base
    # TOML without the key still emits the number the code would use.
    BASE_LOSS_LAYER_RAMP_SHARPNESS="$(base_hyperparameter_value loss_layer_ramp_sharpness)"
    if [ -z "$BASE_LOSS_LAYER_RAMP_SHARPNESS" ]; then
        BASE_LOSS_LAYER_RAMP_SHARPNESS="3.0"
    fi
    # Which per-check residue. sin_residue is the Julia default (every earlier run).
    BASE_BASE_LOSS="$(base_hyperparameter_value base_loss)"
    if [ -z "$BASE_BASE_LOSS" ]; then
        BASE_BASE_LOSS="sin_residue"
    fi
    # Which samples of the training file to train on (src/sample_selection.jl): the
    # set size (0 = the whole file) and the Poisson centre of its error-weight
    # distribution (0 = the file's own). Both are the Julia defaults when absent.
    BASE_TRAINING_SAMPLES="$(base_hyperparameter_value training_samples)"
    if [ -z "$BASE_TRAINING_SAMPLES" ]; then
        BASE_TRAINING_SAMPLES="0"
    fi
    BASE_FAILURE_WEIGHT_BOUNDARY="$(base_hyperparameter_value failure_weight_boundary)"
    if [ -z "$BASE_FAILURE_WEIGHT_BOUNDARY" ]; then
        BASE_FAILURE_WEIGHT_BOUNDARY="0"
    fi
}

# ---------------------------------------------------------- the data sets ---
# Every input a point reads must exist HERE, before the jobs copy them to the node.
# Without this check a missing input is discovered only after the queue wait, as N
# identical "exit 1" lines with no message -- which is how one 240-point sweep
# failed on 2026-09-04.
check_dataset_files() {
    local missing="" key required
    for key in $DATASETS $REF_DATASETS; do
        for required in "training_data/train_${key}.txt" \
                        "testing_data/test_${key}.txt" \
                        "correlated_weights/correlated_weights_${key}.txt"; do
            if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
                missing="${missing}\n    $required"
            fi
        done
    done
    # A test_keys entry needs only the two files the test step reads.
    for key in $TEST_KEYS; do
        for required in "testing_data/test_${key}.txt" \
                        "correlated_weights/correlated_weights_${key}.txt"; do
            if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
                missing="${missing}\n    $required"
            fi
        done
    done
    # A Bethe arm reads bethe_table_<key>.txt next to every CER file it uses: the
    # training key's and every test key's. The table is built from the training
    # data of that key, offline, by misc/build_bethe_tables.py.
    local check_node_arm needs_bethe_tables=0
    for check_node_arm in $CHECK_NODE_ARMS; do
        if [ "${check_node_arm%%:*}" = "bethe" ]; then
            needs_bethe_tables=1
        fi
    done
    if [ "$needs_bethe_tables" -eq 1 ]; then
        for key in $(printf "%s\n" $DATASETS $REF_DATASETS $TEST_KEYS | sort -u); do
            required="correlated_weights/bethe_table_${key}.txt"
            if [ ! -f "$WORKDIR/$CODENAME/$required" ]; then
                missing="${missing}\n    $required   (python3 misc/build_bethe_tables.py --workdir $CODENAME --key $key)"
            fi
        done
    fi
    if [ -n "$missing" ]; then
        echo "missing dataset input(s) under $WORKDIR/$CODENAME:" >&2
        printf "%b\n" "$missing" >&2
        echo "  The jobs copy these files to the node when they start, so they must exist" >&2
        echo "  before submitting. Check the datasets and test_keys in $SWEEP_FILE, and" >&2
        echo "  that a clean-up or rsync has not removed the inputs." >&2
        exit 1
    fi
}

# ------------------------------------------------------------- the points ---
tag_of() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# alpha* of one dataset key: ln((1 - t)/t) / ln((1 - m)/m), t = single_qubit_rescale
# and m the median of the key's raw single-qubit rates (the "<qubit> : <rate>"
# lines of its CER file; the "(<a>, <b>) : <J>" lines are pairs). The factor by
# which the rescale softens the median prior's log-odds -- src/neuralbase.jl,
# prior_softening_factor, mirrored here, including its 1 when the rescale leaves
# the rates alone (t <= 0, no rates, or a rate already at t or above).
alpha_star_of_key() {   # <key>
    local cer_file="$WORKDIR/$CODENAME/correlated_weights/correlated_weights_$1.txt"
    sed -nE 's/^[[:space:]]*[0-9]+[[:space:]]*:[[:space:]]*([0-9.eE+-]+)[[:space:]]*$/\1/p' "$cer_file" \
        | sort -g \
        | awk -v t="${RESCALE:-0}" -v cer_file="$cer_file" '
            { rate[NR] = $1 + 0 }
            END {
                if (NR == 0) {
                    print "alpha_star: no single-qubit rates in " cer_file "; using 1.0" > "/dev/stderr"
                    printf "1.0"
                    exit
                }
                if (t + 0 <= 0 || rate[NR] >= t + 0) { printf "1.0"; exit }
                if (NR % 2 == 1) { median = rate[(NR + 1) / 2] }
                else             { median = (rate[NR / 2] + rate[NR / 2 + 1]) / 2 }
                printf "%.4f", log((1 - t) / t) / log((1 - median) / median)
            }'
}

# An alpha_star arm: say which alpha each training key's points start from.
print_alpha_star_values() {
    local check_node_arm key
    for check_node_arm in $CHECK_NODE_ARMS; do
        case "$check_node_arm" in
            enriched:alpha_star:*|bethe:alpha_star:*)
                echo "[hp_sweep] $check_node_arm starts each point from alpha* of its training key:"
                for key in $DATASETS; do
                    echo "             $key -> $(alpha_star_of_key "$key")"
                done
                ;;
        esac
    done
}

# THE TRAIN/TEST PAIRING. The test set(s) a model trained on <training key> is
# scored on:
#   test_keys = []          its own key, and nothing else. train_<key>.txt and
#                           test_<key>.txt come from one error model -- at sigma = 0
#                           the same gate rates -- so nothing is transferred.
#   test_keys = [k1, ...]   every listed key, whatever the model was trained on
#                           ("train at one rate, test across the range").
# emit_point writes one test command per key this returns, with the CER file of
# that TEST key, and check_test_pairing verifies the written commands against it.
test_keys_of() {   # <training key>
    if [ -z "$(echo $TEST_KEYS)" ]; then
        echo "$1"
    else
        echo "$TEST_KEYS"
    fi
}

# Check-node arm -> the check-node keys of a point's TOML.
#   tanh                              the standard check-to-variable rule
#   enriched:<alpha>:<fixed|learn>    the CER pair couplings inside each check
#                                     factor (src/soft_constraints.jl), alpha the
#                                     scale on J, held or trained from there
#   bethe:<alpha>:<fixed|learn>       the same node with the Bethe table
#                                     (correlated_weights/bethe_table_<key>.txt)
# <alpha> is a number, or alpha_star: alpha* of the point's TRAINING key.
# Sets POINT_CHECK_NODE, POINT_ENRICHED_TABLE, POINT_CHECK_NODE_TAG (the run-tag
# token: "", _enr or _bethe), POINT_COUPLING_SCALE_INIT, POINT_COUPLING_SCALE_LEARNABLE.
# Fields are split on ":" one at a time with ${var%%:*} / ${var#*:}, so a spec with
# too few fields leaves the remainder EQUAL to the last field rather than empty --
# every branch tests the field it expects.
parse_check_node_arm() {   # <spec> <training key> <use_cer>
    local spec="$1" key="$2" use_cer="$3"
    local name="${spec%%:*}"
    POINT_CHECK_NODE="tanh"
    POINT_ENRICHED_TABLE="pairwise"
    POINT_CHECK_NODE_TAG=""
    POINT_COUPLING_SCALE_INIT="1.0"
    POINT_COUPLING_SCALE_LEARNABLE="false"
    if [ "$name" = "tanh" ]; then
        return 0
    fi
    if [ "$name" != "enriched" ] && [ "$name" != "bethe" ]; then
        echo "check node arm '$spec': unknown check node '$name' (tanh, enriched:<alpha>:<fixed|learn> or bethe:<alpha>:<fixed|learn>)." >&2
        exit 1
    fi
    if [ "$use_cer" = "false" ]; then
        echo "check node arm '$spec': a $name check node needs couplings; refusing it on the no-CER arm." >&2
        exit 1
    fi
    POINT_CHECK_NODE="enriched"
    POINT_CHECK_NODE_TAG="_enr"
    if [ "$name" = "bethe" ]; then
        POINT_ENRICHED_TABLE="bethe"
        POINT_CHECK_NODE_TAG="_bethe"
    fi
    local rest="${spec#*:}"
    if [ "$rest" = "$spec" ]; then
        echo "check node arm '$spec' needs :<alpha>:<fixed|learn> after the name." >&2
        exit 1
    fi
    POINT_COUPLING_SCALE_INIT="${rest%%:*}"
    local learn_spec="${rest#*:}"
    if [ "$learn_spec" = "$rest" ]; then
        echo "check node arm '$spec' needs :fixed or :learn after <alpha>." >&2
        exit 1
    fi
    if [ "$learn_spec" = "learn" ]; then
        POINT_COUPLING_SCALE_LEARNABLE="true"
    elif [ "$learn_spec" != "fixed" ]; then
        echo "check node arm '$spec' must end in :fixed or :learn, got '$learn_spec'." >&2
        exit 1
    fi
    if [ "$POINT_COUPLING_SCALE_INIT" = "alpha_star" ]; then
        POINT_COUPLING_SCALE_INIT="$(alpha_star_of_key "$key")"
    elif ! echo "$POINT_COUPLING_SCALE_INIT" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "check node arm '$spec': alpha '$POINT_COUPLING_SCALE_INIT' must be a number or alpha_star." >&2
        exit 1
    fi
}

# Optimizer arm "<tag>[:<key>=<value>,...]" -> the optimizer and layer keys of a
# point's TOML: the base TOML's values (the layer pair: the sweep file's), with the
# arm's overrides applied. Sets POINT_OPTIMIZER_TAG and POINT_<KEY> for the eleven
# sweepable keys. An unknown key is refused: it would be written into the TOML and
# ignored by the Julia side, so the arm would be the baseline under another name.
RAMP_WARMUP_NOTED=0
LAYER_MISMATCH_WARNED=0
parse_optimizer_arm() {   # <spec>
    local spec="$1"
    POINT_OPTIMIZER_TAG="${spec%%:*}"
    POINT_LEARNING_RATE="$BASE_LEARNING_RATE"
    POINT_ADAM_EPS="$BASE_ADAM_EPS"
    POINT_WEIGHT_DECAY="$BASE_WEIGHT_DECAY"
    POINT_INITIAL_CONDITIONS_SCALE="$BASE_INITIAL_CONDITIONS_SCALE"
    POINT_WARMUP_LAYERS="$BASE_WARMUP_LAYERS"
    POINT_LOSS_LAYER_SELECTION="$LOSS_LAYER_SELECTION"
    POINT_COMMIT_LAYER_RULE="$COMMIT_LAYER_RULE"
    POINT_LOSS_LAYER_RAMP_SHARPNESS="$BASE_LOSS_LAYER_RAMP_SHARPNESS"
    POINT_BASE_LOSS="$BASE_BASE_LOSS"
    POINT_TRAINING_SAMPLES="$BASE_TRAINING_SAMPLES"
    POINT_FAILURE_WEIGHT_BOUNDARY="$BASE_FAILURE_WEIGHT_BOUNDARY"
    if [ -z "$POINT_OPTIMIZER_TAG" ]; then
        echo "optimizer arm '$spec' has an empty tag; the tag enters every filename." >&2
        exit 1
    fi
    if [ "$spec" != "$POINT_OPTIMIZER_TAG" ]; then
        local remaining_overrides="${spec#*:}" one_override override_key override_value
        while [ -n "$remaining_overrides" ]; do
            one_override="${remaining_overrides%%,*}"
            if [ "$remaining_overrides" = "$one_override" ]; then
                remaining_overrides=""
            else
                remaining_overrides="${remaining_overrides#*,}"
            fi
            override_key="${one_override%%=*}"
            override_value="${one_override#*=}"
            if [ "$one_override" = "$override_key" ] || [ -z "$override_value" ]; then
                echo "optimizer arm '$spec': '$one_override' is not <key>=<value>." >&2
                exit 1
            fi
            case "$override_key" in
                learning_rate)             POINT_LEARNING_RATE="$override_value" ;;
                adam_eps)                  POINT_ADAM_EPS="$override_value" ;;
                weight_decay)              POINT_WEIGHT_DECAY="$override_value" ;;
                initial_conditions_scale)  POINT_INITIAL_CONDITIONS_SCALE="$override_value" ;;
                warmup_layers)             POINT_WARMUP_LAYERS="$override_value" ;;
                loss_layer_selection)      POINT_LOSS_LAYER_SELECTION="$override_value" ;;
                commit_layer_rule)         POINT_COMMIT_LAYER_RULE="$override_value" ;;
                loss_layer_ramp_sharpness) POINT_LOSS_LAYER_RAMP_SHARPNESS="$override_value" ;;
                base_loss)                 POINT_BASE_LOSS="$override_value" ;;
                training_samples)          POINT_TRAINING_SAMPLES="$override_value" ;;
                failure_weight_boundary)   POINT_FAILURE_WEIGHT_BOUNDARY="$override_value" ;;
                *)
                    echo "optimizer arm '$spec': '$override_key' is not a sweepable key." >&2
                    echo "  Allowed: learning_rate, adam_eps, weight_decay, initial_conditions_scale," >&2
                    echo "           warmup_layers, loss_layer_selection, commit_layer_rule," >&2
                    echo "           loss_layer_ramp_sharpness, base_loss, training_samples," >&2
                    echo "           failure_weight_boundary." >&2
                    exit 1
                    ;;
            esac
        done
    fi

    local inherited_key inherited_value
    for inherited_key in learning_rate adam_eps weight_decay initial_conditions_scale warmup_layers; do
        case "$inherited_key" in
            learning_rate)            inherited_value="$POINT_LEARNING_RATE" ;;
            adam_eps)                 inherited_value="$POINT_ADAM_EPS" ;;
            weight_decay)             inherited_value="$POINT_WEIGHT_DECAY" ;;
            initial_conditions_scale) inherited_value="$POINT_INITIAL_CONDITIONS_SCALE" ;;
            warmup_layers)            inherited_value="$POINT_WARMUP_LAYERS" ;;
        esac
        if [ -z "$inherited_value" ]; then
            echo "optimizer arm '$spec': $inherited_key is set neither by the arm nor by $MODELS_DIR/$BASE_HP." >&2
            exit 1
        fi
    done
    case "$POINT_LOSS_LAYER_SELECTION" in
        softmin|last|mean|ramp) ;;
        *) echo "optimizer arm '$spec': loss_layer_selection '$POINT_LOSS_LAYER_SELECTION' must be softmin, last, mean or ramp." >&2; exit 1 ;;
    esac
    case "$POINT_COMMIT_LAYER_RULE" in
        first|last) ;;
        *) echo "optimizer arm '$spec': commit_layer_rule '$POINT_COMMIT_LAYER_RULE' must be first or last." >&2; exit 1 ;;
    esac
    case "$POINT_BASE_LOSS" in
        sin_residue|smooth_loss) ;;
        *) echo "optimizer arm '$spec': base_loss '$POINT_BASE_LOSS' must be sin_residue or smooth_loss." >&2; exit 1 ;;
    esac
    # The training-set keys: a non-negative integer and a non-negative number. A
    # weighted draw has no natural size, so the boundary needs a set size; the Julia
    # side refuses the same combination, but refuse here, before a queue wait.
    if ! echo "$POINT_TRAINING_SAMPLES" | grep -qE '^[0-9]+$'; then
        echo "optimizer arm '$spec': training_samples '$POINT_TRAINING_SAMPLES' must be a non-negative integer (0 = the whole file)." >&2
        exit 1
    fi
    if ! echo "$POINT_FAILURE_WEIGHT_BOUNDARY" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "optimizer arm '$spec': failure_weight_boundary '$POINT_FAILURE_WEIGHT_BOUNDARY' must be a non-negative number (0 = no weighting)." >&2
        exit 1
    fi
    # Zero in any spelling (0, 0.0, 00) has no digit left once dots and zeros go.
    if [ -n "$(echo "$POINT_FAILURE_WEIGHT_BOUNDARY" | sed 's/[.0]//g')" ] && [ "$POINT_TRAINING_SAMPLES" -eq 0 ]; then
        echo "optimizer arm '$spec': failure_weight_boundary = $POINT_FAILURE_WEIGHT_BOUNDARY needs training_samples > 0." >&2
        exit 1
    fi
    # A pairing that trains one layer and scores another is almost always a
    # mistake; warn rather than refuse, since someone may want it deliberately.
    if [ "$POINT_LOSS_LAYER_SELECTION" = "last" ] && [ "$POINT_COMMIT_LAYER_RULE" = "first" ] && [ "$LAYER_MISMATCH_WARNED" -eq 0 ]; then
        echo "[hp_sweep] WARNING: loss_layer_selection=last with commit_layer_rule=first trains" >&2
        echo "           the final layer and commits to the first clearing one -- the mismatch" >&2
        echo "           these keys exist to remove." >&2
        LAYER_MISMATCH_WARNED=1
    fi
    # The ramp starts from zero at warmup_layers, so the warmup is part of the loss
    # definition there. 0 lets the BP transient (layers 1-5: 35, 7.5, 1.1, 0.24,
    # 0.12 on SOLVED batches) into the weighted mean. Noted, since 5 was the
    # measured knee (and 0 is what the 72q_BB_cycles_1_sweep models used).
    if [ "$POINT_LOSS_LAYER_SELECTION" = "ramp" ] && [ "${POINT_WARMUP_LAYERS:-0}" -lt 3 ] && [ "$RAMP_WARMUP_NOTED" -eq 0 ]; then
        echo "[hp_sweep] note: loss_layer_selection=ramp with warmup_layers=${POINT_WARMUP_LAYERS:-0}:" >&2
        echo "           the ramp weights the BP transient (layers 1-5) too. Measured knee is 5." >&2
        RAMP_WARMUP_NOTED=1
    fi
}

# One point: its TOML (into POINT_TOML_DIR), its training command, its test
# command(s), and the weights file its training writes.
emit_point() {   # <training key> <seed> <use_cer> <check-node arm> <optimizer arm>
    local key="$1" seed="$2" use_cer="$3" check_node_spec="${4:-tanh}" optimizer_spec="${5:-base}"
    local arm="cer" require="true"
    if [ "$use_cer" = "false" ]; then
        arm="nocer"
        require="false"
    fi
    parse_check_node_arm "$check_node_spec" "$key" "$use_cer"
    parse_optimizer_arm "$optimizer_spec"

    # The run tag: _<arm>_<optimizer tag>, e.g. _cer_rampfirst -- or _<arm> alone
    # for a lone, literal "base" arm with no overrides (a sweep that does not use
    # this axis). A non-tanh check node adds its token after the arm
    # (_cer_enr_<opt>, _cer_bethe_<opt>), so it never overwrites the tanh arm's
    # files. Nothing else reaches the filename (2026-10-09): the settings a sweep
    # holds fixed are in the TOML. So two points that would share a filename are
    # refused below, since no other part of the name could tell them apart.
    local run_tag="_${arm}${POINT_CHECK_NODE_TAG}"
    if ! { [ "$N_OPTIMIZER_ARMS" -eq 1 ] && [ "$optimizer_spec" = "base" ]; }; then
        run_tag="_${arm}${POINT_CHECK_NODE_TAG}_${POINT_OPTIMIZER_TAG}"
    fi
    # The TOML keeps the hyperparams_hp_ prefix: cleanup.py and the local runner
    # find sweep TOMLs by it.
    local hp="hyperparams_hp${run_tag}_$(tag_of "$key")_seed${seed}.toml"
    if grep -qxF "$hp" "$EMITTED_TOML_NAMES"; then
        echo "two points would share $hp -- only the arm and the optimizer tag reach the" >&2
        echo "  filename, so two arms that differ in anything else need different tags." >&2
        exit 1
    fi
    echo "$hp" >> "$EMITTED_TOML_NAMES"
    # The weights file the training writes (src/train.jl's name). The code puts
    # _no_cer between the training source and the run tag on a no-CER arm.
    local cer_infix=""
    if [ "$use_cer" = "false" ]; then
        cer_infix="_no_cer"
    fi
    echo "neuralbp_weights_nlayers_${NLAYERS}_epochs_${N_EPOCHS}_trained_using_train_${key}${cer_infix}${run_tag}_seed_${seed}.json" >> "$EXPECTED_MODELS"

    grep -vE "$GENERATED_KEYS_PATTERN" "$MODELS_DIR/$BASE_HP" > "$POINT_TOML_DIR/$hp"
    {
        echo ""
        echo "# generated by sweep_hyperparams.sh $TS"
        echo "retrain = true"
        # The training job runs on a GPU node and needs the hand-written gradient
        # (checked in the base TOML; stated here so every point says so).
        echo "manual_backprop = true"
        echo "run_tag = \"${run_tag}\""
        echo "use_CER = $use_cer"
        echo "seed = $seed"
        echo "single_qubit_rescale = ${RESCALE}"
        echo "require_correlations = ${require}"
        echo "check_node = \"${POINT_CHECK_NODE}\""
        echo "enriched_table = \"${POINT_ENRICHED_TABLE}\""
        echo "coupling_scale_init = ${POINT_COUPLING_SCALE_INIT}"
        echo "coupling_scale_learnable = ${POINT_COUPLING_SCALE_LEARNABLE}"
        echo "loss_layer_selection = \"${POINT_LOSS_LAYER_SELECTION}\""
        echo "commit_layer_rule = \"${POINT_COMMIT_LAYER_RULE}\""
        echo "loss_layer_ramp_sharpness = ${POINT_LOSS_LAYER_RAMP_SHARPNESS}"
        echo "base_loss = \"${POINT_BASE_LOSS}\""
        echo "training_samples = ${POINT_TRAINING_SAMPLES}"
        echo "failure_weight_boundary = ${POINT_FAILURE_WEIGHT_BOUNDARY}"
        echo ""
        echo "# optimizer arm \"${POINT_OPTIMIZER_TAG}\": stated explicitly on every point,"
        echo "# overridden or not, so one filename never means two settings."
        echo "learning_rate = ${POINT_LEARNING_RATE}"
        echo "adam_eps = ${POINT_ADAM_EPS}"
        echo "weight_decay = ${POINT_WEIGHT_DECAY}"
        echo "initial_conditions_scale = ${POINT_INITIAL_CONDITIONS_SCALE}"
        echo "warmup_layers = ${POINT_WARMUP_LAYERS}"
    } >> "$POINT_TOML_DIR/$hp"

    # Training reads the training file and the CER file of its own key.
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${key}.txt --quiet true \
--isdebug true --train train_${key}.txt" >> "$TRAIN_CMDS"
    # Testing: one command per test set of this model (test_keys_of). --train stays
    # at the training key -- it names the weights file that retrain = false loads --
    # while --test and --cer_data name the TEST key, so the decoder is always handed
    # the priors of the data in front of it. The results filename carries both
    # keys, so no two tests share one.
    local test_key
    for test_key in $(test_keys_of "$key"); do
        echo "julia --project=\"./../\" --heap-size-hint=$HEAP neural_bp_experiments.jl \
--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS \
--hyperparams $hp --cer_data correlated_weights_${test_key}.txt --quiet true \
--diagnose true --train train_${key}.txt --test test_${test_key}.txt" >> "$TEST_CMDS"
    done
}

# Every point of the sweep, in the order the array split interleaves. The check
# node is a forward-pass axis, so it crosses every CER cell; the no-CER baseline
# has no couplings to enrich and is emitted once per optimizer arm, with the
# standard rule. ref_datasets (rarely used) get the CER tanh arm and the baseline
# only, on the BASELINE optimizer arm: they are a reference point against earlier
# sweeps, and crossing them with a new axis would change what they refer to.
emit_all_points() {
    local key seed optimizer check_node_arm
    for key in $DATASETS; do
        for seed in $SEEDS; do
            for optimizer in $OPTIMIZER_ARMS; do
                for check_node_arm in $CHECK_NODE_ARMS; do
                    emit_point "$key" "$seed" true "$check_node_arm" "$optimizer"
                done
                if [ "$INCLUDE_NOCER" = "true" ]; then
                    emit_point "$key" "$seed" false "tanh" "$optimizer"
                fi
            done
        done
    done
    for key in $REF_DATASETS; do
        for seed in $SEEDS; do
            emit_point "$key" "$seed" true "tanh" "base"
            if [ "$INCLUDE_NOCER" = "true" ]; then
                emit_point "$key" "$seed" false "tanh" "base"
            fi
        done
    done
    N_POINTS="$(wc -l < "$TRAIN_CMDS" | tr -d ' ')"
    N_TEST_POINTS="$(wc -l < "$TEST_CMDS" | tr -d ' ')"
    # One weights file per training point. emit_point refuses two points with one
    # TOML name, and the weights name is made of the same parts, so a mismatch here
    # is a bug in this file, not in the settings.
    local n_expected_models
    n_expected_models="$(wc -l < "$EXPECTED_MODELS" | tr -d ' ')"
    if [ "$n_expected_models" -ne "$N_POINTS" ]; then
        echo "internal error: $n_expected_models expected weights file(s) for $N_POINTS training point(s)." >&2
        exit 1
    fi
}

# Reads the written test commands back and refuses the sweep if any of them breaks
# the pairing test_keys_of defines. For every test command:
#   - its --hyperparams and --train are those of a training command of this sweep
#     (so retrain = false loads a model this sweep trains, from that training file);
#   - its --cer_data is the CER file of its --test key;
#   - with test_keys = []: its --test key IS its --train key;
#     with test_keys listed: its --test key is one of them.
# And every training command has at least one test.
check_test_pairing() {
    local problems
    problems="$(awk -v matched="$([ -z "$(echo $TEST_KEYS)" ] && echo 1 || echo 0)" \
                    -v listed=" $(echo $TEST_KEYS) " '
        function argument(name,    i) {
            for (i = 1; i < NF; i++) { if ($i == name) { return $(i + 1) } }
            return ""
        }
        function key_of(file, prefix) {
            sub("^" prefix, "", file); sub(/\.txt$/, "", file); return file
        }
        FNR == 1 { file_index++ }
        file_index == 1 {
            trained[argument("--hyperparams") " " argument("--train")] = 1
            next
        }
        {
            hyperparams = argument("--hyperparams")
            train_key = key_of(argument("--train"), "train_")
            test_key  = key_of(argument("--test"), "test_")
            cer_key   = key_of(argument("--cer_data"), "correlated_weights_")
            if (!((hyperparams " " argument("--train")) in trained))
                print "    test " FNR ": " hyperparams " with " argument("--train") " is not a training point of this sweep"
            if (cer_key != test_key)
                print "    test " FNR ": tested on " test_key " with the CER file of " cer_key
            if (matched && test_key != train_key)
                print "    test " FNR ": trained on " train_key " but tested on " test_key " (test_keys = [])"
            if (!matched && index(listed, " " test_key " ") == 0)
                print "    test " FNR ": tested on " test_key ", which is not in test_keys"
            tested[hyperparams " " argument("--train")] = 1
        }
        END {
            for (point in trained) { if (!(point in tested)) print "    no test for " point }
        }' "$TRAIN_CMDS" "$TEST_CMDS")"
    if [ -n "$problems" ]; then
        echo "internal error: the test commands do not pair with the training points as test_keys says:" >&2
        echo "$problems" >&2
        exit 1
    fi
}

# What each training key's models are scored on, counted from the written commands.
print_test_plan() {
    if [ -z "$(echo $TEST_KEYS)" ]; then
        echo "[hp_sweep] test sets: test_keys = [], so each model is scored on the test set of its own"
        echo "           training key only:"
    else
        echo "[hp_sweep] test sets: every model is scored on every key in test_keys:"
    fi
    awk '
        function argument(name,    i) {
            for (i = 1; i < NF; i++) { if ($i == name) { return $(i + 1) } }
            return ""
        }
        {
            pair = argument("--train") " -> " argument("--test")
            if (!(pair in count)) { order[++n] = pair }
            count[pair]++
        }
        END { for (i = 1; i <= n; i++) printf "             %s   (%d test(s))\n", order[i], count[order[i]] }
    ' "$TEST_CMDS"
}

# A point whose weights file is already in models/ is trained again (every point
# TOML says retrain = true), and the old file and its results are replaced at
# stage-out. Re-running a sweep does that on purpose; a new sweep whose tags repeat
# an old one's does it by accident, so say which files.
warn_existing_models() {
    local n_existing=0 expected_model
    while IFS= read -r expected_model; do
        if [ -f "$MODELS_DIR/$expected_model" ]; then
            n_existing=$(( n_existing + 1 ))
            if [ "$n_existing" -eq 1 ]; then
                echo "[hp_sweep] WARNING: model(s) of this sweep already exist in $MODELS_DIR;" >&2
                echo "           the training job will retrain them and replace them, with their results:" >&2
            fi
            if [ "$n_existing" -le 5 ]; then
                echo "             $expected_model" >&2
            fi
        fi
    done < "$EXPECTED_MODELS"
    if [ "$n_existing" -gt 0 ]; then
        echo "           ($n_existing of $N_POINTS.) To keep them, give the optimizer arms new tags." >&2
    fi
}

#!/bin/bash
# Run a NARROWED version of the sweep on an Apple Silicon Mac, no cluster.
#
#   bash misc/run_local.sh                  build datasets, train, then test
#   bash misc/run_local.sh --dry-run        build datasets + command lists, run no julia
#   bash misc/run_local.sh --train-only     stop after training
#   bash misc/run_local.sh --test-only      skip training (weights must exist)
#   bash misc/run_local.sh --jobs 2         concurrent training processes (default 4)
#   bash misc/run_local.sh --samples 500000 samples per dataset (default 200000)
#   bash misc/run_local.sh --layers 50      unrolled BP layers (default 90; 50 measured worse)
#   bash misc/run_local.sh --warmup 10      layers hidden from the loss, design A /
#                                           softmin arms only (default 0)
#   bash misc/run_local.sh --design ramp    optimizer arms: pair (default: design A +
#                                           softmin/first), ramp (ramp loss alone),
#                                           all (the three)
#   bash misc/run_local.sh --ramp-warmup 5 --ramp-sharpness 3
#                                           the ramp's own warmup and k (defaults)
#   bash misc/run_local.sh --design baseloss --layers 50
#                                           the residue A/B: {softmin, ramp} x
#                                           {sin_residue, smooth_loss} x {no-CER, CER}
#                                           x 2 seeds = 16 trainings / 64 tests, both
#                                           losses committing at the LAST layer
#   bash misc/run_local.sh --design baseloss --base-loss smooth --commit first
#                                           restrict to one residue (sin|smooth|both)
#                                           and/or change the commit rule
#   bash misc/run_local.sh --design sampling --training-samples 10000 --failure-weight-boundary 5
#                                           the training-set A/B: softmin + first on
#                                           (a) a uniform set of N samples and (b) a
#                                           set of N whose error weights follow
#                                           Poisson(lambda), x {no-CER, CER} x 2 seeds
#                                           = 8 trainings / 32 tests. The whole-file
#                                           arm is the histpair point of --design all
#                                           (deterministic, same filenames). With any
#                                           other --design the two values, when given,
#                                           are applied to every arm.
#   bash misc/run_local.sh --design icscale --layers 50 --jobs 8
#                                           ramp + last-layer commit + smooth_loss, with
#                                           initial_conditions_scale swept over --scales
#                                           (default 0.1,0.3), x {no-CER, CER} x 2 seeds
#                                           = 8 trainings / 32 tests. Everything else
#                                           (warmup, k, training_samples,
#                                           failure_weight_boundary, optimizer) is the
#                                           baseline TOML's value. Run tags:
#                                           _nocer_init_0p1, _cer_init_0p3, ...
#
# WARMUP ONLY AFFECTS THE SOFTMIN ARM. Under loss_layer_selection = "last" the
# loss reads losses_per_layer[end] and nothing else, so warmup_layers is
# irrelevant to design-A training -- `loss.jl:239` says so outright. Setting it
# to 0 is the fix for the SOFTMIN arm, which was scoring layers 11-90 while
# first-to-clear commits at layer 1.2-3.0.
#
# LAYERS is in every weights and results filename (`nlayers_<N>`), so two layer
# counts coexist without collision -- and the globs below are layer-aware so a
# mixed run directory cannot be miscounted.
#
# WHAT THIS RUNS, and why it is not the cluster sweep
# ---------------------------------------------------
# With --design pair (the default): 4 arms x 2 seeds = 8 trainings, 32 tests. The
# arms are no-CER and CER-enriched, each under both layer designs:
#
#            | design A (last/last) | historical (softmin/first)
#   no-CER   |          x           |            x
#   enriched |          x           |            x
#
# --design ramp swaps in the ramp loss alone (4 trainings, 16 tests);
# --design all runs the three optimizer arms together (12 trainings, 48 tests).
#
# The cluster sweep is 8 arms x 5 seeds = 40 trainings / 160 tests. Dropped here:
# the CER-tanh arm and the enriched+step-schedule arm, and three seeds.
#
# Consequences, stated plainly:
#   - Without the CER-tanh arm, "no-CER vs enriched" conflates the PRIORS with the
#     CHECK NODE. Earlier measurements put those at -20% and -10% separately; this
#     run cannot separate them again.
#   - Two seeds give a difference, not an error bar. Seed sd was ~10%, so a layer-
#     design gap smaller than that is not resolvable here.
# Treat the output as a direction to iterate on, not as the verdict.
#
# WHY TRAINING IGNORES THE GPU
# ----------------------------
# Enzyme cannot differentiate through device-array allocation, so the training
# forward pass is always the CPU one (src/train.jl:20). Metal does nothing for it.
# Training therefore runs USE_GPU=0, $JOBS processes wide; only testing uses Metal.
#
# WHY 4 CONCURRENT AND NOT 8
# --------------------------
# Each training process holds the sample matrix (n_samples x 72 x 8 bytes) plus
# several times that in readdlm intermediates. At 200k samples that is ~0.5 GB
# each; at the full 1e6 it is ~2 GB each, and 8 of those exceeds 16 GB of unified
# memory before macOS takes its share. An M-series chip also has 4 performance
# cores and 4 efficiency cores, and the E-cores run this roughly a third as fast,
# so 8 jobs at once finishes LATER than two waves of 4 as well as risking swap.
#
# WHY THE SAMPLE COUNT IS CUT
# ---------------------------
# Testing, not training, is the cost here: on an A100 a test point is ~129 s, and
# an M-series GPU has ~1/15th the memory bandwidth for a workload that is bound by
# it. 32 tests at 1e6 samples is most of a day. At 200k, failure counts land near
# 480-630 per arm (Poisson error ~4%), which still separates a 20% effect cleanly.
# Training is unaffected either way: with online_training it draws
# n_epochs x n_gradient_updates_per_epoch x batch_size = 5 x 500 x 20 = 50,000
# samples, so a 200k pool is already 4x oversubscribed.
#
# The cut datasets get their own `_<N>` key, so every weights file and results CSV
# is named differently from the cluster run and nothing collides.
set -eu

JOBS=4
SAMPLES=200000
LAYERS=90
WARMUP=0
# Which optimizer arms to run. "pair" is design A + the historical softmin pair,
# "ramp" is the ramp loss alone, "all" is the three together. The ramp carries
# ITS OWN warmup (RAMP_WARMUP, default 5), because for the ramp the warmup is
# part of the loss definition -- it is where the weights start from zero -- and
# the measured knee is 5. --warmup still governs the other two arms.
DESIGN="pair"
DESIGN_GIVEN=0
RAMP_WARMUP=5
RAMP_SHARPNESS=3.0
# For --design baseloss: which commit rule both loss arms use (last, as specified
# -- it isolates the LOSS comparison from the readout), and which residues to run.
BASELOSS_COMMIT="last"
BASELOSS_RESIDUES="both"
# The training set (src/sample_selection.jl): its size (0 = the whole file) and
# the Poisson centre of its error-weight distribution (0 = the file's own).
TRAINING_SAMPLES=0
FAILURE_WEIGHT_BOUNDARY=0
# For --design icscale: the initial_conditions_scale values, comma-separated.
SCALES="0.1,0.3"
DRY_RUN=0
DO_TRAIN=1
DO_TEST=1
GPU_MEM="6G"

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs)       JOBS="${2:-}"; shift ;;
        --jobs=*)     JOBS="${1#*=}" ;;
        --layers)     LAYERS="${2:-}"; shift ;;
        --layers=*)   LAYERS="${1#*=}" ;;
        --warmup)     WARMUP="${2:-}"; shift ;;
        --warmup=*)   WARMUP="${1#*=}" ;;
        --design)     DESIGN="${2:-}"; DESIGN_GIVEN=1; shift ;;
        --design=*)   DESIGN="${1#*=}"; DESIGN_GIVEN=1 ;;
        --ramp-warmup)    RAMP_WARMUP="${2:-}"; shift ;;
        --ramp-warmup=*)  RAMP_WARMUP="${1#*=}" ;;
        --ramp-sharpness)   RAMP_SHARPNESS="${2:-}"; shift ;;
        --ramp-sharpness=*) RAMP_SHARPNESS="${1#*=}" ;;
        --commit)      BASELOSS_COMMIT="${2:-}"; shift ;;
        --commit=*)    BASELOSS_COMMIT="${1#*=}" ;;
        --base-loss)   BASELOSS_RESIDUES="${2:-}"; shift ;;
        --base-loss=*) BASELOSS_RESIDUES="${1#*=}" ;;
        --training-samples)   TRAINING_SAMPLES="${2:-}"; shift ;;
        --training-samples=*) TRAINING_SAMPLES="${1#*=}" ;;
        --failure-weight-boundary)   FAILURE_WEIGHT_BOUNDARY="${2:-}"; shift ;;
        --failure-weight-boundary=*) FAILURE_WEIGHT_BOUNDARY="${1#*=}" ;;
        --scales)     SCALES="${2:-}"; shift ;;
        --scales=*)   SCALES="${1#*=}" ;;
        --samples)    SAMPLES="${2:-}"; shift ;;
        --samples=*)  SAMPLES="${1#*=}" ;;
        --gpu-memory) GPU_MEM="${2:-}"; shift ;;
        --gpu-memory=*) GPU_MEM="${1#*=}" ;;
        --dry-run)    DRY_RUN=1 ;;
        --train-only) DO_TEST=0 ;;
        --test-only)  DO_TRAIN=0 ;;
        --help|-h)    awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_trainable_alpha"
DATA="$EXPTS_DIR/../data/$CODENAME"
TS="$(date +%Y-%m-%d_%H-%M-%S)"

SRC_TRAIN_KEY="p_0.0015_sig_0.0015_s_1"
SRC_TEST_KEYS="p_0.0015_sig_0.0015_s_1 p_0.0007_sig_0.001_s_1 p_0.0005_sig_0.001_s_1 p_0.0005_sig_0.0005_s_1"
SUFFIX="_${SAMPLES}"

# Only TRAINING needs GNU parallel (it runs $JOBS wide). Testing is a plain
# sequential loop, so --test-only works without it.
if [ "$DO_TRAIN" -eq 1 ] && ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel is not installed (training runs $JOBS wide). brew install parallel" >&2
    echo "  --test-only does not need it." >&2
    exit 1
fi
if [ ! -d "$DATA" ]; then
    echo "no run directory: $DATA" >&2
    exit 1
fi

# The training-set overrides, appended to every arm of the other presets when
# given. The generator validates the values.
SELECTION_OVERRIDES=""
if [ "$TRAINING_SAMPLES" != "0" ]; then
    SELECTION_OVERRIDES="$SELECTION_OVERRIDES,training_samples=$TRAINING_SAMPLES"
fi
if [ "$FAILURE_WEIGHT_BOUNDARY" != "0" ]; then
    SELECTION_OVERRIDES="$SELECTION_OVERRIDES,failure_weight_boundary=$FAILURE_WEIGHT_BOUNDARY"
fi

# The optimizer arms, as the generator's spec strings. Every arm states its own
# warmup_layers (tag-neutral) so the run never depends on the base TOML's value.
ARM_DESIGN_A="designA:loss_layer_selection=last,commit_layer_rule=last,initial_conditions_scale=0.3,warmup_layers=$WARMUP$SELECTION_OVERRIDES"
ARM_HISTPAIR="histpair:loss_layer_selection=softmin,commit_layer_rule=first,initial_conditions_scale=0.3,warmup_layers=$WARMUP$SELECTION_OVERRIDES"
ARM_RAMP="ramp:loss_layer_selection=ramp,commit_layer_rule=last,initial_conditions_scale=0.3,warmup_layers=$RAMP_WARMUP,loss_layer_ramp_sharpness=$RAMP_SHARPNESS$SELECTION_OVERRIDES"
# The training-set A/B, on the best arm (softmin + first-to-clear, warmup 0):
# N samples drawn uniformly, and N drawn with Poisson(lambda) error weights.
ARM_UNIFORM_SET="uniform:loss_layer_selection=softmin,commit_layer_rule=first,initial_conditions_scale=0.3,warmup_layers=$WARMUP,training_samples=$TRAINING_SAMPLES"
ARM_POISSON_SET="poisson:loss_layer_selection=softmin,commit_layer_rule=first,initial_conditions_scale=0.3,warmup_layers=$WARMUP,training_samples=$TRAINING_SAMPLES,failure_weight_boundary=$FAILURE_WEIGHT_BOUNDARY"
# The initial-conditions scan under ramp + last + smooth_loss: one arm per value
# in $SCALES, tagged init_<value> (`.` -> `p`). Everything else -- warmup_layers,
# loss_layer_ramp_sharpness, training_samples, failure_weight_boundary, the
# optimizer -- is whatever the baseline TOML says, unless the flags were given.
# The run tags are `_<arm>_<optimizer tag>`:
#     _nocer_init_0p1  _nocer_init_0p3  _cer_init_0p1  _cer_init_0p3
# (the code itself still adds `_no_cer` before the tag on the no-CER arms, and
# `nlayers_<N>` / `epochs_<n>` / the training file's name are the weights
# filename's own fields, not part of the tag).
ICSCALE_ARMS_TOML=""
N_ICSCALE_ARMS=0
for scale in $(echo "$SCALES" | tr ',' ' '); do
    if ! echo "$scale" | grep -qE '^[0-9]+(\.[0-9]+)?$'; then
        echo "--scales: '$scale' is not a number (got '$SCALES')" >&2
        exit 1
    fi
    scale_tag="init_$(echo "$scale" | sed 's/\./p/')"
    spec="${scale_tag}:loss_layer_selection=ramp,commit_layer_rule=last,base_loss=smooth_loss,initial_conditions_scale=$scale$SELECTION_OVERRIDES"
    if [ -n "$ICSCALE_ARMS_TOML" ]; then ICSCALE_ARMS_TOML="$ICSCALE_ARMS_TOML, "; fi
    ICSCALE_ARMS_TOML="$ICSCALE_ARMS_TOML\"$spec\""
    N_ICSCALE_ARMS=$(( N_ICSCALE_ARMS + 1 ))
done
# The residue A/B. Softmin at warmup 0 (the measured fix), ramp at its own warmup;
# both commit at $BASELOSS_COMMIT so the comparison is between LOSSES. The tag of
# each arm names loss, residue and commit so nothing collides with the pair/all
# presets even in the same run directory.
case "$BASELOSS_COMMIT" in first|last) ;; *) echo "--commit must be first or last (got '$BASELOSS_COMMIT')" >&2; exit 1 ;; esac
BL_RESIDUES=""
case "$BASELOSS_RESIDUES" in
    both)   BL_RESIDUES="sin_residue smooth_loss" ;;
    sin)    BL_RESIDUES="sin_residue" ;;
    smooth) BL_RESIDUES="smooth_loss" ;;
    *) echo "--base-loss must be both, sin or smooth (got '$BASELOSS_RESIDUES')" >&2; exit 1 ;;
esac
BASELOSS_ARMS_TOML=""
for residue in $BL_RESIDUES; do
    short="sin"; if [ "$residue" = "smooth_loss" ]; then short="smooth"; fi
    for loss in softmin ramp; do
        if [ "$loss" = "softmin" ]; then
            spec="sm${short}${BASELOSS_COMMIT}:loss_layer_selection=softmin,commit_layer_rule=$BASELOSS_COMMIT,initial_conditions_scale=0.3,warmup_layers=$WARMUP,base_loss=$residue"
        else
            spec="rp${short}${BASELOSS_COMMIT}:loss_layer_selection=ramp,commit_layer_rule=$BASELOSS_COMMIT,initial_conditions_scale=0.3,warmup_layers=$RAMP_WARMUP,loss_layer_ramp_sharpness=$RAMP_SHARPNESS,base_loss=$residue"
        fi
        if [ -n "$BASELOSS_ARMS_TOML" ]; then BASELOSS_ARMS_TOML="$BASELOSS_ARMS_TOML, "; fi
        BASELOSS_ARMS_TOML="$BASELOSS_ARMS_TOML\"$spec\""
    done
done
N_BASELOSS_ARMS=$(( 2 * $(echo $BL_RESIDUES | wc -w | tr -d ' ') ))
OPTIMIZER_ARMS_TOML=""
ARMS_LABEL=""
N_OPT_ARMS=0
case "$DESIGN" in
    pair) OPTIMIZER_ARMS_TOML="\"$ARM_DESIGN_A\", \"$ARM_HISTPAIR\""; ARMS_LABEL="designA + histpair"; N_OPT_ARMS=2 ;;
    ramp) OPTIMIZER_ARMS_TOML="\"$ARM_RAMP\"";                        ARMS_LABEL="ramp (k=$RAMP_SHARPNESS, warmup $RAMP_WARMUP) + last-layer commit"; N_OPT_ARMS=1 ;;
    all)  OPTIMIZER_ARMS_TOML="\"$ARM_DESIGN_A\", \"$ARM_HISTPAIR\", \"$ARM_RAMP\""; ARMS_LABEL="designA + histpair + ramp (k=$RAMP_SHARPNESS, warmup $RAMP_WARMUP)"; N_OPT_ARMS=3 ;;
    baseloss) OPTIMIZER_ARMS_TOML="$BASELOSS_ARMS_TOML"; ARMS_LABEL="{softmin, ramp} x {$BL_RESIDUES}, commit=$BASELOSS_COMMIT"; N_OPT_ARMS=$N_BASELOSS_ARMS ;;
    sampling)
        if [ "$TRAINING_SAMPLES" = "0" ] || [ "$FAILURE_WEIGHT_BOUNDARY" = "0" ]; then
            echo "--design sampling needs --training-samples <N> and --failure-weight-boundary <lambda>, both > 0" >&2
            exit 1
        fi
        OPTIMIZER_ARMS_TOML="\"$ARM_UNIFORM_SET\", \"$ARM_POISSON_SET\""
        ARMS_LABEL="softmin/first on $TRAINING_SAMPLES samples: uniform vs Poisson($FAILURE_WEIGHT_BOUNDARY) error weights"
        N_OPT_ARMS=2 ;;
    icscale)
        OPTIMIZER_ARMS_TOML="$ICSCALE_ARMS_TOML"
        ARMS_LABEL="ramp + last commit + smooth_loss, initial_conditions_scale in {$SCALES}; warmup and k from the baseline"
        N_OPT_ARMS=$N_ICSCALE_ARMS ;;
    *) echo "--design must be pair, ramp, all, baseloss, sampling or icscale (got '$DESIGN')" >&2; exit 1 ;;
esac
# 2 check-node arms (no-CER, CER-enriched) x optimizer arms x 2 seeds; x 4 test sets.
N_TRAIN_EXPECTED=$(( 2 * N_OPT_ARMS * 2 ))
N_TEST_EXPECTED=$(( N_TRAIN_EXPECTED * 4 ))

echo
echo "[run_local] $TS"
echo "  codename   -> $CODENAME"
echo "  samples    -> $SAMPLES per dataset (key suffix '$SUFFIX')"
if [ "$DESIGN" = "icscale" ]; then
    echo "  layers     -> $LAYERS unrolled   warmup from the baseline TOML"
else
    echo "  layers     -> $LAYERS unrolled   warmup $WARMUP (design A / softmin arms)"
fi
echo "  arms       -> no-CER + CER-enriched(0.42, learned), x $ARMS_LABEL"
if [ "$DESIGN_GIVEN" -eq 0 ]; then
    echo "                (--design not given: this is the default 'pair', design A + softmin."
    echo "                 The RAMP loss runs only with --design ramp or --design all;
                 the residue A/B is --design baseloss; the training-set A/B is
                 --design sampling.)"
fi
if [ "$DESIGN" != "sampling" ] && [ -n "$SELECTION_OVERRIDES" ]; then
    echo "  train set  -> every arm: training_samples=$TRAINING_SAMPLES failure_weight_boundary=$FAILURE_WEIGHT_BOUNDARY"
elif [ "$DESIGN" != "sampling" ]; then
    # Inherited from the baseline TOML (the generator restates both on every point).
    baseline_value() {   # <key>
        grep -E "^[[:space:]]*$1[[:space:]]*=" "$DATA/models/hyperparams_baseline.toml" | head -1 |
            sed -E 's/^[^=]*=[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//'
    }
    BASE_TS="$(baseline_value training_samples)"; BASE_FWB="$(baseline_value failure_weight_boundary)"
    echo "  train set  -> from the baseline TOML: training_samples=${BASE_TS:-0} failure_weight_boundary=${BASE_FWB:-0}"
    if [ "$DESIGN" = "icscale" ] && [ "${BASE_TS:-0}" = "0" ]; then
        echo "                (the whole training file, uniformly: set training_samples /"
        echo "                 failure_weight_boundary in the baseline or pass the flags)"
    fi
fi
echo "  seeds      -> 1 2        => $N_TRAIN_EXPECTED training point(s), $N_TEST_EXPECTED test point(s)"
echo "  training   -> CPU (USE_GPU=0), $JOBS concurrent, --heap-size-hint=2G"
echo "  testing    -> Metal (USE_GPU=1), sequential with live output, GPU_MEMORY=$GPU_MEM"

# ----------------------------------------------------------- cut datasets ---
# The files are 72 rows x n_samples SPACE-SEPARATED columns, so a sample is a
# COLUMN and `cut -f1-N -d' '` takes the first N samples. Same trick as the
# --local block in sweep_hyperparams.sh.
cut_dataset() {   # <subdir> <prefix> <src key>
    local subdir="$1" prefix="$2" src="$3"
    local in="$DATA/$subdir/${prefix}${src}.txt"
    local out="$DATA/$subdir/${prefix}${src}${SUFFIX}.txt"
    if [ ! -f "$in" ]; then
        echo "  missing input: $in" >&2
        return 1
    fi
    if [ -f "$out" ]; then
        echo "    $(basename "$out") exists, keeping it"
        return 0
    fi
    # Cut even under --dry-run: the generator VALIDATES that every dataset key it
    # is given exists on disk, so without these files there is no command list to
    # show. --dry-run means "run no julia", not "write nothing".
    cut -d' ' -f1-"$SAMPLES" "$in" > "$out"
    echo "    $(basename "$out")  ($(head -1 "$out" | wc -w | tr -d ' ') samples)"
}

echo
echo "[run_local] datasets"
cut_dataset training_data train_ "$SRC_TRAIN_KEY"
for k in $SRC_TEST_KEYS; do
    cut_dataset testing_data test_ "$k"
done
# Every dataset key needs its own correlated_weights file: emit_point writes
# `--cer_data correlated_weights_<test_key>.txt` per test point. The couplings do
# not depend on the sample count, so these are copies.
for k in $SRC_TRAIN_KEY $SRC_TEST_KEYS; do
    src="$DATA/correlated_weights/correlated_weights_${k}.txt"
    dst="$DATA/correlated_weights/correlated_weights_${k}${SUFFIX}.txt"
    if [ ! -f "$dst" ]; then
        cp -f "$src" "$dst"
    fi
done

TEST_KEYS_TOML=""
for k in $SRC_TEST_KEYS; do
    if [ -n "$TEST_KEYS_TOML" ]; then TEST_KEYS_TOML="$TEST_KEYS_TOML, "; fi
    TEST_KEYS_TOML="$TEST_KEYS_TOML\"${k}${SUFFIX}\""
done

# ------------------------------------------------------------ the settings ---
# Driven through sweep_hyperparams.sh --settings so that arm definitions, tagging
# and TOML generation stay in ONE place. The SLURM .sh files it also writes are
# ignored here; the .txt command lists are what this script runs.
#
# train_array_tasks = 1 and test_array_tasks = 1 because there is no array; the
# walltimes exist only to satisfy the generator's preflight and mean nothing
# locally, so they are set wide.
SETTINGS="$EXPTS_DIR/scripts/local_settings_${TS}.toml"
mkdir -p "$EXPTS_DIR/scripts"
cat > "$SETTINGS" <<EOF
workdir          = "$WORKDIR"
codename         = "$CODENAME"
datasets         = ["${SRC_TRAIN_KEY}${SUFFIX}"]
ref_datasets     = []
test_keys        = [$TEST_KEYS_TOML]
base_hyperparams = "hyperparams_baseline.toml"
n_hidden_layers  = $LAYERS
seeds            = [1, 2]
include_nocer    = true
single_qubit_rescale = 0.1
loss_layer_selection = "last"
commit_layer_rule    = "last"
# ONE check-node arm. The no-CER baseline is emitted separately (it has no
# couplings to enrich), so this yields 4 arms, not 2.
check_node_arms  = ["enriched:0.42:learn"]
# warmup_layers is stated on BOTH arms rather than inherited from the baseline
# TOML, so the run does not silently change when that file is edited. It is a
# tag-neutral override (only loss_layer_selection and commit_layer_rule reach the
# filename), so the four arm tags are unchanged.
optimizer_arms   = [$OPTIMIZER_ARMS_TOML]
cluster_name     = "narval"
account_cpu      = "def-jemerson"
account_gpu      = "def-jemerson_gpu"
email            = "dsannamo@uwaterloo.ca"
julia_module     = "julia/1.12.5"
cuda_module      = "cuda"
heap_size_hint   = "2G"
train_array_tasks = 1
train_cpus       = $JOBS
train_mem_per_cpu = "2G"
cpu_node_memory_mb = 16384
train_wall_time  = "24:00:00"
test_array_tasks = 1
gpu_type         = "a100"
gpu_request_style = "auto"
n_gpus_per_node  = 1
test_jobs        = 1
mem_per_gpu      = "$GPU_MEM"
# Stated explicitly rather than inferred from gpu_type: "a100" would infer 40 GB
# and the generator would print a GPU_MEMORY this machine does not have. Only the
# printed summary and the ignored SLURM script use it -- the real run exports
# GPU_MEMORY=$GPU_MEM below -- but a number that disagrees with reality in a log
# is how an hour gets lost later.
vram_per_gpu     = "${GPU_MEM%G}"
test_cpus        = $JOBS
test_wall_time   = "24:00:00"
seconds_per_test = 129
seconds_per_train_point = 6600
job_startup_seconds = 900
EOF
echo
echo "[run_local] settings -> $SETTINGS"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "[run_local] --dry-run: datasets and command lists ARE written (the"
    echo "            generator validates the dataset keys exist); no julia runs."
fi

( cd "$EXPTS_DIR" && bash sweep_hyperparams.sh --settings "$SETTINGS" --no-edit ) \
    | sed -n '/training point/,/commands/p' | sed 's/^/  /'

TRAIN_TXT=$(ls -t "$DATA/cluster"/hp_sweep_train_*.txt | head -1)
TEST_TXT=$(ls -t "$DATA/cluster"/hp_sweep_test_*.txt | head -1)
GEN_TS=$(basename "$TRAIN_TXT" | sed -E 's/hp_sweep_train_(.*)\.txt/\1/')

# $WORKDIR_RUNTIME is where the SLURM jobs stage the run directory to; locally
# the data never moves, so it is just the real workdir.
LOCAL_TRAIN="$DATA/cluster/local_train_${GEN_TS}.txt"
LOCAL_TEST="$DATA/cluster/local_test_${GEN_TS}.txt"
sed "s|\$WORKDIR_RUNTIME|$WORKDIR|g" "$TRAIN_TXT" > "$LOCAL_TRAIN"
sed "s|\$WORKDIR_RUNTIME|$WORKDIR|g" "$TEST_TXT"  > "$LOCAL_TEST"
echo
echo "[run_local] commands"
echo "  train -> $LOCAL_TRAIN  ($(wc -l < "$LOCAL_TRAIN" | tr -d ' ') point(s))"
echo "  test  -> $LOCAL_TEST  ($(wc -l < "$LOCAL_TEST" | tr -d ' ') point(s))"

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "  first training command:"; head -1 "$LOCAL_TRAIN" | sed 's/^/    /'
    echo "  first test command:";     head -1 "$LOCAL_TEST"  | sed 's/^/    /'
    echo
    echo "[run_local] nothing was run."
    exit 0
fi

# ---------------------------------------------------------------- training ---
if [ "$DO_TRAIN" -eq 1 ]; then
    TRAIN_JOBLOG="$DATA/cluster/local_train_${GEN_TS}.joblog"
    echo
    echo "[run_local] training: $(wc -l < "$LOCAL_TRAIN" | tr -d ' ') point(s), $JOBS at a time, CPU only"
    echo "  started $(date)"
    cd "$EXPTS_DIR"
    USE_GPU=0 JULIA_NUM_THREADS=1 parallel --jobs "$JOBS" --joblog "$TRAIN_JOBLOG" \
        --results "$DATA/cluster/local_train_${GEN_TS}" < "$LOCAL_TRAIN" &
    PARALLEL_PID=$!
    wait "$PARALLEL_PID" || true
    N_OK=$(awk 'NR>1 && $7 == 0' "$TRAIN_JOBLOG" | wc -l | tr -d ' ')
    N_ALL=$(wc -l < "$LOCAL_TRAIN" | tr -d ' ')
    echo "  finished $(date): $N_OK/$N_ALL point(s) exited 0"
    if [ "$N_OK" -ne "$N_ALL" ]; then
        awk -F'\t' 'NR>1 && $7 != 0 {print "    FAILED (exit " $7 "): " $9}' "$TRAIN_JOBLOG"
        echo "  training did not complete; not testing a partial model set." >&2
        exit 1
    fi
fi

if [ "$DO_TEST" -eq 0 ]; then
    echo
    echo "[run_local] --train-only: stopping here."
    exit 0
fi

# ----------------------------------------------------------------- testing ---
# Two steps the SLURM test job also does, and that are easy to miss by hand:
#  1. The generator writes retrain = true (the training job needs it). Left alone,
#     every test point RETRAINS from scratch instead of loading the weights.
#  2. neural_bp_experiments.jl SKIPS testing when the results CSV already exists,
#     and reports the stale numbers as if fresh.
echo
echo "[run_local] preparing to test"
N_FLIPPED=0
for f in "$DATA"/models/hyperparams_hp_*.toml; do
    if grep -qE '^[[:space:]]*retrain[[:space:]]*=[[:space:]]*true' "$f"; then
        sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true|\1false|' "$f" > "$f.tmp"
        mv "$f.tmp" "$f"
        N_FLIPPED=$((N_FLIPPED + 1))
    fi
done
echo "  retrain = false in $N_FLIPPED TOML(s)"

# NOTE THE WILDCARD BETWEEN THE KEY AND THE TAG. A no-CER arm carries `_no_cer`
# there -- `..._s_1_200000_no_cer_nocer_init_0p1_seed_1.json` -- so a glob that
# joins the key straight onto the tag matches only the CER arms and reports 4 of
# 8 after a perfectly good training run. (My stub built filenames without the
# infix, which is exactly why the first version of this check passed its own
# test.) Matched on `_seed_`, which every tag is followed by.
N_MODELS=$(ls "$DATA"/models/*nlayers_"${LAYERS}"_*"${SUFFIX}"*_seed_*.json 2>/dev/null | wc -l | tr -d ' ')
N_EXPECTED=$(wc -l < "$LOCAL_TRAIN" | tr -d ' ')
echo "  $N_MODELS of $N_EXPECTED trained model(s) present"
if [ "$N_MODELS" -lt "$N_EXPECTED" ]; then
    echo "  refusing to test a partial model set. Run without --test-only first." >&2
    exit 1
fi

N_STALE=$(ls "$DATA"/results/simulation_results_*nlayers_"${LAYERS}"_*"${SUFFIX}"*_seed_*.csv 2>/dev/null | wc -l | tr -d ' ')
rm -f "$DATA"/results/simulation_results_*nlayers_"${LAYERS}"_*"${SUFFIX}"*_seed_*.csv
echo "  cleared $N_STALE stale results file(s) for this key"

# A PLAIN SEQUENTIAL LOOP, not GNU parallel. parallel --jobs 1 would also run one
# at a time, but it captures each point's stdout into its --results tree, so you
# watch nothing happen for hours. Here julia writes straight to the terminal.
#
# The list is read on fd 3, not stdin: a command that reads stdin would otherwise
# swallow the remaining lines and the loop would stop after one point.
TEST_LOG="$DATA/cluster/local_test_${GEN_TS}.log"
N_ALL=$(wc -l < "$LOCAL_TEST" | tr -d ' ')
N_OK=0
N_BAD=0
POINT=0
START_ALL=$(date +%s)
printf '# point\texit\tseconds\tarm\ttest_key\tcommand\n' > "$TEST_LOG"
echo
echo "[run_local] testing: $N_ALL point(s), sequential, on Metal (GPU_MEMORY=$GPU_MEM)"
echo "  started $(date)"
cd "$EXPTS_DIR"
while IFS= read -r cmd <&3; do
    POINT=$((POINT + 1))
    ARM=$(printf '%s' "$cmd" | sed -E 's/.*--hyperparams hyperparams_hp_//; s/_p_0p0015[^ ]*_seed([0-9]+)\.toml.*/ seed \1/')
    TESTKEY=$(printf '%s' "$cmd" | sed -E 's/.*--test test_//; s/\.txt.*//')
    echo
    echo "----- [$POINT/$N_ALL] $ARM  ->  $TESTKEY"
    T0=$(date +%s)
    RC=0
    USE_GPU=1 GPU_MEMORY="$GPU_MEM" JULIA_NUM_THREADS=1 bash -c "$cmd" || RC=$?
    T1=$(date +%s)
    DT=$((T1 - T0))
    if [ "$RC" -eq 0 ]; then
        N_OK=$((N_OK + 1))
    else
        N_BAD=$((N_BAD + 1))
        echo "      FAILED (exit $RC)"
    fi
    # Mean-so-far ETA. The first point is the one worth watching: it tells you
    # what the whole run will cost on this machine, which no estimate from an
    # A100 timing can.
    ELAPSED=$((T1 - START_ALL))
    REMAIN=$(( ELAPSED * (N_ALL - POINT) / POINT ))
    printf '      %ds   |   %d ok, %d failed   |   elapsed %dh%02dm, ~%dh%02dm left\n' \
        "$DT" "$N_OK" "$N_BAD" \
        $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) \
        $((REMAIN / 3600)) $(((REMAIN % 3600) / 60))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$POINT" "$RC" "$DT" "$ARM" "$TESTKEY" "$cmd" >> "$TEST_LOG"
done 3< "$LOCAL_TEST"
echo
echo "  finished $(date): $N_OK/$N_ALL point(s) exited 0"
if [ "$N_BAD" -ne 0 ]; then
    echo "  failed point(s) — arm and test set, not the whole command line:"
    awk -F'\t' 'NR>1 && $2 != 0 {printf "    [%s] exit %s   %s  ->  %s\n", $1, $2, $4, $5}' "$TEST_LOG"
    echo "  full commands are in the log, column 6."
fi
echo "  per-point log -> $TEST_LOG"

echo
echo "[run_local] results -> $DATA/results/"
ls -1 "$DATA"/results/simulation_results_*nlayers_"${LAYERS}"_*"${SUFFIX}"*_seed_*.csv 2>/dev/null | wc -l | tr -d ' ' \
    | xargs -I{} echo "  {} results file(s) written"
echo
echo "  next: check the arms actually trained before reading the numbers —"
echo "    bash misc/check_training_health.sh $CODENAME"
echo "  rollbacks are per-arm and were 8/20 on the enriched arms last time, so"
echo "  confirm the completed-epoch count matches across arms before comparing."

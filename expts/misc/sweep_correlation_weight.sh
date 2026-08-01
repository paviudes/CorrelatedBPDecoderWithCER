#!/usr/bin/env bash
# ============================================================================
# sweep_correlation_weight.sh — sweep alpha4 (correlation_weight) x alpha3
#                               (sparsity_importance) for the CER ablation
# ============================================================================
# TWO PHASES, mirroring the main pipeline:
#
#   bash misc/sweep_correlation_weight.sh train   # CPU, trains one model per point
#   bash misc/sweep_correlation_weight.sh test    # GPU, tests those models
#
# Run `train` first, wait for it to finish, then run `test`. Both must be run
# FROM the expts/ directory, and the emitted script must be sbatch'ed from there
# too (all paths inside are relative, e.g. ./../data/...).
#
# WHY THE SPLIT: training is Enzyme reverse-mode AD, which runs on the CPU
# regardless of USE_GPU (Metal/CUDA array allocation is not differentiable), so
# a GPU allocation during training would idle. Testing is a pure forward pass
# over 10^6 samples and is ~25x faster on a GPU.
#
# WHY THE SWEEP: the correlation term
#     L_corr = -(1/(N*|C|)) sum_C J_ik sigma_i sigma_k
# is monotonically decreasing in sigma wherever J > 0, so it always argues for
# MORE predicted errors and has no internal counterweight. In the full log prior
# that counterweight is the single-qubit field term, whose role here is played by
# `sparsity_penalty` (alpha3). So alpha4 alone is not the knob — the alpha3/alpha4
# BALANCE is. This sweeps both.
#
# alpha4 = 0 is the key control: CER priors ON, correlation term OFF. Comparing
# it against the existing no-CER run separates the prior-scale effect from the
# correlation-term effect (the two things `use_CER = false` changes at once).
#
# WHAT IT DOES
#   1. writes one hyperparameters TOML per (alpha4, alpha3) point into
#      <workdir>/<codename>/models/, derived from a base TOML, with:
#        - correlation_weight  = "A,A,0.7,up"   (min == max => CONSTANT, no anneal)
#        - sparsity_importance = "B,B,0.8,up"   (constant)
#        - retrain  = true in TRAIN mode / false in TEST mode
#        - run_tag  = "_a4<A>_a3<B>"
#      `run_tag` is appended to the model AND results filenames, so sweep points
#      never overwrite each other (or the existing baseline runs).
#   2. writes a commands file (one julia invocation per point per p), and
#   3. writes a self-contained SLURM script that runs them with GNU parallel.
#      This deliberately bypasses submit.sh / batch_run.jl.
#
# In TEST mode it first checks that every point's trained weights file exists,
# and drops (with a warning) any point whose model is missing — so a partial
# training run can still be tested rather than silently retraining on the GPU.
#
# USAGE
#   bash misc/sweep_correlation_weight.sh train
#   bash misc/sweep_correlation_weight.sh test
#   bash misc/sweep_correlation_weight.sh train --pvals "0.0005" --jobs 8
#   bash misc/sweep_correlation_weight.sh test  --gpus 2 --jobs 2
# then:
#   sbatch ../data/<codename>/cluster/sweep_<mode>_<timestamp>.sh
#
# NOTE ON NORMALISATION: dividing the correlation term by n_edges (as the loss
# does) is exactly degenerate with alpha4 — if you switch the loss to `/ n_samples`
# alone, multiply these alpha4 values by ~1/n_edges (~540 for the 72q code).
# ============================================================================
set -euo pipefail

usage() { sed -n '2,60p' "$0"; }

MODE="${1:-}"
case "$MODE" in
    train|test) shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "ERROR: first argument must be 'train' or 'test'." >&2
       echo "  bash misc/sweep_correlation_weight.sh train   # CPU training" >&2
       echo "  bash misc/sweep_correlation_weight.sh test    # GPU testing" >&2
       exit 2 ;;
esac

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_20.toml"
PVALS="0.0005 0.002"
ALPHA4="0 0.01 0.1 1.0"        # correlation_weight  (0 = control: CER priors, no corr term)
ALPHA3="0.5 5.0"               # sparsity_importance (0.5 = current; 5.0 ~ |log((1-p)/p)| anchor)
NLAYERS=100
SEED=1
ACCOUNT="def-jemerson"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""                    # e.g. h100, a100, v100; empty = any

# Mode-dependent defaults (override with the flags below).
if [ "$MODE" = "train" ]; then
    JOBS=16                    # 16 points => one wave
    WALLTIME="6:00:00"         # ~6h per point
    MEM_PER_CPU="8G"
    USE_GPU="0"
    GPUS=0
else
    JOBS=1                     # GPU testing measured ~2-3 min/point => 1 slot is plenty
    WALLTIME="3:00:00"
    MEM_PER_CPU="8G"
    USE_GPU="1"
    GPUS=1
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)   WORKDIR="$2";     shift 2;;
        --codename)  CODENAME="$2";    shift 2;;
        --base_hp)   BASE_HP="$2";     shift 2;;
        --pvals)     PVALS="$2";       shift 2;;
        --alpha4)    ALPHA4="$2";      shift 2;;
        --alpha3)    ALPHA3="$2";      shift 2;;
        --nlayers)   NLAYERS="$2";     shift 2;;
        --seed)      SEED="$2";        shift 2;;
        --jobs)      JOBS="$2";        shift 2;;
        --gpus)      GPUS="$2";        shift 2;;
        --gpu_type)  GPU_TYPE="$2";    shift 2;;
        --walltime)  WALLTIME="$2";    shift 2;;
        --mem)       MEM_PER_CPU="$2"; shift 2;;
        --account)   ACCOUNT="$2";     shift 2;;
        --email)     EMAIL="$2";       shift 2;;
        --cuda_module) CUDA_MODULE="$2"; shift 2;;
        -h|--help)   usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
BASE_HP_PATH="$MODELS_DIR/$BASE_HP"

[ -d "$MODELS_DIR" ]   || { echo "no models dir: $MODELS_DIR (run this from expts/)" >&2; exit 1; }
[ -f "$BASE_HP_PATH" ] || { echo "no base hyperparams: $BASE_HP_PATH" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

# n_epochs and the CER tag are needed to reconstruct the weights filename that
# src/train.jl will write/look for (test-mode preflight).
N_EPOCHS=$(grep -E '^[[:space:]]*n_epochs[[:space:]]*=' "$BASE_HP_PATH" | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
if grep -qE '^[[:space:]]*use_CER[[:space:]]*=[[:space:]]*false' "$BASE_HP_PATH"; then
    CER_TAG="_no_cer"
else
    CER_TAG=""
fi
[ -n "$N_EPOCHS" ] || { echo "could not read n_epochs from $BASE_HP_PATH" >&2; exit 1; }

TS=$(date +%Y-%m-%d_%H-%M-%S)
COMMANDS="$CLUSTER_DIR/sweep_commands_${MODE}_${TS}.txt"
SLURM="$CLUSTER_DIR/sweep_${MODE}_${TS}.sh"
: > "$COMMANDS"

# Filename-safe token: 0.01 -> 0p01, 1.0 -> 1p0
tag_of() { echo "$1" | tr '.' 'p' | tr -d '+'; }

RETRAIN=$([ "$MODE" = "train" ] && echo true || echo false)

# --------------------------------------------------- generate hyperparams ---
n_points=0
n_dropped=0
dropped_list=""
for a4 in $ALPHA4; do
  for a3 in $ALPHA3; do
    t4=$(tag_of "$a4"); t3=$(tag_of "$a3")
    run_tag="_a4${t4}_a3${t3}"
    hp_name="hyperparams_sweep${run_tag}.toml"

    # Strip the keys we override (plus stale/legacy ones), then append ours.
    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|correlation_importance|correlation_syndrome_importance)[[:space:]]*=' \
        "$BASE_HP_PATH" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_correlation_weight.sh ($MODE, $TS) ----
# TRAIN mode sets retrain = true (each point trains its own model); TEST mode
# sets it false so the run LOADS that model instead of retraining on the GPU.
retrain = $RETRAIN

# Tag appended to model + results filenames so sweep points never collide.
run_tag = "${run_tag}"

# alpha4: overall weight on the Ising correlation term. min == max => CONSTANT
# (no annealing), so the sweep measures alpha4 itself rather than a schedule.
correlation_weight = "${a4},${a4},0.7,up"

# alpha3: sparsity weight — the counterweight to the correlation term's
# one-directional push toward more predicted errors. Also held CONSTANT.
sparsity_importance = "${a3},${a3},0.8,up"
EOF

    for p in $PVALS; do
      train_source="train_p_${p}_s_${SEED}"
      weights="$MODELS_DIR/neuralbp_weights_nlayers_${NLAYERS}_epochs_${N_EPOCHS}_trained_using_${train_source}${CER_TAG}${run_tag}.json"

      if [ "$MODE" = "test" ] && [ ! -f "$weights" ]; then
          n_dropped=$((n_dropped + 1))
          dropped_list="$dropped_list\n    $(basename "$weights")"
          continue
      fi

      cmd="julia --project=\"./../\" neural_bp_experiments.jl --workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS --hyperparams $hp_name --correlation_strengths_file correlated_weights_p_${p}_s_${SEED}.txt --quiet true --train ${train_source}.txt"
      if [ "$MODE" = "test" ]; then
          cmd="$cmd --test test_p_${p}_s_${SEED}.txt"
      fi
      echo "$cmd" >> "$COMMANDS"
      n_points=$((n_points + 1))
    done
  done
done

if [ "$MODE" = "test" ] && [ "$n_dropped" -gt 0 ]; then
    echo "WARNING: $n_dropped point(s) have no trained model and were EXCLUDED:" >&2
    printf "$dropped_list\n" >&2
    echo "  (run the 'train' phase first, or add the lines back to $COMMANDS by hand)" >&2
fi
if [ "$n_points" -eq 0 ]; then
    echo "ERROR: no commands generated." >&2
    [ "$MODE" = "test" ] && echo "  every point is missing its trained model — run 'train' first." >&2
    exit 1
fi

# ------------------------------------------------- SBATCH resource header ---
GRES_LINE=""
CUDA_LOAD=""
if [ "$MODE" = "test" ] && [ "$GPUS" -gt 0 ]; then
    if [ -n "$GPU_TYPE" ]; then
        GRES_LINE="#SBATCH --gres=gpu:${GPU_TYPE}:${GPUS}"
    else
        GRES_LINE="#SBATCH --gres=gpu:${GPUS}"
    fi
    CUDA_LOAD="module load $CUDA_MODULE"
fi

# With >1 GPU we hand each parallel slot its own device. Capturing SLURM's own
# CUDA_VISIBLE_DEVICES first is essential: on MIG partitions it is a list of
# UUIDs ("MIG-a,MIG-b"), and overwriting it with an integer index silently hides
# the GPU and drops the job to CPU. `cut -d, -f{%}` works for both plain indices
# and MIG UUIDs. `bash -c {}` re-parses the shell-quoted command.
if [ "$MODE" = "test" ] && [ "$GPUS" -gt 1 ]; then
    PARALLEL_LINE="export SLURM_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
parallel --jobs $JOBS --results \"\$LOCAL_LOGS\" 'CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f{%}) bash -c {}' < \"\$COMMANDS_LOCAL\" &"
else
    PARALLEL_LINE="parallel --jobs $JOBS --results \"\$LOCAL_LOGS\" < \"\$COMMANDS_LOCAL\" &"
fi

# ----------------------------------------------------------- SLURM script ---
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cer_sweep_${MODE}_$TS
#SBATCH --output=$CLUSTER_DIR/sweep_${MODE}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/sweep_${MODE}_${TS}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$JOBS
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
$GRES_LINE
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# Standalone CER sweep runner — $MODE phase, $n_points command(s), $JOBS at a time.
# Deliberately does NOT use submit.sh / batch_run.jl.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep [$MODE] started: \$(date)"
echo "points: $n_points   parallel slots: $JOBS   USE_GPU=$USE_GPU"
echo "========================================="

module load $JULIA_MODULE
$CUDA_LOAD

if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi
echo "[depot] \$JULIA_DEPOT_PATH"

# Training is Enzyme reverse-mode AD => CPU only. Testing is a forward pass => GPU.
export USE_GPU=$USE_GPU
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1

cd \$SLURM_SUBMIT_DIR

# Precompile ONCE before fanning out, so N parallel workers don't race on the
# shared depot lock.
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export JULIA_PKG_PRECOMPILE_AUTO=0

# ---- stage in to node-local disk -------------------------------------------
LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
STAGE_IN_START=\$(date +%s)
tar -cf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"
echo "[stage-in] done in \$(( \$(date +%s) - STAGE_IN_START ))s"

# Commands carry a placeholder so the workdir resolves to node-local at runtime.
COMMANDS_LOCAL="\$SLURM_TMPDIR/sweep_commands_${MODE}.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$COMMANDS" > "\$COMMANDS_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/sweep_${MODE}_$TS"
mkdir -p "\$LOCAL_LOGS"

# ---- stage out (fires on normal exit AND on SLURM's pre-walltime TERM) ------
stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    echo "[stage-out] \$(date '+%F %T')"
    DIRS=()
    for d in results models cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    if [ \${#DIRS[@]} -gt 0 ]; then
        tar -cf - -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" | tar -xf - -C "$WORKDIR/$CODENAME"
        echo "[stage-out] copied: \${DIRS[*]}"
    else
        echo "[stage-out] nothing to copy."
    fi
}
term_handler() { stage_out; exit 0; }
trap term_handler TERM
trap stage_out EXIT

# Background + wait so the TERM trap fires immediately; a FOREGROUND parallel
# would defer it until it returned, and the walltime SIGKILL would wipe
# \$SLURM_TMPDIR with every partial result in it.
$PARALLEL_LINE
wait \$!

echo "========================================="
echo "sweep [$MODE] finished: \$(date)"
echo "========================================="
EOF
chmod +x "$SLURM"

echo "[$MODE] generated $n_points command(s): $(echo $ALPHA4 | wc -w) alpha4 x $(echo $ALPHA3 | wc -w) alpha3 x $(echo $PVALS | wc -w) p"
[ "$n_dropped" -gt 0 ] && echo "         ($n_dropped dropped for missing models)"
echo "  hyperparams -> $MODELS_DIR/hyperparams_sweep_a4*_a3*.toml   (retrain = $RETRAIN)"
echo "  commands    -> $COMMANDS"
echo "  slurm       -> $SLURM"
if [ "$MODE" = "train" ]; then
    echo "  resources   -> ${JOBS} CPUs, $WALLTIME, USE_GPU=0 (Enzyme AD is CPU-only)"
else
    echo "  resources   -> ${JOBS} slot(s), ${GPUS} GPU(s), $WALLTIME, USE_GPU=1"
fi
echo
echo "submit with:  sbatch $SLURM"
[ "$MODE" = "train" ] && echo "then, once it finishes:  bash misc/sweep_correlation_weight.sh test"
exit 0

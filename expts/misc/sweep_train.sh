#!/usr/bin/env bash
# ============================================================================
# sweep_train.sh — TRAIN phase of the CER sweep (CPU only)
# ============================================================================
# Trains one model per (use_CER, alpha4, alpha3, p, repeat) point. Emits only
# training commands: no --test, no GPU.
#
# WHY CPU: training is Enzyme reverse-mode AD, which cannot differentiate
# through GPU array allocation, so it runs on the CPU regardless of USE_GPU.
# A GPU here would sit idle for the whole job.
#
# THE EXPERIMENT (defaults below): does CER help when the network has a SMALL
# training budget? With 500 gradient updates per epoch x 20 epochs the network
# can learn the pairwise correlations directly from the samples, which makes an
# explicit prior redundant. The defaults here cut that to 50 updates x 10 epochs
# (batch 200), i.e.
#     samples/epoch  10,000 -> 10,000   (unchanged)
#     total samples 200,000 -> 100,000  (2x less)
#     grad updates   10,000 ->     500  (20x fewer)   <-- the real change
# so this is an OPTIMISATION-budget test more than a data-volume test. Both arms
# get identical budgets; only use_CER differs.
#
# BOTH ARMS ARE GENERATED HERE (use_CER = true and false) so the comparison is
# matched in every other respect. When use_CER = false the correlation term is
# inactive (is_correlated = false), so only alpha4 = 0 is generated for that arm
# — the other alpha4 values would be identical runs and pure waste.
#
# RUN FROM expts/ , and sbatch the emitted script from expts/ too.
#
#     bash misc/sweep_train.sh
#     bash misc/sweep_train.sh --repeats 6 --updates 50 --batch_size 200 --epochs 10
#     sbatch ../data/72q_BB_cycles_1/cluster/sweep_train_<timestamp>.sh
#
# Then:  bash misc/sweep_test.sh   (with the SAME grid flags)
#
# REPEATS: `--repeats N` trains N independent models per point (tagged _r1.._rN).
# Julia seeds its RNG per process, so repeats differ only in random weight
# initialisation — which is exactly the training variance you need to decide
# whether a gap between arms is real. Measured baseline spread was ~2%.
#
# Every generated point is recorded in models/directory.csv (run_tag -> full
# hyperparameter set).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_sweep_common.sh"

usage() { sed -n '2,45p' "$0"; }

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_20.toml"
PVALS="0.0005"                 # one p keeps the ladder affordable; 0.0005 is where
                               # the CER effect was largest
USE_CER_VALUES="true false"
ALPHA4="0 0.1"                 # only applied to the use_CER = true arm.
                               # 0   = CER priors, correlation term OFF
                               # 0.1 = CER priors + correlation term (best so far)
ALPHA3="0.5"
REPEATS=7                      # 3 budgets x (2 CER + 1 no-CER) x 1 p x 7 = 63 runs
# Training budget. With online_training = true the trainer does not sweep a
# fixed dataset: each epoch it draws UPDATES batches of BATCH_SIZE from the pool.
#   samples per epoch    = BATCH_SIZE * UPDATES
#   total samples        = BATCH_SIZE * UPDATES * EPOCHS
#   total gradient steps = UPDATES * EPOCHS          <-- the 20x cut vs before
EPOCHS=10
BATCH_SIZE=50                  # HELD FIXED across the ladder: batch size sets the
                               # SGD gradient-noise scale, which is a different
                               # knob from budget. Varying it too would confound
                               # "fewer updates" with "noisier/cleaner gradients".
UPDATES_LIST="25 100 400"      # TOML key: n_gradient_updates_per_epoch — the LADDER
NLAYERS=100
SEED=1
JOBS=64                        # one Narval node
WALLTIME="6:00:00"
MEM_PER_CPU="4G"
ACCOUNT="def-jemerson"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)    WORKDIR="$2";        shift 2;;
        --codename)   CODENAME="$2";       shift 2;;
        --base_hp)    BASE_HP="$2";        shift 2;;
        --pvals)      PVALS="$2";          shift 2;;
        --use_cer)    USE_CER_VALUES="$2"; shift 2;;
        --alpha4)     ALPHA4="$2";         shift 2;;
        --alpha3)     ALPHA3="$2";         shift 2;;
        --repeats)    REPEATS="$2";        shift 2;;
        --epochs)     EPOCHS="$2";         shift 2;;
        --batch_size) BATCH_SIZE="$2";     shift 2;;
        --updates_per_epoch|--updates) UPDATES_LIST="$2"; shift 2;;
        --nlayers)    NLAYERS="$2";        shift 2;;
        --seed)       SEED="$2";           shift 2;;
        --jobs)       JOBS="$2";           shift 2;;
        --walltime)   WALLTIME="$2";       shift 2;;
        --mem)        MEM_PER_CPU="$2";    shift 2;;
        --account)    ACCOUNT="$2";        shift 2;;
        --email)      EMAIL="$2";          shift 2;;
        -h|--help)    usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
BASE_HP_PATH="$MODELS_DIR/$BASE_HP"
REGISTRY="$MODELS_DIR/directory.csv"

[ -d "$MODELS_DIR" ]   || { echo "no models dir: $MODELS_DIR (run this from expts/)" >&2; exit 1; }
[ -f "$BASE_HP_PATH" ] || { echo "no base hyperparams: $BASE_HP_PATH" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
COMMANDS="$CLUSTER_DIR/sweep_commands_train_${TS}.txt"
SLURM="$CLUSTER_DIR/sweep_train_${TS}.sh"
: > "$COMMANDS"
sweep_registry_init "$REGISTRY"

# ------------------------------------------------ generate points/commands ---
n_budgets=$(echo $UPDATES_LIST | wc -w)
n_points=0
n_skipped=0
for updates in $UPDATES_LIST; do
  for use_cer in $USE_CER_VALUES; do
    for a4 in $ALPHA4; do
      # With use_CER = false the correlation term is inactive, so every alpha4
      # gives the identical model. Generate only alpha4 = 0 for that arm.
      if [ "$use_cer" = "false" ] && [ "$a4" != "0" ]; then
          n_skipped=$((n_skipped + 1))
          continue
      fi
      for a3 in $ALPHA3; do
        for rep in $(seq 1 "$REPEATS"); do
          run_tag=$(sweep_run_tag "$a4" "$a3" "$rep" "$REPEATS" "$updates" "$n_budgets")
          hp_name=$(sweep_hp_name "$run_tag" "$use_cer")

          sweep_write_hyperparams "$BASE_HP_PATH" "$MODELS_DIR/$hp_name" true "$run_tag" \
              "$a4" "$a3" "$use_cer" "$EPOCHS" "$BATCH_SIZE" "$updates" train "$TS"
          sweep_registry_record "$REGISTRY" "$BASE_HP_PATH" "$run_tag" "$hp_name" \
              "$use_cer" "$a4" "$a3" "$EPOCHS" "$BATCH_SIZE" "$updates" \
              "$NLAYERS" "$CODENAME" "$BASE_HP" train "$TS"

          for p in $PVALS; do
            echo "julia --project=\"./../\" neural_bp_experiments.jl --workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS --hyperparams $hp_name --correlation_strengths_file correlated_weights_p_${p}_s_${SEED}.txt --quiet true --train train_p_${p}_s_${SEED}.txt" >> "$COMMANDS"
            n_points=$((n_points + 1))
          done
        done
      done
    done
  done
done

# ----------------------------------------------------------- SLURM script ---
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cer_sweep_train_$TS
#SBATCH --output=$CLUSTER_DIR/sweep_train_${TS}.out
#SBATCH --error=$CLUSTER_DIR/sweep_train_${TS}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$JOBS
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# CER sweep — TRAIN phase. $n_points command(s), $JOBS at a time. CPU only.
# Budget ladder: $EPOCHS epochs x {$UPDATES_LIST} updates x batch $BATCH_SIZE.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep TRAIN started: \$(date)   points: $n_points   slots: $JOBS"
echo "========================================="

module load $JULIA_MODULE

if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi
echo "[depot] \$JULIA_DEPOT_PATH"

# Enzyme AD is CPU-only; a GPU here would sit idle.
export USE_GPU=0
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1

cd \$SLURM_SUBMIT_DIR

# Precompile ONCE before fanning out so the workers don't race on the depot lock.
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export JULIA_PKG_PRECOMPILE_AUTO=0

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
STAGE_IN_START=\$(date +%s)
tar -cf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"
echo "[stage-in] done in \$(( \$(date +%s) - STAGE_IN_START ))s"

COMMANDS_LOCAL="\$SLURM_TMPDIR/sweep_commands_train.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$COMMANDS" > "\$COMMANDS_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/sweep_train_$TS"
mkdir -p "\$LOCAL_LOGS"

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
# would defer it and the walltime SIGKILL would wipe \$SLURM_TMPDIR with every
# trained model in it.
parallel --jobs $JOBS --results "\$LOCAL_LOGS" < "\$COMMANDS_LOCAL" &
wait \$!

echo "========================================="
echo "sweep TRAIN finished: \$(date)"
echo "========================================="
EOF
chmod +x "$SLURM"

n_p=$(echo $PVALS | wc -w)
echo "[train] $n_points command(s)   (skipped $n_skipped redundant no-CER alpha4 point(s))"
echo "  grid        -> ${n_budgets} budget(s) x use_CER {$USE_CER_VALUES} x alpha4 {$ALPHA4} x alpha3 {$ALPHA3} x ${REPEATS} repeat(s) x ${n_p} p"
echo "  ladder      -> batch $BATCH_SIZE (FIXED), $EPOCHS epochs, updates/epoch:"
for u in $UPDATES_LIST; do
    printf "                   %-5s -> %7d samples/epoch, %8d total samples, %6d gradient steps\n" \
        "$u" "$((BATCH_SIZE * u))" "$((BATCH_SIZE * u * EPOCHS))" "$((u * EPOCHS))"
done
echo "  hyperparams -> $MODELS_DIR/hyperparams_sweep*.toml   (retrain = true)"
echo "  registry    -> $REGISTRY"
echo "  commands    -> $COMMANDS"
echo "  slurm       -> $SLURM"
echo "  resources   -> ${JOBS} CPUs, $WALLTIME, USE_GPU=0"
echo
echo "submit with:  sbatch $SLURM"
echo
echo "then, once it finishes:"
echo "  bash misc/sweep_test.sh --pvals \"$PVALS\" --use_cer \"$USE_CER_VALUES\" --alpha4 \"$ALPHA4\" --alpha3 \"$ALPHA3\" --repeats $REPEATS --updates_per_epoch \"$UPDATES_LIST\""
echo "  (only the GRID flags — sweep_test.sh reads epochs/batch_size/updates from the"
echo "   TOMLs written above, so the training config has a single source of truth.)"

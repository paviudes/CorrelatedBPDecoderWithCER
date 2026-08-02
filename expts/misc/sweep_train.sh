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
# THE EXPERIMENT: does CER help when the network has a SMALL training budget?
# At 10,000 gradient steps the network learns the pairwise correlations straight
# from the samples, which makes an explicit prior redundant — and there CER was
# measured to be a wash (298 vs 277 failures). So this sweeps a BUDGET LADDER and
# asks where, if anywhere, the prior starts paying for itself:
#
#     updates/epoch    samples/epoch   total samples   gradient steps
#            25             1,250          12,500             250
#           100             5,000          50,000           1,000
#           400            20,000         200,000           4,000
#
# `batch_size` is HELD FIXED (50) across the ladder on purpose: batch size sets
# the SGD gradient-noise scale, a qualitatively different knob from budget, and
# varying both would leave the result unattributable.
#
# The prediction worth falsifying: near the bottom rung the model is close to its
# initialisation, i.e. roughly plain BP with whatever priors it was given — and
# BP with the TRUE channel priors should beat BP with a mis-specified p=0.1. If
# CER is not ahead even at 250 steps, the prior buys nothing at any budget.
#
# BOTH ARMS ARE GENERATED HERE (use_CER = true and false) so the comparison is
# matched in every other respect. When use_CER = false the correlation term is
# inactive (is_correlated = false), so only alpha4 = 0 is generated for that arm
# — the other alpha4 values would be identical runs and pure waste.
#
# JOB ARRAY: the commands are split into contiguous chunks, one per array task,
# exactly as submission/slurm.jl does — each task sed's out its own slice and
# runs it with GNU parallel across its own cores. 126 points over 2 tasks of 63.
#
# RUN FROM expts/ , and sbatch the emitted script from expts/ too.
#
#     bash misc/sweep_train.sh
#     bash misc/sweep_train.sh --updates_per_epoch "25 100 400" --repeats 7
#     sbatch ../data/72q_BB_cycles_1/cluster/sweep_train_<timestamp>.sh
#
# Then:  bash misc/sweep_test.sh   (with the SAME grid flags)
#
# REPEATS: `--repeats N` trains N independent models per point (tagged _r1.._rN),
# identical in every setting. They differ only in the random weight
# initialisation and in which minibatches get drawn (online_training samples
# randomly) — i.e. they measure training-run variance, which is the uncertainty
# that decides whether a gap between arms is real. Measured baseline spread was
# ~2% (742 / 748 / 773 failures on repeats of one config). There is no explicit
# RNG seeding anywhere in the codebase, so repeats genuinely differ.
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
PVALS="0.0005 0.0015"
USE_CER_VALUES="true false"
ALPHA4="0 0.1"                 # only applied to the use_CER = true arm.
                               # 0   = CER priors, correlation term OFF
                               # 0.1 = CER priors + correlation term (best so far)
ALPHA3="0.5"
REPEATS=7                      # 3 budgets x (2 CER + 1 no-CER) x 2 p x 7 = 126 runs
                               # = 2 array tasks x 63 (of 64 cores each)
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
JOBS=64                        # parallel slots == cores per array task (one Narval node)
MAX_NODES=4                    # cap on the SLURM array size
WALLTIME="8:00:00"             # set by the SLOWEST rung: u=400 is 200,000 total
                               # samples, i.e. the same sample count as the
                               # original 20-epoch runs (~6h). Slack so a timeout
                               # doesn't cost the top rung.
MEM_PER_CPU="3G"               # 64 x 3G = 192G. A Narval standard node has 249G
                               # usable, so 4G/core (256G) would only fit on the
                               # scarce 498G nodes and sit in the queue.
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
        --max_nodes)  MAX_NODES="$2";      shift 2;;
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

# --------------------------------------------------------------- job array ---
# Same split as submission/slurm.jl: each array task owns a contiguous chunk of
# the commands file and runs it with GNU parallel across its own cores.
N_TASKS=$(( (n_points + JOBS - 1) / JOBS ))
[ "$N_TASKS" -lt 1 ] && N_TASKS=1
[ "$N_TASKS" -gt "$MAX_NODES" ] && N_TASKS=$MAX_NODES
CHUNK=$(( (n_points + N_TASKS - 1) / N_TASKS ))
# Never request more cores (or parallel slots) than the chunk actually holds —
# with 126 points over 2 tasks the chunk is 63, so asking for 64 would leave a
# core idle in every task and inflate the memory request for nothing.
CPUS_PER_TASK=$JOBS
[ "$CHUNK" -lt "$CPUS_PER_TASK" ] && CPUS_PER_TASK=$CHUNK
SLOTS=$CPUS_PER_TASK

# ----------------------------------------------------------- SLURM script ---
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cer_sweep_train_$TS
#SBATCH --output=$CLUSTER_DIR/sweep_train_${TS}_%a.out
#SBATCH --error=$CLUSTER_DIR/sweep_train_${TS}_%a.err
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# CER sweep — TRAIN phase. $n_points command(s) split over $N_TASKS array task(s)
# of up to $CHUNK each, $JOBS at a time. CPU only.
# Budget ladder: $EPOCHS epochs x {$UPDATES_LIST} updates x batch $BATCH_SIZE.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep TRAIN task \${SLURM_ARRAY_TASK_ID} started: \$(date)"
echo "total points: $n_points   array tasks: $N_TASKS   chunk: $CHUNK   slots: $SLOTS"
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

# This array task's slice of the command list (1-based, inclusive).
START=\$(( SLURM_ARRAY_TASK_ID * $CHUNK + 1 ))
END=\$(( START + $CHUNK - 1 ))
COMMANDS_LOCAL="\$SLURM_TMPDIR/sweep_commands_train_\${SLURM_ARRAY_TASK_ID}.txt"
sed -n "\${START},\${END}p" "$COMMANDS" | sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" > "\$COMMANDS_LOCAL"
echo "[chunk] lines \${START}-\${END}: \$(wc -l < "\$COMMANDS_LOCAL") command(s)"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/sweep_train_${TS}_\${SLURM_ARRAY_TASK_ID}"
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
parallel --jobs $SLOTS --results "\$LOCAL_LOGS" < "\$COMMANDS_LOCAL" &
wait \$!

echo "========================================="
echo "sweep TRAIN task \${SLURM_ARRAY_TASK_ID} finished: \$(date)"
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
echo "  array       -> ${N_TASKS} task(s) x up to ${CHUNK} command(s), ${JOBS} slots each"
echo "  resources   -> ${CPUS_PER_TASK} CPUs/task x ${MEM_PER_CPU} = $((CPUS_PER_TASK * ${MEM_PER_CPU%G}))G/task, $WALLTIME, USE_GPU=0"
echo
echo "submit with:  sbatch $SLURM"
echo
echo "then, once it finishes:"
echo "  bash misc/sweep_test.sh --pvals \"$PVALS\" --use_cer \"$USE_CER_VALUES\" --alpha4 \"$ALPHA4\" --alpha3 \"$ALPHA3\" --repeats $REPEATS --updates_per_epoch \"$UPDATES_LIST\""
echo "  (only the GRID flags — sweep_test.sh reads epochs/batch_size/updates from the"
echo "   TOMLs written above, so the training config has a single source of truth.)"

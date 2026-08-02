#!/usr/bin/env bash
# ============================================================================
# sweep_test.sh — TEST phase of the CER sweep (GPU)
# ============================================================================
# Tests the models sweep_train.sh produced. Emits ONLY testing commands
# (retrain = false, so each run loads its model rather than retraining).
#
# GPU directives follow the project's normal testing pipeline (submit.sh ->
# batch_run.jl -> submission/slurm.jl):
#   * `--gpus-per-node=<model>:<n>`, NOT `--gres` — the Alliance docs prefer it,
#     and omitting the model specifier "may cause the job to be rejected or be
#     sent to an arbitrary GPU".
#   * default a100_1g.5gb — the smallest Narval MIG slice (1 core, 15 GB), most
#     abundant and near-zero queue wait; the 72q decoder fits easily. Keep
#     cpus/mem inside the bundle ratio or the scheduler charges you for the
#     resources you didn't use and your priority drops:
#     https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
#   * `module load cuda`, `export USE_GPU=1`, `export GPU_BACKEND=cuda`.
#   * one parallel slot per GPU; with >1 GPU each slot gets its own device via
#     MIG-safe CUDA_VISIBLE_DEVICES slicing.
#
# RUN FROM expts/ , and sbatch the emitted script from expts/ too.
#
#     bash misc/sweep_test.sh
#     bash misc/sweep_test.sh --repeats 6            # only if train used non-defaults
#     sbatch ../data/72q_BB_cycles_1/cluster/sweep_test_<timestamp>.sh
#
# NO TRAINING KNOBS HERE. `--epochs`, `--batch_size` and `--updates` are NOT
# flags of this script: it reads them out of the hyperparameters TOML that
# sweep_train.sh already wrote, and flips `retrain` to false IN PLACE. So the
# training configuration has exactly one source of truth and the two phases
# cannot silently disagree.
#
# The remaining flags (--pvals/--use_cer/--alpha4/--alpha3/--repeats) only
# enumerate WHICH points to test — they rebuild the run_tag, and hence the
# hyperparameters filename to look for. If they don't match the training run,
# the corresponding TOMLs simply won't exist and the points are dropped with a
# warning rather than silently retrained on the GPU.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_sweep_common.sh"

usage() { sed -n '2,33p' "$0"; }

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_20.toml"
PVALS="0.0005"
USE_CER_VALUES="true false"
ALPHA4="0 0.1"
ALPHA3="0.5"
REPEATS=7
UPDATES_LIST="25 100 400"      # must match sweep_train.sh: part of the run_tag
NLAYERS=100
SEED=1
GPUS=1
GPU_TYPE="a100_3g.20gb"        # Narval A100 50% MIG: 6 cores, 62 GB
JOBS=""                        # defaults to GPUS (one command per GPU)
CPUS_PER_GPU=6                 # bundle ratio for a100_3g.20gb
MEM_PER_CPU="10G"              # 6 x 10G = 60G, just inside the 62 GB bundle
WALLTIME="6:00:00"
ACCOUNT="def-jemerson"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --workdir)      WORKDIR="$2";        shift 2;;
        --codename)     CODENAME="$2";       shift 2;;
        --base_hp)      BASE_HP="$2";        shift 2;;
        --pvals)        PVALS="$2";          shift 2;;
        --use_cer)      USE_CER_VALUES="$2"; shift 2;;
        --alpha4)       ALPHA4="$2";         shift 2;;
        --alpha3)       ALPHA3="$2";         shift 2;;
        --repeats)      REPEATS="$2";        shift 2;;
        --updates_per_epoch|--updates) UPDATES_LIST="$2"; shift 2;;
        --nlayers)      NLAYERS="$2";        shift 2;;
        --seed)         SEED="$2";           shift 2;;
        --gpus)         GPUS="$2";           shift 2;;
        --gpu_type)     GPU_TYPE="$2";       shift 2;;
        --jobs)         JOBS="$2";           shift 2;;
        --cpus_per_gpu) CPUS_PER_GPU="$2";   shift 2;;
        --mem)          MEM_PER_CPU="$2";    shift 2;;
        --walltime)     WALLTIME="$2";       shift 2;;
        --account)      ACCOUNT="$2";        shift 2;;
        --email)        EMAIL="$2";          shift 2;;
        --cuda_module)  CUDA_MODULE="$2";    shift 2;;
        -h|--help)      usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

[ -n "$JOBS" ] || JOBS="$GPUS"
CPUS_PER_TASK=$((CPUS_PER_GPU * GPUS))

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"

# The registry (models/directory.csv) is owned by sweep_train.sh — it records
# the settings a model was TRAINED with, which testing does not change.
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR (run this from expts/)" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
COMMANDS="$CLUSTER_DIR/sweep_commands_test_${TS}.txt"
SLURM="$CLUSTER_DIR/sweep_test_${TS}.sh"
: > "$COMMANDS"

# ------------------------------------------------ generate points/commands ---
n_budgets=$(echo $UPDATES_LIST | wc -w)
n_points=0
n_dropped=0
n_missing_toml=0
dropped_list=""
budget_note=""
for updates in $UPDATES_LIST; do
  for use_cer in $USE_CER_VALUES; do
    cer_tag=$(sweep_cer_tag_for "$use_cer")
    for a4 in $ALPHA4; do
      if [ "$use_cer" = "false" ] && [ "$a4" != "0" ]; then
          continue      # identical to alpha4 = 0 when the correlation term is off
      fi
      for a3 in $ALPHA3; do
        for rep in $(seq 1 "$REPEATS"); do
          run_tag=$(sweep_run_tag "$a4" "$a3" "$rep" "$REPEATS" "$updates" "$n_budgets")
          hp_name=$(sweep_hp_name "$run_tag" "$use_cer")
          hp_path="$MODELS_DIR/$hp_name"

          # The training config lives in the TOML sweep_train.sh wrote. Read it —
          # never re-derive it from flags.
          if [ ! -f "$hp_path" ]; then
              n_missing_toml=$((n_missing_toml + 1))
              continue
          fi
          sweep_disable_retrain "$hp_path"
          n_epochs=$(sweep_toml_get "$hp_path" n_epochs)
          budget_note="${budget_note}\n    u=${updates}: $n_epochs epochs x $(sweep_toml_get "$hp_path" n_gradient_updates_per_epoch) updates x batch $(sweep_toml_get "$hp_path" batch_size) = $(( $(sweep_toml_get "$hp_path" n_gradient_updates_per_epoch) * n_epochs )) steps"

          for p in $PVALS; do
            train_source="train_p_${p}_s_${SEED}"
            weights=$(sweep_weights_path "$MODELS_DIR" "$NLAYERS" "$n_epochs" "$train_source" "$cer_tag" "$run_tag")
            if [ ! -f "$weights" ]; then
                n_dropped=$((n_dropped + 1))
                dropped_list="$dropped_list\n    $(basename "$weights")"
                continue
            fi
            echo "julia --project=\"./../\" neural_bp_experiments.jl --workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS --hyperparams $hp_name --correlation_strengths_file correlated_weights_p_${p}_s_${SEED}.txt --quiet true --train ${train_source}.txt --test test_p_${p}_s_${SEED}.txt" >> "$COMMANDS"
            n_points=$((n_points + 1))
          done
        done
      done
    done
  done
  # keep only the first budget_note line per rung
  budget_note=$(printf "$budget_note" | awk '!seen[$0]++' | tr '\n' '\n')
done

if [ "$n_missing_toml" -gt 0 ]; then
    echo "WARNING: $n_missing_toml point(s) have no hyperparameters TOML — sweep_train.sh" >&2
    echo "  was not run for them (or with a different --alpha4/--alpha3/--repeats grid)." >&2
fi
if [ "$n_dropped" -gt 0 ]; then
    echo "WARNING: $n_dropped point(s) have a TOML but no trained model, EXCLUDED:" >&2
    printf "$dropped_list\n" | head -20 >&2
fi
if [ "$n_points" -eq 0 ]; then
    echo "ERROR: no commands generated — nothing has been trained for this grid." >&2
    echo "  Run: bash misc/sweep_train.sh   (with matching --alpha4/--alpha3/--repeats)" >&2
    exit 1
fi

# With >1 GPU, hand each parallel slot its own device. Capturing SLURM's own
# CUDA_VISIBLE_DEVICES first is essential: on MIG it is a list of UUIDs
# ("MIG-a,MIG-b"), and overwriting it with an integer index silently hides the
# GPU and drops the run to CPU. `cut -d, -f{%}` handles both forms.
if [ "$GPUS" -gt 1 ]; then
    PARALLEL_LINE="export SLURM_CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
parallel --jobs $JOBS --results \"\$LOCAL_LOGS\" 'CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f{%}) bash -c {}' < \"\$COMMANDS_LOCAL\" &"
else
    PARALLEL_LINE="parallel --jobs $JOBS --results \"\$LOCAL_LOGS\" < \"\$COMMANDS_LOCAL\" &"
fi

# ----------------------------------------------------------- SLURM script ---
cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cer_sweep_test_$TS
#SBATCH --output=$CLUSTER_DIR/sweep_test_${TS}.out
#SBATCH --error=$CLUSTER_DIR/sweep_test_${TS}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
#SBATCH --gpus-per-node=${GPU_TYPE}:${GPUS}
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# CER sweep — TEST phase. $n_points command(s), $JOBS at a time, ${GPUS} x ${GPU_TYPE}.
# NOTE: no 'set -e' — one failing point must not kill the rest of the sweep.
set -uo pipefail
echo "========================================="
echo "sweep TEST started: \$(date)   points: $n_points   slots: $JOBS   gpus: ${GPU_TYPE}:${GPUS}"
echo "========================================="

module load $JULIA_MODULE
module load $CUDA_MODULE

if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi
echo "[depot] \$JULIA_DEPOT_PATH"

export USE_GPU=1
export GPU_BACKEND=cuda
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1

cd \$SLURM_SUBMIT_DIR

# Precompile ONCE, ON THE GPU NODE, before fanning out. Doing it here rather
# than on a login node is also what lets CUDA_Runtime_jll see a real driver —
# precompiling CUDA.jl without one bakes in "no CUDA runtime found".
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export JULIA_PKG_PRECOMPILE_AUTO=0

# Report what CUDA sees, so a silent CPU fallback is visible in the log instead
# of costing the whole allocation.
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @info "CUDA functional: \$(CUDA.functional())"; CUDA.functional() && @info CUDA.device()' || \
    echo "WARNING: CUDA check failed — runs below may fall back to CPU."

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
STAGE_IN_START=\$(date +%s)
tar -cf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"
echo "[stage-in] done in \$(( \$(date +%s) - STAGE_IN_START ))s"

COMMANDS_LOCAL="\$SLURM_TMPDIR/sweep_commands_test.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$COMMANDS" > "\$COMMANDS_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/sweep_test_$TS"
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

$PARALLEL_LINE
wait \$!

echo "========================================="
echo "sweep TEST finished: \$(date)"
echo "========================================="
EOF
chmod +x "$SLURM"

echo "[test] $n_points command(s) generated"
[ "$n_dropped" -gt 0 ] && echo "       ($n_dropped dropped for missing models)"
echo "  budget ladder (read from the training TOMLs, not from flags):"
printf "$budget_note\n" | sed '/^$/d'
echo "  hyperparams -> $MODELS_DIR/hyperparams_sweep*.toml   (retrain flipped to false in place)"
echo "  commands    -> $COMMANDS"
echo "  slurm       -> $SLURM"
echo "  resources   -> ${GPUS} x ${GPU_TYPE}, ${CPUS_PER_TASK} CPU(s), ${MEM_PER_CPU}/cpu, $WALLTIME"
echo
echo "submit with:  sbatch $SLURM"

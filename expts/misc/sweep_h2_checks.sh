#!/usr/bin/env bash
# ============================================================================
# sweep_h2_checks.sh — measurement runs for Checks B, C and D of the H2 handoff
#                      (GPU node: train on the bundled CPU cores, test on the GPU)
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_h2_checks.sh --setup    # once: create the debug codename
#     bash misc/sweep_h2_checks.sh B          # lambda sweep
#     bash misc/sweep_h2_checks.sh C          # sparsity arms at the best lambda
#     bash misc/sweep_h2_checks.sh D          # seed replication of the headline
#
# then:  sbatch ../data/72q_BB_cycles_1_debug/cluster/h2_<check>_<timestamp>.sh
#
# ---------------------------------------------------------------------------
# WHY A GPU NODE FOR WHAT IS MOSTLY CPU WORK
#
# Each point trains AND tests in ONE invocation (`--test` is emitted), so there
# is no model hand-off between a CPU job and a later GPU job — and no filename
# reconstruction for a test-side preflight to get wrong.
#
# Training itself is CPU-only: Enzyme reverse-mode AD cannot differentiate
# through GPU array allocation. The GPU is used only by the closing 10^6-sample
# forward pass. It is therefore IDLE for ~90% of each run's wall time, which is
# the honest cost of this arrangement.
#
# What makes it defensible is Narval's bundle ratio (Alliance "Ratios in
# bundles", 1 RGU = 3.00 cores = 31.1 GB). Billing is
# max(RGU, cores/3.00, memory/31.1), so at the RECOMMENDED ratio the three
# coincide and the GPU costs nothing beyond the cores you were requesting anyway:
#
#   instance        RGU   recommended        charged at that ratio
#   a100_1g.5gb    0.57   1 core,  15 GB     0.57
#   a100_2g.10gb   1.14   3 cores, 31 GB     1.14
#   a100_3g.20gb   2.00   6 cores, 62 GB     2.00      <-- default here
#   a100 (whole)   4.00  12 cores, 124 GB    4.00
#
# Go OVER the core ratio and you are charged for cores while the GPU idles; go
# under and you are charged for the GPU while cores idle. Both are waste.
#
# ONE MIG PER JOB. The MIG documentation is explicit: "requesting more than one
# MIG instance in a job is not permitted. Such a job will be rejected at
# submission time." So a MIG job is capped at its bundle's cores — 6 on a
# 3g.20gb. More concurrency than that requires a whole A100 (12 cores).
#
# Note the RGU-hours are often a wash: 7 points on a 3g.20gb (2.0 RGU, two waves)
# costs the same as 7 on a whole A100 (4.0 RGU, one wave). The MIG wins on QUEUE
# TIME, which is why it is the default.
#
# ---------------------------------------------------------------------------
# WHAT EACH CHECK IS
#
#   B  correlation_weight in {0, 0.01, 0.03, 0.1, 0.3, 1.0} at p = 7e-4, plus an
#      `annealed` reference arm. Does the convergence-failure count fall back to
#      the no-CER level as lambda -> 0? Monotone decline confirms H2 and locates
#      an operating point; failures still elevated at lambda = 0.01 indicts the
#      FORM of the term rather than its weight.
#
#   C  at --best_lambda, sparsity_importance in {0.0, 0.05, 0.15} plus the
#      annealed reference.
#
#      CORRECTION TO THE HANDOFF. It states `sparsity_importance = 0.0` in the
#      runs under investigation. It is not. Both TOMLs carry "0,5e-1,0.8,up",
#      and `compute_hyperparameters` (src/train.jl) evaluates an "up" spec as
#      max - (max-min)*decay^(epoch-1), giving
#
#        epoch    1       2       3       4       5
#        lambda   0.0100  0.3070  0.5149  0.6604  0.7623
#        sparsity 0.0000  0.1000  0.1800  0.2440  0.2952
#
#      Sparsity is 0 only in epoch 1 and reaches 0.295 by epoch 5 — INSIDE the
#      0.3-0.5 band previously measured to be independently harmful (653 -> 309
#      failures when it was dropped). The premise is inverted: the question is
#      not "does adding a counterweight help", it is "is the annealed sparsity
#      already in its harmful range, and is that interacting with lambda?".
#
#   D  CER and no-CER at seeds 1-5, same seed set for both arms, BOTH schedules
#      annealed exactly as in the investigated runs — so the spread it measures
#      is the error bar for the headline comparison, not for some other setting.
#      A previously measured training-seed spread on PROVABLY IDENTICAL
#      configurations was 309 vs 620 failures (2.0x).
#
# ---------------------------------------------------------------------------
# TWO THINGS THIS SCRIPT IS CAREFUL ABOUT
#
#   CONSTANT, NOT ANNEALED. `correlation_weight` and `sparsity_importance` are
#   "min,max,decay,direction" specs. A bare number is written as min == max,
#   pinning it for the whole run — which is what sweeping a VALUE requires. A
#   full spec is passed through verbatim, which is how the reference arms
#   reproduce the investigated configuration.
#
#   --isdebug true ON EVERY POINT. Check A could not be completed because
#   data/72q_BB_cycles_1/logs/ is empty: the runs were launched without it, so
#   `correlation_penalty` and the realised per-epoch `correlation_weight` were
#   never recorded.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CHECK="${1:-}"
case "$CHECK" in
    B|C|D|--setup) shift ;;
    *) echo "usage: bash misc/sweep_h2_checks.sh {--setup|B|C|D} [flags]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_debug"
SOURCE_CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_5_corrs.toml"
PVAL="0.0007"
NLAYERS=90
SEED_B_C=1
LAMBDAS="0 0.01 0.03 0.1 0.3 1.0"
BEST_LAMBDA="0.1"
SPARSITIES="0.0 0.05 0.15"
ANNEALED_LAMBDA="1e-2,1,0.7,up"
ANNEALED_SPARSITY="0,5e-1,0.8,up"
SEEDS_D="1 2 3 4 5"
CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""                      # "" => auto-pick the smallest bundle that fits
WALLTIME="12:00:00"              # ~1.5 h per point (train ~4000 s + test ~400 s),
                                 # two waves at 6 slots, plus slack
HEAP_HINT="4G"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pval)         PVAL="$2"; CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"; shift 2;;
        --lambdas)      LAMBDAS="$2";     shift 2;;
        --best_lambda)  BEST_LAMBDA="$2"; shift 2;;
        --sparsities)   SPARSITIES="$2";  shift 2;;
        --seeds)        SEEDS_D="$2";     shift 2;;
        --nlayers)      NLAYERS="$2";     shift 2;;
        --gpu_type)     GPU_TYPE="$2";    shift 2;;
        --walltime)     WALLTIME="$2";    shift 2;;
        --account)      ACCOUNT="$2";     shift 2;;
        --heap_hint)    HEAP_HINT="$2";   shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

# --------------------------------------------------------------- --setup ---
# Symlink the bulky read-only inputs and copy only what a run writes to. The
# data files are 144 MB each; linking keeps the debug tree small and guarantees
# both codenames see byte-identical inputs, which the comparison depends on.
if [ "$CHECK" = "--setup" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"
    dst="$WORKDIR/$CODENAME"
    [ -d "$src" ] || { echo "source codename missing: $src (run from expts/)" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        if [ ! -e "$dst/$shared" ]; then
            ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"
            echo "  linked  $shared"
        fi
    done
    cp -n "$src"/models/*.toml "$dst/models/" 2>/dev/null || true
    echo "  copied  models/*.toml ($(ls -1 "$dst"/models/*.toml 2>/dev/null | wc -l) files)"
    echo
    echo "  $dst ready. Results and models stay separate from $SOURCE_CODENAME."
    exit 0
fi

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
[ -d "$MODELS_DIR" ] || {
    echo "no models dir: $MODELS_DIR" >&2
    echo "run 'bash misc/sweep_h2_checks.sh --setup' first (from expts/)" >&2
    exit 1
}
[ -f "$MODELS_DIR/$BASE_HP" ] || { echo "no base hyperparams: $MODELS_DIR/$BASE_HP" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
COMMANDS="$CLUSTER_DIR/h2_commands_${CHECK}_${TS}.txt"
SLURM="$CLUSTER_DIR/h2_${CHECK}_${TS}.sh"
: > "$COMMANDS"

# A bare number is pinned CONSTANT (min == max); a full "min,max,decay,direction"
# spec passes through untouched.
spec_for() {
    local value="$1" decay="$2"
    case "$value" in
        *,*) echo "$value" ;;
        *)   echo "${value},${value},${decay},up" ;;
    esac
}

write_point() {   # <hp_name> <run_tag> <use_cer> <lambda> <sparsity> <seed>
    local hp_name="$1" run_tag="$2" use_cer="$3" lambda="$4" sparsity="$5" seed="$6"
    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|correlation_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_h2_checks.sh (check $CHECK, $TS) ----
retrain = true
run_tag = "${run_tag}"
use_CER = $use_cer
seed = $seed
correlation_weight = "$(spec_for "$lambda" 0.7)"
sparsity_importance = "$(spec_for "$sparsity" 0.8)"
single_qubit_rescale = 0.1
EOF
}

emit() {
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $1 --cer_data $CER_DATA --isdebug true --quiet true --diagnose true" \
         "--train train_p_${PVAL}_s_1.txt --test test_p_${PVAL}_s_1.txt" >> "$COMMANDS"
}

n_points=0
tag_of() { echo "$1" | tr '.' 'p'; }

if [ "$CHECK" = "B" ]; then
    # Sparsity pinned at 0.0 across the ladder — a DELIBERATE isolation of
    # lambda, not a copy of the investigated runs.
    for lambda in $LAMBDAS; do
        hp="hyperparams_h2B_lam$(tag_of "$lambda").toml"
        write_point "$hp" "_h2B_lam$(tag_of "$lambda")" true "$lambda" "0.0" "$SEED_B_C"
        emit "$hp"; n_points=$((n_points + 1))
    done
    # Reference: BOTH knobs annealed as in the investigated runs, so the ladder
    # contains the configuration it is trying to explain.
    write_point "hyperparams_h2B_annealed.toml" "_h2B_annealed" true \
                "$ANNEALED_LAMBDA" "$ANNEALED_SPARSITY" "$SEED_B_C"
    emit "hyperparams_h2B_annealed.toml"; n_points=$((n_points + 1))
fi

if [ "$CHECK" = "C" ]; then
    for sparsity in $SPARSITIES; do
        run_tag="_h2C_lam$(tag_of "$BEST_LAMBDA")_sp$(tag_of "$sparsity")"
        hp="hyperparams${run_tag}.toml"
        write_point "$hp" "$run_tag" true "$BEST_LAMBDA" "$sparsity" "$SEED_B_C"
        emit "$hp"; n_points=$((n_points + 1))
    done
    run_tag="_h2C_lam$(tag_of "$BEST_LAMBDA")_spannealed"
    hp="hyperparams${run_tag}.toml"
    write_point "$hp" "$run_tag" true "$BEST_LAMBDA" "$ANNEALED_SPARSITY" "$SEED_B_C"
    emit "$hp"; n_points=$((n_points + 1))
fi

if [ "$CHECK" = "D" ]; then
    for seed in $SEEDS_D; do
        hp="hyperparams_h2D_cer_seed${seed}.toml"
        write_point "$hp" "_h2D_cer" true "$ANNEALED_LAMBDA" "$ANNEALED_SPARSITY" "$seed"
        emit "$hp"; n_points=$((n_points + 1))
        hp="hyperparams_h2D_nocer_seed${seed}.toml"
        write_point "$hp" "_h2D_nocer" false "0" "$ANNEALED_SPARSITY" "$seed"
        emit "$hp"; n_points=$((n_points + 1))
    done
fi

# ------------------------------------------------------- GPU bundle sizing ---
# Pick the SMALLEST Narval bundle whose recommended core count covers the number
# of concurrent runs. Staying exactly on the ratio is what makes the GPU free.
if [ -z "$GPU_TYPE" ]; then
    if   [ "$n_points" -le 1 ]; then GPU_TYPE="a100_1g.5gb"
    elif [ "$n_points" -le 3 ]; then GPU_TYPE="a100_2g.10gb"
    elif [ "$n_points" -le 6 ]; then GPU_TYPE="a100_3g.20gb"
    else                             GPU_TYPE="a100_3g.20gb"   # cap at the MIG;
    fi                                                         # runs in waves
fi
case "$GPU_TYPE" in
    a100_1g.5gb)  SLOTS=1;  MEM="15G";  VRAM_GB=5  ;;
    a100_2g.10gb) SLOTS=3;  MEM="31G";  VRAM_GB=10 ;;
    a100_3g.20gb) SLOTS=6;  MEM="62G";  VRAM_GB=20 ;;
    a100_4g.20gb) SLOTS=6;  MEM="62G";  VRAM_GB=20 ;;
    a100)         SLOTS=12; MEM="124G"; VRAM_GB=40 ;;
    *) echo "unknown --gpu_type: $GPU_TYPE" >&2; exit 2;;
esac
[ "$n_points" -lt "$SLOTS" ] && SLOTS=$n_points
N_WAVES=$(( (n_points + SLOTS - 1) / SLOTS ))

# VRAM is shared by the concurrent workers on the ONE instance, so hand each a
# share rather than the whole thing — predict.jl sizes its prediction batch from
# this, and two workers reaching the test phase together would otherwise each
# budget for the full instance.
GPU_MEMORY_PER_SLOT="$(( (VRAM_GB * 1024) / SLOTS ))M"

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=h2_${CHECK}_$TS
#SBATCH --output=$CLUSTER_DIR/h2_${CHECK}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/h2_${CHECK}_${TS}.err
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=$SLOTS
#SBATCH --mem=$MEM
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# H2 check $CHECK — $n_points point(s), $SLOTS concurrent, $N_WAVES wave(s).
# Each point TRAINS on the bundled CPU cores (Enzyme AD is CPU-only) and then
# TESTS on the GPU, in one invocation. The GPU idles during training; the bundle
# ratio ($GPU_TYPE = $SLOTS cores, $MEM) is what keeps that from costing extra.
set -uo pipefail
echo "h2 check $CHECK started: \$(date)"
nvidia-smi || true

module load $JULIA_MODULE
module load $CUDA_MODULE

if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi

export USE_GPU=1
export GPU_BACKEND=cuda
export GPU_MEMORY=$GPU_MEMORY_PER_SLOT
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_HEAP_SIZE_HINT=$HEAP_HINT
export JULIA_PKG_OFFLINE=true

cd \$SLURM_SUBMIT_DIR

# Precompile HERE, on the GPU node. This cannot move to a login node:
# CUDA_Runtime_jll must see a real driver at precompile time, or "no CUDA
# runtime found" gets baked into the cache. LocalPreferences.toml must keep
# local_toolkit = true — switching to artifact mode makes CUDA.jl try to
# download on a node with no internet, which hangs until the walltime.
if ! timeout 1800 julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    echo "ERROR: precompilation failed or timed out." >&2
    exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @info "CUDA functional: \$(CUDA.functional())"; CUDA.functional() && @info CUDA.device()' \\
    || echo "WARNING: CUDA not functional — runs will silently fall back to the CPU forward pass."

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
# -h dereferences the symlinks --setup created, so the staged copy holds real files.
tar -chf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"

COMMANDS_LOCAL="\$SLURM_TMPDIR/h2_commands_${CHECK}.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$COMMANDS" > "\$COMMANDS_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/h2_${CHECK}_${TS}"
mkdir -p "\$LOCAL_LOGS"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    echo "[stage-out] \$(date '+%F %T')"
    DIRS=()
    for d in results models logs cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    if [ \${#DIRS[@]} -gt 0 ]; then
        # The generator owns the sweep TOMLs; do not copy them back over.
        tar -cf - --exclude='hyperparams_h2*.toml' -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" \\
            | tar -xf - -C "$WORKDIR/$CODENAME"
    fi
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

# Background + wait so the TERM trap fires immediately; a foreground parallel
# would defer it and the walltime SIGKILL would wipe \$SLURM_TMPDIR.
parallel --jobs $SLOTS --results "\$LOCAL_LOGS" < "\$COMMANDS_LOCAL" &
wait \$!
echo "h2 check $CHECK finished: \$(date)"
EOF
chmod +x "$SLURM"

echo "[h2 $CHECK] $n_points point(s)"
case "$CHECK" in
    B) echo "  lambda {$LAMBDAS} + annealed reference, sparsity 0.0, seed $SEED_B_C, p=$PVAL";;
    C) echo "  lambda $BEST_LAMBDA, sparsity {$SPARSITIES} + annealed reference, seed $SEED_B_C";;
    D) echo "  CER and no-CER at seeds {$SEEDS_D}, both schedules annealed";;
esac
echo "  every point trains AND tests in one invocation (--isdebug true, --diagnose true)"
echo
echo "  GPU        -> $GPU_TYPE, $SLOTS core(s), $MEM system RAM"
echo "  concurrency-> $SLOTS slot(s), $N_WAVES wave(s) for $n_points point(s)"
echo "  GPU_MEMORY -> $GPU_MEMORY_PER_SLOT per worker (${VRAM_GB}G instance / $SLOTS)"
if [ "$N_WAVES" -gt 1 ]; then
    echo "  NOTE: $n_points points over $SLOTS slots runs in $N_WAVES waves. A whole a100"
    echo "        (--gpu_type a100, 12 slots) would fit more concurrently at 4.0 RGU"
    echo "        instead of 2.0 — the RGU-HOURS are similar; the MIG queues sooner."
fi
echo "  account    -> $ACCOUNT"
echo "  commands   -> $COMMANDS"
echo "  slurm      -> $SLURM"
echo
echo "submit with:  sbatch $SLURM"

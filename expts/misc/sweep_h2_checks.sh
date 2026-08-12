#!/usr/bin/env bash
# ============================================================================
# sweep_h2_checks.sh — measurement runs for Checks B, C and D of the H2 handoff
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_h2_checks.sh B      # lambda sweep
#     bash misc/sweep_h2_checks.sh C      # sparsity arms at the best lambda
#     bash misc/sweep_h2_checks.sh D      # seed replication of the headline
#     bash misc/sweep_h2_checks.sh all
#
# then:  sbatch ../data/72q_BB_cycles_1/cluster/h2_<check>_<timestamp>.sh
#
# ---------------------------------------------------------------------------
# WHAT EACH CHECK IS
#
#   B  correlation_weight in {0, 0.01, 0.03, 0.1, 0.3, 1.0} at p = 7e-4.
#      Does the convergence-failure count fall back to the no-CER level as
#      lambda -> 0? Monotone decline confirms H2 and locates an operating point;
#      failures still elevated at lambda = 0.01 indicts the FORM of the term
#      rather than its weight.
#
#   C  at --best_lambda, sparsity_importance in {0.0, 0.05, 0.15} plus the
#      ANNEALED reference.
#
#      CORRECTION TO THE HANDOFF. It states `sparsity_importance = 0.0` in the
#      runs under investigation. It is not. Both TOMLs carry
#      "0,5e-1,0.8,up", and `compute_hyperparameters` (src/train.jl) evaluates
#      an "up" spec as  max - (max-min)*decay^(epoch-1), giving
#
#        epoch    1       2       3       4       5
#        lambda   0.0100  0.3070  0.5149  0.6604  0.7623
#        sparsity 0.0000  0.1000  0.1800  0.2440  0.2952
#
#      Sparsity is 0 only in epoch 1 and reaches 0.295 by epoch 5 — INSIDE the
#      0.3-0.5 band previously measured to be independently harmful (653 -> 309
#      failures when it was dropped). So the premise is inverted: the question is
#      not "does adding a counterweight help", it is "is the annealed sparsity
#      already in its harmful range, and is that interacting with lambda?".
#      The `annealed` arm below reproduces the investigated setting so the
#      constant-sparsity arms have something to be compared against.
#
#   D  CER and no-CER at seeds 1-5, same seed set for both arms.
#      A previously measured training-seed spread on PROVABLY IDENTICAL
#      configurations was 309 vs 620 failures (2.0x). Until that error bar is
#      quantified here, no single-run comparison in B or C is interpretable.
#
# ---------------------------------------------------------------------------
# TWO THINGS THIS SCRIPT IS CAREFUL ABOUT
#
#   CONSTANT, NOT ANNEALED. `correlation_weight` and `sparsity_importance` are
#   "min,max,decay,direction" annealing specs. Writing min == max pins the value
#   for the whole run, which is what a sweep over the value requires. The runs
#   under investigation used "1e-2,1,0.7,up" — i.e. lambda RAMPED 0.01 -> 1
#   across epochs, so "the" lambda of those runs is not a single number.
#
#   --isdebug true ON EVERY POINT. The handoff's Check A could not be completed
#   because data/72q_BB_cycles_1/logs/ is empty: the runs were launched without
#   it, so `correlation_penalty` and the realised per-epoch `correlation_weight`
#   were never recorded. Every run here writes
#   logs/debugging_<source>_seed_<n>{,_individual_losses}.csv.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_sweep_common.sh"

CHECK="${1:-}"
case "$CHECK" in
    B|C|D|all|--setup) shift ;;
    *) echo "usage: bash misc/sweep_h2_checks.sh {--setup|B|C|D|all} [flags]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_debug"   # measurement runs go in their own codename so
                                   # they cannot overwrite the production results
                                   # in 72q_BB_cycles_1. Populate it first:
                                   #   bash misc/sweep_h2_checks.sh --setup
BASE_HP="hyperparams_epochs_5_corrs.toml"
NOCER_HP="hyperparams_epochs_5_no_cer.toml"
PVAL="0.0007"                    # largest effect, cheapest runs
NLAYERS=90
EPOCHS=5
SEED_B_C=1                       # ONE seed across B and C: points differ only in the knob
LAMBDAS="0 0.01 0.03 0.1 0.3 1.0"
BEST_LAMBDA="0.1"                # --best_lambda after reading B
SPARSITIES="0.0 0.05 0.15"
# The schedules the investigated runs actually used, reproduced verbatim as a
# reference arm in B and C. Full specs, so they pass through un-pinned.
ANNEALED_LAMBDA="1e-2,1,0.7,up"
ANNEALED_SPARSITY="0,5e-1,0.8,up"
SEEDS_D="1 2 3 4 5"
CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"
ACCOUNT="def-jemerson"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
JOBS=16
WALLTIME="24:00:00"              # CER training measured at 14302 s at this p; 4h
                                 # of slack per point, and lambda=0 runs are fast
MEM_PER_CPU="6G"
HEAP_HINT="4G"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pval)         PVAL="$2"; CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"; shift 2;;
        --lambdas)      LAMBDAS="$2";     shift 2;;
        --best_lambda)  BEST_LAMBDA="$2"; shift 2;;
        --sparsities)   SPARSITIES="$2";  shift 2;;
        --seeds)        SEEDS_D="$2";     shift 2;;
        --nlayers)      NLAYERS="$2";     shift 2;;
        --jobs)         JOBS="$2";        shift 2;;
        --walltime)     WALLTIME="$2";    shift 2;;
        --mem)          MEM_PER_CPU="$2"; shift 2;;
        --account)      ACCOUNT="$2";     shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

SOURCE_CODENAME="72q_BB_cycles_1"    # what --setup mirrors

# --------------------------------------------------------------- --setup ---
# Populate the debug codename by SYMLINKING the bulky read-only inputs and
# COPYING only what a run writes to. Training/testing data is 144 MB per file;
# linking keeps the debug tree small and guarantees both codenames see byte-
# identical inputs, which the whole comparison depends on.
if [ "$CHECK" = "--setup" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"
    dst="$WORKDIR/$CODENAME"
    [ -d "$src" ] || { echo "source codename missing: $src" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        if [ ! -e "$dst/$shared" ]; then
            ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"
            echo "  linked  $shared"
        fi
    done
    for toml in "$src"/models/*.toml; do
        cp -n "$toml" "$dst/models/" 2>/dev/null || true
    done
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

# Write one point's TOML. `lambda` and `sparsity` are pinned CONSTANT by setting
# min == max in the annealing spec. `use_cer=false` also forces lambda to 0 so
# the no-CER arm can never carry a live correlation term.
# `lambda` and `sparsity` may be either a bare number (pinned CONSTANT via
# min == max) or a full "min,max,decay,direction" spec, which is passed through
# verbatim. The latter is how the `annealed` reference arms reproduce the exact
# configuration under investigation.
spec_for() {   # spec_for <value-or-spec> <default_decay>
    local value="$1" decay="$2"
    case "$value" in
        *,*) echo "$value" ;;                      # already a full spec
        *)   echo "${value},${value},${decay},up" ;;
    esac
}

write_point() {   # write_point <hp_name> <run_tag> <use_cer> <lambda> <sparsity> <seed>
    local hp_name="$1" run_tag="$2" use_cer="$3" lambda="$4" sparsity="$5" seed="$6"
    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|correlation_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_h2_checks.sh (check $CHECK, $TS) ----
retrain = true
run_tag = "${run_tag}"
use_CER = $use_cer
seed = $seed

# A bare number here is pinned CONSTANT (min == max) so the sweep measures the
# VALUE. A full "min,max,decay,direction" spec is passed through as-is, which is
# how the `annealed` reference arms reproduce the investigated configuration
# (lambda 0.01->0.762, sparsity 0.0->0.295 across 5 epochs).
correlation_weight = "$(spec_for "$lambda" 0.7)"
sparsity_importance = "$(spec_for "$sparsity" 0.8)"

# The CER arm keeps the rescaled single-qubit priors it was measured with.
single_qubit_rescale = 0.1
EOF
}

emit() {   # emit <hp_name>
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $1 --cer_data $CER_DATA --isdebug true --quiet true --diagnose true" \
         "--train train_p_${PVAL}_s_1.txt --test test_p_${PVAL}_s_1.txt" >> "$COMMANDS"
}

n_points=0
tag_of() { echo "$1" | tr '.' 'p'; }

if [ "$CHECK" = "B" ] || [ "$CHECK" = "all" ]; then
    # Sparsity pinned at 0.0 across the ladder — a DELIBERATE isolation of lambda,
    # not a copy of the investigated runs (those anneal sparsity 0 -> 0.295).
    for lambda in $LAMBDAS; do
        run_tag="_h2B_lam$(tag_of "$lambda")"
        hp="hyperparams_h2B_lam$(tag_of "$lambda").toml"
        write_point "$hp" "$run_tag" true "$lambda" "0.0" "$SEED_B_C"
        emit "$hp"; n_points=$((n_points + 1))
    done
    # Reference point: BOTH knobs annealed exactly as in the investigated runs,
    # so the ladder contains the configuration it is trying to explain. Without
    # it, no point in B reproduces what was actually measured.
    write_point "hyperparams_h2B_annealed.toml" "_h2B_annealed" true \
                "$ANNEALED_LAMBDA" "$ANNEALED_SPARSITY" "$SEED_B_C"
    emit "hyperparams_h2B_annealed.toml"; n_points=$((n_points + 1))
fi

if [ "$CHECK" = "C" ] || [ "$CHECK" = "all" ]; then
    for sparsity in $SPARSITIES; do
        run_tag="_h2C_lam$(tag_of "$BEST_LAMBDA")_sp$(tag_of "$sparsity")"
        hp="hyperparams${run_tag}.toml"
        write_point "$hp" "$run_tag" true "$BEST_LAMBDA" "$sparsity" "$SEED_B_C"
        emit "$hp"; n_points=$((n_points + 1))
    done
    # The investigated sparsity schedule, held against the constant arms. This is
    # the arm the handoff assumed was 0.0; it actually reaches 0.295 by epoch 5.
    run_tag="_h2C_lam$(tag_of "$BEST_LAMBDA")_spannealed"
    hp="hyperparams${run_tag}.toml"
    write_point "$hp" "$run_tag" true "$BEST_LAMBDA" "$ANNEALED_SPARSITY" "$SEED_B_C"
    emit "$hp"; n_points=$((n_points + 1))
fi

if [ "$CHECK" = "D" ] || [ "$CHECK" = "all" ]; then
    for seed in $SEEDS_D; do
        # CER arm: the configuration under investigation, one seed per run.
        run_tag="_h2D_cer"
        hp="hyperparams_h2D_cer_seed${seed}.toml"
        # D reproduces the investigated configuration exactly — both schedules
        # annealed — so the seed spread it measures is the error bar that applies
        # to the headline CER-vs-no-CER comparison, not to some other setting.
        write_point "$hp" "$run_tag" true "$ANNEALED_LAMBDA" "$ANNEALED_SPARSITY" "$seed"
        emit "$hp"; n_points=$((n_points + 1))
        # no-CER arm: identical but for use_CER, and lambda forced to 0.
        run_tag="_h2D_nocer"
        hp="hyperparams_h2D_nocer_seed${seed}.toml"
        write_point "$hp" "$run_tag" false "0" "$ANNEALED_SPARSITY" "$seed"
        emit "$hp"; n_points=$((n_points + 1))
    done
fi

CPUS=$JOBS
[ "$n_points" -lt "$CPUS" ] && CPUS=$n_points

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=h2_${CHECK}_$TS
#SBATCH --output=$CLUSTER_DIR/h2_${CHECK}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/h2_${CHECK}_${TS}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem-per-cpu=$MEM_PER_CPU
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# H2 check $CHECK — $n_points measurement run(s), CPU only (Enzyme AD).
set -uo pipefail
echo "h2 check $CHECK started: \$(date)"

module load $JULIA_MODULE
if [ -z "\${JULIA_DEPOT_PATH:-}" ]; then
    if [ -n "\${SCRATCH:-}" ] && [ -d "\$SCRATCH/.julia" ]; then
        export JULIA_DEPOT_PATH="\$SCRATCH/.julia"
    else
        export JULIA_DEPOT_PATH="\$HOME/.julia"
    fi
fi
export USE_GPU=0
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_HEAP_SIZE_HINT=$HEAP_HINT
export JULIA_PKG_OFFLINE=true
export JULIA_PKG_PRECOMPILE_AUTO=0

cd \$SLURM_SUBMIT_DIR

# Warm the depot on a LOGIN node first: bash misc/precompile_depot.sh
if ! timeout 600 julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CorrelatedBPDecoderWithCER'; then
    echo "ERROR: depot not usable offline. Run 'bash misc/precompile_depot.sh' on a login node." >&2
    exit 1
fi

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
tar -cf - --exclude=testing_data -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"
# testing_data IS needed here (--test is emitted), so stage it separately rather
# than pulling the whole directory twice.
mkdir -p "\$LOCAL_WORK_DIR/testing_data"
cp "$WORKDIR/$CODENAME/testing_data/test_p_${PVAL}_s_1.txt" "\$LOCAL_WORK_DIR/testing_data/"

COMMANDS_LOCAL="\$SLURM_TMPDIR/h2_commands_${CHECK}.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$COMMANDS" > "\$COMMANDS_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/h2_${CHECK}_${TS}"
mkdir -p "\$LOCAL_LOGS"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    DIRS=()
    for d in results models logs cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    if [ \${#DIRS[@]} -gt 0 ]; then
        tar -cf - --exclude='hyperparams_*.toml' -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" | tar -xf - -C "$WORKDIR/$CODENAME"
    fi
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

parallel --jobs $CPUS --results "\$LOCAL_LOGS" < "\$COMMANDS_LOCAL" &
wait \$!
echo "h2 check $CHECK finished: \$(date)"
EOF
chmod +x "$SLURM"

echo "[h2 $CHECK] $n_points run(s)"
case "$CHECK" in
    B|all) echo "  B: lambda {$LAMBDAS}, sparsity 0.0, seed $SEED_B_C, p=$PVAL — CONSTANT specs";;
esac
case "$CHECK" in
    C|all) echo "  C: lambda $BEST_LAMBDA, sparsity {$SPARSITIES}, seed $SEED_B_C";;
esac
case "$CHECK" in
    D|all) echo "  D: CER and no-CER at seeds {$SEEDS_D} — matched seed sets";;
esac
echo "  every point: --isdebug true (Check A needs logs/) and --diagnose true"
echo "  commands -> $COMMANDS"
echo "  slurm    -> $SLURM"
echo
echo "submit with:  sbatch $SLURM"

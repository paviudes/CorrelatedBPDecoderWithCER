#!/usr/bin/env bash
# ============================================================================
# sweep_lambda.sh — test the correlation term at a weight where it can matter
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_lambda.sh --setup      # once: create the debug codename
#     bash misc/sweep_lambda.sh --probe      # SANITY GATE: lambda=100, 1 seed
#     bash misc/sweep_lambda.sh              # primary: 4 lambdas + baseline x 5 seeds
#     bash misc/sweep_lambda.sh --ungated    # deliberate probe: lambda=10, gate OFF
#     bash misc/sweep_lambda.sh --collect    # summarise
#
#     sbatch ../data/72q_BB_cycles_1_debug/cluster/lam_<mode>_<timestamp>.sh
#
# RUN --probe FIRST (handoff 2.3). If lambda=100 is pathological (>5000 failures,
# or min_weight_c2v_v2c going negative in the debug log) stop and report before
# spending the remaining budget.
#
# ---------------------------------------------------------------------------
# WHY THIS SWEEP
#
# The 2x2 gate x CER sweep came back null on every contrast (|t| <= 1.59), and the
# result that motivated the whole line of work did not replicate: convergence
# failures read 735 (CER) vs 374 (no-CER) at one seed, z = +10.8, but across 5
# paired seeds the difference was +59.8 +- 257.1, t = +0.52. A true +361 effect
# would have given t ~ 3.1, so the original effect size is rejected at ~p = 0.03.
# "CER worsens convergence" was seed noise.
#
# That agrees with the magnitude analysis: the correlation term is 0.05% of the
# total loss and cannot mechanically move failure rates by 1.7x. So the term is
# not harmful, it is INERT — and it has never been run at a weight where it could
# do anything.
#
# WHY IT IS INERT (this drives the design). The term is
# -(1/|C|) * sum_(i,k in C) J_ik * sigma_i * sigma_k, quadratic in sigma. Near a
# converged solution sigma is near-binary, so sigma_i*sigma_k = 1 only for pairs
# that genuinely co-flipped. Expected co-flipped pairs per sample is
# |C| * P11 ~ 540 * 1e-4 ~ 0.05, giving
#
#     per-sample corr ~ -J_bar * 0.05 / 540 ~ -1.6e-4
#
# against a MEASURED correlation_penalty mean of -3.1e-4 — agreement to within a
# factor of two, so the mechanism is confirmed, not conjectured. The weakness is
# structural: 1/|C| divides by 540 total edges when ~0.05 are active per sample.
# And the co-firing rate goes as p^2, so no single lambda is right across p.
#
# ---------------------------------------------------------------------------
# THE ARMS
#
#   lam0 / lam1 / lam10 / lam100   use_CER = true, gate ON (tau=0.5),
#                                  correlation_weight CONSTANT at that value
#   nocer                          use_CER = false, gate ON     <-- ADDED
#
# THE NO-CER ARM IS NOT IN THE HANDOFF AND IS ADDED DELIBERATELY. Without it the
# only baseline is lambda = 0, which is NOT a no-CER run: it still carries the
# rescaled CER single-qubit priors (single_qubit_rescale = 0.1). So lambda = 0
# already contains a prior effect, and an improvement at lambda > 0 could not be
# separated from it. The three-way decomposition is what is wanted:
#
#     nocer            flat p = 0.1 priors,      no correlation term
#     lam0             CER single-qubit priors,  no correlation term
#     lam1/10/100      CER single-qubit priors + correlation term
#
# nocer -> lam0 isolates the PRIORS; lam0 -> lam>0 isolates the COUPLINGS. That
# split is the one that mattered before: at p=5e-4 the priors alone moved 620 ->
# 483 failures while raw-vs-rescaled priors moved 3788 -> 303.
#
# It is also directly comparable to the 2x2's `nocer_gated` arm (same seeds, same
# gate, same sparsity schedule), so the two can be cross-checked. Note the 2x2's
# `cer_gated` arm is NOT the lambda=0 point: it ran lambda ANNEALED 0.01 -> 0.762,
# not pinned to 0. Comparing lam0 against it tests whether that anneal did
# anything, which on the magnitude analysis it should not have.
#
# ---------------------------------------------------------------------------
# CONSTANT, NOT ANNEALED. lambda is written as min == max so it is pinned for the
# whole run. The existing "1e-2,1,0.7,up" only reaches 0.762 by epoch 5 and
# confounds "how strong" with "when"; raising its ceiling would not fix that.
# Verify flatness from the debug log before trusting a run — --collect asserts it.
#
# GATE ON THROUGHOUT (handoff 2.2). The gate confines the reward to samples that
# already softly clear the syndrome, where it is ordering-safe by construction.
# Ungated at lambda = 10 or 100 would let it compete with the syndrome objective
# at full strength on failing samples. `--ungated` runs that as a separate probe;
# it is expected to be bad and is kept out of the primary sweep.
#
# require_correlations = true on every CER arm, so a missing-pairs CER file
# raises instead of masquerading as a null result.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_debug"
SOURCE_CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_5_corrs.toml"
PVAL="0.0007"
NLAYERS=90
SEEDS="1 2 3 4 5"
LAMBDAS="0 1 10 100"
GATE_TAU="0.5"
ANNEALED_SPARSITY="0,5e-1,0.8,up"    # unchanged from the 2x2: only lambda varies
CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"
INCLUDE_NOCER=1                      # the added baseline; --no_nocer to drop it

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""
TEST_JOBS=1                          # serial GPU phase: MIG allows one context
WALLTIME="16:00:00"
HEAP_HINT="4G"
MODE="primary"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --setup)     MODE="setup";   shift;;
        --collect)   MODE="collect"; shift;;
        --probe)     MODE="probe";   shift;;
        --ungated)   MODE="ungated"; shift;;
        --no_nocer)  INCLUDE_NOCER=0; shift;;
        --pval)      PVAL="$2"; CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"; shift 2;;
        --lambdas)   LAMBDAS="$2";   shift 2;;
        --seeds)     SEEDS="$2";     shift 2;;
        --tau)       GATE_TAU="$2";  shift 2;;
        --nlayers)   NLAYERS="$2";   shift 2;;
        --gpu_type)  GPU_TYPE="$2";  shift 2;;
        --walltime)  WALLTIME="$2";  shift 2;;
        --account)   ACCOUNT="$2";   shift 2;;
        --outdir)    OUTDIR="$2";    shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

if [ "$MODE" = "setup" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"; dst="$WORKDIR/$CODENAME"
    [ -d "$src" ] || { echo "source codename missing: $src (run from expts/)" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        [ -e "$dst/$shared" ] || { ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"; echo "  linked  $shared"; }
    done
    cp -n "$src"/models/*.toml "$dst/models/" 2>/dev/null || true
    echo "  $dst ready."; exit 0
fi

if [ "$MODE" = "collect" ]; then
    results_dir="$WORKDIR/$CODENAME/results"
    [ -d "$results_dir" ] || { echo "no results dir: $results_dir" >&2; exit 1; }
    extra=""; [ -n "${OUTDIR:-}" ] && extra="--outdir ${OUTDIR}"
    exec julia --project="$SCRIPT_DIR/../../" "$SCRIPT_DIR/collect_lambda.jl" "$results_dir" $extra
fi

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR — run --setup first" >&2; exit 1; }
[ -f "$MODELS_DIR/$BASE_HP" ] || { echo "no base hyperparams: $MODELS_DIR/$BASE_HP" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

# Mode-specific grids.
if [ "$MODE" = "probe" ]; then
    LAMBDAS="100"; SEEDS="1"; INCLUDE_NOCER=0; WALLTIME="6:00:00"
elif [ "$MODE" = "ungated" ]; then
    LAMBDAS="10"; GATE_TAU="-1.0"; INCLUDE_NOCER=0
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_CMDS="$CLUSTER_DIR/lam_${MODE}_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/lam_${MODE}_test_${TS}.txt"
HP_LIST="$CLUSTER_DIR/lam_${MODE}_hp_${TS}.txt"
SLURM="$CLUSTER_DIR/lam_${MODE}_${TS}.sh"
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"; : > "$HP_LIST"

tag_of() { echo "$1" | tr '.' 'p' | tr -d '-'; }

write_point() {   # <hp_name> <run_tag> <use_cer> <lambda> <gate> <seed>
    local hp_name="$1" run_tag="$2" use_cer="$3" lambda="$4" gate="$5" seed="$6"
    local require="true"
    if [ "$use_cer" = "false" ]; then require="false"; fi
    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|require_correlations|correlation_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_lambda.sh ($MODE, $TS) ----
retrain = true
run_tag = "${run_tag}"
use_CER = $use_cer
seed = $seed

# lambda PINNED CONSTANT (min == max). The annealed "1e-2,1,0.7,up" reaches only
# 0.762 by epoch 5 and confounds strength with timing.
correlation_weight = "${lambda},${lambda},0.7,up"

# Unchanged from the 2x2 arm this extends, so lambda is the only variable.
sparsity_importance = "${ANNEALED_SPARSITY}"
syndrome_gate_threshold = ${gate}
single_qubit_rescale = 0.1

# Refuse to run if the CER file yielded no couplings — otherwise a missing-pairs
# file is indistinguishable from a null result in a sweep about couplings.
require_correlations = ${require}
EOF
    echo "$hp_name" >> "$HP_LIST"
}

emit_pair() {
    local hp="$1"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $CER_DATA --isdebug true --quiet true" \
         "--train train_p_${PVAL}_s_1.txt" >> "$TRAIN_CMDS"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $CER_DATA --quiet true --diagnose true" \
         "--train train_p_${PVAL}_s_1.txt --test test_p_${PVAL}_s_1.txt" >> "$TEST_CMDS"
}

n_points=0
gate_label="gated"; [ "$GATE_TAU" = "-1.0" ] && gate_label="ungated"

for seed in $SEEDS; do
    for lambda in $LAMBDAS; do
        run_tag="_lam$(tag_of "$lambda")_${gate_label}"
        hp="hyperparams_lam$(tag_of "$lambda")_${gate_label}_seed${seed}.toml"
        write_point "$hp" "$run_tag" true "$lambda" "$GATE_TAU" "$seed"
        emit_pair "$hp"; n_points=$((n_points + 1))
    done
    if [ "$INCLUDE_NOCER" = "1" ]; then
        run_tag="_lamnocer_${gate_label}"
        hp="hyperparams_lamnocer_${gate_label}_seed${seed}.toml"
        write_point "$hp" "$run_tag" false "0" "$GATE_TAU" "$seed"
        emit_pair "$hp"; n_points=$((n_points + 1))
    fi
done

# ---- GPU bundle, sized by TRAINING concurrency (Narval: 1 RGU = 3.00 cores) ---
if [ -z "$GPU_TYPE" ]; then
    if   [ "$n_points" -le 1 ]; then GPU_TYPE="a100_1g.5gb"
    elif [ "$n_points" -le 3 ]; then GPU_TYPE="a100_2g.10gb"
    elif [ "$n_points" -le 6 ]; then GPU_TYPE="a100_3g.20gb"
    else                             GPU_TYPE="a100"
    fi
fi
case "$GPU_TYPE" in
    a100_1g.5gb)  SLOTS=1;  MEM="15G";  VRAM_GB=5  ;;
    a100_2g.10gb) SLOTS=3;  MEM="31G";  VRAM_GB=10 ;;
    a100_3g.20gb) SLOTS=6;  MEM="62G";  VRAM_GB=20 ;;
    a100)         SLOTS=12; MEM="124G"; VRAM_GB=40 ;;
    *) echo "unknown --gpu_type: $GPU_TYPE" >&2; exit 2;;
esac
[ "$n_points" -lt "$SLOTS" ] && SLOTS=$n_points
TRAIN_WAVES=$(( (n_points + SLOTS - 1) / SLOTS ))
GPU_MEMORY_PER_SLOT="$(( (VRAM_GB * 1024) / TEST_JOBS ))M"

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=lam_${MODE}_$TS
#SBATCH --output=$CLUSTER_DIR/lam_${MODE}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/lam_${MODE}_${TS}.err
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=$SLOTS
#SBATCH --mem=$MEM
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# lambda sweep ($MODE), $n_points point(s).
# PHASE 1 trains at $SLOTS-way concurrency, CPU only (Enzyme AD cannot use a GPU).
# PHASE 2 tests $TEST_JOBS at a time — MIG permits one CUDA context, and four
# concurrent contexts on one 20 GB MIG previously killed 2 of 4 runs at
# cuDevicePrimaryCtxRetain.
set -uo pipefail
echo "lambda sweep ($MODE) started: \$(date)"
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
export GPU_BACKEND=cuda
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JULIA_HEAP_SIZE_HINT=$HEAP_HINT
export JULIA_PKG_OFFLINE=true

cd \$SLURM_SUBMIT_DIR

# Precompile HERE: CUDA_Runtime_jll must see a driver, or "no CUDA runtime found"
# is baked in. LocalPreferences.toml must keep local_toolkit = true.
if ! timeout 1800 julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    echo "ERROR: precompilation failed or timed out." >&2; exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @info "CUDA functional: \$(CUDA.functional())"' \\
    || echo "WARNING: CUDA not functional — the test phase would fall back to CPU."

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
tar -chf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"

TRAIN_LOCAL="\$SLURM_TMPDIR/lam_train.txt"; TEST_LOCAL="\$SLURM_TMPDIR/lam_test.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" > "\$TRAIN_LOCAL"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS"  > "\$TEST_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/lam_${MODE}_${TS}"
mkdir -p "\$LOCAL_LOGS/train" "\$LOCAL_LOGS/test"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    DIRS=()
    for d in results models logs cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    [ \${#DIRS[@]} -gt 0 ] && tar -cf - --exclude='hyperparams_lam*.toml' \\
        -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" | tar -xf - -C "$WORKDIR/$CODENAME"
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

export USE_GPU=0
echo "[phase 1] training \$(wc -l < "\$TRAIN_LOCAL") point(s), $SLOTS at a time: \$(date)"
parallel --jobs $SLOTS --results "\$LOCAL_LOGS/train" < "\$TRAIN_LOCAL" &
wait \$!
echo "[phase 1] done: \$(date)"

while read -r hp; do
    f="\$LOCAL_WORK_DIR/models/\$hp"
    [ -f "\$f" ] || continue
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true([[:space:]]*(#.*)?)\$|\1false\2|' "\$f" > "\$f.tmp" && mv "\$f.tmp" "\$f"
done < "$HP_LIST"

export USE_GPU=1
export GPU_MEMORY=$GPU_MEMORY_PER_SLOT
echo "[phase 2] testing \$(wc -l < "\$TEST_LOCAL") point(s), $TEST_JOBS at a time: \$(date)"
parallel --jobs $TEST_JOBS --results "\$LOCAL_LOGS/test" < "\$TEST_LOCAL" &
wait \$!
echo "lambda sweep ($MODE) finished: \$(date)"
EOF
chmod +x "$SLURM"

n_seeds=$(echo $SEEDS | wc -w); n_lam=$(echo $LAMBDAS | wc -w)
echo "[lambda $MODE] $n_points point(s)"
echo
printf "  %-22s %-9s %-10s %-8s %s\n" "run_tag" "use_CER" "lambda" "gate" "role"
for lambda in $LAMBDAS; do
    printf "  %-22s %-9s %-10s %-8s %s\n" "_lam$(tag_of "$lambda")_${gate_label}" "true" "$lambda (const)" "$GATE_TAU" \
        "$([ "$lambda" = "0" ] && echo 'CER priors, no couplings' || echo 'CER priors + couplings')"
done
if [ "$INCLUDE_NOCER" = "1" ]; then
    printf "  %-22s %-9s %-10s %-8s %s\n" "_lamnocer_${gate_label}" "false" "n/a" "$GATE_TAU" \
        "ADDED baseline: flat p=0.1, no couplings"
fi
echo
echo "  seeds  -> $SEEDS  (same set at every lambda)"
echo "  grid   -> $n_lam lambda(s)$([ "$INCLUDE_NOCER" = "1" ] && echo ' + nocer baseline') x $n_seeds seed(s) = $n_points"
echo "  GPU    -> $GPU_TYPE, $SLOTS core(s), $MEM; train $TRAIN_WAVES wave(s), test serial"
echo "  assert -> require_correlations = true on every CER arm"
echo
if [ "$MODE" = "primary" ]; then
    echo "  RUN --probe FIRST (lambda=100, 1 seed). Stop if it is pathological."
    echo
    echo "  reading protocol, fixed in advance:"
    echo "    totals fall with lambda, coset flat/down  -> the prior helps; find the optimum"
    echo "    COSET climbs, convergence flat            -> ceiling found, H1 live; note where"
    echo "    convergence climbs                        -> gate not confining; check gate_open_fraction"
    echo "    nothing moves even at lambda=100          -> go to the normalization proposal"
    echo "  coset is the PRIMARY endpoint: the gate protects convergence by construction,"
    echo "  so high lambda should CONVERT the failure mode rather than remove it."
fi
echo "submit with:  sbatch $SLURM"

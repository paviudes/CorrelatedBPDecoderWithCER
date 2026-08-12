#!/usr/bin/env bash
# ============================================================================
# sweep_gate_cer.sh — does the syndrome GATE change the CER vs no-CER picture?
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_gate_cer.sh --setup     # once: create the debug codename
#     bash misc/sweep_gate_cer.sh             # generate the 2x2 x seeds sweep
#     sbatch ../data/72q_BB_cycles_1_debug/cluster/gatecer_<timestamp>.sh
#
# ---------------------------------------------------------------------------
# THE HYPOTHESIS UNDER TEST
#
# `syndrome_gate_threshold` was absent from `default_hyperparams`, so the -1.0
# fallback lived only in a `get(...)` at src/train.jl:353 and EVERY run to date
# took the UNGATED path (src/loss.jl:400).
#
# Ungated, the auxiliary terms — certainty, sparsity, and the CER correlation
# reward — are applied to every sample, INCLUDING samples whose residual is
# nowhere near clearing the syndrome. For those samples the correlation reward
# competes with the syndrome objective instead of choosing among solutions that
# already satisfy it. That is a mechanism for CONVERGENCE failures specifically:
# BP is pulled off the path to a syndrome-clearing solution rather than tipped
# into the wrong logical coset once it gets there.
#
# Gated (tau > 0), `syndrome_gate_per_sample` opens the aux terms only where the
# soft syndrome weight is below tau — "tau = 0.5 => open below half a broken
# check" — and layer selection sees base_loss alone. That is the constrained
# form: satisfy the syndrome first, then let the prior choose among the
# solutions that do.
#
# NOTE ON WHAT THIS DOES *NOT* RE-TEST. A separate question — whether the
# correlation term can change WHICH LAYER the softmin commits to — was measured
# directly from logs/debugging_*_individual_losses.csv and answered NO: across
# 473 batches where base_loss actually varies between layers, the correlation
# term moved the argmin exactly 0 times, and its across-layer spread is 0.06% of
# base_loss. The gate hypothesis here is about per-SAMPLE competition, which
# that measurement does not address.
#
# ---------------------------------------------------------------------------
# THE DESIGN — 2 x 2 x seeds, everything else fixed
#
#     arm        use_CER   correlation term   syndrome_gate_threshold
#     cer_ungated  true      active             -1  (historical)
#     cer_gated    true      active             tau
#     nocer_ungated false     inactive           -1  (historical)
#     nocer_gated   false     inactive           tau
#
# The no-CER gated arm is not padding: the gate also governs sparsity and
# certainty, which carry far more across-layer weight than the correlation term
# (sparsity is 23.5% of the total loss, correlation 0.05%). Without it, any
# improvement from gating the CER arm could not be attributed to the correlation
# term rather than to gating in general.
#
# Read the DIAGNOSTIC split, not the total:
#   convergence failures down in cer_gated but flat in nocer_gated
#       -> the ungated correlation reward was obstructing syndrome clearing. H2.
#   convergence failures down in BOTH
#       -> gating helps generally (most likely via sparsity); not a CER story.
#   coset failures move instead
#       -> H1, and the gate is not the relevant knob.
#   nothing moves beyond the seed spread
#       -> the gate is not the explanation; look elsewhere.
#
# ---------------------------------------------------------------------------
# TWO FAILURES FROM THE LAST RUN THAT THIS FIXES
#
#   GPU OOM. Four concurrent workers each opened a CUDA context on ONE 20 GB MIG
#   and two died at `cuDevicePrimaryCtxRetain` — context creation, not our
#   tensors. MIG has no MPS, so contexts do not share. Here TRAINING runs at full
#   concurrency (it is CPU-only: Enzyme cannot differentiate GPU allocation) and
#   TESTING runs SERIALLY afterwards, in the same job. One CUDA context at a time.
#
#   DEBUG LOG COLLISION. `debugging_<source><seed>` carried no run_tag, so four
#   points wrote one file and three were lost. src/train.jl now includes
#   cer_tag/run_tag, so each arm gets its own log.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------- defaults ---
WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_debug"
SOURCE_CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_5_corrs.toml"
PVAL="0.0007"
NLAYERS=90
SEEDS="1 2 3 4 5"
GATE_TAU="0.5"                   # "open below half a broken check" (loss.jl docstring)
ANNEALED_LAMBDA="1e-2,1,0.7,up"  # the schedules the investigated runs used
ANNEALED_SPARSITY="0,5e-1,0.8,up"
CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""                      # "" => auto from the training concurrency
TEST_JOBS=1                      # concurrent CUDA contexts. 1 is what stops the OOM.
WALLTIME="16:00:00"
HEAP_HINT="4G"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --setup)        SETUP=1;          shift;;
        --pval)         PVAL="$2"; CER_DATA="correlated_weights_p_${PVAL}_s_1.txt"; shift 2;;
        --seeds)        SEEDS="$2";       shift 2;;
        --tau)          GATE_TAU="$2";    shift 2;;
        --nlayers)      NLAYERS="$2";     shift 2;;
        --gpu_type)     GPU_TYPE="$2";    shift 2;;
        --test_jobs)    TEST_JOBS="$2";   shift 2;;
        --walltime)     WALLTIME="$2";    shift 2;;
        --account)      ACCOUNT="$2";     shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

if [ "${SETUP:-0}" = "1" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"; dst="$WORKDIR/$CODENAME"
    [ -d "$src" ] || { echo "source codename missing: $src (run from expts/)" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        [ -e "$dst/$shared" ] || { ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"; echo "  linked  $shared"; }
    done
    cp -n "$src"/models/*.toml "$dst/models/" 2>/dev/null || true
    echo "  copied  models/*.toml"; echo; echo "  $dst ready."
    exit 0
fi

MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR — run --setup first" >&2; exit 1; }
[ -f "$MODELS_DIR/$BASE_HP" ] || { echo "no base hyperparams: $MODELS_DIR/$BASE_HP" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_CMDS="$CLUSTER_DIR/gatecer_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/gatecer_test_${TS}.txt"
HP_LIST="$CLUSTER_DIR/gatecer_hp_${TS}.txt"
SLURM="$CLUSTER_DIR/gatecer_${TS}.sh"
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"; : > "$HP_LIST"

write_point() {   # <hp_name> <run_tag> <use_cer> <gate_tau> <seed>
    local hp_name="$1" run_tag="$2" use_cer="$3" gate="$4" seed="$5"
    local lambda="$ANNEALED_LAMBDA"
    # With use_CER = false the correlation term is inactive regardless; pin
    # lambda to 0 so the TOML cannot be misread as carrying a live CER term.
    if [ "$use_cer" = "false" ]; then lambda="0,0,0.7,up"; fi
    grep -vE '^[[:space:]]*(correlation_weight|sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|correlation_importance)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_gate_cer.sh ($TS) ----
retrain = true
run_tag = "${run_tag}"
use_CER = $use_cer
seed = $seed

# THE VARIABLE UNDER TEST. <= 0 keeps the historical ungated path; > 0 applies
# the aux terms only to samples already near a syndrome-clearing residual.
syndrome_gate_threshold = ${gate}

# Both schedules exactly as in the runs being explained, so the only differences
# across arms are use_CER and the gate.
correlation_weight = "${lambda}"
sparsity_importance = "${ANNEALED_SPARSITY}"
single_qubit_rescale = 0.1
EOF
    echo "$hp_name" >> "$HP_LIST"
}

emit_pair() {   # <hp_name>
    local hp="$1"
    # TRAIN only — no --test, so no CUDA context is created. Full concurrency.
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $CER_DATA --isdebug true --quiet true" \
         "--train train_p_${PVAL}_s_1.txt" >> "$TRAIN_CMDS"
    # TEST — retrain flipped to false in between, so this LOADS the model.
    # Run serially: one CUDA context at a time.
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $CER_DATA --quiet true --diagnose true" \
         "--train train_p_${PVAL}_s_1.txt --test test_p_${PVAL}_s_1.txt" >> "$TEST_CMDS"
}

n_points=0
for seed in $SEEDS; do
    for arm in cer nocer; do
        use_cer=true;  [ "$arm" = "nocer" ] && use_cer=false
        for gate_label in ungated gated; do
            gate="-1.0"; [ "$gate_label" = "gated" ] && gate="$GATE_TAU"
            run_tag="_gc_${arm}_${gate_label}"
            hp="hyperparams_gc_${arm}_${gate_label}_seed${seed}.toml"
            write_point "$hp" "$run_tag" "$use_cer" "$gate" "$seed"
            emit_pair "$hp"
            n_points=$((n_points + 1))
        done
    done
done

# --------------------------------------------------- GPU bundle sizing ---
# Sized by the TRAINING concurrency (cores). The GPU is only touched by the
# serial test pass at the end, but the bundle ratio is what keeps the cores
# from costing extra: Narval is 1 RGU = 3.00 cores = 31.1 GB.
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
# Testing is serial, so each worker may use the whole instance.
GPU_MEMORY_PER_SLOT="$(( (VRAM_GB * 1024) / TEST_JOBS ))M"

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=gatecer_$TS
#SBATCH --output=$CLUSTER_DIR/gatecer_${TS}.out
#SBATCH --error=$CLUSTER_DIR/gatecer_${TS}.err
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=$SLOTS
#SBATCH --mem=$MEM
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# 2x2 gate x CER, $n_points point(s).
# PHASE 1 trains all points at $SLOTS-way concurrency, CPU only (Enzyme AD).
# PHASE 2 tests them $TEST_JOBS at a time on the GPU. Splitting the phases is
# what prevents the CUDA-context OOM that killed 2 of 4 points last time.
set -uo pipefail
echo "gate x CER sweep started: \$(date)"
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

# Precompile HERE: CUDA_Runtime_jll must see a real driver or "no CUDA runtime
# found" is baked into the cache. LocalPreferences.toml must keep
# local_toolkit = true — artifact mode tries to download on an offline node.
if ! timeout 1800 julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    echo "ERROR: precompilation failed or timed out." >&2; exit 1
fi
export JULIA_PKG_PRECOMPILE_AUTO=0
julia --project=\$SLURM_SUBMIT_DIR/.. -e 'using CUDA; @info "CUDA functional: \$(CUDA.functional())"' \\
    || echo "WARNING: CUDA not functional — the test phase would fall back to CPU."

LOCAL_WORK_DIR="\$SLURM_TMPDIR/$CODENAME"
echo "staging $CODENAME -> \$SLURM_TMPDIR"
tar -chf - -C "\$(dirname $WORKDIR/$CODENAME)" "\$(basename $WORKDIR/$CODENAME)" | tar -xf - -C "\$SLURM_TMPDIR"

TRAIN_LOCAL="\$SLURM_TMPDIR/gatecer_train.txt"
TEST_LOCAL="\$SLURM_TMPDIR/gatecer_test.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" > "\$TRAIN_LOCAL"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS"  > "\$TEST_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/gatecer_${TS}"
mkdir -p "\$LOCAL_LOGS/train" "\$LOCAL_LOGS/test"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    echo "[stage-out] \$(date '+%F %T')"
    DIRS=()
    for d in results models logs cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    [ \${#DIRS[@]} -gt 0 ] && tar -cf - --exclude='hyperparams_gc_*.toml' \\
        -C "\$LOCAL_WORK_DIR" "\${DIRS[@]}" | tar -xf - -C "$WORKDIR/$CODENAME"
}
trap 'stage_out; exit 0' TERM
trap stage_out EXIT

######################################################################
# PHASE 1 — TRAIN (CPU only; no --test, so no CUDA context is created)
######################################################################
export USE_GPU=0
echo "[phase 1] training \$(wc -l < "\$TRAIN_LOCAL") point(s), $SLOTS at a time: \$(date)"
parallel --jobs $SLOTS --results "\$LOCAL_LOGS/train" < "\$TRAIN_LOCAL" &
wait \$!
echo "[phase 1] done: \$(date)"

# Flip retrain = true -> false in every generated TOML so the test pass LOADS
# the model it just trained instead of training it again on the GPU.
while read -r hp; do
    f="\$LOCAL_WORK_DIR/models/\$hp"
    [ -f "\$f" ] || continue
    sed -E 's|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true([[:space:]]*(#.*)?)\$|\1false\2|' "\$f" > "\$f.tmp" && mv "\$f.tmp" "\$f"
done < "$HP_LIST"
echo "[phase 1] retrain flipped to false in \$(wc -l < "$HP_LIST") TOML(s)"

######################################################################
# PHASE 2 — TEST on the GPU, $TEST_JOBS at a time
######################################################################
export USE_GPU=1
export GPU_MEMORY=$GPU_MEMORY_PER_SLOT
echo "[phase 2] testing \$(wc -l < "\$TEST_LOCAL") point(s), $TEST_JOBS at a time: \$(date)"
parallel --jobs $TEST_JOBS --results "\$LOCAL_LOGS/test" < "\$TEST_LOCAL" &
wait \$!
echo "[phase 2] done: \$(date)"
echo "gate x CER sweep finished: \$(date)"
EOF
chmod +x "$SLURM"

echo "[gate x CER] $n_points point(s) = 2 arms x 2 gate settings x $(echo $SEEDS | wc -w) seed(s)"
echo
printf "  %-16s %-9s %-24s %s\n" "run_tag" "use_CER" "syndrome_gate_threshold" "correlation term"
printf "  %-16s %-9s %-24s %s\n" "_gc_cer_ungated"   "true"  "-1.0"       "active (lambda 0.01->0.76)"
printf "  %-16s %-9s %-24s %s\n" "_gc_cer_gated"     "true"  "$GATE_TAU"  "active, gated"
printf "  %-16s %-9s %-24s %s\n" "_gc_nocer_ungated" "false" "-1.0"       "inactive"
printf "  %-16s %-9s %-24s %s\n" "_gc_nocer_gated"   "false" "$GATE_TAU"  "inactive"
echo
echo "  seeds       -> $SEEDS  (same set in every arm)"
echo "  p           -> $PVAL, $NLAYERS layers"
echo "  GPU         -> $GPU_TYPE, $SLOTS core(s), $MEM"
echo "  phase 1     -> train $n_points point(s), $SLOTS at a time, $TRAIN_WAVES wave(s), USE_GPU=0"
echo "  phase 2     -> test serially ($TEST_JOBS at a time), GPU_MEMORY=$GPU_MEMORY_PER_SLOT"
echo "  every point -> --isdebug true (train) and --diagnose true (test)"
echo
echo "  read the CONVERGENCE vs COSET split, not the totals:"
echo "    convergence down in cer_gated but flat in nocer_gated -> H2 confirmed"
echo "    convergence down in BOTH                              -> gating helps generally"
echo "    coset moves instead                                   -> H1, gate irrelevant"
echo
echo "submit with:  sbatch $SLURM"

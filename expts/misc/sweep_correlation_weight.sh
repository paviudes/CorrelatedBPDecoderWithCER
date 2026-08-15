#!/usr/bin/env bash
# ============================================================================
# sweep_correlation_weight.sh — CER vs no-CER across p, with the REVISED J
#                               convention and sparsity switched OFF
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_correlation_weight.sh --setup      # once: create the codename
#     bash misc/sweep_correlation_weight.sh --check      # verify the CER files only
#     bash misc/sweep_correlation_weight.sh --probe      # 1 p, 1 seed, both arms
#     bash misc/sweep_correlation_weight.sh              # primary: 3 p x 2 arms x 3 seeds
#     bash misc/sweep_correlation_weight.sh --ungated    # contrast: historical tau = -1
#     bash misc/sweep_correlation_weight.sh --collect    # summarise
#
#     sbatch ../data/72q_BB_cycles_1_debug/cluster/cw_<mode>_<timestamp>.sh
#
# ---------------------------------------------------------------------------
# WHY THIS SWEEP EXISTS — TWO CHANGES SINCE THE LAST CER RUN
#
# (1) THE J CONVENTION WAS CORRECTED.
#
# For binary e_i, e_j in {0,1}, the pairwise log-linear (Ising) prior is
#
#     log P(e_i, e_j) = c + h_i e_i + h_j e_j + J_ik e_i e_k
#
# Reading off the four cells and eliminating c, h_i, h_j gives, uniquely,
#
#     J_ik = log P11 - log P10 - log P01 + log P00
#          = log[ P00 * P11 / (P01 * P10) ]          <-- the 2x2 log ODDS RATIO
#
# That is the only combination of the four cell probabilities in which c, h_i
# and h_j all cancel, i.e. the only one that isolates the INTERACTION from the
# single-qubit fields. It is what `src/loss.jl` documents and what the loss term
#
#     L_corr = -(1/(N|C|)) sum_C J_ik sigma_i sigma_k
#
# requires, because the single-qubit fields h_i are already carried separately
# by `initial_llrs = log((1-p_i)/p_i)`.
#
# The previously supplied J was the pointwise mutual information,
# log[P11 / (P_i P_j)]. It is a legitimate association measure and it vanishes
# under independence exactly as the log odds ratio does — but it still contains
# -log P_i - log P_j, i.e. marginal information that `initial_llrs` has already
# supplied. Adding it as a PAIRWISE coupling double-counts the marginals.
#
# HOW MUCH DID IT ACTUALLY CHANGE? Measured on the three files, old vs new:
#
#     p        old mean   new mean   mean delta   Spearman(old,new)   sign flips
#     0.0005    +1.806     +2.011      +0.205         0.99986           29/540
#     0.0007    +1.639     +1.817      +0.178         0.99991           14/540
#     0.0019    +1.131     +1.284      +0.153         0.99995            1/540
#
# The single-qubit rates are byte-identical between the two vintages; only the
# pair block moved. The shift is a near-uniform +0.15..+0.21 (about +10%) with
# rank order essentially preserved. Analytically that is expected: the two
# differ by log P00 - log(1 - P11/P_i) - log(1 - P11/P_j), which is small and
# positive whenever P11 << P_i, P_j.
#
# CONSEQUENCE, STATED PLAINLY: the convention error was real but numerically
# minor. It does NOT explain the previous null results, and it does not
# retroactively invalidate them — a term contributing 0.05% of the total loss
# does not become decisive under a 10% rescale. Expect this sweep to reproduce
# the null unless (2) below is what was actually binding.
#
# (2) SPARSITY IS PINNED TO ZERO.
#
# `sparsity_importance` was previously annealed into the 0.3-0.5 band, where it
# is ~400x the correlation term's contribution. Since sparsity penalises
# predicted errors and the correlation term rewards co-occurring ones, they pull
# opposite ways and sparsity wins by three orders of magnitude. Removing it is
# the cleanest test of whether the couplings do anything once nothing is
# actively cancelling them.
#
#   NOTE: the base TOML hyperparams_epochs_5_corrs.toml ALREADY carries
#   sparsity_importance = "0,0,0.8,up". This script pins it explicitly anyway so
#   the sweep is self-describing and survives edits to the base file.
#
# (3) THE GATE IS ON BY DEFAULT (tau = 0.5). THIS IS NOT WHAT THE BASE TOML DOES.
#
# The base TOML omits `syndrome_gate_threshold`, so it defaults to -1 = ungated,
# and every previous run was ungated. Running THIS sweep ungated would be wrong,
# for three reasons that compound exactly when sparsity is zeroed.
#
#   (a) It contradicts the design. `src/loss.jl` is explicit: base_loss is the
#       only term that identifies the correct answer, and certainty / sparsity /
#       correlation are "selectors WITHIN that zero set" — they are meant to
#       break ties on the flat solution manifold, not to be minimised in their
#       own right on samples that have not yet cleared the syndrome.
#
#   (b) Ungated, the aux terms enter LAYER SELECTION. The ungated path softmins
#       over base + aux combined, so a layer can win selection by being
#       confident and co-activating rather than by clearing the syndrome. The
#       gated path softmins over base ALONE and adds gated aux afterwards. With
#       sparsity zeroed there is nothing left pulling the other way, so this is
#       the configuration in which that failure mode is most available.
#
#   (c) The certainty term is BATCH_SIZE TIMES STRONGER ungated. Compare:
#         gated:    certainty_per_sample -> sum(gate .* aux_j) / n_samples
#         ungated:  syndrome_loss_regularizer = sum(...)   [NO /n_samples]
#       At n_bits = 72 a fully fractional sample carries ~72*log2 = 49.9 nats, so
#       at the annealed ceiling alpha_cert = 1e-2 the gated contribution is ~0.50
#       while the ungated one is ~0.50 * batch_size = ~10.0 — against a base loss
#       scale of ~1 per softly broken check. Ungated, the regulariser rather than
#       the syndrome dominates the objective. That has been true of every earlier
#       run and was survivable because sparsity opposed it; with sparsity = 0 the
#       correlation reward (which lowers loss by raising sigma) and the certainty
#       penalty (which lowers loss by driving sigma binary) both push the same
#       way on coupled pairs, unopposed.
#
# Gated, the correlation reward r_j <= 0 is ordering-safe by construction: it can
# only improve a sample that already clears H, so it can never make a solved
# sample lose to a failing one. That is the property this sweep needs.
#
# `--ungated` reproduces the historical tau = -1 configuration for comparability.
# Expect it to be worse; run it as a contrast, not as the primary.
#
# ---------------------------------------------------------------------------
# THE GRID
#
#   arm    use_CER   couplings   priors                  role
#   cer     true      revised J   CER single-qubit        treatment
#   nocer   false     none        flat p = 0.1            baseline
#
#   x p in {0.0005, 0.0007, 0.0019}   (the three p with CER data present)
#   x seeds {1 2 3}                   (paired: the SAME seeds on both arms, so
#                                      the contrast is a paired t, not a pooled z)
#
# The p axis matters here in a way it did not in the lambda sweep: the co-firing
# rate that drives the correlation term goes as p^2, so the term is ~14x weaker
# at p = 5e-4 than at p = 1.9e-3. If the couplings ever help, the largest p is
# where it should show first.
#
# `require_correlations = true` on every CER arm, so a missing-pairs CER file
# raises instead of quietly masquerading as a null result.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORKDIR="./../data"
CODENAME="72q_BB_cycles_1_debug"
SOURCE_CODENAME="72q_BB_cycles_1"
BASE_HP="hyperparams_epochs_5_corrs.toml"
PVALS="0.0005 0.0007 0.0019"
SEEDS="1 2 3"
NLAYERS=90
SPARSITY="0.0"                 # pinned CONSTANT — the point of the sweep
GATE_TAU="0.5"                 # gate ON by default; --ungated for tau = -1
LAMBDA=""                      # empty = inherit correlation_weight from base TOML
SINGLE_QUBIT_RESCALE="0.1"     # inherited from the base TOML; exposed for clarity

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""
TEST_JOBS=1                    # serial GPU phase: one MIG permits one CUDA context
WALLTIME="20:00:00"
HEAP_HINT="4G"
MODE="primary"

usage() { sed -n '2,120p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --setup)     MODE="setup";   shift;;
        --check)     MODE="check";   shift;;
        --collect)   MODE="collect"; shift;;
        --probe)     MODE="probe";   shift;;
        --ungated)   GATE_TAU="-1.0"; shift;;
        --pvals)     PVALS="$2";     shift 2;;
        --seeds)     SEEDS="$2";     shift 2;;
        --sparsity)  SPARSITY="$2";  shift 2;;
        --lambda)    LAMBDA="$2";    shift 2;;
        --tau)       GATE_TAU="$2";  shift 2;;
        --rescale)   SINGLE_QUBIT_RESCALE="$2"; shift 2;;
        --base_hp)   BASE_HP="$2";   shift 2;;
        --codename)  CODENAME="$2";  shift 2;;
        --nlayers)   NLAYERS="$2";   shift 2;;
        --gpu_type)  GPU_TYPE="$2";  shift 2;;
        --walltime)  WALLTIME="$2";  shift 2;;
        --account)   ACCOUNT="$2";   shift 2;;
        --outdir)    OUTDIR="$2";    shift 2;;
        -h|--help)   usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

CER_DIR="$WORKDIR/$CODENAME/correlated_weights"
MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"

# ------------------------------------------------------------------ setup ---
if [ "$MODE" = "setup" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"; dst="$WORKDIR/$CODENAME"
    [ -d "$src" ] || { echo "source codename missing: $src (run from expts/)" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        [ -e "$dst/$shared" ] || { ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"; echo "  linked  $shared"; }
    done
    cp -n "$src"/models/*.toml "$dst/models/" 2>/dev/null || true
    echo "  $dst ready."
    echo "  NOTE: if correlated_weights is a SYMLINK back to $SOURCE_CODENAME, the"
    echo "        revised-J files must be placed there, not in the debug copy."
    exit 0
fi

# ------------------------------------------------------------------ collect --
if [ "$MODE" = "collect" ]; then
    results_dir="$WORKDIR/$CODENAME/results"
    [ -d "$results_dir" ] || { echo "no results dir: $results_dir" >&2; exit 1; }
    extra=""; [ -n "${OUTDIR:-}" ] && extra="--outdir ${OUTDIR}"
    exec julia --project="$SCRIPT_DIR/../../" "$SCRIPT_DIR/collect_correlation_weight.jl" "$results_dir" $extra
fi

# ------------------------------------------------------- CER file preflight --
# The entire sweep is about the two-qubit couplings, so a CER file that parses to
# zero pairs would produce a "null result" that is really a missing-data bug.
# `require_correlations = true` catches it inside Julia; this catches it before
# burning a GPU allocation, and additionally reports the J statistics so a
# convention regression is visible at submit time.
check_cer_files() {
    local ok=1
    printf "  %-10s %-8s %-7s %-10s %-10s %-10s %s\n" \
           "p" "singles" "pairs" "J mean" "J min" "J max" "% J<0"
    for p in $PVALS; do
        local f="$CER_DIR/correlated_weights_p_${p}_s_1.txt"
        if [ ! -f "$f" ]; then
            printf "  %-10s MISSING: %s\n" "$p" "$f"; ok=0; continue
        fi
        awk -F: -v p="$p" '
            /^\(/ { n_pair++; v=$2+0; s+=v; if (n_pair==1){mn=v;mx=v}
                    if (v<mn) mn=v; if (v>mx) mx=v; if (v<0) neg++; next }
            NF==2 { n_single++ }
            END {
                if (n_pair == 0) { printf "  %-10s %-8d %-7d  NO PAIR ENTRIES\n", p, n_single, 0; exit 3 }
                printf "  %-10s %-8d %-7d %-+10.4f %-+10.4f %-+10.4f %.1f%%\n",
                       p, n_single, n_pair, s/n_pair, mn, mx, 100*neg/n_pair
            }' "$f" || ok=0
    done
    [ "$ok" = "1" ] || return 1
    return 0
}

if [ "$MODE" = "check" ]; then
    echo "CER data in $CER_DIR"
    echo "  (expecting J = log[P00*P11/(P01*P10)], the revised convention)"
    echo
    check_cer_files || { echo "PREFLIGHT FAILED." >&2; exit 1; }
    exit 0
fi

# ------------------------------------------------------------------ primary --
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR — run --setup first" >&2; exit 1; }
[ -f "$MODELS_DIR/$BASE_HP" ] || { echo "no base hyperparams: $MODELS_DIR/$BASE_HP" >&2; exit 1; }
mkdir -p "$CLUSTER_DIR"

if [ "$MODE" = "probe" ]; then
    PVALS="0.0007"; SEEDS="1"; WALLTIME="8:00:00"
fi

echo "CER data preflight:"
check_cer_files || { echo "PREFLIGHT FAILED — refusing to submit." >&2; exit 1; }
echo

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_CMDS="$CLUSTER_DIR/cw_${MODE}_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/cw_${MODE}_test_${TS}.txt"
HP_LIST="$CLUSTER_DIR/cw_${MODE}_hp_${TS}.txt"
SLURM="$CLUSTER_DIR/cw_${MODE}_${TS}.sh"
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"; : > "$HP_LIST"

tag_of() { echo "$1" | tr '.' 'p' | tr -d '-'; }

# `src/loss.jl` switches on `syndrome_gate_threshold <= 0`, so the label must be
# decided by the same NUMERIC test — a string compare against "-1.0" would call
# "-1" or "0" gated and silently mislabel every output file of such a run.
gate_label="ungated"
if awk -v tau="$GATE_TAU" 'BEGIN { exit !(tau > 0) }'; then
    gate_label="gated"
fi

# `run_tag` deliberately omits p and the seed: p is already in the training
# source (train_p_<p>_s_1) and the seed is auto-appended as _seed_<n>, so the
# three axes are all recoverable from the filename without duplication.
write_point() {   # <hp_name> <run_tag> <use_cer> <seed>
    local hp_name="$1" run_tag="$2" use_cer="$3" seed="$4"
    local require="true"
    if [ "$use_cer" = "false" ]; then
        require="false"
    fi
    grep -vE '^[[:space:]]*(sparsity_importance|retrain|run_tag|use_CER|seed|single_qubit_rescale|syndrome_gate_threshold|require_correlations)[[:space:]]*=' \
        "$MODELS_DIR/$BASE_HP" > "$MODELS_DIR/$hp_name"
    cat >> "$MODELS_DIR/$hp_name" <<EOF

# ---- injected by sweep_correlation_weight.sh ($MODE, $TS) ----
retrain = true
run_tag = "${run_tag}"
use_CER = $use_cer
seed = $seed

# PINNED CONSTANT at ${SPARSITY} (min == max, so no annealing). Sparsity penalises
# predicted errors while the correlation term rewards co-occurring ones; at the
# previously annealed 0.3-0.5 it was ~400x larger and simply cancelled the
# couplings. Zero removes the counterweight so the couplings can be judged.
sparsity_importance = "${SPARSITY},${SPARSITY},0.8,up"

# Gate ON (tau > 0) confines certainty / sparsity / correlation to samples whose
# soft H-syndrome is already (near-)satisfied, and keeps layer selection on
# base_loss alone. This OVERRIDES the base TOML, which omits the key and so
# defaults to -1 = ungated. Three reasons, all sharper at sparsity = 0: the aux
# terms are tie-breakers on the solution manifold by design; ungated they enter
# layer selection, so a layer can win by co-activating instead of clearing the
# syndrome; and ungated the certainty regulariser is batch_size (= 20) times
# stronger, because syndrome_loss_regularizer sums over the batch while the
# gated path divides by n_samples. See the header.
#
# NOTE TO EDITORS: this heredoc is UNQUOTED (<<EOF, not <<'EOF') because it has
# to expand the sweep's variables, so backticks and dollar-paren inside it are
# command-substituted — a backtick-quoted identifier here silently vanishes from
# the generated TOML. Keep prose in this block free of both.
syndrome_gate_threshold = ${GATE_TAU}

single_qubit_rescale = ${SINGLE_QUBIT_RESCALE}

# Refuse to run if the CER file yielded no couplings, so a missing-pairs file
# cannot masquerade as a null result in a sweep whose entire content is pairs.
require_correlations = ${require}
EOF
    if [ -n "$LAMBDA" ]; then
        grep -vE '^[[:space:]]*correlation_weight[[:space:]]*=' "$MODELS_DIR/$hp_name" > "$MODELS_DIR/$hp_name.tmp"
        mv "$MODELS_DIR/$hp_name.tmp" "$MODELS_DIR/$hp_name"
        cat >> "$MODELS_DIR/$hp_name" <<EOF

# lambda PINNED CONSTANT by --lambda (overrides the base TOML's anneal).
correlation_weight = "${LAMBDA},${LAMBDA},0.7,up"
EOF
    fi
    echo "$hp_name" >> "$HP_LIST"
}

emit_pair() {   # <hp_name> <p>
    local hp="$1" p="$2"
    local cer_data="correlated_weights_p_${p}_s_1.txt"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $cer_data --isdebug true --quiet true" \
         "--train train_p_${p}_s_1.txt" >> "$TRAIN_CMDS"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $cer_data --quiet true --diagnose true" \
         "--train train_p_${p}_s_1.txt --test test_p_${p}_s_1.txt" >> "$TEST_CMDS"
}

n_points=0
for p in $PVALS; do
  for seed in $SEEDS; do
    for arm in cer nocer; do
        use_cer="true"
        if [ "$arm" = "nocer" ]; then
            use_cer="false"
        fi
        run_tag="_cw${arm}_${gate_label}_sp$(tag_of "$SPARSITY")"
        hp="hyperparams_cw_${arm}_${gate_label}_sp$(tag_of "$SPARSITY")_p$(tag_of "$p")_seed${seed}.toml"
        write_point "$hp" "$run_tag" "$use_cer" "$seed"
        emit_pair "$hp" "$p"
        n_points=$((n_points + 1))
    done
  done
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
#SBATCH --job-name=cw_${MODE}_$TS
#SBATCH --output=$CLUSTER_DIR/cw_${MODE}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/cw_${MODE}_${TS}.err
#SBATCH --gpus=${GPU_TYPE}:1
#SBATCH --cpus-per-task=$SLOTS
#SBATCH --mem=$MEM
#SBATCH --time=$WALLTIME
#SBATCH --signal=B:TERM@600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$EMAIL

# CER vs no-CER across p, revised J convention, sparsity = $SPARSITY ($MODE), $n_points point(s).
# PHASE 1 trains at $SLOTS-way concurrency, CPU only (Enzyme AD cannot use a GPU).
# PHASE 2 tests $TEST_JOBS at a time — one MIG permits one CUDA context, and four
# concurrent contexts on a 20 GB MIG previously killed 2 of 4 runs at
# cuDevicePrimaryCtxRetain.
set -uo pipefail
echo "correlation-weight sweep ($MODE) started: \$(date)"
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

TRAIN_LOCAL="\$SLURM_TMPDIR/cw_train.txt"; TEST_LOCAL="\$SLURM_TMPDIR/cw_test.txt"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TRAIN_CMDS" > "\$TRAIN_LOCAL"
sed "s|\\\$WORKDIR_RUNTIME|\$SLURM_TMPDIR|g" "$TEST_CMDS"  > "\$TEST_LOCAL"

LOCAL_LOGS="\$LOCAL_WORK_DIR/cluster/logs/cw_${MODE}_${TS}"
mkdir -p "\$LOCAL_LOGS/train" "\$LOCAL_LOGS/test"

stage_out_done=0
stage_out() {
    [ "\$stage_out_done" = "1" ] && return 0
    stage_out_done=1
    DIRS=()
    for d in results models logs cluster/logs; do
        [ -d "\$LOCAL_WORK_DIR/\$d" ] && DIRS+=("\$d")
    done
    [ \${#DIRS[@]} -gt 0 ] && tar -cf - --exclude='hyperparams_cw_*.toml' \\
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
echo "correlation-weight sweep ($MODE) finished: \$(date)"
EOF
chmod +x "$SLURM"

n_p=$(echo $PVALS | wc -w); n_seeds=$(echo $SEEDS | wc -w)
echo "[correlation-weight $MODE] $n_points point(s)"
echo
printf "  %-28s %-9s %-12s %-9s %s\n" "run_tag" "use_CER" "couplings" "sparsity" "role"
printf "  %-28s %-9s %-12s %-9s %s\n" \
    "_cwcer_${gate_label}_sp$(tag_of "$SPARSITY")" "true" "revised J" "$SPARSITY" "treatment"
printf "  %-28s %-9s %-12s %-9s %s\n" \
    "_cwnocer_${gate_label}_sp$(tag_of "$SPARSITY")" "false" "none" "$SPARSITY" "baseline: flat p=0.1"
echo
echo "  p      -> $PVALS"
echo "  seeds  -> $SEEDS  (SAME set on both arms => paired contrast)"
echo "  grid   -> $n_p p x 2 arms x $n_seeds seed(s) = $n_points"
echo "  base   -> $MODELS_DIR/$BASE_HP"
echo "  gate   -> syndrome_gate_threshold = $GATE_TAU ($gate_label)"
echo "  lambda -> $([ -n "$LAMBDA" ] && echo "$LAMBDA (pinned)" || echo 'inherited from base TOML (annealed 1e-2 -> 1)')"
echo "  GPU    -> $GPU_TYPE, $SLOTS core(s), $MEM; train $TRAIN_WAVES wave(s), test serial"
echo "  assert -> require_correlations = true on every CER arm"
echo
if [ "$MODE" = "primary" ]; then
    echo "  reading protocol, fixed in advance:"
    echo "    CER beats no-CER at every p, gap grows with p  -> couplings are working;"
    echo "                                                      the p^2 co-firing scaling predicts exactly this"
    echo "    CER beats no-CER uniformly, no p trend         -> it is the single-qubit PRIORS, not the couplings;"
    echo "                                                      confirm with --lambda 0 (priors on, couplings off)"
    echo "    no separation at any p                         -> the revised J did not rescue it either;"
    echo "                                                      the 1/|C| normalisation is the remaining suspect"
    echo "    CER WORSE, especially at large p               -> the reward is outrunning the syndrome term;"
    echo "                                                      check gate_open_fraction, then lower --lambda"
    echo
    if [ "$gate_label" = "ungated" ]; then
        echo "  WARNING: running UNGATED with sparsity = $SPARSITY. The aux terms enter layer"
        echo "  selection and the certainty regulariser is ~batch_size stronger than in the"
        echo "  gated path. This is a deliberate contrast against the default; do not read"
        echo "  it as the primary result."
        echo
    fi
fi
echo "submit with:  sbatch $SLURM"

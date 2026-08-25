#!/usr/bin/env bash
# ============================================================================
# sweep_correlation_weight.sh — CER vs no-CER, sweeping the correlation weight
# ============================================================================
# Run from expts/ :
#
#     bash misc/sweep_correlation_weight.sh --check      # verify the CER files
#     bash misc/sweep_correlation_weight.sh --pin_cer    # lock them by SHA-256
#     bash misc/sweep_correlation_weight.sh              # generate the sweep
#     bash misc/sweep_correlation_weight.sh --collect    # summarise the results
#
#     sbatch ../data/<codename>/cluster/cw_<mode>_<timestamp>.sh
#
# Data profile, selecting the codename and every path derived from it:
#     (default)   72q_BB_cycles_1_spread_comparison    per-CNOT Normal(p, sigma)
#     --uniform   72q_BB_cycles_1_debug                uniform p
#
# Other flags:
#     --datasets / --baseline_datasets / --lambdas / --seeds / --sparsity
#     --tau / --certainty / --gpus / --test_jobs / --walltime / --base_hp
#     --probe      one dataset, one seed          (implies --uniform)
#     --lambda_sweep  the historical lambda grid  (implies --uniform)
#     --ungated    tau = -1
#
# Every mode echoes the directory it resolved to; that line is the check.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORKDIR="./../data"
# CODENAME, BASE_HP, DATASETS and SEEDS are all set by the DATA PROFILE below.
# These are placeholders; nothing reads them before the profile has run.
CODENAME=""
BASE_HP=""
PVALS="0.0005 0.0007 0.0019"
SEEDS=""
# Dataset KEYS: train_<key>.txt / test_<key>.txt / correlated_weights_<key>.txt.
# Empty = derive p_<p>_s_1 from PVALS.
DATASETS=""
NLAYERS=90
SPARSITY="0.0"                 # pinned CONSTANT — the point of the sweep
GATE_TAU="0.5"                 # tau, softly broken checks; --ungated for tau = -1
CERTAINTY="2.2"                # c, LLR units: pair contributes only if both |mu| > c
LAMBDA=""                      # empty = inherit correlation_weight from base TOML
SINGLE_QUBIT_RESCALE="0.1"     # inherited from the base TOML; exposed for clarity
LAMBDAS=""                     # empty = one CER arm inheriting the base TOML's anneal
INCLUDE_NOCER=1                # --no_nocer drops the flat-prior baseline
# The lambda grid for --lambda_sweep. Deliberately BELOW 1, unlike the retired
# sweep_lambda.sh grid {0, 1, 10, 100}: that was built when the term was inert.
# At lambda = 0.76 it now produces a -4.26 sigma coset effect, so the question is
# no longer "is it strong enough" but "where does the benefit stop paying".
# The correlation term now divides by the number of ACTIVE (gated) pairs rather
# than |C|, so lambda is a weight per contributing pair and the old grid does not
# transfer: 1/|C| at lambda = 0.76 equals 1/n_active at lambda ~ 0.00027.
LAMBDA_GRID="0 0.1 0.3 1.0 3.0"
# Dataset PROFILE, orthogonal to MODE: it selects which codename and which dataset
# keys to work on, so `--spread --check`, `--spread --pin_cer` and a plain
# `--spread` run all address the same data.
# Data profile: selects the codename and every path derived from it.
PROFILE="spread"
PROFILE_EXPLICIT=0
BASE_HP_EXPLICIT=0
SEEDS_EXPLICIT=0
WALLTIME_EXPLICIT=0
# Datasets that get ONLY the {lambda = 0, no-CER} pair rather than the full grid.
BASELINE_DATASETS=""
BASELINE_EXPLICIT=0

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""
# GPUS scales the TRAINING core budget: on Narval a whole a100 is 4.0 RGU = 12
# cores, so N GPUs buys 12N cores and the training phase runs 12N-wide. Training
# is Enzyme reverse-mode AD and never touches the GPU, so the extra cards sit
# idle during phase 1 — that is the price of taking cores from a GPU allocation
# rather than submitting a separate CPU job.
#
# An a100 node is 4x A100-40GB / 48 cores, so GPUS <= 4 stays on ONE node. That
# matters: the stage-in/stage-out uses $SLURM_TMPDIR, which is node-local, so a
# job spanning two nodes would silently lose half its results. --nodes=1 below
# enforces it.
# Empty = as many whole GPUs as it takes to train in one wave, capped at a node.
GPUS=""
# Empty = one test process per GPU. More than one per card OOMs and is slower.
TEST_JOBS=""
# MEASURED, not guessed. The 18-point run of 2026-08-14 took 2h41m end to end:
#   precompile + stage-in   8 min
#   phase 1 (train, 2 waves of 12)  1h23m
#   phase 2 (test, serial)          1h11m   (mean 268 s/point, max 1011 s)
# The 2026-08-20 rerun came in at 2h37m for the same 18 points. 4h is a 1.5x
# margin on a twice-measured number, and asking for more only pushes the job
# down the scheduler's queue.
WALLTIME="4:00:00"
HEAP_HINT="4G"
MODE="primary"

usage() { sed -n '2,120p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)     MODE="check";   shift;;
        --pin_cer)   MODE="pin_cer"; shift;;
        --collect)   MODE="collect"; shift;;
        --probe)     MODE="probe";   shift;;
        --ungated)   GATE_TAU="-1.0"; shift;;
        --lambda_sweep) MODE="lambda_sweep"; shift;;
        --spread)    PROFILE="spread"; PROFILE_EXPLICIT=1; shift;;
        --uniform)   PROFILE="uniform"; PROFILE_EXPLICIT=1; shift;;
        --datasets)  DATASETS="$2";  shift 2;;
        --baseline_datasets) BASELINE_DATASETS="$2"; BASELINE_EXPLICIT=1; shift 2;;
        --lambdas)   LAMBDAS="$2";   shift 2;;
        --no_nocer)  INCLUDE_NOCER=0; shift;;
        --pvals)     PVALS="$2";     shift 2;;
        --seeds)     SEEDS="$2"; SEEDS_EXPLICIT=1; shift 2;;
        --sparsity)  SPARSITY="$2";  shift 2;;
        --lambda)    LAMBDAS="$2";   shift 2;;
        --tau)       GATE_TAU="$2";  shift 2;;
        --certainty) CERTAINTY="$2"; shift 2;;
        --rescale)   SINGLE_QUBIT_RESCALE="$2"; shift 2;;
        --base_hp)   BASE_HP="$2"; BASE_HP_EXPLICIT=1; shift 2;;
        --codename)  CODENAME="$2";  shift 2;;
        --nlayers)   NLAYERS="$2";   shift 2;;
        --gpu_type)  GPU_TYPE="$2";  shift 2;;
        --gpus)      GPUS="$2";      shift 2;;
        --test_jobs) TEST_JOBS="$2";  shift 2;;
        --walltime)  WALLTIME="$2"; WALLTIME_EXPLICIT=1; shift 2;;
        --account)   ACCOUNT="$2";   shift 2;;
        --outdir)    OUTDIR="$2";    shift 2;;
        -h|--help)   usage; exit 0;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

# ------------------------------------------------------- per-gate spread mode --
# The dataset where each CNOT's error rate is drawn from Normal(p, sigma) rather
# than fixed at p. Sigma = 0 is the matched uniform-p baseline.
#
# WHY THIS DATASET EXISTS. On the uniform-p data the CER file was almost
# information-free: the 72 single-qubit rates collapsed to TWO sector levels with
# 1.3% spread inside each, and 82% of the variance in J was explained by a single
# structural feature (how many HZ / HX checks the pair shares), with 216 of 540
# couplings sitting at J = 0.003. BP already knows the check structure, so the
# couplings were telling the decoder something it had.
#
# Per-gate sampling breaks that degeneracy. Run --check to see by how much.
# --lambda_sweep and --probe use p values that only the uniform codename has.
if [ "$PROFILE_EXPLICIT" = "0" ]; then
    case "$MODE" in
        lambda_sweep|probe) PROFILE="uniform" ;;
    esac
fi

if [ "$PROFILE" = "spread" ]; then
        # Per-CNOT Normal(p, sigma). Every dataset gets {lambda = 0, no-CER}; only
    # sigma = 0.001, the most informative, also gets the lambda grid.
    CODENAME="72q_BB_cycles_1_spread_comparison"
    if [ "$BASE_HP_EXPLICIT" = "0" ]; then
        BASE_HP="hyperparams_epochs_5_corrs.toml"
    fi
    if [ -z "$DATASETS" ]; then
        DATASETS="p_0.0005_sig_0.001_s_1 p_0.0005_sig_0.001_s_2 p_0.0005_sig_0.001_s_3"
    fi
    # `none` explicitly clears a list, since empty means "use the default".
    if [ "$BASELINE_DATASETS" = "none" ]; then BASELINE_DATASETS=""; fi
    if [ -z "$BASELINE_DATASETS" ] && [ "$BASELINE_EXPLICIT" = "0" ]; then
        BASELINE_DATASETS="p_0.0005_sig_0.0_s_1 p_0.0005_sig_0.0005_s_1 p_0.0005_sig_0.0005_s_2 p_0.0005_sig_0.0005_s_3"
    fi
    if [ -z "$LAMBDAS" ]; then
        LAMBDAS="0 0.3 3.0"
    fi
    if [ "$SEEDS_EXPLICIT" = "0" ]; then
        # The noise samples ARE the replicates; a network-seed axis on top would
        # multiply the grid without adding an independent source of spread.
        SEEDS="1"
    fi
elif [ "$PROFILE" = "uniform" ]; then
    # Uniform p, kept so the earlier sweeps stay reproducible.
    CODENAME="72q_BB_cycles_1_debug"
    if [ "$BASE_HP_EXPLICIT" = "0" ]; then
        BASE_HP="hyperparams_epochs_5_corrs.toml"
    fi
    if [ "$SEEDS_EXPLICIT" = "0" ]; then
        SEEDS="1 2 3"
    fi
else
    echo "unknown data profile: $PROFILE (expected 'spread' or 'uniform')" >&2
    exit 2
fi

CER_DIR="$WORKDIR/$CODENAME/correlated_weights"
MODELS_DIR="$WORKDIR/$CODENAME/models"
CLUSTER_DIR="$WORKDIR/$CODENAME/cluster"

# hyperparams_epochs_10_corrs.toml matches the established base on every knob
# that the script does not itself override, EXCEPT n_epochs: 10 rather than 5.
# learning_rate, loss_layer_temperature, single_qubit_rescale, batch_size and the
# annealing schedules are all identical, so the two sweeps stay comparable. The
# sparsity difference ("0,5e-1" vs "0,0") is moot because the script pins it, and
# the absent `seed` is injected per point.
#
# n_epochs is NOT moot: it doubles the training phase, which is why the walltime
# below is derived from it rather than hard-coded.

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
#
# --pin_cer records a SHA-256 per CER file in J_FINGERPRINT.txt; later runs compare
# against it and refuse to submit on a mismatch. Commit that file.
FINGERPRINT_FILE="$CER_DIR/J_FINGERPRINT.txt"

resolve_datasets() {
    if [ -z "$DATASETS" ]; then
        DATASETS=""
        for p in $PVALS; do
            DATASETS="$DATASETS p_${p}_s_1"
        done
        DATASETS="$(echo $DATASETS)"
    fi
    # The reference datasets read their CER file too, so the preflight covers them.
    CHECKED_DATASETS="$(echo $DATASETS $BASELINE_DATASETS)"
}

cer_file_for() {
    echo "$CER_DIR/correlated_weights_${1}.txt"
}

sha_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

check_cer_files() {
    local ok=1
    printf "  %-10s %-8s %-7s %-10s %-10s %-10s %-7s %s\n" \
           "p" "singles" "pairs" "J mean" "J min" "J max" "% J<0" "sha256"
    for key in $DATASETS; do
        local f
        f="$(cer_file_for "$key")"
        if [ ! -f "$f" ]; then
            printf "  %-28s MISSING: %s\n" "$key" "$f"; ok=0; continue
        fi
        local short
        short="$(sha_of "$f" | cut -c1-12)"
        awk -F: -v p="$key" -v sha="$short" '
            /^\(/ { n_pair++; v=$2+0; s+=v; if (n_pair==1){mn=v;mx=v}
                    if (v<mn) mn=v; if (v>mx) mx=v; if (v<0) neg++; next }
            NF==2 { n_single++ }
            END {
                if (n_pair == 0) { printf "  %-10s %-8d %-7d  NO PAIR ENTRIES\n", p, n_single, 0; exit 3 }
                printf "  %-10s %-8d %-7d %-+10.4f %-+10.4f %-+10.4f %-7.1f %s\n",
                       p, n_single, n_pair, s/n_pair, mn, mx, 100*neg/n_pair, sha
            }' "$f" || ok=0
    done
    [ "$ok" = "1" ] || return 1
    return 0
}

# Compare the files against the pin, if one exists. A missing pin is a warning
# (the guard is opt-in); a MISMATCH is fatal, because it means the couplings are
# not the ones the pinned analysis was built on.
verify_cer_pin() {
    if [ ! -f "$FINGERPRINT_FILE" ]; then
        echo "  no J_FINGERPRINT.txt — the couplings are UNPINNED. After confirming"
        echo "  the files are the intended vintage, run --pin_cer to lock them."
        return 0
    fi
    local mismatched=0
    local unpinned=0
    for key in $CHECKED_DATASETS; do
        local f expected actual
        f="$(cer_file_for "$key")"
        [ -f "$f" ] || continue
        expected="$(awk -v k="$key" '$1 == k { print $2 }' "$FINGERPRINT_FILE")"
        if [ -z "$expected" ]; then
            echo "  $key is NOT IN THE PIN — add it with --pin_cer."
            unpinned=$((unpinned + 1))
            continue
        fi
        actual="$(sha_of "$f")"
        if [ "$expected" != "$actual" ]; then
            echo "  $key FINGERPRINT MISMATCH" >&2
            echo "      pinned : $expected" >&2
            echo "      actual : $actual" >&2
            mismatched=$((mismatched + 1))
        fi
    done
    if [ "$mismatched" -gt 0 ]; then
        echo >&2
        echo "  $mismatched CER file(s) differ from the pin. These are not the couplings" >&2
        echo "  the pin was taken on. Restore the intended files, or re-pin deliberately" >&2
        echo "  with --pin_cer if the change is intended." >&2
        return 1
    fi
    if [ "$unpinned" -gt 0 ]; then
        echo "  $unpinned of $(echo $CHECKED_DATASETS | wc -w) file(s) are not covered by the pin."
        echo "  A partial pin is not a guard. Re-pin once the data is confirmed."
        return 0
    fi
    echo "  J_FINGERPRINT.txt: all $(echo $CHECKED_DATASETS | wc -w) file(s) match the pin."
    return 0
}

if [ "$MODE" = "pin_cer" ]; then
    resolve_datasets
    echo "CER data in $CER_DIR"
    echo
    check_cer_files || { echo "PREFLIGHT FAILED — refusing to pin." >&2; exit 1; }
    {
        echo "# J_FINGERPRINT.txt — SHA-256 of the CER coupling files this analysis assumes."
        echo "# Written by sweep_correlation_weight.sh --pin_cer on $(date +%Y-%m-%d_%H-%M-%S)."
        echo "# Columns: p  sha256"
        for key in $CHECKED_DATASETS; do
            f="$(cer_file_for "$key")"
            [ -f "$f" ] || continue
            echo "$key $(sha_of "$f")"
        done
    } > "$FINGERPRINT_FILE"
    echo
    echo "  pinned -> $FINGERPRINT_FILE"
    echo "  commit this file so a stale sync cannot pass unnoticed."
    exit 0
fi

if [ "$MODE" = "check" ]; then
    resolve_datasets
    echo "CER data in $CER_DIR"
    echo "  (expecting J = log[P00*P11/(P01*P10)], the revised convention)"
    echo
    check_cer_files || { echo "PREFLIGHT FAILED." >&2; exit 1; }
    echo
    verify_cer_pin || exit 1
    exit 0
fi

# ------------------------------------------------------------------ primary --
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR (run this from expts/)" >&2; exit 1; }
if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
    echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
    echo "  Available in $MODELS_DIR:" >&2
    ls "$MODELS_DIR"/hyperparams_epochs*.toml 2>/dev/null | sed 's|.*/|    |' >&2 \
        || echo "    (none)" >&2
    echo "  Name one with --base_hp, or copy the intended file in." >&2
    exit 1
fi
mkdir -p "$CLUSTER_DIR"

if [ "$MODE" = "probe" ]; then
    PVALS="0.0007"; SEEDS="1"; WALLTIME="3:00:00"
fi

if [ "$MODE" = "lambda_sweep" ]; then
    PVALS="0.0007"
    if [ -z "$LAMBDAS" ]; then
        LAMBDAS="$LAMBDA_GRID"
    fi
fi

resolve_datasets

# Refuse a grid that silently outgrows the walltime.
n_planned=$(( $(echo $DATASETS | wc -w) * $(echo $SEEDS | wc -w) * \
              ( $([ -n "$LAMBDAS" ] && echo $LAMBDAS | wc -w || echo 1) + INCLUDE_NOCER ) \
              + $(echo $BASELINE_DATASETS | wc -w) * $(echo $SEEDS | wc -w) * (1 + INCLUDE_NOCER) ))
if [ "$n_planned" -gt 24 ]; then
    echo "ERROR: $n_planned points requested. The 18-point run took 2h37m at 12-way" >&2
    echo "  training concurrency; beyond ~24 the walltime below stops being credible." >&2
    echo "  Narrow --pvals, --seeds or --lambdas, or raise --walltime deliberately." >&2
    exit 2
fi

echo "CER data preflight:"
check_cer_files || { echo "PREFLIGHT FAILED — refusing to submit." >&2; exit 1; }
verify_cer_pin  || { echo "PREFLIGHT FAILED — refusing to submit." >&2; exit 1; }
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

# Pinned constant (min == max, so no annealing).
sparsity_importance = "${SPARSITY},${SPARSITY},0.8,up"

# tau: per-sample detached gate, in softly broken checks.
syndrome_gate_threshold = ${GATE_TAU}

# c: per-pair detached certainty gate on the correlation term, in LLR units.
correlation_certainty_threshold = ${CERTAINTY}

single_qubit_rescale = ${SINGLE_QUBIT_RESCALE}

# Refuse to run if the CER file yielded no couplings.
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

emit_pair() {   # <hp_name> <dataset_key>
    local hp="$1" key="$2"
    local cer_data="correlated_weights_${key}.txt"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $cer_data --isdebug true --quiet true" \
         "--train train_${key}.txt" >> "$TRAIN_CMDS"
    echo "julia --project=\"./../\" --heap-size-hint=$HEAP_HINT neural_bp_experiments.jl" \
         "--workdir \$WORKDIR_RUNTIME --codename $CODENAME --n_hidden_layers $NLAYERS" \
         "--hyperparams $hp --cer_data $cer_data --quiet true --diagnose true" \
         "--train train_${key}.txt --test test_${key}.txt" >> "$TEST_CMDS"
}

# lambda is part of run_tag when pinned, so points never overwrite each other. The
# no-CER arm carries none: with use_CER = false the weight multiplies nothing.
lambda_list="$LAMBDAS"
if [ -z "$lambda_list" ]; then
    lambda_list="__inherit__"
fi

n_points=0
n_cer=0
n_nocer=0

# The baseline datasets run only {lambda = 0, no-CER}; every other dataset runs
# the full lambda grid. Emitting them from one loop keeps write_point/emit_pair
# and the overwrite guard single-sourced.
all_datasets="$DATASETS $BASELINE_DATASETS"
all_datasets="$(echo $all_datasets)"

for key in $all_datasets; do
  key_tag="$(tag_of "$key")"
  point_lambdas="$lambda_list"
  for baseline_key in $BASELINE_DATASETS; do
      if [ "$key" = "$baseline_key" ]; then
          point_lambdas="0"
      fi
  done
  for seed in $SEEDS; do
    for lam in $point_lambdas; do
        LAMBDA=""
        lam_tag=""
        if [ "$lam" != "__inherit__" ]; then
            LAMBDA="$lam"
            lam_tag="_lam$(tag_of "$lam")"
        fi
        run_tag="_cwcer_${gate_label}_sp$(tag_of "$SPARSITY")${lam_tag}"
        hp="hyperparams_cw_cer_${gate_label}_sp$(tag_of "$SPARSITY")${lam_tag}_${key_tag}_seed${seed}.toml"
        write_point "$hp" "$run_tag" "true" "$seed"
        emit_pair "$hp" "$key"
        n_points=$((n_points + 1)); n_cer=$((n_cer + 1))
    done

    if [ "$INCLUDE_NOCER" = "1" ]; then
        LAMBDA=""
        run_tag="_cwnocer_${gate_label}_sp$(tag_of "$SPARSITY")"
        hp="hyperparams_cw_nocer_${gate_label}_sp$(tag_of "$SPARSITY")_${key_tag}_seed${seed}.toml"
        write_point "$hp" "$run_tag" "false" "$seed"
        emit_pair "$hp" "$key"
        n_points=$((n_points + 1)); n_nocer=$((n_nocer + 1))
    fi
  done
done

# ---- overwrite guard: a run_tag with results on disk will be OVERWRITTEN -------
existing=""
n_existing=0
for key in $all_datasets; do
  for seed in $SEEDS; do
    for tag in $(grep -ho 'run_tag = "[^"]*"' "$MODELS_DIR"/hyperparams_cw_*_$(tag_of "$key")_seed${seed}.toml 2>/dev/null \
                 | sed 's/run_tag = "//; s/"//' | sort -u); do
        # Glob the epoch count rather than reading it from the base TOML: the tag
        # plus `_seed_<n>.csv` is already unique, and `_sp0p0` cannot match
        # `_sp0p0_lam0` because `_seed_` must follow immediately.
        hit="$WORKDIR/$CODENAME/results/simulation_results_test_${key}_"*"_trained_using_train_${key}"*"${tag}_seed_${seed}.csv"
        for f in $hit; do
            if [ -f "$f" ]; then
                existing="$existing\n    $(basename "$f")"
                n_existing=$((n_existing + 1))
            fi
        done
    done
  done
done
if [ "$n_existing" -gt 0 ]; then
    echo "  NOTE: $n_existing existing result file(s) carry a run_tag this sweep also writes."
    echo "  They will be retrained and OVERWRITTEN. Configuration-identical arms should"
    echo "  reproduce byte-for-byte (every point is seeded); if they do not, that is itself"
    echo "  a finding. Back them up first if you want to diff them."
    printf "$existing\n" | head -8
    if [ "$n_existing" -gt 8 ]; then
        echo "    ... and $((n_existing - 8)) more"
    fi
    echo
fi

# ---- GPU bundle, sized by TRAINING concurrency (Narval: 1 RGU = 3.00 cores) ---
if [ -z "$GPU_TYPE" ]; then
    if   [ "$n_points" -le 1 ]; then GPU_TYPE="a100_1g.5gb"
    elif [ "$n_points" -le 3 ]; then GPU_TYPE="a100_2g.10gb"
    elif [ "$n_points" -le 6 ]; then GPU_TYPE="a100_3g.20gb"
    else                             GPU_TYPE="a100"
    fi
fi
case "$GPU_TYPE" in
    a100_1g.5gb)  SLOTS_PER_GPU=1;  MEM_PER_GPU=15;  VRAM_GB=5;  IS_MIG=1 ;;
    a100_2g.10gb) SLOTS_PER_GPU=3;  MEM_PER_GPU=31;  VRAM_GB=10; IS_MIG=1 ;;
    a100_3g.20gb) SLOTS_PER_GPU=6;  MEM_PER_GPU=62;  VRAM_GB=20; IS_MIG=1 ;;
    a100)         SLOTS_PER_GPU=12; MEM_PER_GPU=124; VRAM_GB=40; IS_MIG=0 ;;
    *) echo "unknown --gpu_type: $GPU_TYPE" >&2; exit 2;;
esac

# Default: as many whole GPUs as it takes to train in one wave, capped at a node.
if [ -z "$GPUS" ]; then
    if [ "$IS_MIG" = "1" ]; then
        GPUS=1
    else
        GPUS=$(( (n_points + SLOTS_PER_GPU - 1) / SLOTS_PER_GPU ))
        if [ "$GPUS" -lt 1 ]; then GPUS=1; fi
        if [ "$GPUS" -gt 4 ]; then GPUS=4; fi
    fi
fi

# MIG slices cannot be multiply allocated, and $SLURM_TMPDIR is node-local, so a
# job spanning nodes would lose half its results.
if [ "$IS_MIG" = "1" ] && [ "$GPUS" -gt 1 ]; then
    echo "ERROR: --gpus $GPUS with a MIG partition ($GPU_TYPE). MIG slices cannot be" >&2
    echo "  multiply allocated; use --gpu_type a100 for more than one." >&2
    exit 2
fi
if [ "$GPUS" -gt 4 ]; then
    echo "ERROR: --gpus $GPUS exceeds the 4 GPUs on a Narval a100 node. The job would" >&2
    echo "  span nodes, and \$SLURM_TMPDIR is node-local: the stage-out would silently" >&2
    echo "  collect only the first node's results." >&2
    exit 2
fi

SLOTS=$(( SLOTS_PER_GPU * GPUS ))
MEM="$(( MEM_PER_GPU * GPUS ))G"
[ "$n_points" -lt "$SLOTS" ] && SLOTS=$n_points
TRAIN_WAVES=$(( (n_points + SLOTS - 1) / SLOTS ))

# One test process per card. More than one per card OOMs (GPU_MEMORY sizes the
# prediction batch, it does not bound the process) and measured slower than serial.
if [ -z "$TEST_JOBS" ]; then
    TEST_JOBS=$GPUS
    if [ "$TEST_JOBS" -gt "$n_points" ]; then
        TEST_JOBS=$n_points
    fi
fi

GPU_MEMORY_PER_SLOT="$(( (VRAM_GB * 1024 * GPUS * 85) / (TEST_JOBS * 100) ))M"

# ---- walltime, derived from n_epochs -----------------------------------------
# Measured: 8.3 min per epoch per training wave, 4.5 min per test point, ~8 min
# precompile + stage-in. Floored at 4h to cover in-job precompilation.
N_EPOCHS_BASE=$(grep -E '^[[:space:]]*n_epochs[[:space:]]*=' "$MODELS_DIR/$BASE_HP" \
                | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
if [ -z "$N_EPOCHS_BASE" ]; then
    N_EPOCHS_BASE=5
fi
estimated_minutes=$(( (83 * N_EPOCHS_BASE * TRAIN_WAVES) / 10 + (45 * n_points) / (10 * TEST_JOBS) + 8 ))
if [ "$WALLTIME_EXPLICIT" = "0" ]; then
    # 1.3x margin, rounded up to a whole hour, floored at 4h.
    walltime_hours=$(( ((estimated_minutes * 13) / 10 + 59) / 60 ))
    if [ "$walltime_hours" -lt 4 ]; then
        walltime_hours=4
    fi
    WALLTIME="${walltime_hours}:00:00"
fi

cat > "$SLURM" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=cw_${MODE}_$TS
#SBATCH --output=$CLUSTER_DIR/cw_${MODE}_${TS}.out
#SBATCH --error=$CLUSTER_DIR/cw_${MODE}_${TS}.err
#SBATCH --gpus=${GPU_TYPE}:${GPUS}
#SBATCH --nodes=1
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

# Must be submitted, not executed: SLURM_SUBMIT_DIR/SLURM_TMPDIR only exist in a job.
if [ -z "\${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: this script must be SUBMITTED, not executed." >&2
    echo "         sbatch $SLURM" >&2
    echo "  (running it with 'bash' leaves SLURM_SUBMIT_DIR/SLURM_TMPDIR unset," >&2
    echo "   and would put an 18-point training run on a login node.)" >&2
    exit 1
fi

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

# --isdebug writes training logs here; without the directory they are lost.
mkdir -p "\$LOCAL_WORK_DIR/logs"

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
echo "[phase 2] testing \$(wc -l < "\$TEST_LOCAL") point(s), $TEST_JOBS at a time on $GPUS GPU(s): \$(date)"

# Pin each parallel slot to a card, round-robin. Capture SLURM's own list first:
# on MIG it is UUIDs, not indices, so overwriting it with a number hides the GPU.
export SLURM_CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
echo "[phase 2] SLURM gave CUDA_VISIBLE_DEVICES=\$SLURM_CUDA_VISIBLE_DEVICES"
parallel --jobs $TEST_JOBS --results "\$LOCAL_LOGS/test" \\
    'card=\$(( ({%} - 1) % $GPUS + 1 )); export CUDA_VISIBLE_DEVICES=\$(echo \$SLURM_CUDA_VISIBLE_DEVICES | cut -d, -f\$card); bash -c {}' \\
    < "\$TEST_LOCAL" &
wait \$!
echo "correlation-weight sweep ($MODE) finished: \$(date)"
EOF
chmod +x "$SLURM"

n_p=$(echo $DATASETS | wc -w); n_seeds=$(echo $SEEDS | wc -w)
echo "[correlation-weight $MODE] $n_points point(s)"
echo
printf "  %-34s %-9s %-11s %-9s %s\n" "run_tag" "use_CER" "lambda" "sparsity" "role"
for lam in $lambda_list; do
    lam_tag=""; lam_shown="inherited"; role="CER priors + couplings"
    if [ "$lam" != "__inherit__" ]; then
        lam_tag="_lam$(tag_of "$lam")"; lam_shown="$lam"
    fi
    if [ "$lam" = "0" ]; then
        role="CONTROL: CER priors, couplings OFF"
    fi
    printf "  %-34s %-9s %-11s %-9s %s\n" \
        "_cwcer_${gate_label}_sp$(tag_of "$SPARSITY")${lam_tag}" "true" "$lam_shown" "$SPARSITY" "$role"
done
if [ "$INCLUDE_NOCER" = "1" ]; then
    printf "  %-34s %-9s %-11s %-9s %s\n" \
        "_cwnocer_${gate_label}_sp$(tag_of "$SPARSITY")" "false" "n/a" "$SPARSITY" \
        "BASELINE: flat p=0.1, no couplings"
fi
echo
echo "  data   -> $DATASETS"
if [ -n "$BASELINE_DATASETS" ]; then
    echo "  ref    -> $BASELINE_DATASETS   (lambda = 0 and no-CER only)"
fi
echo "  seeds  -> $SEEDS  (SAME set on every arm => paired contrasts)"
echo "  grid   -> $n_cer CER + $n_nocer no-CER = $n_points point(s)"
echo "  base   -> $MODELS_DIR/$BASE_HP"
echo "  gates  -> syndrome tau = $GATE_TAU ($gate_label);  pair certainty c = $CERTAINTY LLR"
echo "  GPU    -> ${GPUS}x $GPU_TYPE, $SLOTS core(s), $MEM, 1 node; train $TRAIN_WAVES wave(s) of $SLOTS, test $TEST_JOBS at a time"
echo "  time   -> $WALLTIME  (estimate ${estimated_minutes} min: $N_EPOCHS_BASE epoch(s) x $TRAIN_WAVES wave(s) train, $n_points tests $TEST_JOBS-way)"
echo "  assert -> require_correlations = true on every CER arm"
echo
if [ "$PROFILE" = "spread" ]; then
    echo "  primary   : lam0 - nocer across sigma (does the CER prior advantage grow?)"
    echo "  secondary : lambda > 0 vs lam0 at sigma = 0.001, the most informative data"
    echo "  NOTE: the sigmas are not matched on physical error rate (mean per-gate rate"
    echo "  5.0e-4 / 5.4e-4 / 7.1e-4), so compare CER vs no-CER WITHIN a dataset."
    echo
fi
if [ "$MODE" = "lambda_sweep" ]; then
    echo "  nocer -> lam0 isolates the PRIORS; lam0 -> lam>0 isolates the COUPLINGS."
    echo
fi
echo "submit with:  sbatch $SLURM"

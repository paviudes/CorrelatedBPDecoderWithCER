#!/usr/bin/env bash
# ============================================================================
# sweep_correlation_weight.sh — CER vs no-CER across p, with the REVISED J
#                               convention and sparsity switched OFF
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/sweep_correlation_weight.sh --check      # verify the CER files
#     bash misc/sweep_correlation_weight.sh --pin_cer    # lock them by SHA-256
#     bash misc/sweep_correlation_weight.sh              # the sweep
#     bash misc/sweep_correlation_weight.sh --collect    # summarise
#
#     sbatch ../data/<codename>/cluster/cw_<mode>_<timestamp>.sh
#
# ---------------------------------------------------------------------------
# WHICH DATA. The codename comes from the DATA PROFILE, and the default is the
# CURRENT dataset — no flag needed:
#
#   (default)   72q_BB_cycles_1_spread_comparison   per-CNOT Normal(p, sigma)
#   --uniform   72q_BB_cycles_1_debug               uniform p, the earlier sweeps
#
# This used to default to the uniform-p codename, so any command missing a flag
# silently addressed superseded data while producing identically-named outputs.
# Every mode echoes the directory it resolved to; that line is the check.
#
# Other modes: --setup (build a derived codename), --probe and --lambda_sweep
# (both imply --uniform, being defined on p values only that dataset has),
# --ungated (historical tau = -1 contrast).
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
# CODENAME, BASE_HP, DATASETS and SEEDS are all set by the DATA PROFILE below.
# These are placeholders; nothing reads them before the profile has run.
CODENAME=""
# The donor codename: where --setup borrows shared data from, and where the base
# hyperparameter files live. This was 72q_BB_cycles_1, which is now EMPTY — no
# correlated_weights, no base TOMLs — so every path through it was dead.
# 72q_BB_cycles_1_debug is the curated one: it holds the revised-J couplings
# (mean J = +2.0111) and all four base hyperparameter files, including a
# hyperparams_epochs_10_corrs.toml byte-identical to the spread codename's.
SOURCE_CODENAME="72q_BB_cycles_1_debug"
BASE_HP=""
PVALS="0.0005 0.0007 0.0019"
SEEDS=""
# DATASETS holds the dataset KEY of each point, i.e. the part of the filename
# shared by the training set, the test set and the CER file:
#
#     training_data/train_<key>.txt
#     testing_data/test_<key>.txt
#     correlated_weights/correlated_weights_<key>.txt
#
# Empty means "derive p_<p>_s_1 from PVALS", which is the historical layout. The
# per-gate-spread datasets add a sigma field (p_0.0005_sig_0.0005_s_2), so the
# key is carried around whole rather than reassembled from parts.
DATASETS=""
NLAYERS=90
SPARSITY="0.0"                 # pinned CONSTANT — the point of the sweep
GATE_TAU="0.5"                 # gate ON by default; --ungated for tau = -1
LAMBDA=""                      # empty = inherit correlation_weight from base TOML
SINGLE_QUBIT_RESCALE="0.1"     # inherited from the base TOML; exposed for clarity
LAMBDAS=""                     # empty = one CER arm inheriting the base TOML's anneal
INCLUDE_NOCER=1                # --no_nocer drops the flat-prior baseline
# The lambda grid for --lambda_sweep. Deliberately BELOW 1, unlike the retired
# sweep_lambda.sh grid {0, 1, 10, 100}: that was built when the term was inert.
# At lambda = 0.76 it now produces a -4.26 sigma coset effect, so the question is
# no longer "is it strong enough" but "where does the benefit stop paying".
LAMBDA_GRID="0 0.1 0.3 0.75 1.5"
# Dataset PROFILE, orthogonal to MODE: it selects which codename and which dataset
# keys to work on, so `--spread --check`, `--spread --pin_cer` and a plain
# `--spread` run all address the same data.
# ---------------------------------------------------------------------------
# DATA PROFILE. This selects the codename and every path derived from it.
#
# THE DEFAULT IS THE CURRENT DATASET. It used to be 72q_BB_cycles_1_debug, which
# meant any command missing a flag silently addressed the superseded uniform-p
# data — and the outputs are named identically either way, so the mistake left no
# trace. Defaulting to the live dataset makes the safe thing automatic and the
# legacy thing explicit.
#
#   (default)   72q_BB_cycles_1_spread_comparison   per-CNOT Normal(p, sigma)
#   --uniform   72q_BB_cycles_1_debug               uniform p, the old sweeps
PROFILE="spread"
PROFILE_EXPLICIT=0
BASE_HP_EXPLICIT=0
SEEDS_EXPLICIT=0
WALLTIME_EXPLICIT=0
# Datasets that get ONLY the {lambda = 0, no-CER} pair rather than the full lambda
# grid. The sigma = 0 reference exists to answer "is CER worth more now than it
# was on uniform-p data?", which needs lam0-vs-nocer at both sigmas and nothing
# else; running five lambdas on it would spend a third of the job re-deriving a
# result we already have.
BASELINE_DATASETS=""

ACCOUNT="def-jemerson_gpu"
EMAIL="pavithran.sridhar@gmail.com"
JULIA_MODULE="julia/1.12.5"
CUDA_MODULE="cuda"
GPU_TYPE=""
TEST_JOBS=1                    # serial GPU phase: one MIG permits one CUDA context
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
        --setup)     MODE="setup";   shift;;
        --check)     MODE="check";   shift;;
        --pin_cer)   MODE="pin_cer"; shift;;
        --collect)   MODE="collect"; shift;;
        --probe)     MODE="probe";   shift;;
        --ungated)   GATE_TAU="-1.0"; shift;;
        --lambda_sweep) MODE="lambda_sweep"; shift;;
        --spread)    PROFILE="spread"; PROFILE_EXPLICIT=1; shift;;
        --uniform)   PROFILE="uniform"; PROFILE_EXPLICIT=1; shift;;
        --datasets)  DATASETS="$2";  shift 2;;
        --baseline_datasets) BASELINE_DATASETS="$2"; shift 2;;
        --lambdas)   LAMBDAS="$2";   shift 2;;
        --no_nocer)  INCLUDE_NOCER=0; shift;;
        --pvals)     PVALS="$2";     shift 2;;
        --seeds)     SEEDS="$2"; SEEDS_EXPLICIT=1; shift 2;;
        --sparsity)  SPARSITY="$2";  shift 2;;
        --lambda)    LAMBDAS="$2";   shift 2;;
        --tau)       GATE_TAU="$2";  shift 2;;
        --rescale)   SINGLE_QUBIT_RESCALE="$2"; shift 2;;
        --base_hp)   BASE_HP="$2"; BASE_HP_EXPLICIT=1; shift 2;;
        --codename)  CODENAME="$2";  shift 2;;
        --nlayers)   NLAYERS="$2";   shift 2;;
        --gpu_type)  GPU_TYPE="$2";  shift 2;;
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
# `--lambda_sweep` and `--probe` are defined in terms of p values that only exist
# in the uniform-p codename, so they imply that profile unless one was named.
if [ "$PROFILE_EXPLICIT" = "0" ]; then
    case "$MODE" in
        lambda_sweep|probe) PROFILE="uniform" ;;
    esac
fi

if [ "$PROFILE" = "spread" ]; then
    # THE CURRENT DATASET. Each CNOT's error rate is drawn from Normal(p, sigma)
    # rather than fixed at p, which is what finally made the CER file carry
    # per-qubit information: within-sector CV on the single-qubit rates went from
    # 1.3% to 17-22%. Three independent noise samples at sigma = p, plus a
    # sigma = 0 reference that gets only {lambda = 0, no-CER}.
    CODENAME="72q_BB_cycles_1_spread_comparison"
    if [ "$BASE_HP_EXPLICIT" = "0" ]; then
        BASE_HP="hyperparams_epochs_10_corrs.toml"
    fi
    if [ -z "$DATASETS" ]; then
        DATASETS="p_0.0005_sig_0.0005_s_1 p_0.0005_sig_0.0005_s_2 p_0.0005_sig_0.0005_s_3"
    fi
    if [ -z "$BASELINE_DATASETS" ]; then
        BASELINE_DATASETS="p_0.0005_sig_0.0_s_1"
    fi
    if [ -z "$LAMBDAS" ]; then
        LAMBDAS="$LAMBDA_GRID"
    fi
    if [ "$SEEDS_EXPLICIT" = "0" ]; then
        # The three noise samples ARE the replicates; a network-seed axis on top
        # would multiply the grid without adding an independent source of spread.
        SEEDS="1"
    fi
elif [ "$PROFILE" = "uniform" ]; then
    # THE SUPERSEDED DATASET, kept so the earlier sweeps stay reproducible: one
    # error rate for every CNOT, which left the CER file almost information-free
    # (2 sector levels, 2 coupling classes, R^2 = 0.82 on structure alone).
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

# ------------------------------------------------------------------ setup ---
# --setup builds a DERIVED codename: an empty working directory that borrows
# code / training_data / testing_data / correlated_weights from a source codename
# by symlink, so a sweep can write models and results without touching the
# original. It is NOT needed for a codename that already owns its data — and
# running it there would copy unrelated TOMLs in from the source.
#
# It also operates on whatever codename the DATA PROFILE selected, so a bare
# `--setup` builds the default (72q_BB_cycles_1_debug) even if you meant the
# spread datasets. The target is printed first for exactly that reason.
if [ "$MODE" = "setup" ]; then
    src="$WORKDIR/$SOURCE_CODENAME"; dst="$WORKDIR/$CODENAME"
    echo "  target codename : $CODENAME"
    echo "  borrowing from  : $SOURCE_CODENAME"
    echo
    if [ "$CODENAME" = "$SOURCE_CODENAME" ]; then
        echo "  Refusing: the target and the donor are the same codename, so there is"
        echo "  nothing to borrow. --setup builds a DERIVED codename; name a new one"
        echo "  with --codename, or skip setup entirely." >&2
        exit 1
    fi

    complete=1
    for shared in code training_data testing_data correlated_weights; do
        if [ ! -e "$dst/$shared" ]; then
            complete=0
        fi
    done
    if [ "$complete" = "1" ]; then
        echo "  Nothing to do — $CODENAME already has code/, training_data/,"
        echo "  testing_data/ and correlated_weights/. It owns its data, so --setup"
        echo "  would only copy unrelated hyperparameter files in from $SOURCE_CODENAME."
        echo
        echo "  Go straight to:"
        echo "    bash misc/sweep_correlation_weight.sh${PROFILE:+ --$PROFILE} --check"
        exit 0
    fi

    [ -d "$src" ] || { echo "source codename missing: $src (run from expts/)" >&2; exit 1; }
    mkdir -p "$dst"/{models,results,logs,cluster}
    for shared in code training_data testing_data correlated_weights; do
        [ -e "$dst/$shared" ] || { ln -s "$(cd "$src/$shared" && pwd)" "$dst/$shared"; echo "  linked  $shared"; }
    done
    cp -n "$src"/models/*.toml "$dst/models/" 2>/dev/null || true
    echo "  $dst ready."
    echo "  NOTE: if correlated_weights is a SYMLINK back to $SOURCE_CODENAME, the"
    echo "        revised-J files must be placed there, not in the derived copy."
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
#
# THE FINGERPRINT EXISTS BECAUSE THIS HAS ALREADY HAPPENED ONCE. The revised-J
# files were updated on the workstation but never reached the cluster, and a
# later restore from a cluster tarball overwrote them locally too — so a sweep
# ran, and was analysed, on the superseded convention. The statistics below make
# that visible to a reader; the SHA-256 pin below makes it FATAL to a script.
#
#     bash misc/sweep_correlation_weight.sh --pin_cer    # record what is there now
#
# writes J_FINGERPRINT.txt next to the data. Every later run compares against it
# and refuses to submit on a mismatch. Commit that file.
FINGERPRINT_FILE="$CER_DIR/J_FINGERPRINT.txt"

# Called after the mode blocks have had their say about PVALS, so `--probe` and
# `--lambda_sweep` narrowing to one p is reflected here rather than overridden.
resolve_datasets() {
    if [ -z "$DATASETS" ]; then
        DATASETS=""
        for p in $PVALS; do
            DATASETS="$DATASETS p_${p}_s_1"
        done
        DATASETS="$(echo $DATASETS)"
    fi
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
    for key in $DATASETS; do
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
        echo "  $unpinned of $(echo $DATASETS | wc -w) file(s) are not covered by the pin."
        echo "  A partial pin is not a guard. Re-pin once the data is confirmed."
        return 0
    fi
    echo "  J_FINGERPRINT.txt: all $(echo $DATASETS | wc -w) file(s) match the pin."
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
        for key in $DATASETS; do
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
[ -d "$MODELS_DIR" ] || { echo "no models dir: $MODELS_DIR — run --setup first" >&2; exit 1; }
# Missing base hyperparameters are a stop, not a silent substitution. An earlier
# version copied a file in from another codename automatically, which quietly
# overrode a deliberately-edited one. Say where the canonical copy lives and let
# the decision be made explicitly.
if [ ! -f "$MODELS_DIR/$BASE_HP" ]; then
    echo "no base hyperparameters: $MODELS_DIR/$BASE_HP" >&2
    donor_hp="$WORKDIR/$SOURCE_CODENAME/models/$BASE_HP"
    if [ -f "$donor_hp" ]; then
        echo >&2
        echo "  A copy exists in the donor codename ($SOURCE_CODENAME). To use it:" >&2
        echo "    cp $donor_hp $MODELS_DIR/" >&2
        echo "  Or name a different file with --base_hp. Available here:" >&2
    else
        echo "  Available in $MODELS_DIR:" >&2
    fi
    ls "$MODELS_DIR"/hyperparams_epochs*.toml 2>/dev/null | sed 's|.*/|    |' >&2 \
        || echo "    (none)" >&2
    exit 1
fi
mkdir -p "$CLUSTER_DIR"

if [ "$MODE" = "probe" ]; then
    PVALS="0.0007"; SEEDS="1"; WALLTIME="3:00:00"
fi

# ONE p, MANY lambda. p = 0.0007 is where the coset effect was largest and
# cleanest (-34.0 +- 7.0 across seeds, t = -8.41, and all three seeds agreeing on
# the shared test set at McNemar z = -4.26). Adding the p axis on top would
# multiply the grid by three for a second-order question.
if [ "$MODE" = "lambda_sweep" ]; then
    PVALS="0.0007"
    if [ -z "$LAMBDAS" ]; then
        LAMBDAS="$LAMBDA_GRID"
    fi
fi

resolve_datasets

# 5 lambda + 1 baseline, 3 seeds = 18 points, the same size as the 2026-08-20 run
# that took 2h37m. Refuse to submit a grid that silently outgrows the walltime.
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

# LAMBDA IS PART OF THE TAG, or two points in a lambda sweep silently overwrite
# each other's weights AND results. Only when lambda is pinned, though: with
# LAMBDAS empty the tag stays exactly `_cw<arm>_<gate>_sp<tag>`, which is what
# the completed 2026-08-20 sweep wrote, so those files are neither collided with
# nor swept up by the collector's lambda-aware pattern.
#
# The no-CER arm carries NO lambda: with use_CER = false there are no couplings,
# so `correlation_weight` multiplies nothing. Emitting it once per lambda would
# be the same run trained N times under N names — fake replicates that would
# shrink the baseline's apparent error bar. It is emitted once per seed.
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

# ---- overwrite guard ---------------------------------------------------------
# A run_tag that already has results on disk will be RETRAINED AND OVERWRITTEN.
# That is sometimes exactly right (the no-CER baseline is configuration-identical
# to the completed sweep, so re-running it is a free determinism check) and
# sometimes a data-loss bug. Either way it should be a decision, not a surprise:
# the previous sweep in this project was analysed against silently stale inputs.
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
    a100_1g.5gb)  SLOTS=1;  MEM="15G";  VRAM_GB=5  ;;
    a100_2g.10gb) SLOTS=3;  MEM="31G";  VRAM_GB=10 ;;
    a100_3g.20gb) SLOTS=6;  MEM="62G";  VRAM_GB=20 ;;
    a100)         SLOTS=12; MEM="124G"; VRAM_GB=40 ;;
    *) echo "unknown --gpu_type: $GPU_TYPE" >&2; exit 2;;
esac
[ "$n_points" -lt "$SLOTS" ] && SLOTS=$n_points
TRAIN_WAVES=$(( (n_points + SLOTS - 1) / SLOTS ))
GPU_MEMORY_PER_SLOT="$(( (VRAM_GB * 1024) / TEST_JOBS ))M"

# ---- walltime, DERIVED from n_epochs rather than hard-coded --------------------
# Two runs of 18 points at n_epochs = 5, 12-way concurrency, measured end to end:
#     training  1h23m over 2 waves          -> 8.3 min per epoch per wave
#     testing   89 min serial over 20 runs  -> 4.5 min per point (mean 268 s)
#     precompile + stage-in                 -> ~8 min
# Training scales with n_epochs; testing does not (it is one forward pass over
# 1e6 samples whatever the model was trained for). The 10-epoch base TOML
# therefore roughly doubles the training phase, which a fixed 4h would not cover:
# 20 points x 10 epochs comes to ~4h30m of work before any margin.
N_EPOCHS_BASE=$(grep -E '^[[:space:]]*n_epochs[[:space:]]*=' "$MODELS_DIR/$BASE_HP" \
                | head -1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')
if [ -z "$N_EPOCHS_BASE" ]; then
    N_EPOCHS_BASE=5
fi
estimated_minutes=$(( (83 * N_EPOCHS_BASE * TRAIN_WAVES) / 10 + (45 * n_points) / 10 + 8 ))
if [ "$WALLTIME_EXPLICIT" = "0" ]; then
    # 1.3x margin on the estimate, rounded up to a whole hour, floor of 2h.
    walltime_hours=$(( ((estimated_minutes * 13) / 10 + 59) / 60 ))
    if [ "$walltime_hours" -lt 2 ]; then
        walltime_hours=2
    fi
    WALLTIME="${walltime_hours}:00:00"
fi

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

# THIS IS A BATCH SCRIPT, NOT A SHELL SCRIPT. Running it with \`bash\` on a login
# node leaves SLURM_SUBMIT_DIR and SLURM_TMPDIR unset, and \`set -u\` then kills it
# at the first reference with the unhelpful "unbound variable". Worse, without
# -u it would try to run an 18-point training job on the login node. Refuse.
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
echo "  gate   -> syndrome_gate_threshold = $GATE_TAU ($gate_label)"
echo "  GPU    -> $GPU_TYPE, $SLOTS core(s), $MEM; train $TRAIN_WAVES wave(s), test serial"
echo "  time   -> $WALLTIME  (estimate ${estimated_minutes} min: $N_EPOCHS_BASE epoch(s) x $TRAIN_WAVES wave(s) train + $n_points serial test)"
echo "  assert -> require_correlations = true on every CER arm"
echo
if [ "$PROFILE" = "spread" ]; then
    echo "  WHAT CHANGED IN THE DATA (measured, uniform-p -> per-gate spread):"
    echo "    single-qubit within-sector CV   1.3%  ->  17-22%     PASSES decisively"
    echo "    J explained by check structure  0.82  ->  0.67-0.75  short of the <0.30 gate"
    echo "    strong-class within-class sd    0.93  ->  1.21-1.42  ~40% more per-edge signal"
    echo "    dead edges (216 at J ~ 0)       216   ->  201-210    essentially unchanged"
    echo
    echo "  So the PRIORS became genuinely informative and the COUPLINGS only partly did."
    echo "  Given lambda = 0 was optimal on the uniform-p data, the honest prior is that"
    echo "  lam0 wins again and the news is how much bigger its margin over no-CER is."
    echo
    echo "  reading protocol, fixed in advance:"
    echo "    lam0 beats nocer by MORE than the 135 seen on uniform-p -> the priors are the"
    echo "                                                               story; per-gate data helps"
    echo "    some lambda > 0 now beats lam0                          -> FIRST evidence the"
    echo "                                                               couplings carry information"
    echo "    lambda > 0 still monotonically worse                    -> couplings are done as a"
    echo "                                                               loss term; move to OSD scoring"
    echo
    echo "  CAUTION: sigma = 0 and sigma > 0 are NOT matched on physical error rate"
    echo "  (mean per-gate rate 5.0e-4 vs 5.3-5.7e-4, because the Normal is clamped at 0),"
    echo "  so compare CER vs no-CER WITHIN a dataset, not failure counts ACROSS sigma."
    echo
fi
if [ "$MODE" = "lambda_sweep" ]; then
    echo "  THE DECOMPOSITION (this is why lambda = 0 and no-CER are both present):"
    echo "    nocer -> lam0     isolates the single-qubit PRIORS"
    echo "    lam0  -> lam>0    isolates the COUPLINGS"
    echo "  Every CER-vs-no-CER number so far has confounded the two; this separates them."
    echo
    echo "  THE PREDICTION UNDER TEST, fixed in advance:"
    echo "    coset selection is a discrete argmax flip, so its benefit should SATURATE in lambda;"
    echo "    the convergence damage is a continuous distortion, so it should grow ~LINEARLY."
    echo "  If so there is an interior optimum and lambda = 0.76 is already past it."
    echo
    echo "  reading protocol:"
    echo "    net minimum at some 0 < lambda < 0.76  -> the trade is tunable; take that lambda to the p axis"
    echo "    coset and convergence scale together   -> no lambda wins; the 1/|C| divisor is the problem"
    echo "                                              (540 edges, ~0.05 firing per sample)"
    echo "    lam0 already beats nocer               -> the win is the PRIORS, not the couplings"
    echo "    nothing separates lam0 from lam0p75    -> the coset effect was not the couplings after all"
    echo
fi
if [ "$MODE" = "primary" ] && [ -z "$PROFILE" ] && [ -z "$LAMBDAS" ]; then
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

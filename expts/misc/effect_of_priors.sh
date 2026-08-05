#!/usr/bin/env bash
#
# effect_of_priors.sh
#
# Isolates the two channels through which CER data enters the neural BP decoder:
#
#   (1) single-qubit error rates  -> priors on the variable nodes
#   (2) two-qubit couplings J_ij  -> correlation term in the loss
#
# This script runs the "priors only" arm: it strips every two-qubit entry from
# the CER file, so channel (2) is switched off while channel (1) stays on.
# Comparing its failure count against the full-CER and no-CER arms tells you
# which channel is responsible for the observed degradation.
#
# Usage (from the `expts/` directory):
#
#     bash misc/effect_of_priors.sh
#
# Everything below the CONFIG block is generic; to run a different arm or a
# different (p, seed) just edit CONFIG.
#
set -euo pipefail

# ============================================================================
# CONFIG
# ============================================================================

CODENAME="72q_BB_cycles_1"
WORKDIR="./../data"

N_HIDDEN_LAYERS=100
TRAIN_FILE="train_p_0.0005_s_1.txt"
TEST_FILE="test_p_0.0005_s_1.txt"

# CER file to use, as supplied. Lives in
# <workdir>/<codename>/correlated_weights/. This script does NOT modify it.
CER_FILE="correlated_weights_p_0.0005_s_1_priors_only_normalized.txt"

# Base hyperparameters TOML to clone (in <workdir>/<codename>/models/).
BASE_TOML="hyperparams_epochs_20.toml"

# Tag for this arm. Appended to the weights JSON and the results CSV, so this
# run cannot collide with the other arms. Keep the leading underscore.
RUN_TAG="_normalized"

# Set to "1" to use the GPU for the testing stage. Training is run on CPU
# first (matching the current workflow), then the GPU is enabled for testing.
USE_GPU_FOR_TEST="1"

# Arms to report side by side at the end, as "tag:label". The first entry is
# the baseline for the ratio and z columns. Results CSVs are located by
# swapping the tag into the standard results filename; arms whose CSV is
# absent are reported as missing rather than failing the run.
COMPARE_ARMS=(
    "_REF:no-CER"
    "_priors_only:CER priors"
    "_normalized:normalized priors"
)

# ============================================================================
# PATHS
# ============================================================================

# All paths below -- WORKDIR, the --project flag, and the experiment script
# itself -- are interpreted relative to the directory you ran this from, NOT
# relative to where this file lives. The script never changes directory.
INVOCATION_DIR="${PWD}"

CER_DIR="${WORKDIR}/${CODENAME}/correlated_weights"
MODELS_DIR="${WORKDIR}/${CODENAME}/models"
RESULTS_DIR="${WORKDIR}/${CODENAME}/results"

CER_SRC="${CER_DIR}/${CER_FILE}"

RESULTS_STEM="simulation_results_${TEST_STEM}_nlayers_${N_HIDDEN_LAYERS}_epochs_${EPOCHS}_trained_using_${TRAIN_STEM}"

# Results CSV for a given run tag.
results_csv_for() { printf '%s/%s%s.csv' "${RESULTS_DIR}" "${RESULTS_STEM}" "$1"; }

RESULTS_CSV="$(results_csv_for "${RUN_TAG}")"

# Extract "<num_failures> <num_samples_per_error_rate>" from a results CSV.
# Columns are located by header name, and CRLF line endings are tolerated.
read_counts() {
    awk -F',' '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                h = $i; gsub(/^[ \t"]+|[ \t"\r]+$/, "", h); col[h] = i
            }
            if (!("num_failures" in col) || !("num_samples_per_error_rate" in col)) exit 1
            next
        }
        NF > 1 {
            f = $(col["num_failures"]);               gsub(/[ \t"\r]/, "", f)
            n = $(col["num_samples_per_error_rate"]); gsub(/[ \t"\r]/, "", n)
            print f, n
            exit
        }
    ' "$1"
}

banner() { printf '\n=== %s ===\n' "$1"; }

# Set `key = value` in a TOML file, replacing the existing line if present and
# appending otherwise. Value is written verbatim, so quote strings yourself.
set_toml_key() {
    local file="$1" key="$2" value="$3"
    if grep -qE "^[[:space:]]*${key}[[:space:]]*=" "${file}"; then
        sed -i.bak -E "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = ${value}|" "${file}"
        rm -f "${file}.bak"
    else
        printf '%s = %s\n' "${key}" "${value}" >>"${file}"
    fi
}

# ============================================================================
# BLOCK 0 -- preflight
# ============================================================================

banner "Preflight"

if [[ ! -f "neural_bp_experiments.jl" ]]; then
    echo "ERROR: neural_bp_experiments.jl not found in the current directory." >&2
    echo "  cwd: ${INVOCATION_DIR}" >&2
    echo "  Run this from the directory containing neural_bp_experiments.jl," >&2
    echo "  e.g.  cd expts && bash misc/effect_of_priors.sh" >&2
    exit 1
fi

for f in "${CER_SRC}" "${TOML_SRC}" "${WORKDIR}/${CODENAME}/${TRAIN_FILE}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f} (resolved from cwd ${INVOCATION_DIR})" >&2; exit 1; }
done
command -v julia >/dev/null || { echo "ERROR: julia not on PATH" >&2; exit 1; }

echo "cwd            : ${INVOCATION_DIR}"
echo "code           : ${CODENAME}"
echo "train / test   : ${TRAIN_FILE} -> ${TEST_FILE}"
echo "run tag        : ${RUN_TAG}"
echo "expected CSV   : ${RESULTS_CSV}"

# ============================================================================
# BLOCK 1 -- inspect the supplied CER file
# ============================================================================

banner "Block 1: inspecting ${CER_FILE}"

# The file is used exactly as supplied; nothing is modified. This block only
# reports what the decoder will be initialised with, so a wrong or stale file
# is obvious before an hour of training is spent on it.
awk '
    function is_pair(l) { return l ~ /^[[:space:]]*\(/ }
    function llr(p)     { return log((1 - p) / p) }
    /^[[:space:]]*$/ { next }
    {
        if (is_pair($0)) { n_pairs++; next }
        i = index($0, ":"); if (i == 0) next
        v = substr($0, i + 1) + 0
        if (v <= 0 || v >= 1) {
            printf "ERROR: rate out of range on line %d: %s\n", FNR, $0 > "/dev/stderr"
            bad = 1; exit
        }
        n++; sum += v
        l = llr(v); sum_l += l
        if (n == 1) { lo = hi = l }
        if (l < lo) lo = l
        if (l > hi) hi = l
    }
    END {
        if (bad) exit 1
        if (n == 0) { print "ERROR: no single-qubit rates found" > "/dev/stderr"; exit 1 }
        printf "priors  : %d qubits, mean rate %.6f\n", n, sum / n
        printf "          LLR mean %.3f  [%.3f, %.3f]  spread %.3f\n", sum_l / n, lo, hi, hi - lo
        printf "          (preset p=0.1 corresponds to LLR %.3f)\n", log(0.9 / 0.1)
        printf "pairs   : %d two-qubit entries%s\n", n_pairs + 0, \
               (n_pairs > 0 ? "  (correlation term ACTIVE)" : "  (correlation term inactive)")
    }
' "${CER_SRC}"

# ============================================================================
# BLOCK 2 -- clone and configure the hyperparameters TOML
# ============================================================================

banner "Block 2: preparing ${TOML_NAME}"

cp "${TOML_SRC}" "${TOML_RUN}"
set_toml_key "${TOML_RUN}" "retrain" "true"
set_toml_key "${TOML_RUN}" "run_tag" "\"${RUN_TAG}\""

grep -E '^[[:space:]]*(retrain|run_tag)[[:space:]]*=' "${TOML_RUN}"

# ============================================================================
# BLOCK 3 -- train (retrain = true)
# ============================================================================

banner "Block 3: training"

unset USE_GPU || true

julia --project="./../" neural_bp_experiments.jl \
    --workdir "${WORKDIR}" \
    --codename "${CODENAME}" \
    --n_hidden_layers "${N_HIDDEN_LAYERS}" \
    --hyperparams "${TOML_NAME}" \
    --correlation_strengths_file "${CER_FILE}" \
    --quiet false \
    --train "${TRAIN_FILE}"

echo "training finished"
ls -la "${MODELS_DIR}"/*"${RUN_TAG}".json 2>/dev/null || \
    echo "NOTE: no weights JSON matching *${RUN_TAG}.json -- check the training log"

# ============================================================================
# BLOCK 4 -- flip to inference mode and test
# ============================================================================

banner "Block 4: testing"

set_toml_key "${TOML_RUN}" "retrain" "false"
grep -E '^[[:space:]]*retrain[[:space:]]*=' "${TOML_RUN}"

if [[ "${USE_GPU_FOR_TEST}" == "1" ]]; then
    export USE_GPU="1"
    echo "USE_GPU=1"
fi

julia --project="./../" neural_bp_experiments.jl \
    --workdir "${WORKDIR}" \
    --codename "${CODENAME}" \
    --n_hidden_layers "${N_HIDDEN_LAYERS}" \
    --hyperparams "${TOML_NAME}" \
    --correlation_strengths_file "${CER_FILE}" \
    --quiet false \
    --train "${TRAIN_FILE}" \
    --test "${TEST_FILE}"

# ============================================================================
# BLOCK 5 -- report
# ============================================================================

banner "Block 5: comparison"

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "ERROR: expected results file not found:" >&2
    echo "  ${RESULTS_CSV}" >&2
    echo "Candidates actually present:" >&2
    ls -1t "${RESULTS_DIR}" 2>/dev/null | head -5 >&2
    exit 1
fi

# Collect every arm, then print one table. The first arm with data is the
# baseline for the ratio and z columns.
rows=""
base_f=""; base_n=""

for arm in "${COMPARE_ARMS[@]}"; do
    tag="${arm%%:*}"
    label="${arm#*:}"
    csv="$(results_csv_for "${tag}")"

    if [[ ! -f "${csv}" ]]; then
        rows+="${label}|-|-|missing"$'\n'
        continue
    fi

    read -r f n < <(read_counts "${csv}") || true
    if [[ -z "${f:-}" ]]; then
        rows+="${label}|-|-|unparseable"$'\n'
        continue
    fi

    if [[ -z "${base_f}" ]]; then
        base_f="${f}"; base_n="${n}"
        rows+="${label}|${f}|${n}|baseline"$'\n'
    else
        stat=$(awk -v f1="${f}" -v n1="${n}" -v f2="${base_f}" -v n2="${base_n}" '
            BEGIN {
                p1 = f1 / n1; p2 = f2 / n2
                pp = (f1 + f2) / (n1 + n2)
                se = sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
                z  = (se > 0) ? (p1 - p2) / se : 0
                printf "%.2fx  z=%+.2f  %s", p1 / p2, z, \
                       (z > 2 ? "WORSE" : (z < -2 ? "BETTER" : "n.s."))
            }')
        rows+="${label}|${f}|${n}|${stat}"$'\n'
    fi
done

printf '\n  %-22s %10s %12s   %s\n' "arm" "failures" "samples" "vs baseline"
printf '  %s\n' "----------------------------------------------------------------------"
printf '%s' "${rows}" | awk -F'|' '{ printf "  %-22s %10s %12s   %s\n", $1, $2, $3, $4 }'

echo
echo "CSV: ${RESULTS_CSV}"

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

# Source CER file (lives in <workdir>/<codename>/correlated_weights/).
CER_FILE="correlated_weights_p_0_0005_s_1.txt"

# Base hyperparameters TOML to clone (in <workdir>/<codename>/models/).
BASE_TOML="hyperparams_epochs_20.toml"

# Tag for this arm. Appended to the weights JSON and the results CSV, so this
# run cannot collide with the baseline. Keep the leading underscore.
RUN_TAG="_priors_only"

# Set to "1" to use the GPU for the testing stage. Training is run on CPU
# first (matching the current workflow), then the GPU is enabled for testing.
USE_GPU_FOR_TEST="1"

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
CER_PRIORS_ONLY_NAME="${CER_FILE%.txt}${RUN_TAG}.txt"
CER_PRIORS_ONLY="${CER_DIR}/${CER_PRIORS_ONLY_NAME}"

TOML_SRC="${MODELS_DIR}/${BASE_TOML}"
TOML_NAME="${BASE_TOML%.toml}${RUN_TAG}.toml"
TOML_RUN="${MODELS_DIR}/${TOML_NAME}"

TRAIN_STEM="${TRAIN_FILE%.txt}"
TEST_STEM="${TEST_FILE%.txt}"
EPOCHS="$(sed -n 's/.*epochs_\([0-9]\+\).*/\1/p' <<<"${BASE_TOML}")"
RESULTS_CSV="${RESULTS_DIR}/simulation_results_${TEST_STEM}_nlayers_${N_HIDDEN_LAYERS}_epochs_${EPOCHS}_trained_using_${TRAIN_STEM}${RUN_TAG}.csv"

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
# BLOCK 1 -- build the priors-only CER file
# ============================================================================

banner "Block 1: stripping two-qubit entries from the CER file"

# Two-qubit lines are exactly those beginning with "(". Everything else (the
# `index : value` single-qubit lines) is kept.
grep -v '^[[:space:]]*(' "${CER_SRC}" >"${CER_PRIORS_ONLY}"

n_singles_src=$(grep -cv '^[[:space:]]*(' "${CER_SRC}" || true)
n_pairs_src=$(grep -c '^[[:space:]]*(' "${CER_SRC}" || true)
n_singles_out=$(wc -l <"${CER_PRIORS_ONLY}")

echo "source  : ${n_singles_src} single-qubit lines, ${n_pairs_src} pair lines"
echo "written : ${CER_PRIORS_ONLY_NAME} (${n_singles_out} single-qubit lines, 0 pair lines)"

if [[ "${n_singles_out}" -ne "${n_singles_src}" ]]; then
    echo "ERROR: single-qubit line count changed while stripping pairs" >&2
    exit 1
fi

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
    --correlation_strengths_file "${CER_PRIORS_ONLY_NAME}" \
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
    --correlation_strengths_file "${CER_PRIORS_ONLY_NAME}" \
    --quiet false \
    --train "${TRAIN_FILE}" \
    --test "${TEST_FILE}"

# ============================================================================
# BLOCK 5 -- report
# ============================================================================

banner "Block 5: result"

if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "ERROR: expected results file not found:" >&2
    echo "  ${RESULTS_CSV}" >&2
    echo "Candidates actually present:" >&2
    ls -1t "${RESULTS_DIR}" 2>/dev/null | head -5 >&2
    exit 1
fi

awk -F',' '
    NR == 1 {
        for (i = 1; i <= NF; i++) {
            h = $i
            gsub(/^[ \t"]+|[ \t"\r]+$/, "", h)
            col[h] = i
        }
        if (!("num_failures" in col) || !("num_samples_per_error_rate" in col)) {
            print "ERROR: required columns not found in CSV header" > "/dev/stderr"
            exit 1
        }
        next
    }
    NF > 1 {
        f = $(col["num_failures"]);           gsub(/[ \t"\r]/, "", f)
        n = $(col["num_samples_per_error_rate"]); gsub(/[ \t"\r]/, "", n)
        print f " failures out of " n
    }
' "${RESULTS_CSV}"

echo
echo "CSV: ${RESULTS_CSV}"

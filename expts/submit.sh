#!/usr/bin/env bash
# ============================================================================
# submit.sh — settings TOML + $EDITOR helper for batch_run.jl
# ============================================================================
#
# Writes a submission_settings_<timestamp>.toml file with commented defaults,
# opens it in your editor, and then runs
#     julia --project="./../" batch_run.jl --settings <toml>
# which writes the job scripts and prints how to submit them.
#
# With train = true and test = true (the defaults) one submission writes both
# jobs: the TRAINING job trains one model per (arm, seed) on the training
# dataset, and the TEST job, chained on it with --dependency=afterok, scores
# every model on every test dataset once every training task has exited 0. Both
# go in with one command, bash <data>/<codename>/cluster/submit_<ts>.sh.
# test = true alone tests models already on disk; train = true alone only trains.
# Datasets are named
#     {train,test,correlated_weights}_p_<p>_sig_<sigma>_s_<sample>.txt
# and the settings name them by (p, sigma) pairs, sigma being a number or a
# multiple of p: [0.0015, "2p"] is p_0.0015_sig_0.003.
#
# When there's no editor available (e.g. a headless SLURM login node with no
# $EDITOR set), the script skips the editor step and just tells you the file
# path and the command to run manually.
#
# HPC resource-ratio caveat (VERY important — this is how you avoid waiting
# forever in the queue). Every Alliance Canada cluster has a bundle ratio
# tying together GPU count, CPU cores, and memory. Requests that exceed the
# ratio get charged for the "missing" resource and drop down the priority
# stack. Bundle sizes are in Table "Ratios in bundles" at:
#     https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
# The defaults below are one whole Narval A100 per array task (12 cores, 32 GB
# of host RAM), which is inside the A100 bundle.
#
# Usage:
#     bash submit.sh                    # write TOML, edit, run
#     bash submit.sh --no-edit          # write TOML, print command, don't edit
#     bash submit.sh --no-run           # write TOML, edit, but don't launch julia
#     bash submit.sh --help             # show this help
# ============================================================================

set -eu

# ----------------------------------------------------------------------------
# Argument handling for the wrapper itself.
# ----------------------------------------------------------------------------
NO_EDIT=0
NO_RUN=0
for arg in "$@"; do
    case "$arg" in
        --no-edit) NO_EDIT=1 ;;
        --no-run)  NO_RUN=1 ;;
        --help|-h)
            awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Try: $0 --help" >&2
            exit 2
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Write the defaults TOML into ./scripts/ (anchored to submit.sh's own dir,
# not the current working directory — so running `bash expts/submit.sh` from
# the repo root and `cd expts && ./submit.sh` both land in the same place).
# The scripts/ folder is git-ignored.
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
mkdir -p "$SCRIPTS_DIR"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
SETTINGS_FILE="$SCRIPTS_DIR/submission_settings_${TS}.toml"

# Defaults mirror batch_run.jl's ArgParse defaults. Change them here (or edit
# the generated file each time) — batch_run.jl treats every entry as optional
# and layers CLI flags on top of TOML values on top of its own defaults.
cat > "$SETTINGS_FILE" <<'EOF'

# ----------------------------------------------------------------------------
# batch_run.jl submission settings
# Edit values below, save, close the editor. This file will be passed to
# batch_run.jl via --settings after the editor exits.
#
# Lists are TOML arrays: ["foo", "bar"] or [0.01, 0.05].
# Strings are quoted. Booleans are lowercase (true / false).
# Lines starting with # are comments and are preserved.
# Only the keys shown below are recognised — a misspelled or unknown key is a
# hard error (batch_run.jl will stop and list the valid keys).
# ----------------------------------------------------------------------------

# --- data ---
workdir          = "./../data"
dirnames         = ["72q_BB_cycles_1_sweep"]

# --- datasets ---------------------------------------------------------------
# Every input of one dataset carries the key  p_<p>_sig_<sigma>_s_<sample>:
#     training_data/train_<key>.txt, testing_data/test_<key>.txt,
#     correlated_weights/correlated_weights_<key>.txt
# A dataset is a (p, sigma) pair, [p, sigma]: sigma is a number, or a multiple of
# p written as a quoted string ("p", "2p", "0.5p"). [0.0015, "2p"] is
# p_0.0015_sig_0.003 and [0.0015, 0] is p_0.0015_sig_0.0. batch_run.jl checks that
# every file exists before writing a job.
#
# TRAINING: ONE pair; every model is trained on it.
train_pair       = [0.0015, "2p"]
train_sample     = 1

# TESTING: every trained model is scored on every pair listed, and only on those:
# nothing is combined into a grid. Here 8 p values x sigma = 0, p, 2p = 24
# datasets. test_pairs = [] tests on the training dataset only.
test_pairs       = [
    [0.0001, 0], [0.0001, "p"], [0.0001, "2p"],
    [0.0003, 0], [0.0003, "p"], [0.0003, "2p"],
    [0.0005, 0], [0.0005, "p"], [0.0005, "2p"],
    [0.0007, 0], [0.0007, "p"], [0.0007, "2p"],
    [0.0009, 0], [0.0009, "p"], [0.0009, "2p"],
    [0.0011, 0], [0.0011, "p"], [0.0011, "2p"],
    [0.0013, 0], [0.0013, "p"], [0.0013, "2p"],
    [0.0015, 0], [0.0015, "p"], [0.0015, "2p"],
]
test_sample      = 1

# --- arms and seeds ---------------------------------------------------------
# One model per (arm, seed). "cer": CER priors (use_CER = true); "nocer": uniform
# priors (use_CER = false). Each arm runs from its own copy of hyperparams_file,
# models/submissions/<timestamp>/<stem>_<arm>_<train|test>.toml; hyperparams_file
# itself is never edited.
cer_arms         = ["cer", "nocer"]
# Passed to every command as --seed; [] = the seed in hyperparams_file. The seed
# fixes the initial weights, the 10^4-sample training draw and the batch order.
# A CER vs no-CER comparison pairs the two arms seed by seed; one seed gives one
# draw of a 3-12% seed-to-seed spread, five can tell a ~15% effect from it.
seeds            = [1, 2, 3, 4, 5]
# Model and result files end in <run tag>_seed_<n>, the run tag being _cer or
# _nocer (after the code's own _no_cer):
#     ..._trained_using_train_<key>_cer_seed_1.json
#     ..._trained_using_train_<key>_no_cer_nocer_seed_1.json
# run_label = "big" makes them _cer_big / _nocer_big, so a second experiment in the
# same data directory does not replace the first one's files.
run_label        = ""

# --- model ---
# Under <workdir>/<dirname>/models/.
hyperparams_file = "hyperparams_baseline.toml"
n_hidden_layers  = 90
# julia --heap-size-hint for every process: the GC otherwise sizes its heap to the
# whole node, and processes sharing one allocation balloon into each other.
heap_size_hint   = "4G"

# --- which jobs this submission writes ---
# Both true: the training job and the test job chained on it (afterok: testing
# starts when every training task has exited 0, and is cancelled if any has not).
# test alone: tests the models already on disk (named by the settings above).
# train alone: only trains.
train            = true
test             = true

# --- devices ---
# Training on a GPU needs manual_backprop = true in hyperparams_file (Enzyme cannot
# differentiate on the GPU). A training at the batch of 20 keeps an A100 only ~20%
# busy, so trainings_per_gpu of them share each card. A test process always has a
# card to itself: its prediction batch is sized to the card's VRAM.
gpu_for_training  = true
gpu_for_testing   = true
trainings_per_gpu = 4

# --- backend selection ---
# "SLURM"     Alliance Canada / any SLURM cluster
# "local"     run on this machine (Metal on Mac, CUDA on Linux, else CPU)
# "Google_VM" Google Cloud VM with auto-shutdown after the job finishes
cluster_backend  = "SLURM"

# ----------------------------------------------------------------------------
# HPC RESOURCE SIZING
# ----------------------------------------------------------------------------
# Every Alliance cluster has a "bundle ratio" tying GPU count, CPU cores, and
# memory together. Requests that exceed the ratio get charged as if you were
# holding the extra resources → the scheduler treats you as a high-usage
# consumer and drops your priority.
#
# Table: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
#
# Narval A100 bundle sizes (per GPU / MIG instance):
#   Model            Fraction   Recommended per-GPU
#   a100             100%       12 cores, 124 GB
#   a100_3g.20gb     50%         6 cores,  62 GB
#   a100_2g.10gb     28%         3 cores,  31 GB
#   a100_1g.5gb      14%         1 core,   15 GB
# Fir/Nibi/Rorqual/Trillium: "h100" and its "h100_*g.*gb" MIG variants.
# If you change gpu_type, change n_cpus and mem_per_gpu with it.
# ----------------------------------------------------------------------------

# --- HPC resources (ignored on non-SLURM backends) ---
account          = "def-jemerson_gpu"
# Per array task. GPU jobs: n_gpus_per_node cards of gpu_type, n_cpus cores and
# mem_per_gpu of HOST memory per card. CPU jobs: n_cpus processes, mem_per_cpu each.
n_gpus_per_node  = 1
gpu_type         = "a100"
cuda_module      = "cuda"
n_cpus           = 12
mem_per_gpu      = "32G"
mem_per_cpu      = "4G"
# VRAM per card in GB for the test batch sizer; 0 = from gpu_type (a100: 40).
vram_per_gpu     = 0
# Upper bound on array tasks, per job. The training job uses one task per
# n_gpus_per_node x trainings_per_gpu models (10 models -> 3 tasks); the test job
# spreads its tests over up to max_nodes tasks (2 arms x 5 seeds x 24 datasets =
# 240 tests -> 24 tasks of 10).
max_nodes        = 24
# Per job (each of the two gets this wall). Measured on Narval (90 layers, 5
# epochs): a wave of 4 tanh trainings on one A100 ~12 min (~25 min with enriched
# ones); a 10^6-sample test ~2.3 min with startup; plus ~6 min per job to prepare
# the depot. Twice the epochs, twice the training. batch_run.jl estimates every
# array task of both jobs from these and refuses to write anything when a job
# cannot finish, saying which of max_nodes / wall_time to raise.
wall_time        = "1:00:00"
email            = "pavithran.sridhar@gmail.com"

EOF

echo "[submit] wrote defaults to: $SETTINGS_FILE"

# ----------------------------------------------------------------------------
# Open the settings file in the user's editor (unless --no-edit).
# ----------------------------------------------------------------------------
open_editor() {
    local editor_cmd=""
    if [ -n "${EDITOR:-}" ]; then
        editor_cmd="$EDITOR"
    else
        # Fallback chain: what most terminals have installed. Skip 'code' by
        # default because it forks and exits immediately — bad for the
        # "wait until you close" contract we need here.
        for cand in nano vim vi; do
            if command -v "$cand" >/dev/null 2>&1; then
                editor_cmd="$cand"
                break
            fi
        done
    fi

    if [ -z "$editor_cmd" ]; then
        echo "[submit] no editor found (set \$EDITOR, or install nano/vim/vi)." >&2
        return 1
    fi

    if [ ! -t 0 ] || [ ! -t 1 ]; then
        echo "[submit] no interactive terminal — skipping editor." >&2
        return 1
    fi

    echo "[submit] opening $SETTINGS_FILE in $editor_cmd..."
    "$editor_cmd" "$SETTINGS_FILE"
}

# batch_run.jl and every path in the settings are relative to expts/.
CMD=(julia --project="./../" batch_run.jl --settings "$SETTINGS_FILE")

if [ "$NO_EDIT" -eq 1 ]; then
    echo "[submit] --no-edit: skipping editor. Edit $SETTINGS_FILE manually, then run from $SCRIPT_DIR:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

if ! open_editor; then
    # Editor missing or non-interactive — print instructions and exit cleanly.
    echo "[submit] Edit $SETTINGS_FILE manually, then run from $SCRIPT_DIR:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

# ----------------------------------------------------------------------------
# Show the constructed command and (unless --no-run) execute it.
# ----------------------------------------------------------------------------
echo ""
echo "[submit] Will run (from $SCRIPT_DIR):"
printf '  '; printf '%q ' "${CMD[@]}"; printf '\n\n'

if [ "$NO_RUN" -eq 1 ]; then
    echo "[submit] --no-run: not launching julia. Copy-Paste the above command to run manually."
    exit 0
fi

cd "$SCRIPT_DIR"
exec "${CMD[@]}"

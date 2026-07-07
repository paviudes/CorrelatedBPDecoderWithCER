#!/usr/bin/env bash
# ============================================================================
# submit.sh — settings TOML + $EDITOR helper for batch_run.jl
# ============================================================================
#
# Writes a submission_settings_<timestamp>.toml file with commented defaults,
# opens it in your editor, and then runs
#     julia --project="./../" batch_run.jl --settings <toml>
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
# The TOML this script writes has defaults sized for a Narval A100 MIG 1g.5gb
# instance (1 core, 15 GB) — most abundant, near-zero queue wait, perfect for
# small tests. See the annotated comments in the TOML for how to size up if
# you actually need a whole A100 or a full CPU node for training.
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
# ----------------------------------------------------------------------------

# --- data & experiment ---
working_dir      = "./../data"
dirnames         = ["72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data"]
pvals            = [0.01]
qvals            = [0.0]
n_samples        = 64

# --- Parameters for the neural BP model ---
# The hyperparams TOML file should be placed in the same directory as the <working_dir>/<dirnames>/<models> folder.
hyperparams_file = "hyperparams_epochs_10.toml"
n_hidden_layers  = 200

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
#   Model            Fraction   Recommended per-GPU        Availability
#   a100             100%       12 cores, 124 GB           scarce, long wait
#   a100_3g.20gb     50%         6 cores,  62 GB           moderate
#   a100_2g.10gb     28%         3 cores,  31 GB           abundant
#   a100_1g.5gb      14%         1 core,   15 GB           MOST abundant, near-zero wait
#
# CPU-only nodes (train mode) don't share the GPU bundle rule but do have
# their own core-per-RGU ratio. Narval CPU nodes have 48 cores each; typical
# training jobs use 32–48 cores per node with 4 GB/core.
#
# The defaults below are sized for the smallest MIG A100 instance — great for
# small tests (72q decoder etc.) with minimal queue wait. If you need bigger,
# bump n_cpus, mem_per_cpu, and gpu_type TOGETHER to stay inside the bundle.
# ----------------------------------------------------------------------------

# --- HPC resources (ignored on non-SLURM backends) ---
account          = "def-jemerson"

# Size per test-mode invocation. For train mode (test=false, no GPU), you can
# safely go up to 32–48 cores with 4G–8G/core on a CPU node.
n_cpus           = 1
mem_per_cpu      = "16G"

wall_time        = "1:00:00"
max_nodes        = 1
email            = "pavithran.sridhar@gmail.com"

# --- test mode ---
# When true: flips retrain = false in the hyperparams TOML, emits a test-mode
# script (GPU on SLURM+CUDA, Metal on Mac local, CPU fallback otherwise).
test             = false

# --- GPU knobs (used when test = true) ---
n_gpus_per_node  = 1

# Alliance Canada GPU model specifier. Pick a MIG instance for small tests
# (near-zero queue wait); pick "a100" only if you actually need the full GPU.
# The 72q/90q/144q BB decoders comfortably fit in a 1g.5gb MIG.
#   Narval:  "a100_1g.5gb"  (default, MIG 14%),  "a100_2g.10gb"  (MIG 28%),
#            "a100_3g.20gb" (MIG 50%),           "a100"          (whole GPU)
#   Fir/Nibi/Rorqual/Trillium:  "h100" and its "h100_*g.*gb" MIG variants
# See the ratios table linked in the header comment above.
gpu_type         = "a100_1g.5gb"

# Cluster's CUDA module name (used only on SLURM test mode).
cuda_module      = "cuda"

# Memory per GPU (test mode only, e.g. "16G"). When non-empty this OVERRIDES
# mem_per_cpu — SLURM disallows both --mem-per-gpu and --mem-per-cpu at once.
# For MIG instances it's usually simpler to set mem_per_cpu = "<bundle mem>"
# and leave this empty. Only use mem_per_gpu when you request multiple GPUs
# per node and want the memory scaled per-GPU rather than per-CPU.
mem_per_gpu      = ""

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

CMD=(julia --project="./../" batch_run.jl --settings "$SETTINGS_FILE")

if [ "$NO_EDIT" -eq 1 ]; then
    echo "[submit] --no-edit: skipping editor. Edit $SETTINGS_FILE manually, then run:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

if ! open_editor; then
    # Editor missing or non-interactive — print instructions and exit cleanly.
    echo "[submit] Edit $SETTINGS_FILE manually, then run:"
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

# ----------------------------------------------------------------------------
# Show the constructed command and (unless --no-run) execute it.
# ----------------------------------------------------------------------------
echo ""
echo "[submit] Will run:"
printf '  '; printf '%q ' "${CMD[@]}"; printf '\n\n'

if [ "$NO_RUN" -eq 1 ]; then
    echo "[submit] --no-run: not launching julia. Copy-Paste the above command to run manually."
    exit 0
fi

exec "${CMD[@]}"
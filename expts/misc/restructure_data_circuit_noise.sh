#!/usr/bin/env bash
# ============================================================================
# restructure_data_circuit_noise.sh
# ============================================================================
#
# Reshape Debankan's circuit-noise data folder into the directory
# layout the Neural BP pipeline expects. Run it FROM INSIDE the data folder
# (the one named like `18q_BB_p_0.0005_cycles_1`):
#
#     cd 18q_BB_p_0.0005_cycles_1
#     bash ./../../expts/restructure_data_circuit_noise.sh --suffix=p_0.0005
#
# Given `--suffix=<suffix>` it performs these moves (creating subdirs as
# needed). The suffix is inserted just before the .txt extension, so you can
# pass whatever tag you like (e.g. p_0.0005, p_0.0005_cycles_1, run3):
#
#   train_errors.txt        -> training_data/train_errors_<suffix>.txt
#   test_errors.txt         -> testing_data/test_errors_<suffix>.txt
#   correlated_weights.txt  -> correlated_weights/correlated_weights_<suffix>.txt
#   *failure_rates_OSD_E_order_2.txt
#                           -> results/            (name unchanged)
#   HX.txt HZ.txt LX.txt LZ.txt clique_edges.txt logical_checks.txt
#                           -> code/               (names unchanged)
#
# It also creates empty models/ and cluster/ directories (for later training
# outputs and cluster job files).
#
# Anything not matched by a rule is LEFT IN PLACE and reported, so nothing is
# silently moved or lost.
#
# The script is safe to re-run: files already in their destination are skipped
# rather than clobbered, and missing sources are reported, not fatal.
#
# Usage:
#     bash restructure_data_circuit_noise.sh --suffix=<suffix>
#     bash restructure_data_circuit_noise.sh --suffix=<suffix> --dry-run   # preview only
#     bash restructure_data_circuit_noise.sh --help
# ============================================================================

set -euo pipefail

usage() {
    # Print the leading comment block (minus the shebang) as help text.
    awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"
}

# ----------------------------------------------------------------------------
# Parse arguments. `--suffix=<suffix>` is required; `--dry-run` and `--help`
# optional. `--suffix <suffix>` (space-separated) is accepted too, as a
# convenience.
# ----------------------------------------------------------------------------
SUFFIX=""
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --suffix=*) SUFFIX="${1#--suffix=}"; shift ;;
        --suffix)   shift
                    [ $# -gt 0 ] || { echo "error: --suffix requires a value" >&2; exit 2; }
                    SUFFIX="$1"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$SUFFIX" ]; then
    echo "error: --suffix=<suffix> is required (e.g. --suffix=p_0.0005)" >&2
    usage >&2
    exit 2
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[restructure] DRY RUN — no files will be moved."
fi
echo "[restructure] working directory: $(pwd)"
echo "[restructure] suffix = $SUFFIX"
echo

moved=0
skipped=0
missing=0

# ----------------------------------------------------------------------------
# relocate <src> <dest_dir> [<dest_name>]
#   Move <src> into <dest_dir>, optionally renaming to <dest_name>
#   (defaults to the original basename). Idempotent + non-destructive:
#     - missing <src>            -> report, count as missing, continue
#     - destination already there -> report, count as skipped, continue
#   Honours --dry-run (prints the intended action without doing it).
# ----------------------------------------------------------------------------
relocate() {
    local src="$1"
    local dest_dir="$2"
    local dest_name="${3:-$(basename "$src")}"
    local dest="$dest_dir/$dest_name"

    if [ ! -e "$src" ]; then
        echo "  [missing] $src  (nothing to move)"
        missing=$((missing + 1))
        return 0
    fi
    if [ -e "$dest" ]; then
        echo "  [skip]    $dest already exists"
        skipped=$((skipped + 1))
        return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry]     $src -> $dest"
    else
        mkdir -p "$dest_dir"
        mv "$src" "$dest"
        echo "  [moved]   $src -> $dest"
    fi
    moved=$((moved + 1))
}

# --- Rules 1-3: rename with the suffix into their own subdirectories. ---------
echo "training_data/ , testing_data/ , correlated_weights/:"
relocate "train_errors.txt"       "training_data"      "train_errors_${SUFFIX}.txt"
relocate "test_errors.txt"        "testing_data"       "test_errors_${SUFFIX}.txt"
relocate "correlated_weights.txt" "correlated_weights" "correlated_weights_${SUFFIX}.txt"
echo

# --- Rule 4: OSD-order-2 failure-rate table(s) into results/ (name kept). ----
# Matched by suffix so the long "<dirname>_BP+OSD_..." prefix doesn't matter.
echo "results/:"
shopt -s nullglob
failure_rate_files=( *failure_rates_OSD_E_order_2.txt )
shopt -u nullglob
if [ "${#failure_rate_files[@]}" -eq 0 ]; then
    echo "  [missing] *failure_rates_OSD_E_order_2.txt  (nothing to move)"
    missing=$((missing + 1))
else
    for f in "${failure_rate_files[@]}"; do
        relocate "$f" "results"
    done
fi
echo

# --- Rule 5: parity-check matrices, logicals, clique edges, logical checks
#     into code/. logical_checks.txt is only moved if present (relocate handles
#     its absence gracefully). -------------------------------------------------
echo "code/:"
for f in HX.txt HZ.txt LX.txt LZ.txt clique_edges.txt logical_checks.txt; do
    relocate "$f" "code"
done
echo

# --- Create empty models/ and cluster/ directories (training outputs and
#     cluster job files, respectively). -----------------------------------------
for d in models cluster; do
    echo "$d/:"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry]     mkdir $d/"
    elif [ -d "$d" ]; then
        echo "  [skip]    $d/ already exists"
    else
        mkdir -p "$d"
        echo "  [mkdir]   $d/"
    fi
    echo
done

# ----------------------------------------------------------------------------
# Report anything left behind that no rule covered (excluding the subdirs we
# just created and this script itself, if it happens to sit in the folder).
# ----------------------------------------------------------------------------
echo "unmatched (left in place):"
self_name="$(basename "$0")"
found_unmatched=0
shopt -s nullglob
for entry in *; do
    [ -f "$entry" ] || continue                 # skip our new subdirectories
    [ "$entry" = "$self_name" ] && continue      # skip this script if copied in
    echo "  [kept]    $entry"
    found_unmatched=1
done
shopt -u nullglob
[ "$found_unmatched" -eq 0 ] && echo "  (none)"
echo

echo "[restructure] done: moved=$moved skipped=$skipped missing=$missing"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[restructure] (dry run — re-run without --dry-run to apply)"
fi

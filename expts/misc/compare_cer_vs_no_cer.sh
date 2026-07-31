#!/usr/bin/env bash
# ============================================================================
# compare_cer_vs_no_cer.sh — CER vs no-CER logical error rates, side by side
# ============================================================================
# Scans a results/ folder for the Neural BP per-simulation CSVs and pairs each
# `..._trained_using_train_p_<x>_s_1.csv` (CER) with its `..._no_cer.csv` twin,
# keyed on the TEST p parsed from the filename. Prints a 3-column CSV:
#
#     p,ler_with_cer,ler_without_cer
#
# rows sorted by ascending p. A missing counterpart leaves that field EMPTY
# (so pandas/DataFrames read it as missing rather than as a string) and logs a
# note to stderr. The LER column is located by header NAME, so column order in
# the input CSVs does not matter.
#
# Usage:
#   bash expts/misc/compare_cer_vs_no_cer.sh <results-folder> [out.csv]
# e.g.
#   bash expts/misc/compare_cer_vs_no_cer.sh ../data/update_23July2026/72q_BB_cycles_1/results
#   bash expts/misc/compare_cer_vs_no_cer.sh <folder> cer_comparison.csv
#
# With no second argument the CSV goes to stdout (pipe it wherever you like).
# ============================================================================
set -euo pipefail

FIELD="average_logical_error_rate"   # change to std_logical_error_rate / num_failures to compare those instead

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <results-folder> [out.csv]" >&2
    exit 2
fi
dir="$1"
out="${2:-}"

if [ ! -d "$dir" ]; then
    echo "not a directory: $dir" >&2
    exit 2
fi

# Pull one named field out of a 1-data-row CSV (header lookup, CRLF tolerant).
get_field() {
    awk -F, -v want="$FIELD" '
        { sub(/\r$/, "") }
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == want) col = i; next }
        NF && col { print $col; exit }
    ' "$1"
}

# Every test-p present under either naming (CER or _no_cer), numerically sorted.
p_values=$(ls "$dir"/simulation_results_test_p_*_s_1_*.csv 2>/dev/null \
           | sed -E 's|.*/simulation_results_test_p_([0-9.]+)_s_1_.*|\1|' \
           | sort -u -g)

if [ -z "$p_values" ]; then
    echo "no simulation_results_test_p_*.csv files found in $dir" >&2
    exit 1
fi

emit() { if [ -n "$out" ]; then cat >> "$out"; else cat; fi; }

[ -n "$out" ] && : > "$out"
printf 'p,ler_with_cer,ler_without_cer\n' | emit

n_pairs=0; n_cer_only=0; n_nocer_only=0
for p in $p_values; do
    # CER = the file WITHOUT the _no_cer suffix; no-CER = the one with it.
    cer_file=$(ls "$dir"/simulation_results_test_p_"${p}"_s_1_*.csv 2>/dev/null | grep -v '_no_cer\.csv$' | head -1 || true)
    nocer_file=$(ls "$dir"/simulation_results_test_p_"${p}"_s_1_*_no_cer.csv 2>/dev/null | head -1 || true)

    cer_val=""; nocer_val=""
    [ -n "$cer_file" ]   && cer_val=$(get_field "$cer_file")
    [ -n "$nocer_file" ] && nocer_val=$(get_field "$nocer_file")

    if [ -n "$cer_val" ] && [ -n "$nocer_val" ]; then
        n_pairs=$((n_pairs + 1))
    elif [ -n "$cer_val" ]; then
        n_cer_only=$((n_cer_only + 1))
        echo "  note: p=$p has CER only (no _no_cer counterpart)" >&2
    else
        n_nocer_only=$((n_nocer_only + 1))
        echo "  note: p=$p has no-CER only (no CER counterpart)" >&2
    fi

    printf '%s,%s,%s\n' "$p" "$cer_val" "$nocer_val" | emit
done

echo "done: $n_pairs paired, $n_cer_only CER-only, $n_nocer_only no-CER-only.${out:+ -> $out}" >&2

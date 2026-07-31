#!/usr/bin/env bash
# ============================================================================
# cer_vs_no_cer.sh — pair CER vs no-CER Neural BP runs by p
# ============================================================================
# Scans a results/ folder for
#   simulation_results_test_p_<P>_s_1_nlayers_*_epochs_*_trained_using_*.csv
# and pairs each CER run with its `_no_cer` counterpart, emitting
#
#   p,num_failures_CER,num_failures_no_CER
#
# sorted numerically by p. `p` comes from the FILENAME (`test_p_<P>_s_`);
# `num_failures` is located in the CSV by HEADER NAME, so column order doesn't
# matter. A p present in only one of the two sets is reported with `NA` in the
# missing column (and listed on stderr).
#
# Usage:
#   bash expts/misc/cer_vs_no_cer.sh <results-folder> [output.csv]
# e.g. from expts/ :
#   bash misc/cer_vs_no_cer.sh ../data/update_23July2026/72q_BB_cycles_1/results
#   bash misc/cer_vs_no_cer.sh ../data/.../results cer_comparison.csv
# With no output path the table goes to stdout.
# ============================================================================
set -euo pipefail

dir="${1:-.}"
out="${2:-}"

if [ ! -d "$dir" ]; then
    echo "not a directory: $dir" >&2
    exit 2
fi

shopt -s nullglob
files=( "$dir"/simulation_results_test_p_*_s_1_nlayers_*_epochs_*_trained_using_*.csv )
shopt -u nullglob

if [ "${#files[@]}" -eq 0 ]; then
    echo "no matching simulation_results_*.csv in $dir" >&2
    exit 1
fi

emit() {
    echo "p,num_failures_CER,num_failures_no_CER"
    awk -F, '
        # Header: find num_failures by name (re-evaluated per file).
        FNR == 1 {
            col = 0
            for (i = 1; i <= NF; i++) if ($i == "num_failures") col = i
            if (col == 0) printf "  !! no num_failures column: %s\n", FILENAME > "/dev/stderr"
            next
        }
        # First data row only (these files hold one record).
        FNR == 2 && col > 0 {
            name = FILENAME
            sub(/.*\//, "", name)
            if (match(name, /test_p_[0-9.]+([eE][-+]?[0-9]+)?_s_/)) {
                tag = substr(name, RSTART, RLENGTH)
                gsub(/^test_p_|_s_$/, "", tag)
                seen[tag] = 1
                if (name ~ /_no_cer\.csv$/) nocer[tag] = $col; else cer[tag] = $col
            } else {
                printf "  !! could not read p from filename: %s\n", name > "/dev/stderr"
            }
        }
        END {
            for (p in seen) {
                c = (p in cer)   ? cer[p]   : "NA"
                n = (p in nocer) ? nocer[p] : "NA"
                if (c == "NA" || n == "NA")
                    printf "  !! unpaired p=%s (CER=%s, no_CER=%s)\n", p, c, n > "/dev/stderr"
                printf "%s,%s,%s\n", p, c, n
            }
        }
    ' "${files[@]}" | sort -t, -k1,1g
}

if [ -n "$out" ]; then
    emit > "$out"
    echo "wrote $out ($(( $(wc -l < "$out") - 1 )) rows)" >&2
else
    emit
fi

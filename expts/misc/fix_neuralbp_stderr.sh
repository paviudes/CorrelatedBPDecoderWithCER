#!/usr/bin/env bash
# ============================================================================
# fix_neuralbp_stderr.sh — ONE-OFF backward-compat fix (NOT part of the pipeline)
# ============================================================================
# Older Neural BP simulation-result CSVs stored `std_logical_error_rate` computed
# with n_layers (=100) in the denominator instead of num_samples_per_error_rate
# (=10^6). For those 10^6-sample runs the reported error is too large by exactly
# sqrt(10^6 / 100) = 100, so dividing the std column by 100 restores the correct
# standard error (identical to recomputing sqrt(p(1-p)/10^6)).
#
# Scope guard — only files named
#   simulation_results_test_p_*_s_1_nlayers_100_epochs_20_trained_using_train_p_*_s_1.csv
# are touched, and only when EVERY data row has num_samples_per_error_rate ==
# 1000000 (the factor is 10^6-specific). Columns are located by header NAME, so
# column order does not matter. The original is copied to <file>.bak first, and a
# file that already has a .bak is skipped — so re-running cannot double-divide.
#
# Usage:
#   bash expts/misc/fix_neuralbp_stderr.sh <folder> [<folder> ...]
# e.g. from expts/ :
#   bash misc/fix_neuralbp_stderr.sh \
#       ../data/update_23July2026/72q_BB_cycles_1/results \
#       ../data/update_23July2026/18q_BB_cycles_1/results
#
# After verifying the result, delete the backups with:  rm <folder>/*.csv.bak
# ============================================================================
set -euo pipefail

FACTOR=100
EXPECTED_N=1000000
PATTERN='simulation_results_test_p_*_s_1_nlayers_100_epochs_20_trained_using_train_p_*_s_1.csv'

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <folder> [<folder> ...]" >&2
    exit 2
fi

total_fixed=0
total_skipped=0

for dir in "$@"; do
    if [ ! -d "$dir" ]; then
        echo "!! not a directory, skipping: $dir" >&2
        continue
    fi

    shopt -s nullglob
    files=( "$dir"/$PATTERN )
    shopt -u nullglob

    if [ "${#files[@]}" -eq 0 ]; then
        echo "-- $dir: no matching CSVs"
        continue
    fi
    echo "== $dir: ${#files[@]} matching file(s)"

    for f in "${files[@]}"; do
        base=$(basename "$f")

        # Idempotency guard: a .bak means this file was already corrected.
        if [ -e "$f.bak" ]; then
            echo "   skip (already has .bak): $base"
            total_skipped=$((total_skipped + 1))
            continue
        fi

        # Rewrite into a temp file; commit only if the whole file validates.
        # awk locates columns by header name, checks N == 10^6 on every row, and
        # divides only std_logical_error_rate by FACTOR (untouched columns keep
        # their exact original text; %.17g keeps full double precision).
        if awk -F, -v OFS=, -v factor="$FACTOR" -v expN="$EXPECTED_N" '
            BEGIN { CONVFMT = "%.17g"; OFMT = "%.17g"; scol = 0; ncol = 0; shown = 0 }
            NR == 1 {
                for (i = 1; i <= NF; i++) {
                    if ($i == "std_logical_error_rate")     scol = i
                    if ($i == "num_samples_per_error_rate") ncol = i
                }
                if (scol == 0) { print "  no std_logical_error_rate column"     > "/dev/stderr"; exit 3 }
                if (ncol == 0) { print "  no num_samples_per_error_rate column" > "/dev/stderr"; exit 4 }
                print; next
            }
            NF == 0 { print; next }                         # preserve blank lines
            {
                if ($ncol + 0 != expN) {
                    printf "  row %d: num_samples=%s (expected %d) - not a 10^6 run\n", NR, $ncol, expN > "/dev/stderr"
                    exit 5
                }
                old = $scol
                $scol = $scol / factor
                if (!shown) { printf "        sample: std %s -> %s\n", old, $scol > "/dev/stderr"; shown = 1 }
                print
            }
        ' "$f" > "$f.tmp"; then
            cp -p "$f" "$f.bak"     # backup the original only after awk succeeds
            mv "$f.tmp" "$f"
            echo "   fixed: $base"
            total_fixed=$((total_fixed + 1))
        else
            rc=$?
            rm -f "$f.tmp"
            echo "   SKIP (validation failed, rc=$rc): $base" >&2
            total_skipped=$((total_skipped + 1))
        fi
    done
done

echo "done: $total_fixed fixed, $total_skipped skipped."

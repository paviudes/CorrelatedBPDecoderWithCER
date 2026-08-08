#!/usr/bin/env bash
# ============================================================================
# cer_vs_no_cer.sh — thin wrapper around misc/cer_vs_no_cer.jl
# ============================================================================
# RUN FROM expts/ :
#
#     bash misc/cer_vs_no_cer.sh
#     bash misc/cer_vs_no_cer.sh ./../data/<other>/72q_BB_cycles_1/results
#     bash misc/cer_vs_no_cer.sh <results_dir> --csv /tmp/summary.csv
#
# Compares the Neural BP CER arm against the no-CER baseline at every error
# rate, and puts the plain-BP and BP-OSD failure rates alongside for scale.
#
# The work is in the Julia script (this repo's analysis code lives in Julia, and
# Julia is what is guaranteed present on the cluster). This wrapper exists only
# so the old `bash misc/cer_vs_no_cer.sh` entry point keeps working.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec julia --project="$SCRIPT_DIR/../../" "$SCRIPT_DIR/cer_vs_no_cer.jl" "$@"

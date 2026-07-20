#!/usr/bin/env python3
# ============================================================================
# reformat_standard_bp_failures.py
# ============================================================================
#
# The BP-OSD (standard-decoder) baseline is delivered as one file per (p, sample),
# each containing a SINGLE number: the failure count. e.g.
#
#     72q_BB_p_0.0005_cycles_1_BP+OSD_failure_rates_OSD_E_order_2.txt   ->  298
#
# `collect_standard_decoder_statistics(:Circuit; ...)` instead expects a single
# aggregated file whose rows are the 6-column circuit layout:
#
#     <p> <sample> <failures> <total_trials> <average> <sigma>
#
# This script bridges the two. For every matching file in a directory it:
#   - reads the failure count from the file body,
#   - pulls `p` from `p_<p>` and `sample` from `cycles_<sample>` in the filename,
#   - uses a fixed trial count (default 100000; override with --trials),
#   - computes  average = failures / trials  and the binomial standard error of
#     that rate,  sigma = sqrt(average * (1 - average) / trials),
# and writes one aggregated file (default: standard_bp_failure_rates.txt),
# sorted by (p, sample). Originals are left untouched.
#
# Usage (the folder can be given positionally or with --directory):
#     python3 reformat_standard_bp_failures.py <dir> [--trials N] [--out FILE]
#         [--glob '*BP+OSD_failure_rates_OSD_E_order_2.txt'] [--dry-run]
#     python3 reformat_standard_bp_failures.py --directory <dir> --trials 100000 \
#         --out 18q_BB_cycles_1_BP+OSD_failure_rates_OSD_E_order_2.txt
# ============================================================================

import argparse
import glob
import math
import os
import re
import sys


def extract_p_and_sample(filename):
    """Pull the `p` value and the `sample` (from `cycles_<n>`) out of a filename."""
    base = os.path.basename(filename)
    p_match = re.search(r"p_([0-9]*\.?[0-9]+)", base)
    s_match = re.search(r"cycles_([0-9]+)", base)
    if p_match is None:
        raise ValueError(f"could not find 'p_<value>' in filename: {base}")
    if s_match is None:
        raise ValueError(f"could not find 'cycles_<sample>' in filename: {base}")
    return p_match.group(1), int(s_match.group(1))


def read_failures(path):
    """Read the single integer failure count from a file body."""
    with open(path) as fh:
        tokens = fh.read().split()
    if len(tokens) != 1:
        raise ValueError(
            f"expected exactly one number in {os.path.basename(path)}, got {len(tokens)}: {tokens}"
        )
    return int(tokens[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", default=None,
                    help="folder containing the per-(p, sample) failure files (positional)")
    ap.add_argument("--directory", "-d", dest="directory_flag", default=None,
                    help="same folder, as a named flag (equivalent to the positional form)")
    ap.add_argument("--trials", type=int, default=100_000,
                    help="fixed number of trials per file (default: 100000)")
    ap.add_argument("--glob", default="*BP+OSD_failure_rates_OSD_E_order_2.txt",
                    help="glob for the input files (default matches the OSD_E_order_2 files)")
    ap.add_argument("--out", default="standard_bp_failure_rates.txt",
                    help="aggregated output filename, written inside <directory> (default: standard_bp_failure_rates.txt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rows that would be written, but don't write the file")
    args = ap.parse_args(argv)

    directory = args.directory_flag or args.directory
    if directory is None:
        ap.error("provide the folder positionally or via --directory/-d")

    paths = sorted(glob.glob(os.path.join(directory, args.glob)))
    if not paths:
        print(f"[reformat] no files matched {args.glob!r} in {args.directory}", file=sys.stderr)
        return 1

    rows = []
    for path in paths:
        base = os.path.basename(path)
        if base == args.out:
            continue  # never fold the current aggregate back into itself
        # Only per-(p, sample) input files carry `p_<value>` and `cycles_<n>`
        # tags. Skip anything else that happens to match the glob — e.g. a
        # previously-written aggregate (which has no `p_` tag) — instead of
        # crashing on it.
        if (re.search(r"p_[0-9]*\.?[0-9]+", base) is None
                or re.search(r"cycles_[0-9]+", base) is None):
            print(f"[reformat] skipping (no p_/cycles_ tag): {base}", file=sys.stderr)
            continue
        p_str, sample = extract_p_and_sample(path)
        failures = read_failures(path)
        trials = args.trials
        average = failures / trials
        sigma = math.sqrt(average * (1.0 - average) / trials)
        # sort key uses the numeric p; the emitted p keeps the filename's exact text
        rows.append((float(p_str), sample, p_str, failures, trials, average, sigma))

    rows.sort(key=lambda r: (r[0], r[1]))

    lines = [f"{p_str} {sample} {failures} {trials} {average:.10g} {sigma:.10g}"
             for (_, sample, p_str, failures, trials, average, sigma) in rows]

    print(f"[reformat] {len(rows)} file(s) -> {len(lines)} row(s), trials={args.trials}")
    for line in lines:
        print("   " + line)

    if args.dry_run:
        print("[reformat] --dry-run: nothing written")
        return 0

    out_path = os.path.join(directory, args.out)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[reformat] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

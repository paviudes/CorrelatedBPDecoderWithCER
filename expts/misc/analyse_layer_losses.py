#!/usr/bin/env python3
"""
Which layer actually holds the minimum of the per-layer base loss?

    python3 misc/analyse_layer_losses.py --workdir <codename>
    python3 misc/analyse_layer_losses.py --workdir <codename> --layers 90
    python3 misc/analyse_layer_losses.py --workdir <codename> --tag opthistpair

Reads `logs/debugging_*_individual_losses.csv`, which `train.jl` writes for every
run started with `--isdebug true` (the sweep generator always passes it). The
`base_loss` column is the FULL per-layer vector for the scored layers -- a quoted
comma-separated list -- and `total_loss` is whatever `loss_layer_selection`
reduced it to. So the per-layer logging the question asks for already exists; this
script just reads it.

WHY THIS MATTERS. The design-A argument is that training on the last layer is
wrong because the loss minimum sits somewhere else. That is checkable directly:
for each batch, compare `base_loss[end]` against `min(base_loss)`.

Watch for the FLOOR. The base loss does not decay to zero, it decays to a
saturation floor set by sigma(mu) on already-correct decodes (CLAUDE.md: 96.2% of
the base loss is that saturation, only 3.8% is coset signal). Once a batch reaches
the floor, EVERY later layer sits on it, the argmin is just the first layer to get
there, and `base_loss[end]` equals the minimum exactly. A batch like that cannot
distinguish the two designs no matter which layer the loss reads. The interesting
population is the batches whose minimum is ABOVE the floor -- those are the ones
carrying gradient, and they are the only ones where "which layer" has an answer.
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(10 ** 7)

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# A batch counts as sitting ON the floor when its minimum is within this factor of
# the smallest minimum seen in that file. The floor is a property of the run (it
# depends on the priors and the rescale), so it is measured per file, not assumed.
FLOOR_TOLERANCE: float = 1.5
# `last == min` in float terms. The values are Float32 written through CSV, so an
# exact tie is what the plateau produces; this only absorbs the last decimal.
TIE_TOLERANCE: float = 1e-6
# A batch carries usable gradient when its MINIMUM is still this far above zero.
# An ABSOLUTE threshold, not a multiple of the per-file floor: the floor test
# above finds a sharp plateau (the CER arms sit on one) but calls a batch
# "informative" whenever the values merely spread out, which over-counts badly on
# the no-CER arms where the minima are tiny but not identical. The base loss
# starts around 32 at layer 1, so 1e-3 is five orders of magnitude down -- a batch
# below it is a solved batch whichever layer is read.
INFORMATIVE_MINIMUM: float = 1e-3


def arm_name_from_path(path: str) -> str:
    stem: str = os.path.basename(path)
    stem = stem.replace("_individual_losses.csv", "")
    marker: str = "_s_1_"
    if marker in stem:
        stem = stem.split(marker, 1)[1]
    for sample_suffix in ("200000_", "1000000_"):
        if stem.startswith(sample_suffix):
            stem = stem[len(sample_suffix):]
    stem = stem.replace("no_cer_", "")
    return stem


def read_layer_losses(path: str) -> Tuple[List[List[float]], List[float], int]:
    per_batch_losses: List[List[float]] = []
    total_losses: List[float] = []
    scored_layers: int = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            layer_losses: List[float] = [float(value) for value in row["base_loss"].split(",")]
            per_batch_losses.append(layer_losses)
            total_losses.append(float(row["total_loss"]))
            scored_layers = int(row["layers"])
    return per_batch_losses, total_losses, scored_layers


def summarise_one_file(path: str) -> Optional[Dict[str, object]]:
    per_batch_losses, total_losses, scored_layers = read_layer_losses(path)
    if len(per_batch_losses) == 0:
        return None

    batch_minima: List[float] = [min(losses) for losses in per_batch_losses]
    floor_value: float = min(batch_minima)
    floor_cutoff: float = floor_value * FLOOR_TOLERANCE

    argmin_layers_informative: List[int] = []
    excess_ratios: List[float] = []
    n_on_floor: int = 0
    n_last_is_min: int = 0
    n_monotone: int = 0
    n_last_at_floor: int = 0

    for losses in per_batch_losses:
        minimum: float = min(losses)
        last: float = losses[-1]
        argmin_layer: int = losses.index(minimum) + 1

        if last <= minimum * (1.0 + TIE_TOLERANCE):
            n_last_is_min += 1
        else:
            excess_ratios.append(last / minimum)
        if all(losses[i + 1] <= losses[i] * (1.0 + TIE_TOLERANCE) for i in range(len(losses) - 1)):
            n_monotone += 1
        if last <= floor_cutoff:
            n_last_at_floor += 1
        if minimum <= floor_cutoff:
            n_on_floor += 1
        if minimum > INFORMATIVE_MINIMUM:
            argmin_layers_informative.append(argmin_layer)

    n_batches: int = len(per_batch_losses)
    summary: Dict[str, object] = {
        "arm": arm_name_from_path(path),
        "n_batches": n_batches,
        "scored_layers": scored_layers,
        "floor": floor_value,
        "median_minimum": statistics.median(batch_minima),
        "pct_on_floor": 100.0 * n_on_floor / n_batches,
        "pct_last_is_min": 100.0 * n_last_is_min / n_batches,
        "pct_last_at_floor": 100.0 * n_last_at_floor / n_batches,
        "pct_monotone": 100.0 * n_monotone / n_batches,
        "n_informative": len(argmin_layers_informative),
        "argmin_informative": argmin_layers_informative,
        "excess_ratios": excess_ratios,
        "mean_total_loss": statistics.mean(total_losses),
    }
    return summary


def print_summary_table(summaries: List[Dict[str, object]]) -> None:
    print()
    print("PER-LAYER BASE LOSS: where is the minimum, and does the last layer hold it?")
    print()
    header: str = (
        f'{"ARM":<46}{"batches":>8}{"layers":>7}{"floor":>10}'
        f'{"median min":>12}{"last==min":>11}{"monotone":>10}'
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f'{summary["arm"]:<46}{summary["n_batches"]:>8}{summary["scored_layers"]:>7}'
            f'{summary["floor"]:>10.2g}{summary["median_minimum"]:>12.2g}'
            f'{summary["pct_last_is_min"]:>10.1f}%{summary["pct_monotone"]:>9.1f}%'
        )

    print()
    print(f"ARGMIN LAYER, restricted to batches whose minimum exceeds {INFORMATIVE_MINIMUM:g}")
    print("(the batches that actually carry gradient; the rest are solved whichever layer is read)")
    print()
    header2: str = (
        f'{"ARM":<46}{"informative":>12}{"of":>8}{"median":>8}{"p10":>6}{"p90":>6}'
        f'{"max":>6}      last layer above min'
    )
    print(header2)
    print("-" * (len(header2) + 2))
    for summary in summaries:
        argmins: List[int] = summary["argmin_informative"]  # type: ignore[assignment]
        excess: List[float] = summary["excess_ratios"]       # type: ignore[assignment]
        share: str = f'{100.0 * len(argmins) / int(summary["n_batches"]):.1f}%'
        if len(argmins) == 0:
            print(f'{summary["arm"]:<46}{0:>12}{share:>8}{"-":>8}{"-":>6}{"-":>6}{"-":>6}')
            continue
        ordered: List[int] = sorted(argmins)
        p10: int = ordered[int(0.10 * (len(ordered) - 1))]
        p90: int = ordered[int(0.90 * (len(ordered) - 1))]
        if len(excess) == 0:
            excess_text: str = "never"
        else:
            excess_text = f"{len(excess)} batch(es), median x{statistics.median(excess):.2f}"
        print(
            f'{summary["arm"]:<46}{len(argmins):>12}{share:>8}{statistics.median(ordered):>8.0f}'
            f'{p10:>6}{p90:>6}{max(ordered):>6}      {excess_text}'
        )


def print_argmin_histogram(summaries: List[Dict[str, object]]) -> None:
    print()
    print("ARGMIN LAYER HISTOGRAM (informative batches, pooled over the arms above)")
    print()
    pooled: List[int] = []
    for summary in summaries:
        pooled.extend(summary["argmin_informative"])  # type: ignore[arg-type]
    if len(pooled) == 0:
        print("  no informative batches: every batch reached the saturation floor.")
        return
    buckets: Dict[str, int] = {}
    edges: List[Tuple[int, int]] = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 70), (71, 90)]
    for low, high in edges:
        label: str = f"{low}-{high}"
        buckets[label] = sum(1 for layer in pooled if low <= layer <= high)
    widest: int = max(buckets.values())
    for label, count in buckets.items():
        if count == 0 and label.startswith(("51", "71")):
            continue
        bar: str = "#" * int(40 * count / widest) if widest else ""
        print(f"  layers {label:<7} {count:>6}  {100.0 * count / len(pooled):>5.1f}%  {bar}")
    print(f"  total {len(pooled)} informative batch(es)")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Locate the minimum of the per-layer base loss in training debug logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workdir", required=True, metavar="RUN_DIR",
                        help="run directory: a path, or a codename resolved under data/")
    parser.add_argument("--layers", default="", metavar="N",
                        help="only files whose scored-layer count is N")
    parser.add_argument("--tag", default="", metavar="SUBSTRING",
                        help="only files whose name contains SUBSTRING (e.g. opthistpair)")
    arguments = parser.parse_args(argv)

    run_directory: str = arguments.workdir
    if not os.path.isdir(run_directory):
        run_directory = os.path.join(REPO_ROOT, "data", arguments.workdir)
    logs_directory: str = os.path.join(run_directory, "logs")
    if not os.path.isdir(logs_directory):
        print(f"no logs directory: {logs_directory}", file=sys.stderr)
        return 1

    pattern: str = os.path.join(logs_directory, "*_individual_losses.csv")
    paths: List[str] = sorted(glob.glob(pattern))
    if arguments.tag:
        paths = [path for path in paths if arguments.tag in os.path.basename(path)]
    if len(paths) == 0:
        print(f"no *_individual_losses.csv in {logs_directory}", file=sys.stderr)
        print("  Training must run with --isdebug true (the sweep generator always passes it).",
              file=sys.stderr)
        return 1

    summaries: List[Dict[str, object]] = []
    for path in paths:
        summary: Optional[Dict[str, object]] = summarise_one_file(path)
        if summary is None:
            continue
        if arguments.layers and str(summary["scored_layers"]) != str(arguments.layers):
            continue
        summaries.append(summary)
    if len(summaries) == 0:
        print("no files matched the filters.", file=sys.stderr)
        return 1

    print_summary_table(summaries)
    print_argmin_histogram(summaries)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

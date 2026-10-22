#!/usr/bin/env python3
"""
compare_sweep_arms.py -- seed-paired comparison of the arms of a sweep_hyperparams.sh sweep.

    python3 misc/compare_sweep_arms.py --workdir 72q_BB_cycles_1_trainable_alpha \
        --arms _cer_enr_rampw0_last _cer_enr_rampw0_first

An arm is a run tag, leading underscore included, as it appears in the file names
(_cer_enr_rampw0_last, _nocer_ramp_last, ...); a no-CER arm's _no_cer infix is
handled for you. The FIRST arm is the reference. For every test set the script
prints each arm's failures summed over the seeds, as total (coset | convergence),
the change against the reference, in how many seeds the arm is lower, and the
seed-paired t of the log ratio (df = seeds - 1); then the same over all test sets.

Before the counts it compares the arms' WEIGHTS seed by seed (weights_c2v_v2c,
weights_llrs, weights_c2v_readout, coupling_logit). Arms that differ only in a
test-time setting -- commit_layer_rule -- train the same model, so their weights
files should be identical; if they are not, the comparison mixes training with the
readout, and the script says so.

reads  <run>/results/simulation_results_test_<test key>_nlayers_<L>_epochs_<n>_trained_using_train_<train key>[_no_cer]<run tag>_seed_<s>.csv
       <run>/models/neuralbp_weights_nlayers_<L>_epochs_<n>_trained_using_train_<train key>[_no_cer]<run tag>_seed_<s>.json

--train-key, --epochs, --layers pick one training configuration when an arm's
files name several; --seeds restricts the seeds (default: every seed that all
arms have on every test set).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_KEY = r"p_[0-9.eE+-]+_sig_[0-9.eE+-]+_s_\d+"
RESULTS_NAME = re.compile(
    rf"^simulation_results_test_(?P<test_key>{DATASET_KEY})_nlayers_(?P<layers>\d+)_epochs_(?P<epochs>\d+)"
    rf"_trained_using_train_(?P<train_key>{DATASET_KEY})(?P<no_cer>_no_cer)?(?P<run_tag>_.+?)?_seed_(?P<seed>\d+)\.csv$")
WEIGHT_FIELDS = ("weights_c2v_v2c", "weights_llrs", "weights_c2v_readout", "coupling_logit")


def read_results(results_directory: str) -> List[Dict[str, object]]:
    """Every results CSV whose name parses, with the counts of its one row."""
    rows: List[Dict[str, object]] = []
    for path in sorted(glob.glob(os.path.join(results_directory, "simulation_results_test_*.csv"))):
        name_match = RESULTS_NAME.match(os.path.basename(path))
        if name_match is None:
            continue
        with open(path) as handle:
            record = next(csv.DictReader(handle), None)
        if record is None:
            continue

        def count(field: str) -> Optional[int]:
            value = record.get(field, "")
            return int(float(value)) if value not in ("", None) else None

        rows.append(dict(
            test_key=name_match["test_key"], train_key=name_match["train_key"],
            layers=int(name_match["layers"]), epochs=int(name_match["epochs"]),
            no_cer=name_match["no_cer"] is not None, run_tag=name_match["run_tag"] or "",
            seed=int(name_match["seed"]), total=count("num_failures"), coset=count("num_coset_failures"),
            convergence=count("num_convergence_failures"),
            committed_layer=float(record["mean_committed_layer"]) if record.get("mean_committed_layer") else None,
            commit_rule=record.get("commit_layer_rule", "?"), path=path))
    return rows


def weights_name(layers: int, epochs: int, train_key: str, no_cer: bool, run_tag: str, seed: int) -> str:
    """src/train.jl's weights file name."""
    infix = "_no_cer" if no_cer else ""
    name = f"neuralbp_weights_nlayers_{layers}_epochs_{epochs}_trained_using_train_{train_key}{infix}{run_tag}_seed_{seed}.json"
    return name


def paired_statistics(reference: np.ndarray, other: np.ndarray) -> Tuple[float, int, float]:
    """(change in % of the summed counts, seeds where other < reference, paired t of log(other/reference))."""
    change = 100.0 * (other.sum() / reference.sum() - 1.0) if reference.sum() > 0 else float("nan")
    lower = int((other < reference).sum())
    t_value = float("nan")
    if len(reference) > 1 and np.all(reference > 0) and np.all(other > 0):
        log_ratio = np.log(other / reference)
        spread = log_ratio.std(ddof=1)
        if spread > 0:
            t_value = float(log_ratio.mean() / (spread / math.sqrt(len(log_ratio))))
    return change, lower, t_value


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, help="run directory: a path, or a codename under data/")
    parser.add_argument("--arms", nargs="+", required=True, help="run tags; the first is the reference")
    parser.add_argument("--train-key", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    arguments = parser.parse_args(argv)
    if len(arguments.arms) < 2:
        raise SystemExit("--arms needs at least two run tags")
    for arm in arguments.arms:
        if not arm.startswith("_"):
            raise SystemExit(f"arm '{arm}': give the run tag with its leading underscore, as in the file names")

    run_directory = arguments.workdir if os.path.isdir(arguments.workdir) else os.path.join(REPO_ROOT, "data", arguments.workdir)
    rows = read_results(os.path.join(run_directory, "results"))
    by_arm: Dict[str, List[Dict[str, object]]] = {}
    for arm in arguments.arms:
        selected = [row for row in rows if row["run_tag"] == arm
                    and (not arguments.train_key or row["train_key"] == arguments.train_key)
                    and (not arguments.epochs or row["epochs"] == arguments.epochs)
                    and (not arguments.layers or row["layers"] == arguments.layers)]
        configurations = sorted({(row["train_key"], row["epochs"], row["layers"]) for row in selected})
        if not selected:
            raise SystemExit(f"no results for arm {arm} in {run_directory}/results")
        if len(configurations) > 1:
            listed = ", ".join(f"train {key} / {epochs} epochs / {layers} layers" for key, epochs, layers in configurations)
            raise SystemExit(f"arm {arm} has results of several trainings ({listed}); pick one with "
                             "--train-key / --epochs / --layers")
        by_arm[arm] = selected

    # Seeds and test sets that every arm has: a paired comparison needs both halves.
    cells = {arm: {(row["test_key"], row["seed"]): row for row in arm_rows} for arm, arm_rows in by_arm.items()}
    test_keys = sorted(set.intersection(*[{key for key, _ in arm_cells} for arm_cells in cells.values()]))
    seeds = sorted(set.intersection(*[{seed for _, seed in arm_cells} for arm_cells in cells.values()]))
    if arguments.seeds:
        seeds = [seed for seed in seeds if seed in arguments.seeds]
    complete_seeds = [seed for seed in seeds if all((key, seed) in arm_cells for arm_cells in cells.values() for key in test_keys)]
    for seed in sorted(set(seeds) - set(complete_seeds)):
        print(f"note: seed {seed} lacks a test set in some arm; left out")
    seeds = complete_seeds
    if not seeds or not test_keys:
        raise SystemExit("no seed has every test set in every arm")
    for arm, arm_cells in cells.items():
        extra = sorted({key for key, _ in arm_cells} - set(test_keys))
        if extra:
            print(f"note: {arm} also has {', '.join(extra)}, which another arm lacks; left out")

    reference_arm = arguments.arms[0]
    first_row = by_arm[reference_arm][0]
    print(f"{run_directory}: trained on {first_row['train_key']}, {first_row['epochs']} epochs, "
          f"{first_row['layers']} layers; seeds {seeds}; {len(test_keys)} test set(s)")
    for arm in arguments.arms:
        rules = sorted({row["commit_rule"] for row in by_arm[arm]})
        print(f"  {arm}: commit_layer_rule {', '.join(rules)}")

    # Weights: identical files mean the arms are one model read out two ways.
    print("\nweights, seed by seed, against the reference arm:")
    for arm in arguments.arms[1:]:
        verdicts: List[str] = []
        for seed in seeds:
            reference_path = os.path.join(run_directory, "models", weights_name(
                first_row["layers"], first_row["epochs"], first_row["train_key"],
                bool(by_arm[reference_arm][0]["no_cer"]), reference_arm, seed))
            arm_row = by_arm[arm][0]
            arm_path = os.path.join(run_directory, "models", weights_name(
                arm_row["layers"], arm_row["epochs"], arm_row["train_key"], bool(arm_row["no_cer"]), arm, seed))
            if not (os.path.isfile(reference_path) and os.path.isfile(arm_path)):
                verdicts.append(f"seed {seed}: weights file missing")
                continue
            reference_weights, arm_weights = json.load(open(reference_path)), json.load(open(arm_path))
            largest_difference = 0.0
            for field in WEIGHT_FIELDS:
                reference_values = np.asarray(reference_weights.get(field, []), dtype=np.float64)
                arm_values = np.asarray(arm_weights.get(field, []), dtype=np.float64)
                if reference_values.shape != arm_values.shape:
                    largest_difference = float("inf")
                    break
                if reference_values.size:
                    largest_difference = max(largest_difference, float(np.abs(reference_values - arm_values).max()))
            verdicts.append(f"seed {seed}: " + ("identical" if largest_difference == 0.0
                                                else f"DIFFERENT (largest |difference| {largest_difference:.3g})"))
        print(f"  {arm}: " + "; ".join(verdicts))

    def series(arm: str, key: str, field: str) -> np.ndarray:
        values = [cells[arm][(key, seed)][field] for seed in seeds]
        return np.array([np.nan if value is None else value for value in values], dtype=np.float64)

    print(f"\nfailures summed over {len(seeds)} seed(s): total (coset | convergence); the change, seeds lower and "
          f"paired t are against {reference_arm}")
    totals = {arm: np.zeros(len(seeds)) for arm in arguments.arms}
    for key in test_keys + ["ALL"]:
        print(f"  {key}")
        for arm in arguments.arms:
            if key == "ALL":
                total = totals[arm]
                coset = sum(series(arm, test_key, "coset") for test_key in test_keys)
                convergence = sum(series(arm, test_key, "convergence") for test_key in test_keys)
                layer = np.nanmean([series(arm, test_key, "committed_layer") for test_key in test_keys])
            else:
                total = series(arm, key, "total")
                totals[arm] = totals[arm] + total
                coset, convergence = series(arm, key, "coset"), series(arm, key, "convergence")
                layer = np.nanmean(series(arm, key, "committed_layer"))
            split = (f"({int(np.nansum(coset))} | {int(np.nansum(convergence))})"
                     if not (np.isnan(coset).all() or np.isnan(convergence).all()) else "")
            line = f"    {arm:32s} {int(total.sum()):7d} {split:17s} mean committed layer {layer:5.2f}"
            if arm != reference_arm:
                reference_total = totals[reference_arm] if key == "ALL" else series(reference_arm, key, "total")
                change, lower, t_value = paired_statistics(reference_total, total)
                line += f"   {change:+6.1f}%, lower in {lower}/{len(seeds)} seeds, t {t_value:+.1f}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

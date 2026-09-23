#!/usr/bin/env python3
"""
check_parameter_movement.py — did the trainable parameters actually move?

    python3 misc/check_parameter_movement.py --workdir <run directory>
    python3 misc/check_parameter_movement.py --workdir <run dir> --tag opt

The primary outcome of the optimizer sweep is NOT the failure count. It is
whether alpha and the step schedule move at all. They did not in any run so far:

    alpha  0.42 -> 0.4156 +- 0.008   over 2500 updates
    T0     12   -> 12.0006 +- 0.039  over 2500 updates

At lr = 0.001 a parameter taking full Adam steps travels 2.5 in that budget, so
those moved by ~1e-4 of nominal. The suspected cause is `adam_eps = 1e-4`,
10000x the 1e-8 default: Adam's step is lr * m/(sqrt(v) + eps), and for a
small-gradient parameter sqrt(v) ~ eps, so the denominator is eps-dominated and
the step collapses. The message weights have large enough gradients that
sqrt(v) >> eps and move normally — which is the asymmetry observed.

This script reads the trained weights JSONs and reports, per (optimizer arm,
check-node arm):

    alpha       final value, spread over seeds, and |final - init|
    T0, w       same, for the step schedule
    weight sd   spread of the message weights, against THIS arm's own
                initial_conditions_scale — a swept ic_scale changes the "never
                moved" reference, so a fixed 0.058 would be wrong here
    epochs      completed epochs (a rolled-back epoch reverts the parameters,
                which is itself a way to see no movement)

Read `alpha move` and `T0 move` first. If they stay ~1e-3 at every setting, the
optimizer is not the binding constraint and per-layer alpha cannot work either.
"""

import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = REPO_ROOT / "data"

# A uniform draw on 1 +- s has standard deviation s/sqrt(3); `random_values_around_one`
# is that draw, so this is the weight sd of an untrained model at a given
# initial_conditions_scale. 0.1 -> 0.0577, which is the 0.058 quoted elsewhere.
UNIFORM_SD_FACTOR: float = 1.0 / math.sqrt(3.0)

WEIGHTS_PATTERN: re.Pattern = re.compile(
    r"neuralbp_weights_nlayers_(?P<layers>\d+)_epochs_(?P<epochs>\d+)_"
    r"trained_using_train_(?P<dataset>p_[0-9.]+_sig_[0-9.]+_s_\d+)(?P<tag>.*?)_seed_(?P<seed>\d+)\.json$")


@dataclass
class ModelRecord:
    dataset: str
    arm_tag: str
    optimizer_tag: str
    check_node_tag: str
    seed: int
    alpha: Optional[float]
    step_layer: Optional[float]
    step_width: Optional[float]
    schedule_kind: str
    weight_sd: float
    initial_conditions_scale: Optional[float]
    alpha_init: Optional[float]
    step_layer_init: Optional[float]
    step_width_init: Optional[float]
    completed_epochs: Optional[int]


def split_tag(arm_tag: str) -> Tuple[str, str]:
    """
    Separate the optimizer suffix from the rest of the run tag.
    `_hpcer_cnenr0p42L_sch12w3L_optreps1em8` -> ("eps1em8", "cer_cnenr0p42L_sch12w3L").
    An untagged run reports its optimizer arm as "base".
    """
    optimizer_tag: str = "base"
    remainder: str = arm_tag
    marker: str = "_opt"
    if marker in arm_tag:
        head, _, tail = arm_tag.rpartition(marker)
        optimizer_tag = tail
        remainder = head
    remainder = remainder.replace("_no_cer", "").lstrip("_")
    if remainder.startswith("hp"):
        remainder = remainder[2:]
    if remainder.startswith("xf"):
        remainder = remainder[2:]
    return optimizer_tag, remainder


def read_toml_number(toml_path: Path, key: str) -> Optional[float]:
    if not toml_path.is_file():
        return None
    pattern: re.Pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*([-+0-9.eE]+)")
    for line in toml_path.read_text().splitlines():
        match: Optional[re.Match] = pattern.match(line)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def completed_epochs(run_directory: Path, dataset: str, arm_tag: str, seed: int) -> Optional[int]:
    """
    An epoch that logged fewer rows than the widest one broke early at
    train.jl:418 and was rolled back at :495, reverting every parameter.
    """
    log_path: Path = run_directory / "logs" / f"debugging_train_{dataset}{arm_tag}_seed_{seed}.csv"
    if not log_path.is_file():
        return None
    rows_per_epoch: collections.Counter = collections.Counter()
    with log_path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                epoch: int = int(row["epoch"])
            except (KeyError, ValueError):
                continue
            if epoch > 0:
                rows_per_epoch[epoch] += 1
    if not rows_per_epoch:
        return None
    full_width: int = max(rows_per_epoch.values())
    return sum(1 for count in rows_per_epoch.values() if count == full_width)


def collect(run_directory: Path, tag_filter: str) -> List[ModelRecord]:
    records: List[ModelRecord] = []
    models_directory: Path = run_directory / "models"
    for weights_path in sorted(models_directory.glob("neuralbp_weights_*.json")):
        match: Optional[re.Match] = WEIGHTS_PATTERN.match(weights_path.name)
        if match is None:
            continue
        fields: Dict[str, str] = match.groupdict()
        arm_tag: str = fields["tag"]
        if tag_filter and tag_filter not in arm_tag:
            continue
        try:
            weights: dict = json.loads(weights_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        message_weights: List[float] = (
            weights.get("weights_c2v_v2c", []) + weights.get("weights_llrs", [])
            + weights.get("weights_c2v_readout", []))
        if not message_weights:
            continue
        optimizer_tag, check_node_tag = split_tag(arm_tag)
        seed: int = int(fields["seed"])
        # The generated TOML carries this point's initial conditions, which is
        # what "did it move" has to be measured against. The weights filename is
        #     ..._trained_using_train_<dataset><cer_tag><run_tag>_seed_<n>.json
        # and the TOML is
        #     hyperparams<run_tag>_<tag_of(dataset)>_seed<n>.toml
        # so the name is the captured tag with `_no_cer` (the cer_tag, which the
        # TOML name does not carry) removed. `run_tag` already begins `_hp`.
        # ... except that the generator builds the two names from the same parts
        # with different separators: run_tag is "_hp" + arm (no underscore
        # between), while the TOML is "hyperparams_hp_" + arm. So the prefix
        # gains one underscore.
        run_tag: str = arm_tag.replace("_no_cer", "")
        toml_tag: str = run_tag
        for prefix in ("_hp", "_xf"):
            if toml_tag.startswith(prefix):
                toml_tag = prefix + "_" + toml_tag[len(prefix):]
                break
        dataset_tag: str = fields["dataset"].replace(".", "p")
        toml_path: Path = models_directory / f"hyperparams{toml_tag}_{dataset_tag}_seed{seed}.toml"
        records.append(ModelRecord(
            dataset=fields["dataset"],
            arm_tag=arm_tag,
            optimizer_tag=optimizer_tag,
            check_node_tag=check_node_tag or "tanh",
            seed=seed,
            alpha=(weights.get("coupling_scale") or [None])[0],
            step_layer=weights.get("coupling_schedule_layer"),
            step_width=weights.get("coupling_schedule_width"),
            schedule_kind=weights.get("coupling_schedule_kind", "constant"),
            weight_sd=statistics.pstdev(message_weights),
            initial_conditions_scale=read_toml_number(toml_path, "initial_conditions_scale"),
            alpha_init=read_toml_number(toml_path, "coupling_scale_init"),
            step_layer_init=read_toml_number(toml_path, "coupling_schedule_layer_init"),
            step_width_init=read_toml_number(toml_path, "coupling_schedule_width_init"),
            completed_epochs=completed_epochs(run_directory, fields["dataset"], arm_tag, seed),
        ))
    return records


def summarise(values: List[Optional[float]]) -> Optional[Tuple[float, float]]:
    present: List[float] = [value for value in values if value is not None]
    if not present:
        return None
    spread: float = 0.0
    if len(present) > 1:
        spread = statistics.pstdev(present)
    return (statistics.mean(present), spread)


def format_movement(final: Optional[Tuple[float, float]], init: Optional[Tuple[float, float]]) -> str:
    if final is None:
        return f"{'—':>20}"
    if init is None:
        return f"{final[0]:>9.4f}±{final[1]:<5.3f}{'':>5}"
    movement: float = abs(final[0] - init[0])
    return f"{final[0]:>9.4f}±{final[1]:<5.3f} {movement:>5.3f}"


def report(records: List[ModelRecord]) -> None:
    grouped: Dict[Tuple[str, str], List[ModelRecord]] = collections.defaultdict(list)
    for record in records:
        grouped[(record.optimizer_tag, record.check_node_tag)].append(record)

    print()
    print(f"  {len(records)} trained model(s)")
    print()
    header: str = ("  {:<10} {:<26} {:>3} {:>5} {:>20} {:>20} {:>20} {:>16}".format(
        "optimizer", "arm", "n", "ep", "alpha  (±sd) move", "T0  (±sd) move", "w  (±sd) move", "weight sd / init"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (optimizer_tag, check_node_tag) in sorted(grouped):
        group: List[ModelRecord] = grouped[(optimizer_tag, check_node_tag)]
        epochs: List[Optional[float]] = [record.completed_epochs for record in group]
        epochs_text: str = "—"
        epoch_summary = summarise(epochs)
        if epoch_summary is not None:
            epochs_text = f"{epoch_summary[0]:.1f}"
        alpha_text: str = format_movement(
            summarise([record.alpha for record in group]),
            summarise([record.alpha_init for record in group]))
        has_schedule: bool = any(record.schedule_kind == "step" for record in group)
        layer_text: str = f"{'—':>20}"
        width_text: str = f"{'—':>20}"
        if has_schedule:
            layer_text = format_movement(
                summarise([record.step_layer for record in group]),
                summarise([record.step_layer_init for record in group]))
            width_text = format_movement(
                summarise([record.step_width for record in group]),
                summarise([record.step_width_init for record in group]))
        weight_summary = summarise([record.weight_sd for record in group])
        scale_summary = summarise([record.initial_conditions_scale for record in group])
        weight_text: str = "—"
        if weight_summary is not None:
            if scale_summary is None:
                weight_text = f"{weight_summary[0]:.4f} / ?"
            else:
                untrained_sd: float = scale_summary[0] * UNIFORM_SD_FACTOR
                weight_text = f"{weight_summary[0]:.4f} / {untrained_sd:.4f}"
        print("  {:<10} {:<26} {:>3} {:>5} {} {} {} {:>16}".format(
            optimizer_tag, check_node_tag, len(group), epochs_text,
            alpha_text, layer_text, width_text, weight_text))
    print()
    print("  alpha/T0/w columns are  <final mean>±<sd over seeds>  <|final - init|>.")
    print("  A move of ~1e-3 is no movement: at lr = 1e-3 over 2500 updates, full")
    print("  Adam steps would travel ~2.5. weight sd is against this arm's own")
    print("  initial_conditions_scale * 1/sqrt(3), the untrained value.")
    print()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="check_parameter_movement.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, metavar="RUN_DIR",
                        help="run directory: a path, or a codename under data/")
    parser.add_argument("--tag", default="", metavar="SUBSTRING",
                        help="only arms whose run tag contains this (e.g. 'opt')")
    arguments: argparse.Namespace = parser.parse_args(argv)

    candidate: Path = Path(arguments.workdir)
    if not candidate.is_dir():
        candidate = DATA_ROOT / arguments.workdir
    if not candidate.is_dir():
        print(f"No such run directory: '{arguments.workdir}'", file=sys.stderr)
        return 1
    run_directory: Path = candidate.resolve()

    records: List[ModelRecord] = collect(run_directory, arguments.tag)
    if not records:
        print(f"No trained models matched under {run_directory}/models.", file=sys.stderr)
        return 1
    report(records)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

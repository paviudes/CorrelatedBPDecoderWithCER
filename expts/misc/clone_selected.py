#!/usr/bin/env python3
"""
clone_selected.py -- copy a selection of the 72q_BB_cycles_1_sweep data sets into a new
run directory, data/72q_BB_cycles_1_sweep_filtered.

From expts/:
    python3 misc/clone_selected.py --p-values 0.0007,0.0011,0.0015 --sig-values 0 --sample-values 1
    python3 misc/clone_selected.py ... --dry-run     show what would be copied, copy nothing

A data set p_<p>_sig_<sigma>_s_<sample> is selected when its p is one of --p-values, its
sigma one of --sig-values and its sample one of --sample-values (every combination of the
three lists). The new directory gets:

    every top-level folder of the source     the same skeleton; empty unless listed below
    code/                                    everything in the source's code/
    training_data/                           train_<key>.txt                of each selected key
    testing_data/                            test_<key>.txt                 of each selected key
    correlated_weights/                      correlated_weights_<key>.txt   of each selected key
    models/                                  hyperparams_baseline.toml and
                                             hyperparams_cer_enriched.toml, nothing else
    info.txt                                 what was selected, the file counts, and when

Values are compared as numbers, so 0, 0.0 and 0e0 all select sig_0.0 and 7e-4 selects
p_0.0007. A value that selects no file is refused, with the values the source has (it is
almost always a typo: 0.007 for 0.0007). A combination the source does not have
(p_0.0007 with sig 0.0015, say) is skipped and listed in info.txt.

The destination must not exist. The copy is built in a hidden folder next to it and
renamed at the end, so an interrupted run leaves nothing half-made under the real name.
"""

import argparse
import math
import os
import re
import shutil
import signal
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = REPO_ROOT / "data"
SOURCE_CODENAME: str = "72q_BB_cycles_1_sweep"
DESTINATION_CODENAME: str = "72q_BB_cycles_1_sweep_filtered"

# The three folders whose files are selected by key: (folder, file prefix, column label).
DATASET_FOLDERS: Tuple[Tuple[str, str, str], ...] = (
    ("training_data", "train_", "train"),
    ("testing_data", "test_", "test"),
    ("correlated_weights", "correlated_weights_", "CER"),
)
KEY_PATTERN: str = r"p_(?P<p>[^_]+)_sig_(?P<sigma>[^_]+)_s_(?P<sample>[0-9]+)"

MODEL_FILES: Tuple[str, ...] = ("hyperparams_baseline.toml", "hyperparams_cer_enriched.toml")
# Used only when the source's models/ lacks the file: the copy tracked in the repository.
MODEL_FILE_FALLBACKS: Dict[str, Path] = {
    "hyperparams_cer_enriched.toml": REPO_ROOT / "expts" / "sweeps" / "hyperparams_cer_enriched.toml",
}


class DatasetFile:
    """One file of a data set folder, with the numbers its name carries."""

    def __init__(self, path: Path, key: str, p_value: float, sigma_value: float, sample_index: int) -> None:
        self.path: Path = path
        self.key: str = key
        self.p_value: float = p_value
        self.sigma_value: float = sigma_value
        self.sample_index: int = sample_index

    def numbers(self) -> Tuple[float, float, int]:
        key_numbers: Tuple[float, float, int] = (self.p_value, self.sigma_value, self.sample_index)
        return key_numbers


def same_number(first_value: float, second_value: float) -> bool:
    """Equal as data set numbers: 0.0007 typed as 7e-4 or 0.00070 is the same p."""
    is_same: bool = math.isclose(first_value, second_value, rel_tol=1e-9, abs_tol=0.0)
    return is_same


def key_number_text(value: float) -> str:
    """A number as the data generator writes it in file names (Python's repr): 0 -> 0.0."""
    number_text: str = repr(float(value))
    return number_text


def joined_numbers(values: List[float]) -> str:
    joined_text: str = ", ".join(key_number_text(value) for value in values)
    return joined_text


def joined_integers(values: List[int]) -> str:
    joined_text: str = ", ".join(str(value) for value in values)
    return joined_text


def human_size(n_bytes: int) -> str:
    size_in_unit: float = float(n_bytes)
    unit: str = "B"
    for next_unit in ("KB", "MB", "GB", "TB"):
        if size_in_unit < 1024:
            break
        size_in_unit /= 1024
        unit = next_unit
    size_text: str = "{:.1f} {}".format(size_in_unit, unit)
    return size_text


def parse_number_list(option_text: str, option_name: str) -> List[float]:
    numbers: List[float] = []
    for item in option_text.split(","):
        stripped_item: str = item.strip()
        if stripped_item == "":
            continue
        try:
            number: float = float(stripped_item)
        except ValueError:
            raise SystemExit("{}: '{}' is not a number.".format(option_name, stripped_item))
        if not math.isfinite(number) or number < 0:
            raise SystemExit("{}: '{}' must be a non-negative number.".format(option_name, stripped_item))
        already_listed: bool = any(same_number(number, listed_number) for listed_number in numbers)
        if not already_listed:
            numbers.append(number)
    if len(numbers) == 0:
        raise SystemExit("{} is empty.".format(option_name))
    return numbers


def parse_integer_list(option_text: str, option_name: str) -> List[int]:
    integers: List[int] = []
    for item in option_text.split(","):
        stripped_item: str = item.strip()
        if stripped_item == "":
            continue
        if not stripped_item.isdigit() or int(stripped_item) < 1:
            raise SystemExit("{}: '{}' is not a positive integer.".format(option_name, stripped_item))
        integer: int = int(stripped_item)
        if integer not in integers:
            integers.append(integer)
    if len(integers) == 0:
        raise SystemExit("{} is empty.".format(option_name))
    return integers


def list_dataset_files(folder: Path, file_prefix: str) -> Tuple[List[DatasetFile], List[str]]:
    """Every <prefix><key>.txt in the folder, and the names of matching links that lead nowhere."""
    name_pattern = re.compile("^" + re.escape(file_prefix) + KEY_PATTERN + r"\.txt$")
    dataset_files: List[DatasetFile] = []
    broken_links: List[str] = []
    for path in sorted(folder.iterdir()):
        name_match = name_pattern.match(path.name)
        if name_match is None:
            continue
        if not path.is_file():
            broken_links.append(path.name)
            continue
        try:
            p_value: float = float(name_match.group("p"))
            sigma_value: float = float(name_match.group("sigma"))
        except ValueError:
            continue
        key: str = path.name[len(file_prefix):-len(".txt")]
        dataset_files.append(DatasetFile(path, key, p_value, sigma_value, int(name_match.group("sample"))))
    return dataset_files, broken_links


def matches_numbers(numbers: Tuple[float, float, int], p_value: float, sigma_value: float, sample_index: int) -> bool:
    matches: bool = (same_number(numbers[0], p_value) and same_number(numbers[1], sigma_value)
                     and numbers[2] == sample_index)
    return matches


def is_selected(dataset_file: DatasetFile, p_values: List[float], sigma_values: List[float],
                sample_indices: List[int]) -> bool:
    selected: bool = (any(same_number(dataset_file.p_value, p_value) for p_value in p_values)
                      and any(same_number(dataset_file.sigma_value, sigma_value) for sigma_value in sigma_values)
                      and dataset_file.sample_index in sample_indices)
    return selected


def refuse_values_that_select_nothing(all_files: List[DatasetFile], selected_files: List[DatasetFile],
                                      p_values: List[float], sigma_values: List[float],
                                      sample_indices: List[int]) -> None:
    """A requested value that selects no file at all is almost always a typo."""
    problems: List[str] = []
    axes = (("--p-values", p_values, 0), ("--sig-values", sigma_values, 1), ("--sample-values", sample_indices, 2))
    for option_name, requested_values, position in axes:
        source_values: List = sorted(set(dataset_file.numbers()[position] for dataset_file in all_files))
        for requested_value in requested_values:
            selects_something: bool = any(same_number(dataset_file.numbers()[position], requested_value)
                                          for dataset_file in selected_files)
            if selects_something:
                continue
            in_source: bool = any(same_number(source_value, requested_value) for source_value in source_values)
            if position == 2:
                value_text: str = str(requested_value)
                available_text: str = joined_integers(source_values)
            else:
                value_text = key_number_text(requested_value)
                available_text = joined_numbers(source_values)
            if in_source:
                problems.append("  {} {}: in the source, but not with the other values given".format(
                    option_name, value_text))
            else:
                problems.append("  {} {}: no file of the source has it; the source has {}".format(
                    option_name, value_text, available_text))
    if len(problems) > 0:
        raise SystemExit("These values select no file:\n" + "\n".join(problems) + "\nNothing was created.")


def build_info_text(source: Path, p_values: List[float], sigma_values: List[float], sample_indices: List[int],
                    files_by_folder: Dict[str, List[DatasetFile]], model_sources: Dict[str, Path],
                    code_files: List[Path], skeleton: List[str], warnings: List[str]) -> str:
    created_at: str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    numbers_by_key: Dict[str, Tuple[float, float, int]] = {}
    keys_by_folder: Dict[str, Set[str]] = {}
    for folder, _, _ in DATASET_FOLDERS:
        keys_by_folder[folder] = set()
        for dataset_file in files_by_folder[folder]:
            numbers_by_key[dataset_file.key] = dataset_file.numbers()
            keys_by_folder[folder].add(dataset_file.key)
    sorted_keys: List[str] = sorted(numbers_by_key, key=lambda key: numbers_by_key[key])
    requested_combinations: List[Tuple[float, float, int]] = [
        (p_value, sigma_value, sample_index)
        for p_value in p_values for sigma_value in sigma_values for sample_index in sample_indices]
    absent_combinations: List[str] = [
        "p_{}_sig_{}_s_{}".format(key_number_text(combination[0]), key_number_text(combination[1]), combination[2])
        for combination in requested_combinations
        if not any(matches_numbers(numbers, *combination) for numbers in numbers_by_key.values())]

    model_descriptions: List[str] = []
    for model_file in MODEL_FILES:
        if model_sources[model_file] == source / "models" / model_file:
            model_descriptions.append(model_file)
        else:
            model_descriptions.append("{} (from {})".format(model_file, model_sources[model_file]))

    lines: List[str] = [
        DESTINATION_CODENAME,
        "created : {}".format(created_at),
        "by      : python3 misc/clone_selected.py " + " ".join(sys.argv[1:]),
        "from    : {}".format(source),
        "",
        "p-values selected     : {}".format(joined_numbers(p_values)),
        "sigma values selected : {}".format(joined_numbers(sigma_values)),
        "samples selected      : {}".format(joined_integers(sample_indices)),
        "",
        "training files : {}".format(len(files_by_folder["training_data"])),
        "testing files  : {}".format(len(files_by_folder["testing_data"])),
        "CER files      : {}   (correlated_weights/)".format(len(files_by_folder["correlated_weights"])),
        "models         : {}".format(", ".join(model_descriptions)),
        "code           : {} file(s), copied whole".format(len(code_files)),
        "folders        : {}".format(", ".join(skeleton)),
        "",
        "data sets: {} of the {} combinations asked for are in the source".format(
            len(requested_combinations) - len(absent_combinations), len(requested_combinations)),
    ]
    key_width: int = max([len(key) for key in sorted_keys] + [len("key")])
    lines.append("  {:<{width}}  train  test  CER".format("key", width=key_width))
    for key in sorted_keys:
        marks: List[str] = ["yes" if key in keys_by_folder[folder] else "NO" for folder, _, _ in DATASET_FOLDERS]
        lines.append("  {:<{width}}  {:<5}  {:<4}  {}".format(key, marks[0], marks[1], marks[2], width=key_width))
    lines.append("")
    if len(absent_combinations) == 0:
        lines.append("not in the source: none")
    else:
        lines.append("not in the source (no file in any of the three folders; skipped):")
        lines.extend("  " + combination for combination in absent_combinations)
    if len(warnings) > 0:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend("  " + warning for warning in warnings)
    info_text: str = "\n".join(lines) + "\n"
    return info_text


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected data sets of data/{} into data/{}.".format(SOURCE_CODENAME, DESTINATION_CODENAME),
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-values", required=True, help="comma-separated, e.g. 0.0007,0.0011,0.0015")
    parser.add_argument("--sig-values", required=True, help="comma-separated, e.g. 0 or 0.0,0.0015")
    parser.add_argument("--sample-values", required=True, help="comma-separated integers, e.g. 1 or 1,2,3")
    parser.add_argument("--dry-run", action="store_true", help="show what would be copied, create nothing")
    arguments: argparse.Namespace = parser.parse_args(argv)
    return arguments


def main(argv: List[str]) -> int:
    arguments: argparse.Namespace = parse_arguments(argv)
    p_values: List[float] = parse_number_list(arguments.p_values, "--p-values")
    sigma_values: List[float] = parse_number_list(arguments.sig_values, "--sig-values")
    sample_indices: List[int] = parse_integer_list(arguments.sample_values, "--sample-values")

    source: Path = DATA_ROOT / SOURCE_CODENAME
    destination: Path = DATA_ROOT / DESTINATION_CODENAME
    if not source.is_dir():
        raise SystemExit("no source run directory {}".format(source))
    for required_folder in ["code", "models"] + [folder for folder, _, _ in DATASET_FOLDERS]:
        if not (source / required_folder).is_dir():
            raise SystemExit("the source has no {}/ ({})".format(required_folder, source / required_folder))
    if os.path.lexists(str(destination)):
        raise SystemExit("{} already exists. Move or remove it first (rm -rf {}), then run this again."
                         .format(destination, destination))

    # The selection, folder by folder.
    warnings: List[str] = []
    all_files: List[DatasetFile] = []
    files_by_folder: Dict[str, List[DatasetFile]] = {}
    for folder, file_prefix, _ in DATASET_FOLDERS:
        folder_files, broken_links = list_dataset_files(source / folder, file_prefix)
        all_files.extend(folder_files)
        files_by_folder[folder] = [dataset_file for dataset_file in folder_files
                                   if is_selected(dataset_file, p_values, sigma_values, sample_indices)]
        for broken_link in broken_links:
            warnings.append("{}/{} is a link to a missing file; not copied".format(folder, broken_link))
    selected_files: List[DatasetFile] = [dataset_file for folder, _, _ in DATASET_FOLDERS
                                         for dataset_file in files_by_folder[folder]]
    refuse_values_that_select_nothing(all_files, selected_files, p_values, sigma_values, sample_indices)
    # A selected key needs all three files to be trained and tested; say which lack one.
    keys_by_folder: Dict[str, Set[str]] = {folder: set(dataset_file.key for dataset_file in files_by_folder[folder])
                                           for folder, _, _ in DATASET_FOLDERS}
    for key in sorted(set().union(*keys_by_folder.values())):
        missing_folders: List[str] = [folder for folder, _, _ in DATASET_FOLDERS if key not in keys_by_folder[folder]]
        if len(missing_folders) > 0:
            warnings.append("{} has no file in {} of the source".format(key, ", ".join(missing_folders)))

    model_sources: Dict[str, Path] = {}
    for model_file in MODEL_FILES:
        in_source: Path = source / "models" / model_file
        if in_source.is_file():
            model_sources[model_file] = in_source
        elif model_file in MODEL_FILE_FALLBACKS and MODEL_FILE_FALLBACKS[model_file].is_file():
            model_sources[model_file] = MODEL_FILE_FALLBACKS[model_file]
            warnings.append("models/{} is not in the source; copied {} instead".format(
                model_file, MODEL_FILE_FALLBACKS[model_file]))
        else:
            raise SystemExit("no {} in {}".format(model_file, source / "models"))

    # The skeleton: every top-level folder of the source (hidden ones aside).
    skeleton: List[str] = sorted(entry.name for entry in source.iterdir()
                                 if entry.is_dir() and not entry.name.startswith("."))
    code_files: List[Path] = sorted(path for path in (source / "code").rglob("*") if path.is_file())
    copy_bytes: int = (sum(dataset_file.path.stat().st_size for dataset_file in selected_files)
                       + sum(path.stat().st_size for path in code_files)
                       + sum(path.stat().st_size for path in model_sources.values()))

    print("[clone_selected] {} -> {}".format(source, destination))
    print("  p-values {}  |  sigma values {}  |  samples {}".format(
        joined_numbers(p_values), joined_numbers(sigma_values), joined_integers(sample_indices)))
    print("  {} training, {} testing, {} CER file(s), {} code file(s), {} model file(s): {} in all".format(
        len(files_by_folder["training_data"]), len(files_by_folder["testing_data"]),
        len(files_by_folder["correlated_weights"]), len(code_files), len(model_sources), human_size(copy_bytes)))
    for warning in warnings:
        print("  WARNING: " + warning)
    if arguments.dry_run:
        for dataset_file in selected_files:
            print("    {}/{}".format(dataset_file.path.parent.name, dataset_file.path.name))
        print("[clone_selected] --dry-run: nothing was created.")
        return 0

    # Built in a hidden folder next to the destination, then renamed, so the destination
    # exists only once it is complete. SIGTERM is turned into an exit so the clean-up runs.
    signal.signal(signal.SIGTERM, lambda signal_number, frame: sys.exit("[clone_selected] terminated; nothing was created."))
    staging: Path = Path(tempfile.mkdtemp(prefix="." + DESTINATION_CODENAME + ".partial-", dir=str(DATA_ROOT)))
    try:
        for folder_name in skeleton:
            if folder_name != "code":
                (staging / folder_name).mkdir()
        shutil.copytree(str(source / "code"), str(staging / "code"))
        for model_file, model_source in model_sources.items():
            shutil.copy2(str(model_source), str(staging / "models" / model_file))
        for copy_number, dataset_file in enumerate(selected_files, start=1):
            folder_name = dataset_file.path.parent.name
            print("  [{}/{}] {}/{}  ({})".format(copy_number, len(selected_files), folder_name,
                                                dataset_file.path.name, human_size(dataset_file.path.stat().st_size)),
                  flush=True)
            shutil.copy2(str(dataset_file.path), str(staging / folder_name / dataset_file.path.name))
        info_text: str = build_info_text(source, p_values, sigma_values, sample_indices, files_by_folder,
                                         model_sources, code_files, skeleton, warnings)
        (staging / "info.txt").write_text(info_text)
        # mkdtemp makes the folder private (0700); give it the mode a plain mkdir would.
        current_umask: int = os.umask(0)
        os.umask(current_umask)
        os.chmod(str(staging), 0o777 & ~current_umask)
        if os.path.lexists(str(destination)):
            raise SystemExit("{} appeared while copying; nothing was renamed.".format(destination))
        os.rename(str(staging), str(destination))
    finally:
        if staging.exists():
            shutil.rmtree(str(staging), ignore_errors=True)

    print()
    print(info_text, end="")
    print("[clone_selected] written: {}  (this summary is in {})".format(destination, destination / "info.txt"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit("[clone_selected] interrupted; nothing was created.")

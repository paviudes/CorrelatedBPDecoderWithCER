#!/usr/bin/env python3
"""
cleanup.py — reset a run directory back to its inputs.

    python3 misc/cleanup.py --workdir <run directory>              show, ask, clean
    python3 misc/cleanup.py --workdir <run directory> --dry-run    show, delete nothing
    python3 misc/cleanup.py --workdir <run directory> --yes        skip the prompt

<run directory> is a path, or a bare codename resolved under the repo's data/,
e.g.  --workdir 72q_BB_cycles_1_soft_constraints

What is removed is defined by ONE table, here and in the RULES tuple below
(the two must say the same thing; everything the script prints comes from RULES):

    DIR         REMOVED            EXCEPT
    ---------   ----------------   -------------------------------------------------
    logs/       *                  —
    cluster/    *                  —
    results/    *                  —
    models/     *.json  *.toml     *.toml not written by a sweep, i.e. not matching
                                   hyperparams_hp_*.toml or hyperparams_xf_*.toml —
                                   the base config every sweep starts from

Not in the table, never touched:  code/  correlated_weights/  training_data/
testing_data/  — the inputs, measured in gigabytes.

The models/ exception is defined negatively on purpose. The base config has gone
by two names already (hyperparams_epochs_5_corrs.toml, hyperparams_baseline.toml),
so a positive "keep this name" rule would have deleted one of them; what IS stable
is the prefix a sweep puts on the files it generates. As a second line of defence
the script refuses to run at all if the rules would leave models/ with no .toml
when it has one now.

Progress bar: tqdm when it is installed, a plain one when it is not, none when
output is not a terminal.
"""

import argparse
import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


# ============================================================================ #
#  The table. One record per row of the block in the module docstring.         #
# ============================================================================ #
@dataclass(frozen=True)
class CleanupRule:
    """
    One row: inside `directory` (relative to the run dir, walked recursively),
    every file matching a `removed` glob is a candidate; a candidate is spared
    if it matches a `kept` glob and does NOT also match a `kept_unless` glob.
    `kept_note` is the text of the EXCEPT column.
    """
    directory: str
    removed: Tuple[str, ...]
    kept: Tuple[str, ...]
    kept_unless: Tuple[str, ...]
    kept_note: str


SWEEP_TOML_GLOBS: Tuple[str, ...] = ("hyperparams_hp_*.toml", "hyperparams_xf_*.toml")

RULES: Tuple[CleanupRule, ...] = (
    CleanupRule("logs",    ("*",),               (),          (),               "—"),
    CleanupRule("cluster", ("*",),               (),          (),               "—"),
    CleanupRule("results", ("*",),               (),          (),               "—"),
    CleanupRule("models",  ("*.json", "*.toml"), ("*.toml",), SWEEP_TOML_GLOBS,
                "*.toml not written by a sweep (not " + ", ".join(SWEEP_TOML_GLOBS) + "): the base config"),
)

# Directories a run needs and this script must never enter.
KEPT_DIRECTORIES: Tuple[str, ...] = ("code", "correlated_weights", "training_data", "testing_data")

# A run directory has at least one of these; anything without any of them is
# not a run directory, and a typo must fail loudly rather than empty something.
EXPECTED_SUBDIRECTORIES: Tuple[str, ...] = tuple(rule.directory for rule in RULES) + KEPT_DIRECTORIES

# This file lives at expts/misc/cleanup.py; the repository root is two up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = REPO_ROOT / "data"


# ============================================================================ #
#  Matching                                                                    #
# ============================================================================ #
def matches_any(name: str, globs: Iterable[str]) -> bool:
    for glob in globs:
        if fnmatch.fnmatch(name, glob):
            return True
    return False


def is_spared(name: str, rule: CleanupRule) -> bool:
    """
    The EXCEPT column: matched by `kept`, and not by `kept_unless`.
    """
    if not matches_any(name, rule.kept):
        return False
    if matches_any(name, rule.kept_unless):
        return False
    return True


@dataclass
class RuleInventory:
    rule: CleanupRule
    to_remove: List[Path]
    spared: List[Path]
    bytes_to_remove: int


def inventory_rule(run_directory: Path, rule: CleanupRule) -> RuleInventory:
    """
    Walk `rule.directory` once and sort every regular file into removed or
    spared. Files that match no `removed` glob are neither, and are left alone.
    """
    target_directory: Path = run_directory / rule.directory
    to_remove: List[Path] = []
    spared: List[Path] = []
    bytes_to_remove: int = 0
    if not target_directory.is_dir():
        return RuleInventory(rule, to_remove, spared, bytes_to_remove)
    for path in sorted(target_directory.rglob("*")):
        if not path.is_file():
            continue
        if not matches_any(path.name, rule.removed):
            continue
        if is_spared(path.name, rule):
            spared.append(path)
            continue
        to_remove.append(path)
        try:
            bytes_to_remove += path.stat().st_size
        except OSError:
            pass
    return RuleInventory(rule, to_remove, spared, bytes_to_remove)


# ============================================================================ #
#  Presentation                                                                #
# ============================================================================ #
def human_size(n_bytes: int) -> str:
    size: float = float(n_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024.0 or unit == "T":
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}T"


def rules_table_text() -> str:
    lines: List[str] = []
    lines.append(f"  {'DIR':<10} {'REMOVED':<18} EXCEPT")
    lines.append(f"  {'-' * 10} {'-' * 18} {'-' * 60}")
    for rule in RULES:
        lines.append(f"  {rule.directory + '/':<10} {' '.join(rule.removed):<18} {rule.kept_note}")
    lines.append("")
    lines.append(f"  never touched: {'  '.join(directory + '/' for directory in KEPT_DIRECTORIES)}")
    return "\n".join(lines)


def print_inventory(run_directory: Path, inventories: List[RuleInventory]) -> int:
    print()
    print("  Cleaning run directory")
    print(f"    {run_directory}")
    print()
    print(f"  {'DIR':<10} {'REMOVED':<18} {'files':>7} {'size':>9}   EXCEPT (spared)")
    print(f"  {'-' * 10} {'-' * 18} {'-' * 7} {'-' * 9}   {'-' * 40}")
    total_files: int = 0
    for inventory in inventories:
        rule: CleanupRule = inventory.rule
        spared_text: str = "—"
        if rule.kept:
            if inventory.spared:
                spared_text = f"{len(inventory.spared)} file(s): " + ", ".join(path.name for path in inventory.spared[:3])
                if len(inventory.spared) > 3:
                    spared_text += f", … (+{len(inventory.spared) - 3})"
            else:
                spared_text = "0 file(s)"
        print(f"  {rule.directory + '/':<10} {' '.join(rule.removed):<18} {len(inventory.to_remove):>7} "
              f"{human_size(inventory.bytes_to_remove):>9}   {spared_text}")
        total_files += len(inventory.to_remove)
    print(f"  {'-' * 10} {'-' * 18} {'-' * 7} {'-' * 9}")
    total_bytes: int = sum(inventory.bytes_to_remove for inventory in inventories)
    print(f"  {'TOTAL':<10} {'':<18} {total_files:>7} {human_size(total_bytes):>9}")
    print()
    print("  Never touched:")
    for directory in KEPT_DIRECTORIES:
        kept_directory: Path = run_directory / directory
        if kept_directory.is_dir():
            n_files: int = sum(1 for path in kept_directory.rglob("*") if path.is_file())
            print(f"    {directory + '/':<22} {n_files} file(s)")
    print()
    return total_files


# ============================================================================ #
#  Safety                                                                      #
# ============================================================================ #
def resolve_run_directory(target_argument: str) -> Path:
    """
    Accept a real path or a bare codename under data/.
    """
    as_path: Path = Path(target_argument)
    if as_path.is_dir():
        return as_path.resolve()
    as_codename: Path = DATA_ROOT / target_argument
    if as_codename.is_dir():
        return as_codename.resolve()
    print(f"No such run directory: '{target_argument}'", file=sys.stderr)
    print(f"  tried: {as_path}", file=sys.stderr)
    print(f"  tried: {as_codename}", file=sys.stderr)
    sys.exit(1)


def refuse_if_not_a_run_directory(run_directory: Path) -> None:
    """
    This script deletes recursively, so it refuses anything that does not look
    like a run directory. A codename typo must fail loudly, not empty $HOME.
    """
    data_root: Path = DATA_ROOT.resolve()
    forbidden: List[Path] = [Path("/").resolve(), Path.home().resolve(), REPO_ROOT, data_root]
    for forbidden_path in forbidden:
        if run_directory == forbidden_path:
            print(f"Refusing to clean '{run_directory}' — that is not a run directory.", file=sys.stderr)
            sys.exit(1)
    if data_root not in run_directory.parents:
        print(f"Refusing to clean '{run_directory}' — it is outside {data_root}.", file=sys.stderr)
        print("  Pass a directory under data/ if this really is a run directory.", file=sys.stderr)
        sys.exit(1)
    if not any((run_directory / expected).is_dir() for expected in EXPECTED_SUBDIRECTORIES):
        print(f"Refusing to clean '{run_directory}' — none of", file=sys.stderr)
        print(f"  {' '.join(directory + '/' for directory in EXPECTED_SUBDIRECTORIES)}", file=sys.stderr)
        print("  is present, so this does not look like a run directory.", file=sys.stderr)
        sys.exit(1)


def refuse_if_no_base_config_would_survive(run_directory: Path, inventories: List[RuleInventory]) -> None:
    """
    Second line of defence for the models/ exception: if models/ holds any
    .toml now and the rules would remove every one of them, the base config is
    not named the way the EXCEPT column expects. Stop, rather than leave a run
    directory no sweep can start from.
    """
    models_directory: Path = run_directory / "models"
    if not models_directory.is_dir():
        return
    tomls_now: List[Path] = [path for path in models_directory.rglob("*.toml") if path.is_file()]
    if not tomls_now:
        return
    tomls_removed: set = set()
    for inventory in inventories:
        if inventory.rule.directory == "models":
            tomls_removed = {path for path in inventory.to_remove if path.suffix == ".toml"}
    if len(tomls_removed) >= len(tomls_now):
        print("Refusing: the rules would remove EVERY .toml from models/, leaving no base config.", file=sys.stderr)
        print("  models/ currently holds:", file=sys.stderr)
        for path in sorted(tomls_now):
            print(f"    {path.name}", file=sys.stderr)
        print(f"  Only files matching {', '.join(SWEEP_TOML_GLOBS)} are treated as sweep products;", file=sys.stderr)
        print("  anything else is the base config and is kept. If one of the files above IS a", file=sys.stderr)
        print("  sweep product under a new prefix, add that prefix to SWEEP_TOML_GLOBS.", file=sys.stderr)
        sys.exit(1)


# ============================================================================ #
#  Deletion                                                                    #
# ============================================================================ #
def progress_iterator(paths: List[Path]) -> Iterable[Path]:
    """
    tqdm when importable and stdout is a terminal; a plain carriage-return bar
    when tqdm is absent; the bare list when output is not a terminal.
    """
    if not sys.stdout.isatty():
        return paths
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(paths, desc="  removing", unit="file", ncols=80, leave=True)
    except ImportError:
        pass

    def plain_bar() -> Iterator[Path]:
        total: int = len(paths)
        width: int = 40
        for index, path in enumerate(paths, start=1):
            yield path
            filled: int = index * width // total if total else width
            percent: int = index * 100 // total if total else 100
            sys.stdout.write(f"\r  removing [{'#' * filled}{'.' * (width - filled)}] {percent:3d}%  {index}/{total}")
            sys.stdout.flush()
        sys.stdout.write("\n")

    return plain_bar()


def remove_files(paths: List[Path]) -> int:
    n_removed: int = 0
    for path in progress_iterator(paths):
        try:
            path.unlink()
            n_removed += 1
        except FileNotFoundError:
            pass
    return n_removed


def remove_empty_subdirectories(run_directory: Path) -> None:
    """
    The files lived in subdirectories (cluster/logs/hp_<ts>_train_task0/,
    results/summary/, ...) that are now empty shells. Remove them bottom-up,
    but keep the top-level directory of every rule so the next sweep does not
    have to recreate them.
    """
    for rule in RULES:
        top: Path = run_directory / rule.directory
        if not top.is_dir():
            continue
        subdirectories: List[Path] = [path for path in top.rglob("*") if path.is_dir()]
        for directory in sorted(subdirectories, key=lambda path: len(path.parts), reverse=True):
            try:
                directory.rmdir()          # only succeeds when empty
            except OSError:
                pass


# ============================================================================ #
#  Main                                                                        #
# ============================================================================ #
def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cleanup.py",
        description="Reset a run directory back to its inputs. What is removed:\n\n" + rules_table_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workdir", required=True, metavar="RUN_DIR",
        help="the run directory: a path, or a codename resolved under data/ "
             "(e.g. 72q_BB_cycles_1_soft_constraints)")
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="show what would be removed and delete nothing")
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="skip the confirmation prompt")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments: argparse.Namespace = parse_arguments(argv)

    run_directory: Path = resolve_run_directory(arguments.workdir)
    refuse_if_not_a_run_directory(run_directory)

    inventories: List[RuleInventory] = [inventory_rule(run_directory, rule) for rule in RULES]
    refuse_if_no_base_config_would_survive(run_directory, inventories)
    total_files: int = print_inventory(run_directory, inventories)

    if total_files == 0:
        print("  Nothing to remove — already clean.")
        return 0
    if arguments.dry_run:
        print("  --dry-run: nothing was deleted.")
        return 0
    if not arguments.yes:
        try:
            confirmation: str = input(f"  Delete these {total_files} file(s)? Type 'yes' to confirm: ")
        except EOFError:
            confirmation = ""
        if confirmation.strip() != "yes":
            print("  Aborted; nothing was deleted.")
            return 0
        print()

    all_paths: List[Path] = [path for inventory in inventories for path in inventory.to_remove]
    n_removed: int = remove_files(all_paths)
    remove_empty_subdirectories(run_directory)

    print()
    print(f"  Done: {n_removed} file(s) removed. Remaining in the run directory:")
    for rule in RULES:
        top: Path = run_directory / rule.directory
        if top.is_dir():
            n_left: int = sum(1 for path in top.rglob("*") if path.is_file())
            print(f"    {rule.directory + '/':<22} {n_left} file(s)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

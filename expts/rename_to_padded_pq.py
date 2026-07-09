#!/usr/bin/env python3
"""Rename `p_<X>_q_<Y>` tags in filenames to the padded `fmt_probs` form,
and (optionally) strip the obsolete `ballistic_` infix.

Background
----------
The main Julia package now formats every filename that carries a
`(per_qubit_prob, neighbour_prob)` pair with `fmt_probs`, which zero-pads
both numbers to the *maximum* decimal count of the pair.  Concretely:

    julia> fmt_probs(0.01, 0.001)
    "p_0.010_q_0.001"

Before that change, filenames came from plain Julia interpolation
(`"$(p)_q_$(q)"`), which produced the minimum-decimal form —

    "p_0.01_q_0.001"    ← old
    "p_0.010_q_0.001"   ← new (canonical)

Separately, the older code path called the Ballistic error model just
"ballistic" and stamped a `_ballistic_` infix into most artefact names:

    "train_ballistic_p_0.01_q_0.001_s_1.txt"      ← old
    "train_p_0.010_q_0.001_s_1.txt"               ← new (canonical)

The `--strip-ballistic` flag additionally rewrites this: any `_ballistic`
followed by `_` or by `.` is dropped from the basename.  This is opt-in
because dropping the infix is a semantic decision, not a formatting fix.

This script walks a directory tree and renames every file (and, if
--rename-dirs is passed, every directory) whose basename matches either
transform.  A basename can contain multiple matches
(e.g. `..._test_ballistic_p_0.01_q_0.001_s_1_..._train_ballistic_p_0.01_q_0.001_s_1.csv`);
all of them are rewritten with `re.sub`.

The script is dry-run by default — it prints the planned renames but
does not touch the filesystem.  Pass `--apply` to actually rename.

Safety
------
- Idempotent: files already in the canonical form are left alone.
- Refuses to overwrite an existing target — it warns and skips.
- Walks bottom-up when renaming directories, so parent renames don't
  invalidate paths of children that haven't been processed yet.

Usage
-----

    # dry-run: preview padding-only renames under a codename directory
    python3 rename_to_padded_pq.py \\
        /scratch/debankan/CorrelatedBPDecoderWithCER/data/72q_BB_...

    # actually rename files (not directories):
    python3 rename_to_padded_pq.py <root> --apply

    # include directory renames too:
    python3 rename_to_padded_pq.py <root> --apply --rename-dirs

    # ALSO drop `_ballistic` from names in the same pass:
    python3 rename_to_padded_pq.py <root> --apply --strip-ballistic
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Matches `p_<num>_q_<num>` where each num is an unsigned decimal literal
# (integer or "digits.digits").  Anchored on the literal `p_` prefix and
# on the `_q_` separator, so it won't misfire on things like
# "p_0.010_std_0.01_q_0.001" (where the p and q pieces aren't adjacent).
PQ_PATTERN = re.compile(r"p_(\d+(?:\.\d+)?)_q_(\d+(?:\.\d+)?)")

# Matches `_ballistic` immediately followed by another underscore (so it's
# embedded, e.g. `train_ballistic_p_...`) or by a period (so it's the tail
# of the stem, e.g. `decoder_statistics_ballistic.csv`).  Lookahead means
# the delimiter itself is preserved when we substitute with the empty
# string — the result is `train_p_...` / `decoder_statistics.csv`.
BALLISTIC_PATTERN = re.compile(r"_ballistic(?=[_.])")


def _decimals(num_str: str) -> int:
    """Return the count of digits after the decimal point, or 0 if none."""
    return len(num_str.split(".", 1)[1]) if "." in num_str else 0


def pad_pair(match: "re.Match[str]") -> str:
    """Substitute an individual `p_X_q_Y` match with its padded form."""
    p_str, q_str = match.group(1), match.group(2)
    n = max(_decimals(p_str), _decimals(q_str))
    p_val = float(p_str)
    q_val = float(q_str)
    return f"p_{p_val:.{n}f}_q_{q_val:.{n}f}"


def rewrite_name(name: str, strip_ballistic: bool) -> str:
    """Apply the transform chain to a basename: padding first, then (opt) strip.

    Padding always runs.  `--strip-ballistic` is applied second so the
    `_ballistic` removal sees the already-padded name — the two transforms
    are commutative, but this ordering makes the intent obvious when
    reading the diff for a single basename."""
    new_name = PQ_PATTERN.sub(pad_pair, name)
    if strip_ballistic:
        new_name = BALLISTIC_PATTERN.sub("", new_name)
    return new_name


def collect_renames(root: Path, include_dirs: bool, strip_ballistic: bool):
    """Walk `root` bottom-up and return the list of (old_path, new_path) pairs
    whose basenames actually change.  Bottom-up order matters when renaming
    directories, so a parent isn't renamed before its children are visited."""
    renames = []
    for cur_dir, subdirs, files in os.walk(root, topdown=False):
        cur = Path(cur_dir)
        for fname in files:
            new_name = rewrite_name(fname, strip_ballistic)
            if new_name != fname:
                renames.append((cur / fname, cur / new_name))
        if include_dirs:
            for dname in subdirs:
                new_name = rewrite_name(dname, strip_ballistic)
                if new_name != dname:
                    renames.append((cur / dname, cur / new_name))
    return renames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", type=Path, help="Directory to walk.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename.  Without this the script prints a "
                         "preview and exits.")
    ap.add_argument("--rename-dirs", action="store_true",
                    help="Also rename directories whose basenames match "
                         "the pattern (bottom-up).")
    ap.add_argument("--strip-ballistic", action="store_true",
                    help="Additionally drop `_ballistic` when followed by "
                         "`_` or `.`, e.g. train_ballistic_p_... → "
                         "train_p_... and decoder_statistics_ballistic.csv → "
                         "decoder_statistics.csv.")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    renames = collect_renames(root, include_dirs=args.rename_dirs,
                              strip_ballistic=args.strip_ballistic)
    if not renames:
        print(f"No filenames under {root} matched the requested "
              f"transform(s) — nothing to do.")
        return 0

    # Print a preview, relative to root for readability.
    for old, new in renames:
        print(f"  {old.relative_to(root)}")
        print(f"    -> {new.name}")
    print()
    print(f"{len(renames)} basename(s) would be rewritten.")

    if not args.apply:
        print("DRY-RUN.  Pass --apply to commit.  "
              "Add --rename-dirs to include directory basenames." if not args.rename_dirs
              else "DRY-RUN.  Pass --apply to commit.")
        return 0

    # Apply.
    n_ok, n_skip = 0, 0
    for old, new in renames:
        if new.exists():
            print(f"warning: target already exists, skipping: {new}",
                  file=sys.stderr)
            n_skip += 1
            continue
        old.rename(new)
        n_ok += 1
    print(f"Renamed {n_ok} basename(s); skipped {n_skip}.")
    return 0 if n_skip == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

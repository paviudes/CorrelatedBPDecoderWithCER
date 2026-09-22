#!/usr/bin/env bash
# check_training_health.sh — did each arm actually train, and equally?
#
#   bash misc/check_training_health.sh <codename> [tag-prefix]
#
#   bash misc/check_training_health.sh 72q_BB_cycles_1_soft_constraints xf
#
# Reports, per arm:
#
#   completed epochs   An epoch that logs fewer than n_gradient_updates_per_epoch
#                      rows broke early at train.jl:418 (the 6th non-finite
#                      gradient) and was then ROLLED BACK at train.jl:495 —
#                      weights AND Adam state restored to the epoch's start. Its
#                      work is discarded, so only COMPLETED epochs are training.
#                      This is the number that matters: on the lr = 0.01 run the
#                      enriched arms completed 1.0 of 5 epochs while the tanh
#                      arms completed ~3.3, which made every arm comparison a
#                      comparison of training budgets rather than of check nodes.
#
#   weight sd          Spread of all three weight vectors in the saved model.
#                      ~0.058 is the initialisation, i.e. never moved. The
#                      lr = 0.01 run sat at 0.20-0.40. A run that completes its
#                      epochs but collapses toward 0.06 is UNDERtrained, which
#                      is the opposite failure and wants more epochs, not a
#                      higher learning rate.
#
# Read them together. Equal completed epochs across arms is the precondition for
# any arm comparison meaning anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "usage: bash misc/check_training_health.sh <codename> [tag-prefix]" >&2
    echo "  <codename>    a directory under $REPO_ROOT/data" >&2
    echo "  [tag-prefix]  only arms whose run_tag CONTAINS this (default: all)" >&2
    exit 2
fi

CODENAME="$1"
TAG_PREFIX="${2:-}"
DATA_DIR="$REPO_ROOT/data/$CODENAME"

if [ ! -d "$DATA_DIR/logs" ]; then
    echo "no logs/ under $DATA_DIR" >&2
    exit 1
fi

CODENAME="$CODENAME" TAG_PREFIX="$TAG_PREFIX" DATA_DIR="$DATA_DIR" python3 - <<'PYTHON'
import csv, glob, json, os, re, statistics, collections, sys

data_dir = os.environ["DATA_DIR"]
tag_prefix = os.environ["TAG_PREFIX"]

# debugging_train_<key>_<tag>_seed_<n>.csv — the per-batch hyperparameter log.
# The _individual_losses sibling is skipped: it carries the same epoch column
# but one row per scored-layer vector, so counting it would double-count.
#
# The dataset key is matched EXPLICITLY rather than as a lazy .+?, because the
# tag itself contains underscores (`no_cer_xfnocer`) and a lazy key would eat
# part of it or vice versa depending on which side won.
log_pattern = re.compile(
    r"debugging_train_(?P<key>p_[0-9.]+_sig_[0-9.]+_s_\d+)_(?P<tag>.+)_seed_(?P<seed>\d+)\.csv$")

runs = []
for path in sorted(glob.glob(os.path.join(data_dir, "logs", "debugging_*.csv"))):
    name = os.path.basename(path)
    if name.endswith("_individual_losses.csv"):
        continue
    match = log_pattern.match(name)
    if match is None:
        continue
    fields = match.groupdict()
    # Substring, not prefix: the no-CER arm's tag is `no_cer_xfnocer`, so its
    # sweep marker sits in the middle. A prefix test would silently drop the
    # baseline and leave every comparison one arm short.
    if tag_prefix and tag_prefix not in fields["tag"]:
        continue
    rows_per_epoch = collections.Counter()
    for row in csv.DictReader(open(path)):
        try:
            rows_per_epoch[int(row["epoch"])] += 1
        except (KeyError, ValueError):
            continue
    # Slots are pre-allocated as zeros, so an epoch that broke early leaves its
    # remaining rows unwritten and they read back as epoch 0. The full width is
    # therefore the widest epoch seen, not a constant we have to be told.
    real = {e: n for e, n in rows_per_epoch.items() if e > 0}
    if not real:
        continue
    full_width = max(real.values())
    n_epochs = max(real)
    completed = sum(1 for e in range(1, n_epochs + 1) if real.get(e, 0) == full_width)
    runs.append(dict(key=fields["key"], tag=fields["tag"], seed=int(fields["seed"]),
                     completed=completed, n_epochs=n_epochs,
                     per_epoch=[real.get(e, 0) for e in range(1, n_epochs + 1)]))

if not runs:
    print("no training logs matched.", file=sys.stderr)
    sys.exit(1)

# Weight spread, keyed on the same tag the log filename carries.
weight_sd = collections.defaultdict(list)
for path in glob.glob(os.path.join(data_dir, "models", "*.json")):
    weights = json.load(open(path))
    try:
        values = (weights["weights_c2v_v2c"] + weights["weights_llrs"]
                  + weights["weights_c2v_readout"])
    except KeyError:
        continue
    name = os.path.basename(path)
    for run in runs:
        if run["tag"] in name and f"_seed_{run['seed']}." in name and run["key"] in name:
            weight_sd[(run["key"], run["tag"])].append(statistics.pstdev(values))
            break

by_arm = collections.defaultdict(list)
for run in runs:
    by_arm[(run["key"], run["tag"])].append(run)

n_epochs_max = max(r["n_epochs"] for r in runs)
print()
print(f"  {len(runs)} training run(s) under {os.path.basename(data_dir)}")
print()
print(f"  {'train set':<26}{'arm tag':<22}{'completed / ' + str(n_epochs_max):>15}{'mean':>7}{'weight sd':>12}")
print(f"  {'-'*26}{'-'*22}{'-'*15}{'-'*7}{'-'*12}")
for (key, tag) in sorted(by_arm):
    group = sorted(by_arm[(key, tag)], key=lambda r: r["seed"])
    completed = [r["completed"] for r in group]
    sds = weight_sd.get((key, tag), [])
    sd_text = "n/a"
    if sds:
        sd_text = f"{statistics.mean(sds):.4f}"
    print(f"  {key:<26}{tag:<22}{str(completed):>15}{statistics.mean(completed):>7.1f}{sd_text:>12}")

print()
worst = min(statistics.mean([r["completed"] for r in g]) for g in by_arm.values())
best = max(statistics.mean([r["completed"] for r in g]) for g in by_arm.values())
if best - worst >= 1.0:
    print(f"  UNEVEN: arms differ by {best - worst:.1f} completed epochs.")
    print("    Arm comparisons are confounded by training budget. The arm with fewer")
    print("    completed epochs is being rolled back more (train.jl:495); lower")
    print("    learning_rate, raise adam_eps, or tighten max_grad_norm.")
else:
    print(f"  EVEN: arms are within {best - worst:.1f} completed epochs of each other.")
    all_sd = [v for vs in weight_sd.values() for v in vs]
    if all_sd and statistics.mean(all_sd) < 0.10:
        print(f"    But mean weight sd is {statistics.mean(all_sd):.4f}, near the ~0.058")
        print("    initialisation — the runs are UNDERtrained. Raise n_epochs or")
        print("    n_gradient_updates_per_epoch rather than the learning rate.")
    elif all_sd:
        print(f"    Mean weight sd {statistics.mean(all_sd):.4f}, well clear of the ~0.058")
        print("    initialisation. Arm comparisons from this run are fair.")
print()
PYTHON

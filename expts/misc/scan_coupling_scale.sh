#!/usr/bin/env bash
# Classical alpha scan for the enriched check node: standard BP (unit weights,
# nothing trained) with the CER couplings inside the check factor at a fixed
# alpha, on the spread_comparison test sets. Training-free, ~5 min per point on
# an M-series Mac with USE_GPU=1.
#
# The alpha = 1 result exists already (run_tag _cnenriched) and is ~10x WORSE
# than tanh, almost entirely convergence failures on weight-2/3 errors. The
# Python reference kernel reproduces that on the exact failing samples, so it is
# the rule, not the implementation. 0.42 is the temperature-consistent value
# (see the TOML); the scan brackets it.
#
# Usage, from expts/:
#   bash misc/scan_coupling_scale.sh                # all alphas, seeds 1 2 3
#   bash misc/scan_coupling_scale.sh 1              # one seed
#   ALPHAS="0p3 0p42" bash misc/scan_coupling_scale.sh 1 2
#
# Results land in data/<codename>/results/ as
#   simulation_results_test_p_0.0005_sig_0.001_s_<seed>_standard_bp_iters_90_cnenriched_a<alpha>.csv
# alongside the existing _cntanh and _cnenriched (alpha = 1) files, so the
# whole scan is one glob away from a table.

set -u

codename="72q_BB_cycles_1_spread_comparison"
n_iterations=90
alphas="${ALPHAS:-0p2 0p3 0p42 0p6}"

if [ "$#" -gt 0 ]; then
    seeds="$*"
else
    seeds="1 2 3"
fi

if [ ! -f "standard_bp_experiments.jl" ]; then
    echo "Run this from the expts/ directory (standard_bp_experiments.jl not found here)."
    exit 1
fi

export USE_GPU="${USE_GPU:-1}"

for seed in ${seeds}; do
    for alpha in ${alphas}; do
        hyperparams_file="hyperparams_standard_bp_enriched_a${alpha}.toml"
        results_file="./../data/${codename}/results/simulation_results_test_p_0.0005_sig_0.001_s_${seed}_standard_bp_iters_${n_iterations}_cnenriched_a${alpha}.csv"
        if [ -f "${results_file}" ]; then
            echo "== seed ${seed} alpha ${alpha}: results exist, skipping"
            continue
        fi
        echo "== seed ${seed} alpha ${alpha}: $(date '+%H:%M:%S')"
        julia --project="./../" standard_bp_experiments.jl \
            --workdir ./../data \
            --codename "${codename}" \
            --n_iterations_BP "${n_iterations}" \
            --hyperparams "${hyperparams_file}" \
            --cer_data "correlated_weights_p_0.0005_sig_0.001_s_${seed}.txt" \
            --test "test_p_0.0005_sig_0.001_s_${seed}.txt" \
            --diagnose true
    done
done

echo
echo "== summary (failures / 10^6; coset + convergence) =="
python3 - "${codename}" "${n_iterations}" <<'PYEOF'
import csv, glob, re, sys, os
codename, iters = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(f"./../data/{codename}/results/simulation_results_test_p_0.0005_sig_0.001_s_*_standard_bp_iters_{iters}_cn*.csv")):
    name = os.path.basename(path)
    seed = re.search(r"_s_(\d+)_standard", name).group(1)
    tag = re.search(r"_cn(\w+)\.csv$", name).group(1)
    record = next(csv.DictReader(open(path)))
    rows.append((tag, int(seed), int(record["num_failures"]),
                 int(record.get("num_coset_failures", -1)), int(record.get("num_convergence_failures", -1))))
def alpha_of(tag):
    if tag == "tanh":
        return 0.0
    match = re.search(r"a(\d+)p(\d+)", tag)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 1.0
rows.sort(key=lambda r: (alpha_of(r[0]), r[1]))
print(f"{'alpha':>6s} {'seed':>4s} {'failures':>9s} {'coset':>6s} {'conv':>6s}")
for tag, seed, fails, coset, conv in rows:
    print(f"{alpha_of(tag):6.2f} {seed:4d} {fails:9d} {coset:6d} {conv:6d}")
PYEOF

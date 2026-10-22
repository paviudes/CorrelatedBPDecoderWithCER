#!/usr/bin/env python3
"""
compare_decoders.py -- BP, BP+OSD and the Neural BP arms side by side, per test
dataset, at one fixed seed and at each arm's best seed.

    python3 misc/compare_decoders.py --workdir 72q_BB_cycles_1_sweep             # seed 1
    python3 misc/compare_decoders.py --workdir 72q_BB_cycles_1_sweep --seed 2

reads  <run>/results/decoder_statistics_NeuralBP.csv   (postprocess.jl --task=gather)
       <run>/results/decoder_statistics_BP.csv
       <run>/results/decoder_statistics_BP_OSD.csv
       <run>/results/reference_decoders_rerun.csv      (optional: reference_decoders.py)
writes <run>/results/<name>.csv    one row per test dataset: failures and LER of every decoder
       <run>/plots/<name>.pdf, .png
       (name = decoder_comparison unless --name says otherwise)

Seeds. `--seed` is the FIXED seed: chosen before looking, so the fair number.
The BEST seed of an arm is the one with the lowest geometric-mean LER over the
test datasets compared; it is chosen on the test data, so it is optimistic. The
min / median / max over all seeds come along as the band in the figure.

The shipped BP and BP+OSD tables decode with a uniform prior of 0.1, 500
iterations and OSD-E of order 2 (see reference_decoders.py, which reproduces them
exactly); the legend says so. Re-runs with other settings, where present, are
drawn as markers (--rerun-configs picks which).

Arms. Every arm in the gathered CSV unless `--arms` names some: nocer first (the
reference of the ratios), then cer, then the rest -- cer_enr (the enriched check
node, sweeps/enriched_submission.toml), cer_bethe (the Bethe table) -- once the
gather includes them (cer_arms = ["cer", "nocer", "cer_enr", "cer_bethe"]).

Figure: top row LER vs p, one panel per sigma / p ratio (sigma = 0, p, 2p); bottom
row the LER ratio of every other arm to the first: at the fixed seed (with
Poisson 95% error bars), of the medians over seeds, and -- with one compared arm
-- at the best seeds.
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RERUN_CONFIGS = ["bp:cer:500", "osd_cs10:cer:500", "osd_cs10:mean:500"]
ARM_COLORS = ["#1f77b4", "#d95f02", "#7570b3", "#e7298a"]
RERUN_STYLES = {
    "bp:cer:500": dict(marker="^", color="#1b9e77", fillstyle="full"),
    "bp:mean:500": dict(marker="^", color="#1b9e77", fillstyle="none"),
    "osd_cs10:cer:500": dict(marker="*", color="#1b9e77", fillstyle="full"),
    "osd_cs10:mean:500": dict(marker="*", color="#1b9e77", fillstyle="none"),
}
ARM_NAMES = {"cer_enr_alphastar": "Neural BP, CER priors + couplings, α carried to each test set",
             "nocer": "Neural BP, no CER", "cer": "Neural BP, CER priors",
             "cer_enr": "Neural BP, CER priors + couplings (enriched)",
             "cer_bethe": "Neural BP, CER priors + Bethe table"}
ARM_MARKERS = ["o", "s", "D", "^", "v", "P"]


def default_arm_order(present_arms: List[str]) -> List[str]:
    """nocer, then cer, then every other arm in alphabetical order."""
    leading = [arm for arm in ("nocer", "cer") if arm in present_arms]
    ordered = leading + sorted(arm for arm in present_arms if arm not in leading)
    return ordered


def geometric_mean(values: np.ndarray) -> float:
    mean_value = float(np.exp(np.mean(np.log(values))))
    return mean_value


def sigma_group(test_p: float, test_sigma: float) -> Tuple[float, str]:
    """(sort order, label) of a dataset's sigma / p ratio: 'σ = 0', 'σ = p', 'σ = 2p'."""
    ratio = round(test_sigma / test_p, 3)
    label = "σ = 0"
    if ratio == 1:
        label = "σ = p"
    elif ratio != 0:
        label = f"σ = {ratio:g}p"
    return ratio, label


def poisson_ratio_interval(numerator: np.ndarray, denominator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """95% interval of a ratio of two Poisson counts, from the log-normal
    approximation (independent counts, so conservative for paired test sets)."""
    ratio = numerator / denominator
    log_spread = 1.96 * np.sqrt(1.0 / np.maximum(numerator, 0.5) + 1.0 / np.maximum(denominator, 0.5))
    interval = (ratio * np.exp(-log_spread), ratio * np.exp(log_spread))
    return interval


def load_tables(results_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def read(name: str, required: bool = True) -> pd.DataFrame:
        path = os.path.join(results_directory, name)
        if not os.path.isfile(path):
            if required:
                raise SystemExit(f"no {path}: run `julia --project=\"./\" postprocess.jl --task=gather` first")
            return pd.DataFrame()
        return pd.read_csv(path)
    neural = read("decoder_statistics_NeuralBP.csv")
    bp = read("decoder_statistics_BP.csv")
    bposd = read("decoder_statistics_BP_OSD.csv")
    rerun = read("reference_decoders_rerun.csv", required=False)
    for frame, name in [(neural, "NeuralBP"), (bp, "BP"), (bposd, "BP_OSD")]:
        if "test_key" not in frame.columns:
            raise SystemExit(f"decoder_statistics_{name}.csv has no test_key column: re-run the gather with the "
                             "current postprocess.jl")
    return neural, bp, bposd, rerun


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare BP, BP+OSD and Neural BP arms.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--workdir", required=True, help="run directory: a path, or a codename under data/")
    parser.add_argument("--seed", type=int, default=1, help="the fixed seed (default: %(default)s)")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Neural BP arms (the `arm` column), the first being the reference of the ratios "
                             "(default: every arm gathered, nocer first)")
    parser.add_argument("--rerun-configs", nargs="*", default=DEFAULT_RERUN_CONFIGS,
                        help="re-run reference configs to draw, where present (default: %(default)s)")
    parser.add_argument("--name", default="decoder_comparison", help="output file stem")
    arguments = parser.parse_args(argv)

    run_directory: str = arguments.workdir
    if not os.path.isdir(run_directory):
        run_directory = os.path.join(REPO_ROOT, "data", arguments.workdir)
    if not os.path.isdir(run_directory):
        print(f"no run directory: {arguments.workdir}", file=sys.stderr)
        return 1
    results_directory = os.path.join(run_directory, "results")
    neural, bp, bposd, rerun = load_tables(results_directory)

    arms: List[str] = arguments.arms or default_arm_order(sorted(set(neural["arm"].dropna())))
    missing_arms = [arm for arm in arms if arm not in set(neural["arm"])]
    if missing_arms:
        print(f"arm(s) {missing_arms} not in the gathered Neural BP results (arms: {sorted(set(neural['arm']))})",
              file=sys.stderr)
        return 1
    neural = neural[neural["arm"].isin(arms)]
    failures = neural.pivot_table(index="test_key", columns=["arm", "seed"], values="num_failures", aggfunc="sum")
    duplicates = neural.groupby(["test_key", "arm", "seed"]).size()
    if (duplicates > 1).any():
        print("more than one results file per (test dataset, arm, seed); narrow the gather (train_instance):",
              file=sys.stderr)
        print(duplicates[duplicates > 1].head(), file=sys.stderr)
        return 1
    datasets = neural.drop_duplicates("test_key").set_index("test_key")[
        ["test_p", "test_sigma", "test_sample", "num_samples_per_error_rate", "train_key"]]
    if datasets["num_samples_per_error_rate"].nunique() != 1:
        print("warning: the test datasets differ in size; LERs are per dataset", file=sys.stderr)

    # References on the same datasets.
    for frame, name in [(bp, "BP"), (bposd, "BP+OSD")]:
        missing_keys = sorted(set(datasets.index) - set(frame["test_key"]))
        if missing_keys:
            print(f"{name}: no table for {len(missing_keys)} test dataset(s), e.g. {missing_keys[:3]}",
                  file=sys.stderr)
            return 1
    bp = bp.set_index("test_key")
    bposd = bposd.set_index("test_key")

    seeds_by_arm: Dict[str, List[int]] = {arm: sorted(failures[arm].columns) for arm in arms}
    for arm in arms:
        if arguments.seed not in seeds_by_arm[arm]:
            print(f"seed {arguments.seed} not in arm {arm} (seeds {seeds_by_arm[arm]})", file=sys.stderr)
            return 1
        incomplete = failures[arm].isna().any()
        if incomplete.any():
            print(f"arm {arm}: seed(s) {list(incomplete[incomplete].index)} miss some test datasets", file=sys.stderr)
            return 1

    # Seed choice.
    print(f"{len(datasets)} test dataset(s); models trained on {sorted(set(datasets['train_key']))}")
    best_seed: Dict[str, int] = {}
    for arm in arms:
        sample_counts = datasets["num_samples_per_error_rate"].values
        per_seed_gm = {seed: geometric_mean(np.maximum(failures[(arm, seed)].values, 0.5) / sample_counts)
                       for seed in seeds_by_arm[arm]}
        per_seed_total = {seed: int(failures[(arm, seed)].sum()) for seed in seeds_by_arm[arm]}
        best_seed[arm] = min(per_seed_gm, key=per_seed_gm.get)
        print(f"  {arm}: geometric-mean LER by seed " +
              ", ".join(f"{seed}: {value:.3e} ({per_seed_total[seed]} failures)" for seed, value in per_seed_gm.items()) +
              f"  -> best seed {best_seed[arm]}")

    # The table.
    table = datasets[["test_p", "test_sigma", "test_sample", "num_samples_per_error_rate"]].rename(
        columns={"num_samples_per_error_rate": "num_samples"}).copy()
    table["sigma_over_p"] = [sigma_group(p, s)[0] for p, s in zip(table["test_p"], table["test_sigma"])]
    columns: Dict[str, pd.Series] = {
        "bp_shipped": bp.loc[table.index, "num_failures"],
        "bposd_shipped": bposd.loc[table.index, "num_failures"],
    }
    for arm in arms:
        columns[f"{arm}_seed{arguments.seed}"] = failures[(arm, arguments.seed)]
        columns[f"{arm}_best_seed{best_seed[arm]}"] = failures[(arm, best_seed[arm])]
        columns[f"{arm}_min"] = failures[arm].min(axis=1)
        columns[f"{arm}_median"] = failures[arm].median(axis=1)
        columns[f"{arm}_max"] = failures[arm].max(axis=1)
    rerun_shown: List[str] = []
    rerun_labels: Dict[str, str] = {}
    if not rerun.empty:
        for config in arguments.rerun_configs:
            rows = rerun[rerun["config"] == config].drop_duplicates("test_key", keep="last").set_index("test_key")
            rows = rows[rows.index.isin(table.index)]
            if rows.empty:
                continue
            rerun_shown.append(config)
            rerun_labels[config] = f"{rows['label'].iloc[0]} (re-run)"
            columns[f"rerun_{config.replace(':', '_')}"] = rows["num_failures"].reindex(table.index)
    for column_name, series in columns.items():
        table[f"{column_name}_failures"] = series
        table[f"{column_name}_ler"] = series / table["num_samples"]
    table = table.sort_values(["sigma_over_p", "test_p"])
    table_path = os.path.join(results_directory, f"{arguments.name}.csv")
    table.reset_index().to_csv(table_path, index=False)

    # Summary.
    reference_arm, compared_arms = arms[0], arms[1:]
    fixed = arguments.seed
    print(f"\nfailures per group (Neural BP at seed {fixed}; best seeds "
          + ", ".join(f"{arm} {best_seed[arm]}" for arm in arms) + ")")
    groups = sorted({sigma_group(p, s) for p, s in zip(table["test_p"], table["test_sigma"])})
    summary_rows = []
    for ratio, label in groups + [(None, "all")]:
        rows = table if ratio is None else table[table["sigma_over_p"] == ratio]
        row = {"group": label, "n": len(rows), "BP": int(rows["bp_shipped_failures"].sum()),
               "BP+OSD": int(rows["bposd_shipped_failures"].sum())}
        for arm in arms:
            row[f"{arm} s{fixed}"] = int(rows[f"{arm}_seed{fixed}_failures"].sum())
            row[f"{arm} best"] = int(rows[f"{arm}_best_seed{best_seed[arm]}_failures"].sum())
            row[f"gm {arm}/BP+OSD"] = round(geometric_mean(rows[f"{arm}_seed{fixed}_failures"].values /
                                                            rows["bposd_shipped_failures"].values), 3)
        summary_rows.append(row)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(summary_rows).to_string(index=False))
    # Every other arm against the first: geometric mean of the per-dataset LER
    # ratio, and on how many datasets the arm is lower.
    if compared_arms:
        print(f"\nLER ratio to {reference_arm}, geometric mean over the group's datasets (datasets where lower):")
        ratio_rows = []
        for compared_arm in compared_arms:
            for ratio, label in groups + [(None, "all")]:
                rows = table if ratio is None else table[table["sigma_over_p"] == ratio]
                fixed_ratio = rows[f"{compared_arm}_seed{fixed}_failures"].values / rows[f"{reference_arm}_seed{fixed}_failures"].values
                best_ratio = (rows[f"{compared_arm}_best_seed{best_seed[compared_arm]}_failures"].values /
                              rows[f"{reference_arm}_best_seed{best_seed[reference_arm]}_failures"].values)
                median_ratio = rows[f"{compared_arm}_median_failures"].values / rows[f"{reference_arm}_median_failures"].values
                ratio_rows.append({
                    "arm": compared_arm, "group": label, "n": len(rows),
                    f"seed {fixed} / seed {fixed}": f"{geometric_mean(fixed_ratio):.3f} ({int((fixed_ratio < 1).sum())}/{len(rows)})",
                    "best / best": f"{geometric_mean(best_ratio):.3f} ({int((best_ratio < 1).sum())}/{len(rows)})",
                    "median / median": f"{geometric_mean(median_ratio):.3f} ({int((median_ratio < 1).sum())}/{len(rows)})",
                })
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(pd.DataFrame(ratio_rows).to_string(index=False))
    if rerun_shown:
        print("\nre-run references on the datasets where they exist (failures):")
        shown = table.dropna(subset=[f"rerun_{rerun_shown[0].replace(':', '_')}_failures"])
        listing = shown[["bposd_shipped_failures"] + [f"rerun_{config.replace(':', '_')}_failures" for config in rerun_shown]
                        + [f"{arm}_seed{fixed}_failures" for arm in arms]
                        + [f"{arm}_best_seed{best_seed[arm]}_failures" for arm in arms if best_seed[arm] != fixed]]
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(listing.astype("Int64").to_string())
    print(f"\ntable -> {table_path}")

    # The figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    n_columns = len(groups)
    figure, axes = plt.subplots(2, n_columns, figsize=(4.6 * n_columns, 7.6), sharex=True, sharey="row",
                                gridspec_kw=dict(height_ratios=[2.2, 1]), squeeze=False)
    handles: Dict[str, object] = {}
    # Every fixed-seed ratio drawn in the bottom row, which shares one y axis.
    plotted_ratios: List[float] = []
    for column_index, (ratio, label) in enumerate(groups):
        rows = table[table["sigma_over_p"] == ratio]
        p_values = rows["test_p"].values
        top, bottom = axes[0, column_index], axes[1, column_index]
        handles["BP"], = top.plot(p_values, rows["bp_shipped_ler"], ":", marker="v", color="#7f7f7f", ms=5,
                                  label="BP (shipped table: prior 0.1, 500 it)")
        handles["BP+OSD"], = top.plot(p_values, rows["bposd_shipped_ler"], "--", marker="D", color="black", ms=4,
                                      label="BP+OSD (shipped table: prior 0.1, OSD-E2, 500 it)")
        for arm_index, arm in enumerate(arms):
            color = ARM_COLORS[arm_index % len(ARM_COLORS)]
            name = ARM_NAMES.get(arm, f"Neural BP, {arm}")
            top.fill_between(p_values, rows[f"{arm}_min_ler"], rows[f"{arm}_max_ler"], color=color, alpha=0.13,
                             linewidth=0)
            fixed_label = f"{name}, seed {fixed}"
            if best_seed[arm] == fixed:
                fixed_label += " (also its best seed)"
            handles[f"{arm} fixed"], = top.plot(p_values, rows[f"{arm}_seed{fixed}_ler"], "-",
                                                marker=ARM_MARKERS[arm_index % len(ARM_MARKERS)],
                                                color=color, ms=5, label=fixed_label)
            if best_seed[arm] != fixed:
                handles[f"{arm} best"], = top.plot(p_values, rows[f"{arm}_best_seed{best_seed[arm]}_ler"], "--",
                                                   color=color, lw=1.4,
                                                   label=f"{name}, best seed ({best_seed[arm]})")
        for config_index, config in enumerate(rerun_shown):
            column = f"rerun_{config.replace(':', '_')}_ler"
            present = rows[column].notna()
            if present.any():
                style = RERUN_STYLES.get(config, dict(marker="P", color="#1b9e77", fillstyle="full"))
                # Spread the re-run markers a little along p so they do not hide each other.
                jitter = np.exp(0.07 * (config_index - (len(rerun_shown) - 1) / 2))
                handles[config], = top.plot(p_values[present.values] * jitter, rows.loc[present, column],
                                            linestyle="none", ms=10 if style["marker"] == "*" else 6.5, mew=1.2,
                                            label=rerun_labels[config], zorder=5, **style)
        top.set_xscale("log")
        top.set_yscale("log")
        top.set_title(label)
        top.grid(True, which="both", alpha=0.25)
        if column_index == 0:
            top.set_ylabel("logical error rate")

        if compared_arms:
            denominator = rows[f"{reference_arm}_seed{fixed}_failures"].values.astype(float)
            for arm_index, compared_arm in enumerate(arms[1:], start=1):
                color = ARM_COLORS[arm_index % len(ARM_COLORS)]
                short_name = compared_arm
                if len(compared_arms) == 1:
                    short_name = ""
                numerator = rows[f"{compared_arm}_seed{fixed}_failures"].values.astype(float)
                fixed_ratio = numerator / denominator
                plotted_ratios.extend(float(value) for value in fixed_ratio if np.isfinite(value) and value > 0)
                lower, upper = poisson_ratio_interval(numerator, denominator)
                bottom.errorbar(p_values, fixed_ratio, yerr=[fixed_ratio - lower, upper - fixed_ratio], fmt="-",
                                marker=ARM_MARKERS[arm_index % len(ARM_MARKERS)], color=color, ms=4, capsize=2,
                                lw=1.2, label=f"{short_name} seed {fixed} / seed {fixed}".strip())
                if len(compared_arms) == 1:
                    bottom.plot(p_values, rows[f"{compared_arm}_best_seed{best_seed[compared_arm]}_failures"].values /
                                rows[f"{reference_arm}_best_seed{best_seed[reference_arm]}_failures"].values, "--s",
                                color=color, ms=3.5, lw=1.0, mfc="white",
                                label=f"best seed ({best_seed[compared_arm]}) / best seed ({best_seed[reference_arm]})")
                median_color = "#555555"
                if len(compared_arms) > 1:
                    median_color = color
                bottom.plot(p_values, rows[f"{compared_arm}_median_failures"].values /
                            rows[f"{reference_arm}_median_failures"].values, ":", color=median_color, lw=1.2,
                            label=f"{short_name} median over seeds / median".strip())
            bottom.axhline(1.0, color="black", lw=0.8)
            bottom.set_yscale("log")
            bottom.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            bottom.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}" if value in (0.5, 0.7, 2, 3, 5) else ""))
            bottom.grid(True, which="both", alpha=0.25)
            bottom.set_xlabel("p")
            if column_index == 0:
                compared_text = "other arm"
                if len(compared_arms) == 1:
                    compared_text = compared_arms[0]
                bottom.set_ylabel(f"LER {compared_text} / {reference_arm}")
            if column_index == n_columns - 1:
                bottom.legend(fontsize=7.5, loc="upper right")
        else:
            bottom.set_visible(False)

    # One limit for the shared ratio axis: 0.4–8 unless a ratio falls outside it
    # (the enriched arm reaches ~90x at p = 1e-4).
    if plotted_ratios:
        ratio_floor: float = min(0.4, 0.8 * min(plotted_ratios))
        ratio_ceiling: float = max(8.0, 1.3 * max(plotted_ratios))
        axes[1, 0].set_ylim(ratio_floor, ratio_ceiling)

    train_text = ", ".join(sorted(set(datasets["train_key"])))
    sample_text = ", ".join(str(sample) for sample in sorted(set(datasets["test_sample"])))
    figure.suptitle(f"{os.path.basename(os.path.normpath(run_directory))}: Neural BP trained on {train_text}; "
                    f"test sample {sample_text}, {int(datasets['num_samples_per_error_rate'].iloc[0]):,} shots per "
                    f"dataset; bands = min–max over seeds {seeds_by_arm[arms[0]]}", fontsize=10)
    figure.legend(handles=list(handles.values()), loc="lower center", ncol=3, fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.005))
    figure.tight_layout(rect=(0, 0.1, 1, 0.97))
    plots_directory = os.path.join(run_directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)
    for extension in ("pdf", "png"):
        figure_path = os.path.join(plots_directory, f"{arguments.name}.{extension}")
        figure.savefig(figure_path, dpi=200, bbox_inches="tight")
        print(f"figure -> {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

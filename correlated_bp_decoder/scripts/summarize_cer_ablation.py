#!/usr/bin/env python3
"""Summarize one or more CER ablation result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Load CER ablation JSON files and print a compact summary."""

    parser = argparse.ArgumentParser(
        description="Print a compact comparison table for CER ablation result JSON files."
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="One or more JSON files written by cer_ablation_explicit.py",
    )
    args = parser.parse_args(argv)

    for path in args.results:
        payload = json.loads(path.read_text())
        print(path)
        print(
            "  dataset={dataset} sample={sample} train={train} test={test} "
            "layers={layers} epochs={epochs}".format(
                dataset=payload["dataset"],
                sample=payload["sample"],
                train=payload["train_samples_used"],
                test=payload["test_samples_used"],
                layers=payload["n_layers"],
                epochs=payload["n_epochs"],
            )
        )

        standard_bp = payload.get("standard_bp")
        if standard_bp is not None:
            print(
                "  standard_bp: logical_success={success:.5f} logical_error={error:.5f}".format(
                    success=standard_bp["logical_success_rate"],
                    error=standard_bp["logical_error_rate"],
                )
            )

        bp_osd = payload.get("bp_osd_reference")
        if bp_osd is not None:
            print(
                "  bp_osd_ref: logical_success={success:.5f} logical_error={error:.5f}".format(
                    success=bp_osd["logical_success_rate"],
                    error=bp_osd["logical_error_rate"],
                )
            )

        for mode, result in payload["neural_results"].items():
            comparisons = payload["comparisons"].get(mode, {})
            improvement = comparisons.get("improvement_factor_over_standard_bp")
            improvement_text = (
                "n/a" if improvement is None else f"{improvement:.3f}x"
            )
            print(
                "  {mode}: logical_success={success:.5f} logical_error={error:.5f} "
                "improvement_vs_standard={improvement}".format(
                    mode=mode,
                    success=result["logical_success_rate"],
                    error=result["logical_error_rate"],
                    improvement=improvement_text,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

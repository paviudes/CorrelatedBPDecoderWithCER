"""Smoke tests for the thin Phase 4 experiment CLI."""

from __future__ import annotations

import json
from pathlib import Path

from correlated_bp_decoder.experiments.run_neural_bp import main

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_run_neural_bp_cli_train_and_evaluate_smoke(tmp_path, capsys) -> None:
    """Exercise the thin train/evaluate CLI flow on the toy fixture code."""

    weights_path = tmp_path / "trained_weights.json"
    errors_path = tmp_path / "test_errors.txt"
    errors_path.write_text("0 0 0\n0 0 0\n0 0 0\n")

    exit_code = main(
        [
            "train-synthetic",
            "--parity-check-file",
            str(FIXTURE_DIR / "parity_check.txt"),
            "--logicals-file",
            str(FIXTURE_DIR / "logicals.txt"),
            "--n-layers",
            "2",
            "--n-samples",
            "8",
            "--error-rate",
            "0.15",
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--seed",
            "0",
            "--weights-out",
            str(weights_path),
        ]
    )
    train_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert weights_path.is_file()
    assert train_output["command"] == "train-synthetic"

    exit_code = main(
        [
            "evaluate-errors",
            "--parity-check-file",
            str(FIXTURE_DIR / "parity_check.txt"),
            "--logicals-file",
            str(FIXTURE_DIR / "logicals.txt"),
            "--n-layers",
            "2",
            "--weights-file",
            str(weights_path),
            "--test-errors-file",
            str(errors_path),
            "--batch-size",
            "2",
        ]
    )
    evaluate_output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert evaluate_output["command"] == "evaluate-errors"
    assert evaluate_output["n_samples"] == 3
    assert evaluate_output["n_correct"] == 3

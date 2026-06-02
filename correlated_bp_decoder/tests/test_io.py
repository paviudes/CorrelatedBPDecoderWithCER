"""Tests for file-backed model loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from correlated_bp_decoder import (
    NachmaniNeuralBP,
    load_base_bp_model,
    load_binary_matrix,
    load_trained_neuralbp_model,
    load_trained_weights,
    save_trained_neuralbp_model,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_load_binary_matrix_preserves_two_dimensional_shape() -> None:
    """Keep single-row logical-operator files as 2D matrices."""

    logicals = load_binary_matrix(FIXTURE_DIR / "logicals.txt")
    assert logicals.shape == (1, 3)
    np.testing.assert_array_equal(logicals, np.asarray([[1, 0, 1]], dtype=np.int64))


def test_load_base_bp_model_uses_cer_rates_and_default_llrs() -> None:
    """Load parity checks, logicals, CER data, and compiled graph metadata."""

    base = load_base_bp_model(
        FIXTURE_DIR / "parity_check.txt",
        FIXTURE_DIR / "logicals.txt",
        5,
        correlation_strengths_file=FIXTURE_DIR / "correlated_weights.txt",
    )

    np.testing.assert_array_equal(
        base.parity_check_matrix.astype(np.int64),
        np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        base.parity_check_matrix_dual.astype(np.int64),
        np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        base.initial_llrs,
        np.asarray(
            [
                np.log((1.0 - 0.1) / 0.1),
                np.log((1.0 - 0.2) / 0.2),
                np.log(9.0),
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        base.connectivity,
        np.asarray([[1, 3], [2, 1], [3, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        base.correlation_strengths,
        np.asarray(
            [
                0.005851149427861176,
                0.0063269667097774155,
                0.006110318845201688,
            ],
            dtype=np.float32,
        ),
    )
    assert base.n_layers == 5
    assert base.is_correlated is True


def test_load_base_bp_model_defaults_to_uncorrelated_setup() -> None:
    """Use the Julia default LLR when no CER file is supplied."""

    base = load_base_bp_model(
        FIXTURE_DIR / "parity_check.txt",
        FIXTURE_DIR / "logicals.txt",
        2,
    )

    np.testing.assert_allclose(
        base.initial_llrs,
        np.full(3, np.log(9.0), dtype=np.float32),
    )
    assert base.is_correlated is False
    assert base.connectivity.shape == (0, 2)
    assert base.correlation_strengths.shape == (0,)


def test_load_base_bp_model_raises_for_missing_cer_file() -> None:
    """Reject explicit CER file paths that do not exist."""

    with pytest.raises(FileNotFoundError):
        load_base_bp_model(
            FIXTURE_DIR / "parity_check.txt",
            FIXTURE_DIR / "logicals.txt",
            1,
            correlation_strengths_file=FIXTURE_DIR / "missing.txt",
        )


def test_save_and_load_trained_neuralbp_model_roundtrip(tmp_path) -> None:
    """Round-trip a structured Python checkpoint through disk."""

    base = load_base_bp_model(
        FIXTURE_DIR / "parity_check.txt",
        FIXTURE_DIR / "logicals.txt",
        2,
    )
    model = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=torch.arange(1, base.nb_weights_c2v_v2c * base.n_layers + 1),
        weights_llrs=torch.arange(1, base.code_n_bits * base.n_layers + 1),
        weights_c2v_readout=torch.arange(1, base.nb_weights_c2v_readout + 1),
    )
    weights_path = tmp_path / "trained_weights.json"

    save_trained_neuralbp_model(weights_path, model, metadata={"tag": "roundtrip"})
    payload = json.loads(weights_path.read_text())
    loaded_model = load_trained_neuralbp_model(weights_path, base)

    assert payload["schema_version"] == 1
    assert payload["base_summary"]["n_layers"] == 2
    assert payload["metadata"]["tag"] == "roundtrip"
    assert torch.equal(loaded_model.weights_c2v_v2c.detach(), model.weights_c2v_v2c.detach())
    assert torch.equal(loaded_model.weights_llrs.detach(), model.weights_llrs.detach())
    assert torch.equal(
        loaded_model.weights_c2v_readout.detach(),
        model.weights_c2v_readout.detach(),
    )


def test_load_trained_weights_accepts_legacy_julia_format(tmp_path) -> None:
    """Remain compatible with the original Julia flat-weight JSON format."""

    legacy_payload = {
        "weights_c2v_v2c": [1.0, 2.0, 3.0],
        "weights_llrs": [4.0, 5.0],
        "weights_c2v_readout": [6.0, 7.0],
    }
    weights_path = tmp_path / "legacy_weights.json"
    weights_path.write_text(json.dumps(legacy_payload))

    loaded = load_trained_weights(weights_path)

    np.testing.assert_allclose(loaded["weights_c2v_v2c"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(loaded["weights_llrs"], np.asarray([4.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(
        loaded["weights_c2v_readout"],
        np.asarray([6.0, 7.0], dtype=np.float32),
    )

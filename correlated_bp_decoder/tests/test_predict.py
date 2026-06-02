"""Tests for Phase 4 prediction helpers."""

from __future__ import annotations

import numpy as np
import torch

from correlated_bp_decoder import (
    NachmaniNeuralBP,
    NeuralBPBase,
    check_bp_solutions,
    neuralbp_test_predictions,
    predict_and_check_neuralbp,
    predict_neuralbp,
)


def _build_prediction_model() -> NachmaniNeuralBP:
    """Build a deterministic tiny model for prediction tests."""

    parity_check = [[1, 1, 0], [0, 1, 1]]
    parity_check_dual = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    initial_llrs = [2.1972246, 2.1972246, 2.1972246]
    base = NeuralBPBase(parity_check, parity_check_dual, initial_llrs, n_layers=2)
    return NachmaniNeuralBP(
        base,
        weights_c2v_v2c=torch.ones(base.nb_weights_c2v_v2c * base.n_layers),
        weights_llrs=torch.ones(base.code_n_bits * base.n_layers),
        weights_c2v_readout=torch.ones(base.nb_weights_c2v_readout),
    )


def test_predict_neuralbp_matches_thresholded_forward_pass() -> None:
    """Threshold the posterior LLRs batchwise in the same way as Julia."""

    model = _build_prediction_model()
    syndromes = torch.tensor(
        [
            [False, True, False],
            [False, False, True],
        ]
    )
    expected = model(
        model.expand_initial_llrs(syndromes.shape[1]),
        syndromes,
    ) < 0

    actual = predict_neuralbp(model, syndromes, batch_size=2)

    assert actual.dtype == torch.bool
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_check_bp_solutions_accepts_any_successful_layer() -> None:
    """Mark a sample correct when any unfolded layer fixes the residual error."""

    parity_check_dual = torch.eye(3, dtype=torch.int64)
    errors = torch.tensor(
        [
            [False, True],
            [False, False],
            [False, False],
        ]
    )
    proposed_recoveries = torch.tensor(
        [
            [[False, True], [False, False]],
            [[False, False], [False, False]],
            [[False, False], [False, False]],
        ]
    )

    is_correct = check_bp_solutions(
        parity_check_dual,
        errors,
        proposed_recoveries,
    )

    assert torch.equal(is_correct, torch.tensor([True, False]))


def test_predict_and_check_neuralbp_decodes_zero_errors() -> None:
    """Return success on the zero-syndrome, zero-error fixture."""

    model = _build_prediction_model()
    syndromes = torch.zeros((2, 3), dtype=torch.bool)
    errors = torch.zeros((3, 3), dtype=torch.bool)

    is_correct = predict_and_check_neuralbp(model, syndromes, errors, batch_size=2)

    assert torch.equal(is_correct, torch.tensor([True, True, True]))


def test_neuralbp_test_predictions_loads_file_backed_errors(tmp_path) -> None:
    """Load Julia-style explicit error files and evaluate them end to end."""

    model = _build_prediction_model()
    errors = np.zeros((3, 4), dtype=np.int64)
    errors_path = tmp_path / "errors.txt"
    np.savetxt(errors_path, errors, fmt="%d")

    is_correct = neuralbp_test_predictions(model, errors_path, batch_size=2)

    assert torch.equal(is_correct, torch.tensor([True, True, True, True]))

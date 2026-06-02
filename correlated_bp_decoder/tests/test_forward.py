"""Tests for the first torch-based neural forward pass."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from correlated_bp_decoder import (
    NachmaniNeuralBP,
    NeuralBPBase,
    forward_pass_with_weights,
    run_bp,
)


def _build_test_model() -> NachmaniNeuralBP:
    parity_check = [[1, 1, 0], [0, 1, 1]]
    parity_check_dual = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    initial_llrs = [2.1972246, 2.1972246, 2.1972246]
    base = NeuralBPBase(parity_check, parity_check_dual, initial_llrs, n_layers=3)

    weights_c2v_v2c = torch.ones(base.nb_weights_c2v_v2c * base.n_layers, dtype=torch.float32)
    weights_llrs = torch.ones(base.code_n_bits * base.n_layers, dtype=torch.float32)
    weights_c2v_readout = torch.ones(base.nb_weights_c2v_readout, dtype=torch.float32)
    return NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout,
    )


def test_forward_module_matches_explicit_weight_path() -> None:
    """Match the Julia forward test's in-place vs functional comparison."""

    model = _build_test_model()
    initial_llrs_batch = torch.full((3, 2), 2.1972246, dtype=torch.float32)
    syndromes_batch = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )

    output_module = model(initial_llrs_batch, syndromes_batch)
    output_explicit = forward_pass_with_weights(
        model.weights_c2v_v2c,
        model.weights_llrs,
        model.weights_c2v_readout,
        model.base,
        initial_llrs_batch,
        syndromes_batch,
    )

    assert output_module.shape == (3, 2, 3)
    assert torch.allclose(output_module, output_explicit, atol=1e-6)


def test_forward_final_layer_matches_standard_bp_with_unity_weights() -> None:
    """Match the Julia final-layer check against standard BP.

    The Julia test uses a file-backed Hamming fixture. We keep the same
    behavioral target here with an in-test Hamming matrix so the regression
    remains self-contained and deterministic under pytest.
    """

    parity_check = np.asarray(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    initial_llrs = np.full(7, np.log(9.0), dtype=np.float32)
    base = NeuralBPBase(parity_check, parity_check, initial_llrs, n_layers=3)
    model = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=torch.ones(base.nb_weights_c2v_v2c * base.n_layers),
        weights_llrs=torch.ones(base.code_n_bits * base.n_layers),
        weights_c2v_readout=torch.ones(base.nb_weights_c2v_readout),
    )
    syndrome = np.asarray([1, 0, 1], dtype=np.int64)

    neural_output = model(
        torch.as_tensor(initial_llrs[:, None]),
        torch.as_tensor(syndrome[:, None], dtype=torch.bool),
    )
    standard_bp_llrs, _ = run_bp(
        parity_check,
        4,
        syndrome,
        initial_llrs.astype(np.float64),
        base.n_layers,
    )

    np.testing.assert_allclose(
        neural_output[:, 0, -1].detach().numpy(),
        standard_bp_llrs,
        atol=1e-6,
    )


def test_forward_batch_first_and_input_preparation_are_consistent() -> None:
    """Support a batch-first public API without changing the internal layout."""

    model = _build_test_model()
    batch_first_llrs = model.expand_initial_llrs(2, batch_first=True)
    batch_first_syndromes = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )

    prepared_llrs, prepared_syndromes = model.prepare_inputs(
        batch_first_llrs,
        batch_first_syndromes,
        batch_first=True,
    )
    output_internal = model(prepared_llrs, prepared_syndromes)
    output_batch_first = model.forward_batch_first(
        batch_first_llrs,
        batch_first_syndromes,
    )

    assert model.device == torch.device("cpu")
    assert prepared_llrs.shape == (3, 2)
    assert prepared_syndromes.shape == (2, 2)
    assert output_batch_first.shape == (2, 3, 3)
    assert torch.allclose(
        output_batch_first,
        output_internal.permute(1, 0, 2),
        atol=1e-6,
    )


def test_forward_regression_fixture_is_stable() -> None:
    """Pin a small deterministic forward result for later regression checks."""

    model = _build_test_model()
    initial_llrs_batch = torch.full((3, 2), 2.1972246, dtype=torch.float32)
    syndromes_batch = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )

    output = model(initial_llrs_batch, syndromes_batch)
    expected = torch.tensor(
        [
            [[0.0, -2.1972246, -2.1972246], [4.3944492, 2.1972246, 2.1972246]],
            [[2.1972246, 2.1972246, 2.1972246], [2.1972246, 2.1972246, 2.1972246]],
            [[4.3944492, 2.1972246, 2.1972246], [0.0, -2.1972246, -2.1972246]],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(output, expected, atol=1e-6)


def test_forward_matches_accelerator_when_available() -> None:
    """Keep the CPU and accelerator forward paths numerically aligned."""

    accelerator: str | None = None
    if torch.cuda.is_available():
        accelerator = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerator = "mps"

    if accelerator is None:
        pytest.skip("No CUDA or MPS accelerator is available in this environment.")

    cpu_model = _build_test_model()
    device_model = _build_test_model().to(accelerator)
    initial_llrs_batch = torch.full((3, 2), 2.1972246, dtype=torch.float32)
    syndromes_batch = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )

    cpu_output = cpu_model(initial_llrs_batch, syndromes_batch)
    device_output = device_model(
        initial_llrs_batch.to(accelerator),
        syndromes_batch.to(accelerator),
    ).cpu()

    assert torch.allclose(cpu_output, device_output, atol=1e-5, rtol=1e-5)


def test_forward_rejects_mismatched_input_shapes() -> None:
    """Validate forward input dimensions before entering message passing."""

    model = _build_test_model()
    initial_llrs_batch = torch.full((2, 1), 2.1972246, dtype=torch.float32)
    syndromes_batch = torch.tensor([[True], [False]], dtype=torch.bool)

    try:
        model(initial_llrs_batch, syndromes_batch)
    except ValueError as exc:
        assert "bit rows" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for mismatched bit dimensions.")

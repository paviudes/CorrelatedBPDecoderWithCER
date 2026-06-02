"""Tests for Phase 2 numerical helpers and message updates."""

from __future__ import annotations

import torch

from correlated_bp_decoder import (
    NeuralBPBase,
    c2v_to_v2c,
    c2v_to_v2c_with_weights_,
    readout,
    readout_with_weights_,
    safe_atanh_exp_signed,
    safe_atanh_exp_signed_,
    safe_log_tanh_split,
    safe_log_tanh_split_,
    v2c_to_c2v,
    v2c_to_c2v_,
)


def _build_test_base() -> NeuralBPBase:
    parity_check = [[1, 1, 0], [0, 1, 1]]
    parity_check_dual = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    initial_llrs = [2.0, 2.0, 2.0]
    return NeuralBPBase(parity_check, parity_check_dual, initial_llrs, n_layers=3)


def test_activation_functional_and_inplace_paths_match() -> None:
    """Match the Julia utility test for the split activation helpers."""

    matrix = torch.tensor(
        [
            [0.2, -0.5, 1.0],
            [-1.3, 0.7, -0.1],
        ],
        dtype=torch.float32,
    )

    magnitudes_functional, signs_functional = safe_log_tanh_split(matrix)
    magnitudes_inplace = torch.empty_like(magnitudes_functional)
    signs_inplace = torch.empty_like(signs_functional)
    safe_log_tanh_split_(magnitudes_inplace, signs_inplace, matrix)

    assert torch.allclose(magnitudes_functional, magnitudes_inplace, atol=1e-6)
    assert torch.equal(signs_functional, signs_inplace)

    output_functional = safe_atanh_exp_signed(magnitudes_functional, signs_functional)
    output_inplace = torch.empty_like(output_functional)
    safe_atanh_exp_signed_(output_inplace, magnitudes_functional, signs_functional)

    assert torch.allclose(output_functional, output_inplace, atol=1e-6)


def test_c2v_to_v2c_functional_and_inplace_paths_match() -> None:
    """Match the Julia message-update equivalence test for C2V-to-V2C."""

    base = _build_test_base()
    messages_c2v_previous = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.5],
            [-0.4, 0.7],
            [0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    weights_llrs = torch.tensor([1.01, 0.99, 1.02], dtype=torch.float32)
    weights_messages = torch.tensor([1.0, 0.95], dtype=torch.float32)
    channel_llrs = torch.full((3, 2), 2.1972246, dtype=torch.float32)

    functional_magnitudes, functional_signs = c2v_to_v2c(
        messages_c2v_previous,
        weights_llrs,
        weights_messages,
        channel_llrs,
        base,
    )

    messages_v2c_inplace = torch.empty_like(functional_magnitudes)
    magnitudes_inplace = torch.empty_like(functional_magnitudes)
    signs_inplace = torch.empty_like(functional_signs)
    weighted_channel_llrs = torch.empty_like(channel_llrs)
    c2v_to_v2c_with_weights_(
        magnitudes_inplace,
        signs_inplace,
        messages_v2c_inplace,
        messages_c2v_previous,
        weighted_channel_llrs,
        weights_llrs,
        weights_messages,
        channel_llrs,
        base,
    )

    assert torch.allclose(functional_magnitudes, magnitudes_inplace, atol=1e-6)
    assert torch.equal(functional_signs, signs_inplace)


def test_v2c_to_c2v_functional_and_inplace_paths_match() -> None:
    """Match the Julia message-update equivalence test for V2C-to-C2V."""

    base = _build_test_base()
    activated_m_v2c_magnitudes = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.5],
            [-0.4, 0.7],
            [0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    activated_m_v2c_signs = torch.tensor(
        [
            [False, True],
            [True, False],
            [False, False],
            [True, True],
        ],
        dtype=torch.bool,
    )
    syndromes_batch = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )

    functional_messages = v2c_to_c2v(
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base,
    )

    messages_inplace = torch.empty_like(functional_messages)
    magnitudes_inplace = torch.empty_like(functional_messages)
    signs_inplace = torch.empty_like(activated_m_v2c_signs)
    v2c_to_c2v_(
        messages_inplace,
        magnitudes_inplace,
        signs_inplace,
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base,
    )

    assert torch.allclose(functional_messages, messages_inplace, atol=1e-6)


def test_readout_functional_and_inplace_paths_match() -> None:
    """Match the Julia readout equivalence test."""

    base = _build_test_base()
    messages_c2v = torch.tensor(
        [
            [0.1, -0.2],
            [0.3, 0.5],
            [-0.4, 0.7],
            [0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    weights_readout = torch.tensor([1.02, 0.97, 1.01, 0.99], dtype=torch.float32)
    weights_llrs = torch.tensor([1.01, 0.99, 1.02], dtype=torch.float32)
    channel_llrs = torch.full((3, 2), 2.1972246, dtype=torch.float32)

    functional_posterior = readout(
        messages_c2v,
        weights_readout,
        weights_llrs,
        channel_llrs,
        base,
    )

    posterior_inplace = torch.empty_like(functional_posterior)
    readout_with_weights_(
        posterior_inplace,
        messages_c2v,
        weights_readout,
        weights_llrs,
        channel_llrs,
        base,
    )

    assert torch.allclose(functional_posterior, posterior_inplace, atol=1e-6)

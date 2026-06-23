"""Tests for Phase 3 neural-BP loss functions."""

from __future__ import annotations

import math

import torch

from correlated_bp_decoder import (
    compute_additional_loss_from_ising_correlations,
    compute_loss_breakdown,
    compute_loss_including_correlations,
    compute_sine_residue_loss_from_llrs,
    linear_ramp_loss,
    sparsity_penalty,
    syndrome_loss_regularizer,
)


def _julia_loss_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixture values used by the Julia loss tests."""

    parity_check_dual = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    posterior_llrs = torch.tensor(
        [
            [2.0663514264498881],
            [1.0663514264498881],
            [-3.3280977282225512],
            [-2.1972245773362196],
            [-0.0663514264498881],
            [-0.06452172443644333],
            [2.1972245773362196],
            [1.0663514264498881],
        ],
        dtype=torch.float32,
    )
    expected_recoveries = torch.tensor(
        [[1], [0], [1], [1], [0], [0], [0], [0]],
        dtype=torch.bool,
    )
    return parity_check_dual, posterior_llrs, expected_recoveries


def test_sine_residue_loss_matches_julia_fixture() -> None:
    """Match the Julia loss test on the same fixture values."""

    parity_check_dual, posterior_llrs, expected_recoveries = _julia_loss_fixture()

    actual = compute_sine_residue_loss_from_llrs(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
    )

    manual = 0.0
    for row in range(parity_check_dual.shape[0]):
        commutation = 0.0
        for bit in range(parity_check_dual.shape[1]):
            if parity_check_dual[row, bit].item() == 0.0:
                continue
            probability = 1.0 / (1.0 + math.exp(posterior_llrs[bit, 0].item()))
            recovery = float(expected_recoveries[bit, 0].item())
            commutation += probability + recovery
        manual += abs(math.sin(math.pi * commutation / 2.0))

    assert torch.isclose(actual, torch.tensor(manual, dtype=torch.float32), atol=1e-6)
    assert torch.isclose(actual, torch.tensor(3.3923538, dtype=torch.float32), atol=1e-6)


def test_correlated_loss_matches_live_julia_formula_on_julia_fixture() -> None:
    """Use the Julia fixture values with the current live correlation formula.

    The prose in ``pavi/tests/test_loss.jl`` predates the current
    squared-difference penalty in ``pavi/src/loss.jl``. This test keeps the
    Julia fixture data but binds it to the live implementation that the
    Python port follows.
    """

    parity_check_dual, posterior_llrs, expected_recoveries = _julia_loss_fixture()
    connectivity = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int64)
    strengths = torch.full((4,), 0.5, dtype=torch.float32)

    correlation_penalty = compute_additional_loss_from_ising_correlations(
        posterior_llrs,
        connectivity,
        strengths,
    )
    base_loss = compute_sine_residue_loss_from_llrs(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
    )
    aggregate = compute_loss_including_correlations(
        posterior_llrs.unsqueeze(-1),
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=True,
        correlation_importance=1.0,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.0,
        sparsity_importance=0.0,
        warmup_loss_layers=0,
        aggregation="last_layer",
    )

    manual_correlation = 0.0
    for edge_index, edge in enumerate(connectivity):
        i = edge[0].item() - 1
        j = edge[1].item() - 1
        prob_i = 1.0 / (1.0 + math.exp(posterior_llrs[i, 0].item()))
        prob_j = 1.0 / (1.0 + math.exp(posterior_llrs[j, 0].item()))
        manual_correlation += strengths[edge_index].item() * (prob_i - prob_j) ** 2
    manual_correlation /= connectivity.shape[0]

    assert torch.isclose(
        correlation_penalty,
        torch.tensor(manual_correlation, dtype=torch.float32),
        atol=1e-6,
    )
    assert torch.isclose(
        aggregate,
        base_loss + correlation_penalty,
        atol=1e-6,
    )


def test_loss_breakdown_and_aggregate_loss_are_consistent() -> None:
    """Return a finite aggregate loss and aligned per-layer diagnostics."""

    posterior_llrs = torch.tensor(
        [
            [[0.0, 1.0], [0.5, 0.0]],
            [[-1.0, 0.5], [0.0, -0.5]],
        ],
        dtype=torch.float32,
    )
    expected_recoveries = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )
    parity_check_dual = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    connectivity = torch.tensor([[1, 2]], dtype=torch.int64)
    strengths = torch.tensor([0.2], dtype=torch.float32)

    breakdown = compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=True,
        correlation_importance=0.3,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.1,
        sparsity_importance=0.05,
        warmup_loss_layers=0,
        aggregation="linear_ramp",
    )
    aggregate = compute_loss_including_correlations(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=True,
        correlation_importance=0.3,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.1,
        sparsity_importance=0.05,
        warmup_loss_layers=0,
        aggregation="linear_ramp",
    )

    assert len(breakdown.per_layer) == 2
    assert torch.isfinite(breakdown.aggregate_loss)
    assert torch.isclose(breakdown.aggregate_loss, aggregate, atol=1e-6)

    layer_totals = torch.stack([layer.total_loss for layer in breakdown.per_layer])
    assert torch.isclose(
        breakdown.aggregate_loss,
        linear_ramp_loss(layer_totals),
        atol=1e-6,
    )


def test_loss_breakdown_matches_manual_per_layer_components() -> None:
    """Vectorized loss breakdown should match the original per-layer formulas."""

    posterior_llrs = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[-0.3, -0.1, 0.1], [0.2, -0.2, -0.4]],
        ],
        dtype=torch.float32,
    )
    expected_recoveries = torch.tensor(
        [
            [True, False],
            [False, True],
        ],
        dtype=torch.bool,
    )
    parity_check_dual = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    connectivity = torch.tensor([[1, 2]], dtype=torch.int64)
    strengths = torch.tensor([0.2], dtype=torch.float32)

    breakdown = compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=True,
        correlation_importance=0.3,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.1,
        sparsity_importance=0.05,
        warmup_loss_layers=0,
        aggregation="linear_ramp",
    )

    for layer_index, layer_breakdown in enumerate(breakdown.per_layer):
        post = posterior_llrs[:, :, layer_index]
        expected_base = compute_sine_residue_loss_from_llrs(
            post,
            expected_recoveries,
            parity_check_dual,
        )
        expected_llr = syndrome_loss_regularizer(post) * torch.tanh(
            torch.tensor(
                (layer_index + 1) / posterior_llrs.shape[2],
                dtype=post.dtype,
            )
        )
        expected_sparse = sparsity_penalty(post)
        expected_corr = compute_additional_loss_from_ising_correlations(
            post,
            connectivity,
            strengths,
        )
        expected_total = (
            expected_base
            + 0.1 * expected_llr
            + 0.3 * expected_corr
            + 0.05 * expected_sparse
        )

        assert torch.isclose(layer_breakdown.base_loss, expected_base, atol=1e-6)
        assert torch.isclose(layer_breakdown.llr_regularizer, expected_llr, atol=1e-6)
        assert torch.isclose(
            layer_breakdown.correlation_penalty,
            expected_corr,
            atol=1e-6,
        )
        assert torch.isclose(
            layer_breakdown.sparsity_penalty,
            expected_sparse,
            atol=1e-6,
        )
        assert torch.isclose(layer_breakdown.total_loss, expected_total, atol=1e-6)


def test_julia_compat_loss_preserves_global_layer_indexing() -> None:
    """Julia-compatible aggregation should keep the full layer axis."""

    posterior_llrs = torch.tensor(
        [
            [[0.0, 0.3, 0.6, 0.9]],
            [[-0.2, -0.4, -0.6, -0.8]],
        ],
        dtype=torch.float32,
    )
    expected_recoveries = torch.tensor([[True], [False]], dtype=torch.bool)
    parity_check_dual = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    connectivity = torch.zeros((0, 2), dtype=torch.int64)
    strengths = torch.zeros((0,), dtype=torch.float32)

    julia_breakdown = compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=False,
        correlation_importance=0.0,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.0,
        sparsity_importance=0.0,
        warmup_loss_layers=2,
        aggregation="linear_ramp",
        julia_loss_compat=True,
    )
    compressed_breakdown = compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        parity_check_dual,
        connectivity,
        strengths,
        is_correlated=False,
        correlation_importance=0.0,
        loss_layer_temperature=1.0,
        llr_certainty_importance=0.0,
        sparsity_importance=0.0,
        warmup_loss_layers=2,
        aggregation="linear_ramp",
        julia_loss_compat=False,
    )

    participating_totals = torch.stack(
        [layer.total_loss for layer in julia_breakdown.per_layer]
    )
    full_totals = torch.zeros((4,), dtype=participating_totals.dtype)
    full_totals[1:] = participating_totals

    assert len(julia_breakdown.per_layer) == 3
    assert torch.isclose(
        julia_breakdown.aggregate_loss,
        linear_ramp_loss(full_totals),
        atol=1e-6,
    )
    assert not torch.isclose(
        julia_breakdown.aggregate_loss,
        compressed_breakdown.aggregate_loss,
        atol=1e-6,
    )

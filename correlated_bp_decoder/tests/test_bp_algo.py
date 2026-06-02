"""Tests for Phase 3 training helpers."""

from __future__ import annotations

import numpy as np
import torch

from correlated_bp_decoder import (
    AnnealingSchedule,
    NachmaniNeuralBP,
    NeuralBPBase,
    TrainingConfig,
    compute_loss_hyperparameters,
    generate_training_data,
    run_bp,
    train_nachmani_neuralbp,
)


def _build_training_model() -> NachmaniNeuralBP:
    parity_check = [[1, 1, 0], [0, 1, 1]]
    parity_check_dual = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    initial_llrs = [2.1972246, 2.1972246, 2.1972246]
    base = NeuralBPBase(parity_check, parity_check_dual, initial_llrs, n_layers=2)
    return NachmaniNeuralBP(base)


def test_neural_bp_matches_standard_bp_with_unity_weights() -> None:
    """Port the Julia ``test_neural_BP`` regression with a fixed syndrome.

    The Julia test samples a random syndrome from a file-backed code. We keep
    the same all-ones-vs-standard-BP comparison but use a deterministic
    Hamming fixture so the translated pytest stays stable.
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
    final_llrs_standard_bp, _ = run_bp(
        parity_check,
        4,
        syndrome,
        initial_llrs.astype(np.float64),
        base.n_layers,
    )

    np.testing.assert_allclose(
        neural_output[:, 0, -1].detach().numpy(),
        final_llrs_standard_bp,
        atol=1e-6,
    )


def test_annealing_schedules_produce_expected_epoch_values() -> None:
    """Match the Julia annealing semantics for up/down schedules."""

    config = TrainingConfig(
        n_epochs=3,
        correlation_importance=AnnealingSchedule(
            maximum=1.0,
            minimum=0.1,
            decay=0.5,
            direction="down",
        ),
        llr_certainty_importance=AnnealingSchedule(
            maximum=1.0,
            minimum=0.5,
            decay=0.5,
            direction="up",
        ),
    )

    epoch_one = compute_loss_hyperparameters(1, config)
    epoch_two = compute_loss_hyperparameters(2, config)

    assert epoch_one.correlation_importance == 1.0
    assert epoch_two.correlation_importance == 0.5
    assert epoch_one.llr_certainty_importance == 0.5
    assert epoch_two.llr_certainty_importance == 0.75


def test_generate_training_data_returns_internal_layout() -> None:
    """Generate binary syndromes and recoveries with the expected shapes."""

    parity_check = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    syndromes, expected_recoveries = generate_training_data(parity_check, 5, 0.2)

    assert syndromes.shape == (2, 5)
    assert expected_recoveries.shape == (3, 5)
    assert syndromes.dtype == torch.bool
    assert expected_recoveries.dtype == torch.bool


def test_training_step_updates_weights_and_returns_epoch_summary() -> None:
    """Train on a tiny synthetic dataset and verify parameter movement."""

    torch.manual_seed(0)
    model = _build_training_model()
    initial_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    syndromes, expected_recoveries = generate_training_data(
        torch.tensor(model.base.parity_check_matrix.astype("float32")),
        8,
        0.15,
    )
    config = TrainingConfig(
        n_epochs=2,
        batch_size=4,
        learning_rate=5e-2,
        max_grad_norm=1.0,
        llr_certainty_importance=AnnealingSchedule(0.1, 0.1),
        sparsity_importance=AnnealingSchedule(0.05, 0.05),
    )

    summary = train_nachmani_neuralbp(
        model,
        syndromes,
        expected_recoveries,
        config,
    )

    assert len(summary.epochs) == 2
    assert all(epoch.applied_batches > 0 for epoch in summary.epochs)
    assert all(epoch.nan_skip_count == 0 for epoch in summary.epochs)
    assert all(epoch.rolled_back is False for epoch in summary.epochs)
    assert all(epoch.mean_loss is None or epoch.mean_loss >= 0.0 for epoch in summary.epochs)

    changed = False
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        if not torch.allclose(parameter.detach(), initial_weights[name]):
            changed = True
    assert changed, "Expected at least one trainable parameter to change."

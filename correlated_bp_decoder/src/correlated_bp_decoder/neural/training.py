"""Training helpers for the neural belief-propagation decoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from .losses import LossBreakdown, compute_loss_breakdown
from .nachmani import NachmaniNeuralBP


@dataclass(slots=True)
class AnnealingSchedule:
    """Schedule specification for a scalar training hyperparameter.

    Parameters
    ----------
    maximum
        Starting or target maximum value.
    minimum
        Floor or starting minimum value.
    decay
        Multiplicative decay applied once per epoch.
    direction
        Whether the schedule anneals ``"down"`` from ``maximum`` toward
        ``minimum`` or ``"up"`` from ``minimum`` toward ``maximum``.
    """

    maximum: float
    minimum: float
    decay: float = 1.0
    direction: Literal["down", "up"] = "down"


@dataclass(slots=True)
class LossHyperparameters:
    """Concrete per-epoch loss hyperparameters.

    Parameters
    ----------
    loss_layer_temperature
        Soft-min temperature or compatibility placeholder for linear ramp.
    correlation_importance
        Weight on the correlation penalty.
    llr_certainty_importance
        Weight on the confidence regularizer.
    sparsity_importance
        Weight on the sparsity penalty.
    """

    loss_layer_temperature: float
    correlation_importance: float
    llr_certainty_importance: float
    sparsity_importance: float


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for Phase 3 neural-BP training.

    Parameters
    ----------
    n_epochs
        Number of passes over the training dataset.
    batch_size
        Number of samples per optimization batch.
    learning_rate
        Base learning rate for Adam or AdamW.
    weight_decay
        Weight decay coefficient. When zero, plain Adam is used.
    max_grad_norm
        Norm threshold for gradient clipping.
    adam_eps
        Numerical epsilon passed to the optimizer.
    max_nan_skips_per_epoch
        Number of non-finite batches tolerated before rolling back the epoch.
    warmup_loss_layers
        Number of initial layers to ignore in the aggregate loss.
    aggregation
        Loss aggregation mode across unfolded layers.
    use_quadratic_residue
        Whether to use the quadratic-residue base loss instead of the active
        sine-residue loss.
    julia_loss_compat
        Whether to mirror the current Julia loss indexing/aggregation
        semantics exactly for apples-to-apples comparisons.
    batching_mode
        Batch-ordering strategy. ``"global_shuffle"`` applies a fresh epoch
        permutation before chunking into batches, while ``"julia_batch_local"``
        preserves Julia's current sequential batch partition with in-batch
        shuffling only.
    loss_layer_temperature
        Schedule for the soft-min temperature.
    correlation_importance
        Schedule for the correlation penalty weight.
    llr_certainty_importance
        Schedule for the confidence regularizer weight.
    sparsity_importance
        Schedule for the sparsity penalty weight.
    """

    n_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 5.0
    adam_eps: float = 1e-4
    max_nan_skips_per_epoch: int = 0
    warmup_loss_layers: int = 0
    aggregation: Literal["linear_ramp", "softmin", "last_layer"] = "linear_ramp"
    use_quadratic_residue: bool = False
    julia_loss_compat: bool = False
    batching_mode: Literal["global_shuffle", "julia_batch_local"] = "global_shuffle"
    loss_layer_temperature: AnnealingSchedule = field(
        default_factory=lambda: AnnealingSchedule(1.0, 1.0, 1.0, "down")
    )
    correlation_importance: AnnealingSchedule = field(
        default_factory=lambda: AnnealingSchedule(0.0, 0.0, 1.0, "down")
    )
    llr_certainty_importance: AnnealingSchedule = field(
        default_factory=lambda: AnnealingSchedule(0.0, 0.0, 1.0, "down")
    )
    sparsity_importance: AnnealingSchedule = field(
        default_factory=lambda: AnnealingSchedule(0.0, 0.0, 1.0, "down")
    )


@dataclass(slots=True)
class EpochTrainingSummary:
    """Summary statistics for one training epoch.

    Parameters
    ----------
    epoch
        One-based epoch index.
    mean_loss
        Mean loss over successfully applied batches, or ``None`` if none ran.
    applied_batches
        Number of optimizer steps performed in the epoch.
    nan_skip_count
        Number of batches skipped due to non-finite gradients or loss.
    rolled_back
        Whether the epoch ended with a rollback to the previous checkpoint.
    hyperparameters
        Concrete per-epoch loss hyperparameters.
    """

    epoch: int
    mean_loss: float | None
    applied_batches: int
    nan_skip_count: int
    rolled_back: bool
    hyperparameters: LossHyperparameters


@dataclass(slots=True)
class TrainingRunSummary:
    """Aggregate result of a training run.

    Parameters
    ----------
    epochs
        Per-epoch summaries in chronological order.
    """

    epochs: list[EpochTrainingSummary]


def compute_loss_hyperparameters(
    epoch: int,
    config: TrainingConfig,
) -> LossHyperparameters:
    """Compute concrete loss hyperparameters for one epoch.

    Parameters
    ----------
    epoch
        One-based epoch index.
    config
        Training configuration containing annealing schedules.

    Returns
    -------
    LossHyperparameters
        Concrete scalar weights for the epoch.
    """

    return LossHyperparameters(
        loss_layer_temperature=_schedule_value(config.loss_layer_temperature, epoch),
        correlation_importance=_schedule_value(config.correlation_importance, epoch),
        llr_certainty_importance=_schedule_value(
            config.llr_certainty_importance, epoch
        ),
        sparsity_importance=_schedule_value(config.sparsity_importance, epoch),
    )


def compute_loss_value(
    model: NachmaniNeuralBP,
    llrs_batch: torch.Tensor,
    syndromes_batch: torch.Tensor,
    expected_recoveries: torch.Tensor,
    hyperparameters: LossHyperparameters,
    *,
    warmup_loss_layers: int = 0,
    aggregation: str = "linear_ramp",
    use_quadratic_residue: bool = False,
    julia_loss_compat: bool = False,
) -> torch.Tensor:
    """Compute the total neural-BP loss for one batch.

    Parameters
    ----------
    model
        Neural BP model to evaluate.
    llrs_batch
        Channel LLR tensor with shape ``(n_bits, n_samples)``.
    syndromes_batch
        Syndrome tensor with shape ``(n_checks, n_samples)``.
    expected_recoveries
        Binary expected-recovery tensor with shape ``(n_bits, n_samples)``.
    hyperparameters
        Concrete scalar loss hyperparameters for the epoch.
    warmup_loss_layers
        Number of initial layers to skip in the aggregate loss.
    aggregation
        Loss aggregation mode across unfolded layers.
    use_quadratic_residue
        Whether to use the quadratic-residue base penalty.
    julia_loss_compat
        Whether to mirror the current Julia loss indexing/aggregation
        semantics exactly.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """

    posterior_llrs = model(llrs_batch, syndromes_batch)
    return compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        model.base.parity_check_matrix_dual,
        model.base.connectivity,
        model.base.correlation_strengths,
        is_correlated=model.base.is_correlated,
        correlation_importance=hyperparameters.correlation_importance,
        loss_layer_temperature=hyperparameters.loss_layer_temperature,
        llr_certainty_importance=hyperparameters.llr_certainty_importance,
        sparsity_importance=hyperparameters.sparsity_importance,
        warmup_loss_layers=warmup_loss_layers,
        aggregation=aggregation,
        use_quadratic_residue=use_quadratic_residue,
        julia_loss_compat=julia_loss_compat,
    ).aggregate_loss


def compute_loss_diagnostics(
    model: NachmaniNeuralBP,
    llrs_batch: torch.Tensor,
    syndromes_batch: torch.Tensor,
    expected_recoveries: torch.Tensor,
    hyperparameters: LossHyperparameters,
    *,
    warmup_loss_layers: int = 0,
    aggregation: str = "linear_ramp",
    use_quadratic_residue: bool = False,
    julia_loss_compat: bool = False,
) -> LossBreakdown:
    """Compute full loss diagnostics for one batch.

    Parameters
    ----------
    model
        Neural BP model to evaluate.
    llrs_batch
        Channel LLR tensor with shape ``(n_bits, n_samples)``.
    syndromes_batch
        Syndrome tensor with shape ``(n_checks, n_samples)``.
    expected_recoveries
        Binary expected-recovery tensor with shape ``(n_bits, n_samples)``.
    hyperparameters
        Concrete scalar loss hyperparameters for the epoch.
    warmup_loss_layers
        Number of initial layers to skip in the aggregate loss.
    aggregation
        Loss aggregation mode across unfolded layers.
    use_quadratic_residue
        Whether to use the quadratic-residue base penalty.
    julia_loss_compat
        Whether to mirror the current Julia loss indexing/aggregation
        semantics exactly.

    Returns
    -------
    LossBreakdown
        Aggregate loss and per-layer diagnostics.
    """

    posterior_llrs = model(llrs_batch, syndromes_batch)
    return compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        model.base.parity_check_matrix_dual,
        model.base.connectivity,
        model.base.correlation_strengths,
        is_correlated=model.base.is_correlated,
        correlation_importance=hyperparameters.correlation_importance,
        loss_layer_temperature=hyperparameters.loss_layer_temperature,
        llr_certainty_importance=hyperparameters.llr_certainty_importance,
        sparsity_importance=hyperparameters.sparsity_importance,
        warmup_loss_layers=warmup_loss_layers,
        aggregation=aggregation,
        use_quadratic_residue=use_quadratic_residue,
        julia_loss_compat=julia_loss_compat,
    )


def train_nachmani_neuralbp(
    model: NachmaniNeuralBP,
    syndromes: torch.Tensor | object,
    expected_recoveries: torch.Tensor | object,
    config: TrainingConfig,
    *,
    initial_llrs_batch: torch.Tensor | object | None = None,
) -> TrainingRunSummary:
    """Train the neural-BP model on an in-memory dataset.

    Parameters
    ----------
    model
        Neural BP model to train in place.
    syndromes
        Binary syndrome tensor with shape ``(n_checks, n_samples)``.
    expected_recoveries
        Binary expected-recovery tensor with shape ``(n_bits, n_samples)``.
    config
        Training configuration.
    initial_llrs_batch
        Optional channel LLR tensor with shape ``(n_bits, n_samples)``. When
        omitted, the model's base initial LLR prior is repeated across samples.

    Returns
    -------
    TrainingRunSummary
        Per-epoch training statistics.
    """

    syndromes_tensor, expected_recoveries_tensor, llrs_tensor = _prepare_training_tensors(
        model,
        syndromes,
        expected_recoveries,
        initial_llrs_batch,
    )
    optimizer = _build_optimizer(model, config)
    epoch_summaries: list[EpochTrainingSummary] = []

    for epoch in range(1, config.n_epochs + 1):
        hyperparameters = compute_loss_hyperparameters(epoch, config)
        checkpoint_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        batch_losses: list[float] = []
        nan_skip_count = 0
        applied_batches = 0
        rolled_back = False

        for batch_index, indices in enumerate(
            _iter_epoch_batches(
                syndromes_tensor.shape[1],
                config.batch_size,
                syndromes_tensor.device,
                config.batching_mode,
            ),
            start=1,
        ):
            syndromes_batch = syndromes_tensor.index_select(1, indices)
            expected_batch = expected_recoveries_tensor.index_select(1, indices)
            llrs_batch = llrs_tensor.index_select(1, indices)

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss_value(
                model,
                llrs_batch,
                syndromes_batch,
                expected_batch,
                hyperparameters,
                warmup_loss_layers=config.warmup_loss_layers,
                aggregation=config.aggregation,
                use_quadratic_residue=config.use_quadratic_residue,
                julia_loss_compat=config.julia_loss_compat,
            )

            if not torch.isfinite(loss):
                nan_skip_count += 1
                if nan_skip_count > config.max_nan_skips_per_epoch:
                    rolled_back = True
                    break
                continue

            loss.backward()
            if not _gradients_are_finite(model):
                optimizer.zero_grad(set_to_none=True)
                nan_skip_count += 1
                if nan_skip_count > config.max_nan_skips_per_epoch:
                    rolled_back = True
                    break
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            applied_batches += 1
            batch_losses.append(float(loss.detach().cpu()))

        if rolled_back:
            model.load_state_dict(checkpoint_state)
            optimizer.load_state_dict(optimizer_state)

        epoch_summaries.append(
            EpochTrainingSummary(
                epoch=epoch,
                mean_loss=(sum(batch_losses) / len(batch_losses)) if batch_losses else None,
                applied_batches=applied_batches,
                nan_skip_count=nan_skip_count,
                rolled_back=rolled_back,
                hyperparameters=hyperparameters,
            )
        )

    return TrainingRunSummary(epochs=epoch_summaries)


def generate_training_data(
    parity_check_matrix: torch.Tensor | object,
    n_samples: int,
    error_probability: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic binary training data for the decoder.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix with shape ``(n_checks, n_bits)``.
    n_samples
        Number of error patterns to generate.
    error_probability
        Independent per-bit error probability.
    device
        Optional output device.

    Returns
    -------
    tuple
        ``(syndromes, expected_recoveries)`` in the internal
        ``(features, samples)`` layout.
    """

    matrix = torch.as_tensor(
        parity_check_matrix,
        dtype=torch.float32,
        device=None if device is None else torch.device(device),
    )
    n_checks, n_bits = matrix.shape
    expected_recoveries = (
        torch.rand((n_bits, n_samples), device=matrix.device) < error_probability
    )
    syndromes = ((matrix @ expected_recoveries.to(matrix.dtype)) % 2).to(torch.bool)
    return syndromes, expected_recoveries


def _schedule_value(schedule: AnnealingSchedule, epoch: int) -> float:
    """Evaluate a scalar annealing schedule at a one-based epoch index."""

    if schedule.direction == "down":
        return max(schedule.minimum, schedule.maximum * schedule.decay ** (epoch - 1))
    return schedule.maximum - (
        schedule.maximum - schedule.minimum
    ) * schedule.decay ** (epoch - 1)


def _build_optimizer(
    model: NachmaniNeuralBP,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create the optimizer for the current training configuration."""

    if config.weight_decay > 0.0:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_eps,
    )


def _prepare_training_tensors(
    model: NachmaniNeuralBP,
    syndromes: torch.Tensor | object,
    expected_recoveries: torch.Tensor | object,
    initial_llrs_batch: torch.Tensor | object | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize in-memory training tensors to the model device."""

    syndromes_tensor = torch.as_tensor(
        syndromes,
        dtype=torch.bool,
        device=model.device,
    )
    expected_recoveries_tensor = torch.as_tensor(
        expected_recoveries,
        dtype=torch.bool,
        device=model.device,
    )
    if syndromes_tensor.ndim != 2 or syndromes_tensor.shape[0] != model.base.code_n_checks:
        raise ValueError("syndromes must have shape (n_checks, n_samples).")
    if (
        expected_recoveries_tensor.ndim != 2
        or expected_recoveries_tensor.shape[0] != model.base.code_n_bits
    ):
        raise ValueError("expected_recoveries must have shape (n_bits, n_samples).")
    if expected_recoveries_tensor.shape[1] != syndromes_tensor.shape[1]:
        raise ValueError("syndromes and expected_recoveries must share the sample axis.")

    if initial_llrs_batch is None:
        llrs_tensor = model.expand_initial_llrs(syndromes_tensor.shape[1])
    else:
        llrs_tensor = torch.as_tensor(
            initial_llrs_batch,
            dtype=torch.float32,
            device=model.device,
        )
        if llrs_tensor.ndim != 2 or llrs_tensor.shape != expected_recoveries_tensor.shape:
            raise ValueError("initial_llrs_batch must have shape (n_bits, n_samples).")

    return syndromes_tensor, expected_recoveries_tensor, llrs_tensor


def _iter_batch_indices(
    permutation: torch.Tensor,
    batch_size: int,
):
    """Yield shuffled batch-index tensors."""

    for start in range(0, permutation.shape[0], batch_size):
        yield permutation[start : start + batch_size]


def _iter_epoch_batches(
    n_samples: int,
    batch_size: int,
    device: torch.device,
    batching_mode: str,
):
    """Yield batch-index tensors according to the selected batching mode."""

    if batching_mode == "global_shuffle":
        permutation = torch.randperm(n_samples, device=device)
        yield from _iter_batch_indices(permutation, batch_size)
        return

    if batching_mode == "julia_batch_local":
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            indices = torch.arange(start, stop, device=device)
            local_permutation = torch.randperm(indices.shape[0], device=device)
            yield indices.index_select(0, local_permutation)
        return

    raise ValueError(f"Unsupported batching mode: {batching_mode!r}")


def _gradients_are_finite(model: NachmaniNeuralBP) -> bool:
    """Check that all present gradients on the model are finite."""

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True

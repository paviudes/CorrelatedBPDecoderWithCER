"""Loss functions for the neural belief-propagation decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..math_utils import binary_entropy_of_sigmoid, sigmoid


@dataclass(slots=True)
class LayerLossBreakdown:
    """Loss components for a single unfolded BP layer.

    Parameters
    ----------
    base_loss
        Syndrome-consistency penalty for the layer.
    llr_regularizer
        Confidence regularizer derived from binary entropy.
    correlation_penalty
        Pairwise correlation penalty for the layer.
    sparsity_penalty
        L1-style low-weight preference on the predicted error pattern.
    total_loss
        Weighted sum of the layer contributions.
    """

    base_loss: torch.Tensor
    llr_regularizer: torch.Tensor
    correlation_penalty: torch.Tensor
    sparsity_penalty: torch.Tensor
    total_loss: torch.Tensor


@dataclass(slots=True)
class LossBreakdown:
    """Aggregate loss and per-layer diagnostics.

    Parameters
    ----------
    aggregate_loss
        Final loss used for optimization.
    per_layer
        Loss components for each participating layer.
    """

    aggregate_loss: torch.Tensor
    per_layer: list[LayerLossBreakdown]


def sine_residue(x: torch.Tensor) -> torch.Tensor:
    """Compute the sine-residue penalty ``|sin(pi * x / 2)|``.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    torch.Tensor
        Elementwise sine-residue values.
    """

    return torch.abs(torch.sin(torch.pi * x * 0.5))


def quadratic_residue(x: torch.Tensor) -> torch.Tensor:
    """Compute the squared distance to the nearest even integer.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    torch.Tensor
        Elementwise quadratic-residue values.
    """

    return (x - 2.0 * torch.round(x * 0.5)) ** 2


def compute_sine_residue_loss_from_llrs(
    posterior_llrs: torch.Tensor,
    expected_recoveries: torch.Tensor,
    parity_check_matrix_dual: torch.Tensor | object,
) -> torch.Tensor:
    """Compute the active Julia sine-residue loss from posterior LLRs.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples)``.
    expected_recoveries
        Binary expected-recovery tensor of shape ``(n_bits, n_samples)``.
    parity_check_matrix_dual
        Dual parity-check matrix with shape ``(n_dual_checks, n_bits)``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """

    dual = _as_float_tensor(parity_check_matrix_dual, posterior_llrs)
    recoveries = expected_recoveries.to(dtype=posterior_llrs.dtype)
    e_total_matrix = sigmoid(posterior_llrs) + recoveries
    commutation_relations = dual @ e_total_matrix
    return sine_residue(commutation_relations).sum() / posterior_llrs.shape[1]


def compute_quadratic_residue_loss_from_llrs(
    posterior_llrs: torch.Tensor,
    expected_recoveries: torch.Tensor,
    parity_check_matrix_dual: torch.Tensor | object,
) -> torch.Tensor:
    """Compute the quadratic-residue variant of the decoder loss.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples)``.
    expected_recoveries
        Binary expected-recovery tensor of shape ``(n_bits, n_samples)``.
    parity_check_matrix_dual
        Dual parity-check matrix with shape ``(n_dual_checks, n_bits)``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """

    dual = _as_float_tensor(parity_check_matrix_dual, posterior_llrs)
    recoveries = expected_recoveries.to(dtype=posterior_llrs.dtype)
    e_total_matrix = sigmoid(posterior_llrs) + recoveries
    commutation_relations = dual @ e_total_matrix
    return quadratic_residue(commutation_relations).sum() / posterior_llrs.shape[1]


def syndrome_loss_regularizer(posterior_llrs: torch.Tensor) -> torch.Tensor:
    """Penalize uncertain posteriors via binary entropy.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples)``.

    Returns
    -------
    torch.Tensor
        Scalar regularizer value.
    """

    return binary_entropy_of_sigmoid(posterior_llrs).sum()


def compute_additional_loss_from_ising_correlations(
    posterior_llrs: torch.Tensor,
    connectivity: torch.Tensor | object,
    correlation_strengths: torch.Tensor | object,
) -> torch.Tensor:
    """Compute the Julia-style Ising correlation penalty.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples)``.
    connectivity
        1-based qubit-pair array with shape ``(n_edges, 2)``.
    correlation_strengths
        Correlation strengths aligned with ``connectivity``.

    Returns
    -------
    torch.Tensor
        Scalar correlation penalty averaged over samples and edges.
    """

    connectivity_tensor = torch.as_tensor(
        connectivity,
        dtype=torch.long,
        device=posterior_llrs.device,
    )
    if connectivity_tensor.numel() == 0:
        return posterior_llrs.new_zeros(())

    strengths = torch.as_tensor(
        correlation_strengths,
        dtype=posterior_llrs.dtype,
        device=posterior_llrs.device,
    ).reshape(-1, 1)
    zero_based = connectivity_tensor - 1
    probs = sigmoid(posterior_llrs)
    sigma_i = probs.index_select(0, zero_based[:, 0])
    sigma_j = probs.index_select(0, zero_based[:, 1])
    pairwise = strengths * (sigma_i - sigma_j) ** 2
    return pairwise.sum() / (posterior_llrs.shape[1] * zero_based.shape[0])


def softmin_loss(losses_per_layer: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute a stable soft-min aggregation across layers.

    Parameters
    ----------
    losses_per_layer
        One-dimensional tensor of per-layer losses.
    temperature
        Soft-min temperature.

    Returns
    -------
    torch.Tensor
        Aggregated loss scalar.
    """

    min_loss = losses_per_layer.min()
    return min_loss - temperature * torch.log(
        torch.exp(-(losses_per_layer - min_loss) / temperature).sum()
    )


def linear_ramp_loss(losses_per_layer: torch.Tensor) -> torch.Tensor:
    """Aggregate per-layer losses with the Julia late-layer weighting.

    Parameters
    ----------
    losses_per_layer
        One-dimensional tensor of per-layer losses.

    Returns
    -------
    torch.Tensor
        Aggregated loss scalar.
    """

    n_layers = losses_per_layer.shape[0]
    weights = torch.tanh(
        torch.arange(
            1,
            n_layers + 1,
            device=losses_per_layer.device,
            dtype=losses_per_layer.dtype,
        )
        / n_layers
    )
    return (losses_per_layer * weights).sum()


def last_layer_only_loss(losses_per_layer: torch.Tensor) -> torch.Tensor:
    """Return only the final layer loss.

    Parameters
    ----------
    losses_per_layer
        One-dimensional tensor of per-layer losses.

    Returns
    -------
    torch.Tensor
        Final layer loss.
    """

    return losses_per_layer[-1]


def sparsity_penalty(posterior_llrs: torch.Tensor) -> torch.Tensor:
    """Compute the Julia-style sparsity penalty on predicted errors.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples)``.

    Returns
    -------
    torch.Tensor
        Scalar sparsity penalty.
    """

    return sigmoid(posterior_llrs).sum() / posterior_llrs.shape[1]


def compute_loss_breakdown(
    posterior_llrs: torch.Tensor,
    expected_recoveries: torch.Tensor,
    parity_check_matrix_dual: torch.Tensor | object,
    connectivity: torch.Tensor | object,
    correlation_strengths: torch.Tensor | object,
    *,
    is_correlated: bool,
    correlation_importance: float,
    loss_layer_temperature: float,
    llr_certainty_importance: float,
    sparsity_importance: float,
    warmup_loss_layers: int = 0,
    aggregation: str = "linear_ramp",
    use_quadratic_residue: bool = False,
    julia_loss_compat: bool = False,
) -> LossBreakdown:
    """Compute the aggregate loss and its per-layer components.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples, n_layers)``.
    expected_recoveries
        Binary expected-recovery tensor of shape ``(n_bits, n_samples)``.
    parity_check_matrix_dual
        Dual parity-check matrix with shape ``(n_dual_checks, n_bits)``.
    connectivity
        1-based qubit-pair array with shape ``(n_edges, 2)``.
    correlation_strengths
        Correlation strengths aligned with ``connectivity``.
    is_correlated
        Whether to include the correlation penalty.
    correlation_importance
        Weight on the correlation penalty.
    loss_layer_temperature
        Temperature used by the soft-min aggregation mode.
    llr_certainty_importance
        Weight on the confidence regularizer.
    sparsity_importance
        Weight on the sparsity penalty.
    warmup_loss_layers
        Number of initial layers to skip in the aggregate loss.
    aggregation
        Aggregation mode: ``"linear_ramp"``, ``"softmin"``, or
        ``"last_layer"``.
    use_quadratic_residue
        Whether to use the quadratic-residue base penalty instead of the
        active sine-residue penalty.
    julia_loss_compat
        Whether to mirror the current Julia indexing and aggregation
        semantics exactly when excluding warmup layers.

    Returns
    -------
    LossBreakdown
        Aggregate loss and per-layer breakdown.
    """

    if posterior_llrs.ndim != 3:
        raise ValueError("posterior_llrs must have shape (n_bits, n_samples, n_layers).")
    n_layers = posterior_llrs.shape[2]
    if not 0 <= warmup_loss_layers < n_layers:
        raise ValueError("warmup_loss_layers must be smaller than the number of layers.")

    dual = _as_float_tensor(parity_check_matrix_dual, posterior_llrs)
    start_layer = warmup_loss_layers
    if julia_loss_compat:
        # Mirror the current Julia loop `for layer in warmup_loss_layers:n_layers`,
        # where the CLI value is treated as a 1-based first included layer index.
        start_layer = max(warmup_loss_layers - 1, 0)
    active_posteriors = posterior_llrs[:, :, start_layer:]
    active_layer_count = active_posteriors.shape[2]

    if active_layer_count == 0:
        raise ValueError("At least one loss layer must participate in the aggregate loss.")

    base_losses = _compute_base_losses_per_layer(
        active_posteriors,
        expected_recoveries,
        dual,
        use_quadratic_residue=use_quadratic_residue,
    )
    llr_regularizers = _compute_llr_regularizers_per_layer(
        active_posteriors,
        start_layer=start_layer,
        n_layers=n_layers,
    )
    sparsity_penalties = _compute_sparsity_penalties_per_layer(active_posteriors)
    if is_correlated:
        correlation_penalties = _compute_correlation_penalties_per_layer(
            active_posteriors,
            connectivity,
            correlation_strengths,
        )
    else:
        correlation_penalties = active_posteriors.new_zeros((active_layer_count,))

    total_losses = (
        base_losses
        + llr_certainty_importance * llr_regularizers
        + correlation_importance * correlation_penalties
        + sparsity_importance * sparsity_penalties
    )

    per_layer = [
        LayerLossBreakdown(
            base_loss=base_losses[layer_offset],
            llr_regularizer=llr_regularizers[layer_offset],
            correlation_penalty=correlation_penalties[layer_offset],
            sparsity_penalty=sparsity_penalties[layer_offset],
            total_loss=total_losses[layer_offset],
        )
        for layer_offset in range(active_layer_count)
    ]

    if julia_loss_compat:
        losses_per_layer = posterior_llrs.new_zeros((n_layers,))
        losses_per_layer[start_layer:] = total_losses
    else:
        losses_per_layer = total_losses
    if aggregation == "linear_ramp":
        aggregate_loss = linear_ramp_loss(losses_per_layer)
    elif aggregation == "softmin":
        aggregate_loss = softmin_loss(losses_per_layer, loss_layer_temperature)
    elif aggregation == "last_layer":
        aggregate_loss = last_layer_only_loss(losses_per_layer)
    else:
        raise ValueError(f"Unsupported loss aggregation mode: {aggregation!r}")

    return LossBreakdown(aggregate_loss=aggregate_loss, per_layer=per_layer)


def compute_loss_including_correlations(
    posterior_llrs: torch.Tensor,
    expected_recoveries: torch.Tensor,
    parity_check_matrix_dual: torch.Tensor | object,
    connectivity: torch.Tensor | object,
    correlation_strengths: torch.Tensor | object,
    *,
    is_correlated: bool,
    correlation_importance: float,
    loss_layer_temperature: float,
    llr_certainty_importance: float,
    sparsity_importance: float,
    warmup_loss_layers: int = 0,
    aggregation: str = "linear_ramp",
    use_quadratic_residue: bool = False,
) -> torch.Tensor:
    """Compute the aggregate neural-BP loss across all unfolded layers.

    Parameters
    ----------
    posterior_llrs
        Posterior LLR tensor of shape ``(n_bits, n_samples, n_layers)``.
    expected_recoveries
        Binary expected-recovery tensor of shape ``(n_bits, n_samples)``.
    parity_check_matrix_dual
        Dual parity-check matrix with shape ``(n_dual_checks, n_bits)``.
    connectivity
        1-based qubit-pair array with shape ``(n_edges, 2)``.
    correlation_strengths
        Correlation strengths aligned with ``connectivity``.
    is_correlated
        Whether to include the correlation penalty.
    correlation_importance
        Weight on the correlation penalty.
    loss_layer_temperature
        Temperature used by the soft-min aggregation mode.
    llr_certainty_importance
        Weight on the confidence regularizer.
    sparsity_importance
        Weight on the sparsity penalty.
    warmup_loss_layers
        Number of initial layers to skip in the aggregate loss.
    aggregation
        Aggregation mode: ``"linear_ramp"``, ``"softmin"``, or
        ``"last_layer"``.
    use_quadratic_residue
        Whether to use the quadratic-residue base penalty instead of the
        active sine-residue penalty.

    Returns
    -------
    torch.Tensor
        Scalar aggregate loss.
    """

    return compute_loss_breakdown(
        posterior_llrs,
        expected_recoveries,
        parity_check_matrix_dual,
        connectivity,
        correlation_strengths,
        is_correlated=is_correlated,
        correlation_importance=correlation_importance,
        loss_layer_temperature=loss_layer_temperature,
        llr_certainty_importance=llr_certainty_importance,
        sparsity_importance=sparsity_importance,
        warmup_loss_layers=warmup_loss_layers,
        aggregation=aggregation,
        use_quadratic_residue=use_quadratic_residue,
    ).aggregate_loss


def _as_float_tensor(
    value: torch.Tensor | object,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Convert an input to the reference tensor's dtype and device."""

    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _compute_base_losses_per_layer(
    posterior_llrs: torch.Tensor,
    expected_recoveries: torch.Tensor,
    parity_check_matrix_dual: torch.Tensor,
    *,
    use_quadratic_residue: bool,
) -> torch.Tensor:
    """Compute the base residue loss for every unfolded layer at once."""

    recoveries = expected_recoveries.to(dtype=posterior_llrs.dtype).unsqueeze(2)
    e_total_matrix = sigmoid(posterior_llrs) + recoveries
    commutation_relations = torch.matmul(
        parity_check_matrix_dual.unsqueeze(0),
        e_total_matrix.permute(2, 0, 1),
    )
    residue_fn = quadratic_residue if use_quadratic_residue else sine_residue
    return residue_fn(commutation_relations).sum(dim=(1, 2)) / posterior_llrs.shape[1]


def _compute_llr_regularizers_per_layer(
    posterior_llrs: torch.Tensor,
    *,
    start_layer: int,
    n_layers: int,
) -> torch.Tensor:
    """Compute the Julia-style confidence regularizer for every layer."""

    layer_indices = torch.arange(
        start_layer + 1,
        start_layer + posterior_llrs.shape[2] + 1,
        device=posterior_llrs.device,
        dtype=posterior_llrs.dtype,
    )
    layer_weights = torch.tanh(layer_indices / n_layers)
    entropy = binary_entropy_of_sigmoid(posterior_llrs).sum(dim=(0, 1))
    return entropy * layer_weights


def _compute_sparsity_penalties_per_layer(posterior_llrs: torch.Tensor) -> torch.Tensor:
    """Compute the sparsity penalty for every layer."""

    return sigmoid(posterior_llrs).sum(dim=(0, 1)) / posterior_llrs.shape[1]


def _compute_correlation_penalties_per_layer(
    posterior_llrs: torch.Tensor,
    connectivity: torch.Tensor | object,
    correlation_strengths: torch.Tensor | object,
) -> torch.Tensor:
    """Compute the CER pairwise penalty for every unfolded layer."""

    connectivity_tensor = torch.as_tensor(
        connectivity,
        dtype=torch.long,
        device=posterior_llrs.device,
    )
    if connectivity_tensor.numel() == 0:
        return posterior_llrs.new_zeros((posterior_llrs.shape[2],))

    strengths = torch.as_tensor(
        correlation_strengths,
        dtype=posterior_llrs.dtype,
        device=posterior_llrs.device,
    ).reshape(-1, 1, 1)
    zero_based = connectivity_tensor - 1
    probs = sigmoid(posterior_llrs)
    sigma_i = probs.index_select(0, zero_based[:, 0])
    sigma_j = probs.index_select(0, zero_based[:, 1])
    pairwise = strengths * (sigma_i - sigma_j) ** 2
    return pairwise.sum(dim=(0, 1)) / (posterior_llrs.shape[1] * zero_based.shape[0])

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
    per_layer: list[LayerLossBreakdown] = []

    base_loss_fn = (
        compute_quadratic_residue_loss_from_llrs
        if use_quadratic_residue
        else compute_sine_residue_loss_from_llrs
    )

    for layer_index in range(warmup_loss_layers, n_layers):
        post = posterior_llrs[:, :, layer_index]
        base_loss = base_loss_fn(post, expected_recoveries, dual)
        llr_reg = syndrome_loss_regularizer(post) * torch.tanh(
            torch.tensor(
                (layer_index + 1) / n_layers,
                dtype=post.dtype,
                device=post.device,
            )
        )
        sparse_pen = sparsity_penalty(post)
        if is_correlated:
            corr_pen = compute_additional_loss_from_ising_correlations(
                post,
                connectivity,
                correlation_strengths,
            )
        else:
            corr_pen = post.new_zeros(())
        total_loss = (
            base_loss
            + llr_certainty_importance * llr_reg
            + correlation_importance * corr_pen
            + sparsity_importance * sparse_pen
        )
        per_layer.append(
            LayerLossBreakdown(
                base_loss=base_loss,
                llr_regularizer=llr_reg,
                correlation_penalty=corr_pen,
                sparsity_penalty=sparse_pen,
                total_loss=total_loss,
            )
        )

    losses_per_layer = torch.stack([layer.total_loss for layer in per_layer])
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

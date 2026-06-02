"""Numerically stable tensor helpers for neural BP."""

from __future__ import annotations

import torch

EPSILON = torch.finfo(torch.float32).eps


def random_values_around_one(
    shape: tuple[int, ...] | list[int],
    *,
    scale: float = 0.01,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample values uniformly from a narrow interval around one.

    Parameters
    ----------
    shape
        Output tensor shape.
    scale
        Half-width of the interval around ``1.0``.
    device
        Optional output device.

    Returns
    -------
    torch.Tensor
        Tensor with dtype ``torch.float32``.
    """

    return 1.0 + scale * (2.0 * torch.rand(*shape, device=device) - 1.0)


def safe_log_tanh_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``log(tanh(x / 2))`` in magnitude/sign form.

    Parameters
    ----------
    x
        Real-valued tensor.

    Returns
    -------
    magnitudes, signs
        ``magnitudes`` stores ``log(tanh(abs(x) / 2))`` and ``signs`` stores
        the negative-input mask.
    """

    t = torch.tanh(torch.abs(x) * 0.5)
    t_clipped = torch.clamp(t, min=EPSILON, max=1.0 - EPSILON)
    magnitudes = torch.log(t_clipped)
    signs = x < 0.0
    return magnitudes, signs


def safe_log_tanh_split_(
    out_magnitudes: torch.Tensor,
    out_signs: torch.Tensor,
    x: torch.Tensor,
) -> None:
    """In-place version of :func:`safe_log_tanh_split`.

    Parameters
    ----------
    out_magnitudes
        Output tensor receiving log magnitudes.
    out_signs
        Output boolean tensor receiving sign bits.
    x
        Real-valued tensor.
    """

    magnitudes, signs = safe_log_tanh_split(x)
    out_magnitudes.copy_(magnitudes)
    out_signs.copy_(signs)


def safe_atanh_exp_signed(
    magnitudes: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Compute ``2 * atanh(exp(x))`` from split magnitude/sign data.

    Parameters
    ----------
    magnitudes
        Real tensor holding the log magnitudes.
    signs
        Boolean tensor holding the sign bits.

    Returns
    -------
    torch.Tensor
        Real-valued tensor after the stable inverse BP nonlinearity.
    """

    e = torch.exp(magnitudes)
    e_clipped = torch.clamp(e, min=EPSILON, max=1.0 - EPSILON)
    e_signed = torch.where(signs, -e_clipped, e_clipped)
    return 2.0 * torch.atanh(e_signed)


def safe_atanh_exp_signed_(
    out: torch.Tensor,
    magnitudes: torch.Tensor,
    signs: torch.Tensor,
) -> None:
    """In-place version of :func:`safe_atanh_exp_signed`.

    Parameters
    ----------
    out
        Output tensor receiving the inverse-nonlinearity result.
    magnitudes
        Real tensor holding the log magnitudes.
    signs
        Boolean tensor holding the sign bits.
    """

    out.copy_(safe_atanh_exp_signed(magnitudes, signs))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Apply the decoder's sigmoid convention.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    torch.Tensor
        ``1 / (1 + exp(x))`` evaluated elementwise.
    """

    return torch.reciprocal(1.0 + torch.exp(x))


def binary_entropy_of_sigmoid(mu: torch.Tensor) -> torch.Tensor:
    """Compute the binary entropy of ``sigmoid(mu)`` stably.

    Parameters
    ----------
    mu
        Tensor of log-likelihood ratios.

    Returns
    -------
    torch.Tensor
        Elementwise binary entropy.
    """

    softplus = torch.where(mu > 0, mu + torch.log1p(torch.exp(-mu)), torch.log1p(torch.exp(mu)))
    return softplus - (1.0 - sigmoid(mu)) * mu


def xor_affine(
    adjacency: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute ``(adjacency @ x + y) mod 2`` for boolean tensors.

    Parameters
    ----------
    adjacency
        Integer or boolean adjacency matrix of shape ``(m, n)``.
    x
        Boolean tensor of shape ``(n, batch)``.
    y
        Boolean tensor of shape ``(m, batch)``.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(m, batch)``.
    """

    adjacency_int = adjacency.to(dtype=torch.int64)
    x_int = x.to(dtype=torch.int64)
    parity = adjacency_int @ x_int
    return ((parity % 2) == 1) ^ y


def sparse_multiply(
    rows: torch.Tensor,
    cols: torch.Tensor,
    weights: torch.Tensor,
    x: torch.Tensor,
    *,
    n_rows: int,
) -> torch.Tensor:
    """Multiply an implicit weighted binary matrix by a dense tensor.

    Parameters
    ----------
    rows
        Row indices of the nonzero entries.
    cols
        Column indices of the nonzero entries.
    weights
        Weight associated with each nonzero entry.
    x
        Dense input tensor with shape ``(n_cols, batch)``.
    n_rows
        Number of output rows.

    Returns
    -------
    torch.Tensor
        Dense output tensor with shape ``(n_rows, batch)``.
    """

    out = torch.zeros(
        (n_rows, x.shape[1]),
        dtype=x.dtype,
        device=x.device,
    )
    sparse_multiply_(out, rows, cols, weights, x)
    return out


def sparse_multiply_(
    out: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    weights: torch.Tensor,
    x: torch.Tensor,
) -> None:
    """In-place weighted sparse-dense multiply using ``index_add_``.

    Parameters
    ----------
    out
        Output tensor to overwrite.
    rows
        Row indices of the nonzero entries.
    cols
        Column indices of the nonzero entries.
    weights
        Weight associated with each nonzero entry.
    x
        Dense input tensor with shape ``(n_cols, batch)``.
    """

    out.zero_()
    weighted_inputs = x.index_select(0, cols) * weights.unsqueeze(1)
    out.index_add_(0, rows, weighted_inputs)

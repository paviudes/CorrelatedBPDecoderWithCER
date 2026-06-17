"""Prediction helpers for the neural belief-propagation decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..tanner_graph import coerce_binary_matrix
from .nachmani import NachmaniNeuralBP


def predict_neuralbp(
    model: NachmaniNeuralBP,
    syndromes: torch.Tensor | object,
    *,
    batch_size: int = 1024,
    batch_first: bool = False,
    return_batch_first: bool | None = None,
) -> torch.Tensor:
    """Predict hard-decision recoveries for a batch of syndromes.

    Parameters
    ----------
    model
        Trained neural-BP model.
    syndromes
        Syndrome batch with shape ``(n_checks, n_samples)`` unless
        ``batch_first`` is ``True``, in which case ``(n_samples, n_checks)``
        is accepted.
    batch_size
        Number of samples to process per forward batch.
    batch_first
        Whether the provided syndromes use batch-first layout.
    return_batch_first
        Whether to return recoveries as ``(n_samples, n_bits, n_layers)``.
        When omitted, it defaults to the same value as ``batch_first``.

    Returns
    -------
    torch.Tensor
        Boolean tensor of predicted recoveries with shape
        ``(n_bits, n_samples, n_layers)`` by default.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    syndromes_tensor = _prepare_syndromes(model, syndromes, batch_first=batch_first)
    n_samples = syndromes_tensor.shape[1]
    predicted_recoveries = torch.empty(
        (model.base.code_n_bits, n_samples, model.base.n_layers),
        dtype=torch.bool,
        device=model.device,
    )

    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            chunk_syndromes = syndromes_tensor[:, start:stop]
            chunk_llrs = model.expand_initial_llrs(stop - start)
            chunk_posteriors = model(chunk_llrs, chunk_syndromes)
            predicted_recoveries[:, start:stop, :] = chunk_posteriors < 0

    if return_batch_first is None:
        return_batch_first = batch_first
    if return_batch_first:
        return predicted_recoveries.permute(1, 0, 2).contiguous()
    return predicted_recoveries


def check_bp_solutions(
    parity_check_matrix_dual: torch.Tensor | object,
    errors: torch.Tensor | object,
    proposed_recoveries: torch.Tensor | object,
    *,
    batch_first: bool = False,
) -> torch.Tensor:
    """Check whether any predicted layer exactly fixes each error pattern.

    Parameters
    ----------
    parity_check_matrix_dual
        Validation matrix with shape ``(n_checks_or_dual_rows, n_bits)``.
        Passing the dual matrix reproduces the strict exact-recovery check.
    errors
        Binary error patterns with shape ``(n_bits, n_samples)`` unless
        ``batch_first`` is ``True``.
    proposed_recoveries
        Predicted recoveries with shape ``(n_bits, n_samples, n_layers)``
        unless ``batch_first`` is ``True``.
    batch_first
        Whether ``errors`` and ``proposed_recoveries`` use batch-first layout.

    Returns
    -------
    torch.Tensor
        Boolean vector of length ``n_samples`` indicating which samples are
        corrected by at least one unfolded layer.
    """

    recoveries_tensor = _prepare_recoveries(proposed_recoveries, batch_first=batch_first)
    errors_tensor = _prepare_errors(
        errors,
        recoveries_tensor.shape[:2],
        batch_first=batch_first,
        device=recoveries_tensor.device,
    )
    use_float_validation = recoveries_tensor.device.type == "mps"
    validation_dtype = torch.float32 if use_float_validation else torch.int64
    validation_matrix = torch.as_tensor(
        parity_check_matrix_dual,
        dtype=validation_dtype,
        device=recoveries_tensor.device,
    )
    if (
        validation_matrix.ndim != 2
        or validation_matrix.shape[1] != recoveries_tensor.shape[0]
    ):
        raise ValueError(
            "parity_check_matrix_dual must have shape (n_rows, n_bits) aligned "
            "with the proposed recoveries."
        )

    residual_errors = torch.logical_xor(errors_tensor.unsqueeze(2), recoveries_tensor)
    residual_dtype = torch.float32 if use_float_validation else torch.int64
    parity_sums = (
        validation_matrix
        @ residual_errors.to(residual_dtype).reshape(validation_matrix.shape[1], -1)
    )
    if use_float_validation:
        parity_sums = torch.remainder(parity_sums, 2.0)
    else:
        parity_sums = parity_sums % 2
    layer_success = torch.all(parity_sums == 0, dim=0).reshape(
        recoveries_tensor.shape[1],
        recoveries_tensor.shape[2],
    )
    return layer_success.any(dim=1)


def predict_and_check_neuralbp(
    model: NachmaniNeuralBP,
    syndromes: torch.Tensor | object,
    errors: torch.Tensor | object,
    *,
    batch_size: int = 1024,
    batch_first: bool = False,
    validation_matrix: torch.Tensor | object | None = None,
) -> torch.Tensor:
    """Predict recoveries and check whether they fix each error pattern.

    Parameters
    ----------
    model
        Trained neural-BP model.
    syndromes
        Syndrome batch with shape ``(n_checks, n_samples)`` unless
        ``batch_first`` is ``True``.
    errors
        Error-pattern batch with shape ``(n_bits, n_samples)`` unless
        ``batch_first`` is ``True``.
    batch_size
        Number of samples to process per forward batch.
    batch_first
        Whether the provided arrays use batch-first layout.
    validation_matrix
        Optional matrix used to validate the residual errors. When omitted,
        the model's stored parity-check matrix is used to match the current
        Julia experiment path.

    Returns
    -------
    torch.Tensor
        Boolean vector of length ``n_samples``.
    """

    proposed_recoveries = predict_neuralbp(
        model,
        syndromes,
        batch_size=batch_size,
        batch_first=batch_first,
        return_batch_first=batch_first,
    )
    if validation_matrix is None:
        validation_matrix = model.base.parity_check_matrix
    return check_bp_solutions(
        validation_matrix,
        errors,
        proposed_recoveries,
        batch_first=batch_first,
    )


def neuralbp_test_predictions(
    model: NachmaniNeuralBP,
    test_errors_file: str | Path,
    *,
    batch_size: int = 4096,
    validation_matrix: torch.Tensor | object | None = None,
) -> torch.Tensor:
    """Evaluate the model on a file-backed matrix of explicit error patterns.

    Parameters
    ----------
    model
        Trained neural-BP model.
    test_errors_file
        Text file containing binary error patterns in the Julia column-major
        convention: shape ``(n_bits, n_samples)``.
    batch_size
        Number of samples to process per forward batch.
    validation_matrix
        Optional matrix used to validate residual errors. When omitted, the
        model's parity-check matrix is used to match the current Julia
        experiment path.

    Returns
    -------
    torch.Tensor
        Boolean vector indicating which samples were successfully decoded.
    """

    test_errors = _load_binary_matrix(test_errors_file)
    test_syndromes = np.mod(
        model.base.parity_check_matrix.astype(np.int64) @ test_errors,
        2,
    )
    return predict_and_check_neuralbp(
        model,
        test_syndromes,
        test_errors,
        batch_size=batch_size,
        validation_matrix=validation_matrix,
    )


def _prepare_syndromes(
    model: NachmaniNeuralBP,
    syndromes: torch.Tensor | object,
    *,
    batch_first: bool,
) -> torch.Tensor:
    """Normalize a syndrome batch to the internal ``(n_checks, n_samples)`` layout."""

    syndromes_tensor = torch.as_tensor(
        syndromes,
        dtype=torch.bool,
        device=model.device,
    )
    if batch_first:
        if syndromes_tensor.ndim != 2:
            raise ValueError("batch-first syndromes must have shape (n_samples, n_checks).")
        syndromes_tensor = syndromes_tensor.transpose(0, 1).contiguous()
    if syndromes_tensor.ndim != 2:
        raise ValueError("syndromes must have shape (n_checks, n_samples).")
    if syndromes_tensor.shape[0] != model.base.code_n_checks:
        raise ValueError(
            f"Expected {model.base.code_n_checks} syndrome rows, got "
            f"{syndromes_tensor.shape[0]}."
        )
    return syndromes_tensor


def _prepare_errors(
    errors: torch.Tensor | object,
    expected_shape: tuple[int, int],
    *,
    batch_first: bool,
    device: torch.device,
) -> torch.Tensor:
    """Normalize error patterns to the internal ``(n_bits, n_samples)`` layout."""

    errors_tensor = torch.as_tensor(errors, dtype=torch.bool)
    if batch_first:
        if errors_tensor.ndim != 2:
            raise ValueError("batch-first errors must have shape (n_samples, n_bits).")
        errors_tensor = errors_tensor.transpose(0, 1).contiguous()
    if errors_tensor.ndim != 2:
        raise ValueError("errors must have shape (n_bits, n_samples).")
    if errors_tensor.shape != expected_shape:
        raise ValueError(
            f"errors must have shape {expected_shape}, got {tuple(errors_tensor.shape)}."
        )
    return errors_tensor.to(device=device)


def _prepare_recoveries(
    proposed_recoveries: torch.Tensor | object,
    *,
    batch_first: bool,
) -> torch.Tensor:
    """Normalize recoveries to the internal ``(n_bits, n_samples, n_layers)`` layout."""

    recoveries_tensor = torch.as_tensor(proposed_recoveries, dtype=torch.bool)
    if batch_first:
        if recoveries_tensor.ndim != 3:
            raise ValueError(
                "batch-first proposed_recoveries must have shape "
                "(n_samples, n_bits, n_layers)."
            )
        recoveries_tensor = recoveries_tensor.permute(1, 0, 2).contiguous()
    if recoveries_tensor.ndim != 3:
        raise ValueError(
            "proposed_recoveries must have shape (n_bits, n_samples, n_layers)."
        )
    return recoveries_tensor


def _load_binary_matrix(path: str | Path) -> np.ndarray:
    """Load a file-backed binary matrix using the shared Julia text convention."""

    matrix = np.loadtxt(Path(path), dtype=np.int64, ndmin=2)
    return coerce_binary_matrix(matrix, name=f"matrix loaded from {path}")

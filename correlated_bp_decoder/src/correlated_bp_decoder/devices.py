"""Helpers for selecting and synchronizing torch devices."""

from __future__ import annotations

from typing import Any

import torch


def cuda_available() -> bool:
    """Return whether the active torch build can use CUDA."""

    return torch.cuda.is_available()


def mps_built() -> bool:
    """Return whether the active torch build includes MPS support."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def mps_available() -> bool:
    """Return whether the active torch build can use Apple's MPS backend."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def describe_torch_runtime() -> dict[str, Any]:
    """Summarize the local torch accelerator/runtime state."""

    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available(),
        "mps_built": mps_built(),
        "mps_available": mps_available(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def resolve_torch_device(requested: str | torch.device | None = "cpu") -> torch.device:
    """Resolve a user-facing device request into a concrete torch device.

    Parameters
    ----------
    requested
        Device request string. Supported values are ``"cpu"``, ``"mps"``,
        ``"cuda"``, and ``"auto"``.

    Returns
    -------
    torch.device
        Resolved device object.

    Raises
    ------
    RuntimeError
        If a specific accelerator was requested but is unavailable.
    ValueError
        If the request is not one of the supported device strings.
    """

    if isinstance(requested, torch.device):
        return requested

    request = "cpu" if requested is None else str(requested).strip().lower()
    if request == "auto":
        if cuda_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "Requested device 'cuda', but CUDA is unavailable in the active "
                f"torch runtime: {describe_torch_runtime()}"
            )
        return torch.device("cuda")
    if request == "mps":
        if not mps_built():
            raise RuntimeError(
                "Requested device 'mps', but this torch build was not compiled "
                f"with MPS support: {describe_torch_runtime()}"
            )
        if not mps_available():
            raise RuntimeError(
                "Requested device 'mps', but the MPS backend is unavailable in "
                f"the active runtime: {describe_torch_runtime()}"
            )
        return torch.device("mps")
    raise ValueError(
        f"Unsupported device request {requested!r}. Expected one of "
        "'cpu', 'mps', 'cuda', or 'auto'."
    )


def synchronize_torch_device(device: str | torch.device) -> None:
    """Synchronize pending work on a torch accelerator when needed."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        return
    if resolved.type == "mps":
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "synchronize"):
            mps_module.synchronize()

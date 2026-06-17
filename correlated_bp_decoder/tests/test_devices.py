"""Tests for torch device selection helpers."""

from __future__ import annotations

import pytest
import torch

from correlated_bp_decoder import (
    describe_torch_runtime,
    resolve_torch_device,
)
import correlated_bp_decoder.devices as devices


def test_resolve_torch_device_cpu_is_stable() -> None:
    """Always allow explicit CPU selection."""

    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_resolve_torch_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pick CPU when no accelerator backend is available."""

    monkeypatch.setattr(devices, "cuda_available", lambda: False)
    monkeypatch.setattr(devices, "mps_available", lambda: False)

    assert resolve_torch_device("auto") == torch.device("cpu")


def test_resolve_torch_device_rejects_unavailable_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a readable error when MPS is requested but unavailable."""

    monkeypatch.setattr(devices, "mps_built", lambda: True)
    monkeypatch.setattr(devices, "mps_available", lambda: False)

    with pytest.raises(RuntimeError, match="Requested device 'mps'"):
        resolve_torch_device("mps")


def test_describe_torch_runtime_has_expected_keys() -> None:
    """Return a lightweight runtime summary for logging/debugging."""

    info = describe_torch_runtime()

    assert "torch_version" in info
    assert "cuda_available" in info
    assert "mps_built" in info
    assert "mps_available" in info

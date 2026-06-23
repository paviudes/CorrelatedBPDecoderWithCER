"""Tests for torch device selection helpers."""

from __future__ import annotations

import pytest
import torch

from correlated_bp_decoder import (
    describe_torch_runtime,
    maybe_compile_torch_module,
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
    assert "torch_compile_available" in info
    assert "cuda_available" in info
    assert "mps_built" in info
    assert "mps_available" in info


def test_maybe_compile_torch_module_disabled_returns_original() -> None:
    """Leave the module untouched when compilation is disabled."""

    module = torch.nn.Linear(2, 2)

    assert maybe_compile_torch_module(module, enabled=False) is module


def test_maybe_compile_torch_module_forwards_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass through the requested compile settings when enabled."""

    captured: dict[str, object] = {}
    module = torch.nn.Linear(2, 2)
    sentinel = object()

    def fake_compile(target: object, **kwargs: object) -> object:
        captured["target"] = target
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled = maybe_compile_torch_module(
        module,
        enabled=True,
        backend="eager",
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=True,
    )

    assert compiled is sentinel
    assert captured["target"] is module
    assert captured["kwargs"] == {
        "backend": "eager",
        "mode": "reduce-overhead",
        "fullgraph": True,
        "dynamic": True,
    }

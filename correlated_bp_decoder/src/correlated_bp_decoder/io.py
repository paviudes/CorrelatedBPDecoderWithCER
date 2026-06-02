"""File-loading and checkpoint helpers for the Python decoder workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from .cer import parse_cer_data
from .neural.base import NeuralBPBase
from .neural.nachmani import NachmaniNeuralBP
from .tanner_graph import coerce_binary_matrix

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]

DEFAULT_ERROR_RATE = 0.1
DEFAULT_LLR = np.float32(np.log((1.0 - DEFAULT_ERROR_RATE) / DEFAULT_ERROR_RATE))
CHECKPOINT_SCHEMA_VERSION = 1


def load_binary_matrix(path: str | Path) -> IntArray:
    """Load a whitespace-delimited binary matrix from disk.

    Parameters
    ----------
    path
        Path to a text file containing rows of ``0`` and ``1`` values.

    Returns
    -------
    numpy.ndarray
        Two-dimensional binary integer matrix.
    """

    matrix = np.loadtxt(Path(path), dtype=np.int64, ndmin=2)
    return coerce_binary_matrix(matrix, name=f"matrix loaded from {path}")


def build_initial_llrs(
    n_bits: int,
    single_qubit_error_rates: dict[int, float],
    *,
    default_error_rate: float = DEFAULT_ERROR_RATE,
) -> FloatArray:
    """Build channel LLRs from 1-based qubit error-rate metadata.

    Parameters
    ----------
    n_bits
        Number of data bits in the code.
    single_qubit_error_rates
        Mapping from 1-based qubit indices to physical error probabilities.
    default_error_rate
        Fallback probability used for qubits absent from the mapping.

    Returns
    -------
    numpy.ndarray
        ``float32`` vector of length ``n_bits`` containing initial channel LLRs.
    """

    llrs = np.full(n_bits, _probability_to_llr(default_error_rate), dtype=np.float32)
    for qubit_index in range(1, n_bits + 1):
        if qubit_index in single_qubit_error_rates:
            llrs[qubit_index - 1] = _probability_to_llr(
                single_qubit_error_rates[qubit_index]
            )
    return llrs


def load_base_bp_model(
    parity_check_matrix_file: str | Path,
    logicals_file: str | Path,
    n_hidden_layers: int,
    *,
    correlation_strengths_file: str | Path | None = None,
) -> NeuralBPBase:
    """Load the Python equivalent of Pavi's Julia ``NeuralBPBase`` model.

    Parameters
    ----------
    parity_check_matrix_file
        File containing the parity-check matrix.
    logicals_file
        File containing logical-operator rows to append to the dual matrix.
    n_hidden_layers
        Number of unfolded BP iterations.
    correlation_strengths_file
        Optional CER-style file containing single-qubit rates and pairwise
        correlation weights.

    Returns
    -------
    NeuralBPBase
        Compiled neural BP base structure ready for later model layers.
    """

    parity_check_matrix = load_binary_matrix(parity_check_matrix_file)
    logicals = load_binary_matrix(logicals_file)
    dual_parity_check_matrix = np.vstack((parity_check_matrix, logicals))
    n_bits = parity_check_matrix.shape[1]

    if correlation_strengths_file is None or str(correlation_strengths_file) == "":
        connectivity = np.zeros((0, 2), dtype=np.int64)
        correlation_strengths = np.zeros((0,), dtype=np.float32)
        initial_llrs = np.full(n_bits, DEFAULT_LLR, dtype=np.float32)
    else:
        correlation_path = Path(correlation_strengths_file)
        if not correlation_path.is_file():
            raise FileNotFoundError(correlation_path)
        cer_data = parse_cer_data(correlation_path)
        connectivity = cer_data.connectivity
        correlation_strengths = cer_data.correlation_strengths
        initial_llrs = build_initial_llrs(
            n_bits,
            cer_data.single_qubit_error_rates,
        )

    return NeuralBPBase(
        parity_check_matrix,
        dual_parity_check_matrix,
        initial_llrs,
        n_hidden_layers,
        connectivity=connectivity,
        correlation_strengths=correlation_strengths,
    )


def load_trained_weights(path: str | Path) -> dict[str, FloatArray]:
    """Load trainable Neural-BP weights from a JSON checkpoint.

    Parameters
    ----------
    path
        Path to a JSON weight file. Both the new structured Python checkpoint
        format and the original Julia flat-weight format are accepted.

    Returns
    -------
    dict
        Dictionary with ``weights_c2v_v2c``, ``weights_llrs``, and
        ``weights_c2v_readout`` arrays as ``float32`` vectors.
    """

    checkpoint = _load_checkpoint_payload(path)
    weights_payload = checkpoint["weights"]
    return {
        name: np.asarray(weights_payload[name]["values"], dtype=np.float32).reshape(-1)
        for name in _weight_names()
    }


def load_trained_neuralbp_model(
    weights_file: str | Path,
    base_or_model: NeuralBPBase | NachmaniNeuralBP,
    *,
    device: torch.device | str | None = None,
) -> NachmaniNeuralBP:
    """Load a trained Nachmani model from disk.

    Parameters
    ----------
    weights_file
        Path to the JSON checkpoint or legacy Julia weights file.
    base_or_model
        Either the compiled :class:`NeuralBPBase` or an existing
        :class:`NachmaniNeuralBP` whose base structure should be reused.
    device
        Optional target device for the reconstructed model.

    Returns
    -------
    NachmaniNeuralBP
        Newly reconstructed model with the loaded trainable weights.
    """

    if isinstance(base_or_model, NachmaniNeuralBP):
        base = base_or_model.base
        inferred_device = base_or_model.device
    else:
        base = base_or_model
        inferred_device = torch.device("cpu")

    checkpoint = _load_checkpoint_payload(weights_file)
    weights = {
        name: torch.as_tensor(
            checkpoint["weights"][name]["values"],
            dtype=torch.float32,
        )
        for name in _weight_names()
    }
    _validate_checkpoint_against_base(checkpoint, base, weights_file)

    model = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights["weights_c2v_v2c"],
        weights_llrs=weights["weights_llrs"],
        weights_c2v_readout=weights["weights_c2v_readout"],
    )
    target_device = inferred_device if device is None else torch.device(device)
    return model.to(target_device)


def save_trained_neuralbp_model(
    weights_file: str | Path,
    model: NachmaniNeuralBP,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a trained Nachmani model to a structured JSON checkpoint.

    Parameters
    ----------
    weights_file
        Output JSON file.
    model
        Neural-BP model whose trainable weights should be serialized.
    metadata
        Optional JSON-serializable metadata to store alongside the checkpoint.

    Returns
    -------
    pathlib.Path
        Path to the written checkpoint file.
    """

    output_path = Path(weights_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": "NachmaniNeuralBP",
        "edge_ordering": (
            "row-major edge ordering from numpy.argwhere(parity_check_matrix == 1), "
            "matching Julia check-major traversal"
        ),
        "base_summary": {
            "n_layers": int(model.base.n_layers),
            "code_n_bits": int(model.base.code_n_bits),
            "code_n_checks": int(model.base.code_n_checks),
            "n_edges": int(model.base.n_edges),
            "nb_weights_c2v_v2c": int(model.base.nb_weights_c2v_v2c),
            "nb_weights_c2v_readout": int(model.base.nb_weights_c2v_readout),
        },
        "weights": {
            name: _serialize_weight_tensor(getattr(model, name))
            for name in _weight_names()
        },
        "metadata": {} if metadata is None else metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _probability_to_llr(probability: float) -> np.float32:
    """Convert a physical error probability into a channel LLR."""

    if not 0.0 < probability < 1.0:
        raise ValueError("Error probabilities must lie strictly between 0 and 1.")
    return np.float32(np.log((1.0 - probability) / probability))


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """Load either the structured Python checkpoint or the legacy Julia format."""

    raw_payload = json.loads(Path(path).read_text())
    if _is_structured_checkpoint(raw_payload):
        return raw_payload
    if _is_legacy_weight_payload(raw_payload):
        return {
            "schema_version": 0,
            "model_type": "NachmaniNeuralBP",
            "edge_ordering": "legacy Julia flat-weight checkpoint",
            "base_summary": {},
            "weights": {
                name: _serialize_weight_array(raw_payload[name])
                for name in _weight_names()
            },
            "metadata": {},
        }
    raise ValueError(
        "Unsupported checkpoint format. Expected a structured Python checkpoint "
        "or a legacy Julia flat-weight JSON file."
    )


def _validate_checkpoint_against_base(
    checkpoint: dict[str, Any],
    base: NeuralBPBase,
    weights_file: str | Path,
) -> None:
    """Validate that a checkpoint matches the compiled base structure."""

    base_summary = checkpoint.get("base_summary", {})
    expected_lengths = {
        "weights_c2v_v2c": base.nb_weights_c2v_v2c * base.n_layers,
        "weights_llrs": base.code_n_bits * base.n_layers,
        "weights_c2v_readout": base.nb_weights_c2v_readout,
    }
    for name, expected_length in expected_lengths.items():
        actual_length = int(checkpoint["weights"][name]["shape"][0])
        if actual_length != expected_length:
            raise ValueError(
                f"Checkpoint {weights_file} has {actual_length} values for {name}, "
                f"expected {expected_length} for the provided NeuralBPBase."
            )

    if "n_layers" in base_summary and int(base_summary["n_layers"]) != base.n_layers:
        raise ValueError(
            f"Checkpoint {weights_file} was saved for {base_summary['n_layers']} "
            f"layers, but the provided base uses {base.n_layers}."
        )
    if (
        "code_n_bits" in base_summary
        and int(base_summary["code_n_bits"]) != base.code_n_bits
    ):
        raise ValueError(
            f"Checkpoint {weights_file} was saved for {base_summary['code_n_bits']} "
            f"bits, but the provided base uses {base.code_n_bits}."
        )


def _serialize_weight_tensor(parameter: torch.Tensor) -> dict[str, Any]:
    """Serialize a parameter tensor to a stable JSON structure."""

    return _serialize_weight_array(parameter.detach().cpu().numpy())


def _serialize_weight_array(values: Any) -> dict[str, Any]:
    """Serialize a one-dimensional weight array to a JSON-friendly mapping."""

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "dtype": "float32",
        "shape": [int(array.shape[0])],
        "values": array.tolist(),
    }


def _is_structured_checkpoint(payload: Any) -> bool:
    """Return whether the payload matches the structured checkpoint schema."""

    return isinstance(payload, dict) and "weights" in payload and "schema_version" in payload


def _is_legacy_weight_payload(payload: Any) -> bool:
    """Return whether the payload matches the original Julia weight layout."""

    return isinstance(payload, dict) and all(name in payload for name in _weight_names())


def _weight_names() -> tuple[str, str, str]:
    """Return the canonical serialized Neural-BP weight names."""

    return ("weights_c2v_v2c", "weights_llrs", "weights_c2v_readout")

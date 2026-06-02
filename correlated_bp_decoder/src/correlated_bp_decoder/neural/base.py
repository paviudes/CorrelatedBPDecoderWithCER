"""Base graph compilation for the neural BP model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..tanner_graph import TannerGraph, coerce_binary_matrix

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class NeuralBPBase:
    """Compiled message-passing structure for an unfolded BP decoder.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix with shape ``(n_checks, n_bits)``.
    parity_check_matrix_dual
        Binary dual parity-check matrix used by later loss functions.
    initial_llrs
        Channel log-likelihood ratios with shape ``(n_bits,)``.
    n_layers
        Number of unfolded BP iterations.
    connectivity
        Optional CER connectivity array with shape ``(n_edges, 2)``. The
        qubit indices are preserved in the original 1-based file convention.
    correlation_strengths
        Optional correlation strengths aligned with ``connectivity``.

    Notes
    -----
    The edge ordering follows ``numpy.argwhere(parity_check_matrix == 1)``,
    which is row-major and therefore matches the Julia implementation's
    check-major loop order.
    """

    __slots__ = (
        "tanner_graph",
        "parity_check_matrix",
        "parity_check_matrix_dual",
        "initial_llrs",
        "n_layers",
        "connectivity",
        "correlation_strengths",
        "is_correlated",
        "code_n_checks",
        "code_n_bits",
        "n_edges",
        "edge_to_check_bit",
        "edge_to_checks",
        "edge_to_bits",
        "adj_initialize_v2c",
        "adj_v2c_c2v",
        "n_v2c_to_c2v_weights",
        "adj_c2v_v2c",
        "n_c2v_to_v2c_weights",
        "adj_c2v_readout",
        "n_c2v_to_readout_weights",
        "non_zero_rows_v2c_c2v",
        "non_zero_cols_v2c_c2v",
        "non_zero_rows_c2v_v2c",
        "non_zero_cols_c2v_v2c",
        "non_zero_rows_c2v_readout",
        "non_zero_cols_c2v_readout",
    )

    def __init__(
        self,
        parity_check_matrix: ArrayLike,
        parity_check_matrix_dual: ArrayLike,
        initial_llrs: ArrayLike,
        n_layers: int,
        *,
        connectivity: ArrayLike | None = None,
        correlation_strengths: ArrayLike | None = None,
    ) -> None:
        parity_check = coerce_binary_matrix(
            parity_check_matrix, name="parity_check_matrix"
        )
        dual_check = coerce_binary_matrix(
            parity_check_matrix_dual, name="parity_check_matrix_dual"
        )

        if dual_check.shape[1] != parity_check.shape[1]:
            raise ValueError(
                "parity_check_matrix_dual must have the same number of columns as "
                "parity_check_matrix."
            )

        n_checks, n_bits = parity_check.shape
        llrs = _coerce_initial_llrs(initial_llrs, n_bits)
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")

        connectivity_array = _coerce_connectivity(connectivity)
        correlation_array = _coerce_correlation_strengths(
            correlation_strengths, connectivity_array.shape[0]
        )

        edge_to_check_bit = _build_edge_to_check_bit(parity_check)
        edge_to_checks = edge_to_check_bit[:, 0]
        edge_to_bits = edge_to_check_bit[:, 1]

        adj_initialize_v2c = _build_initialize_adjacency(edge_to_bits, n_bits)
        adj_v2c_c2v = _build_v2c_to_c2v_adjacency(edge_to_checks, edge_to_bits)
        adj_c2v_v2c = _build_c2v_to_v2c_adjacency(edge_to_checks, edge_to_bits)
        adj_c2v_readout = _build_readout_adjacency(edge_to_bits, n_bits)

        self.tanner_graph = TannerGraph(parity_check)
        self.parity_check_matrix = parity_check.astype(np.bool_, copy=True)
        self.parity_check_matrix_dual = dual_check.astype(np.bool_, copy=True)
        self.initial_llrs = llrs
        self.n_layers = int(n_layers)
        self.connectivity = connectivity_array
        self.correlation_strengths = correlation_array
        self.is_correlated = connectivity_array.shape[0] > 0
        self.code_n_checks = n_checks
        self.code_n_bits = n_bits
        self.n_edges = int(edge_to_check_bit.shape[0])
        self.edge_to_check_bit = edge_to_check_bit
        self.edge_to_checks = edge_to_checks
        self.edge_to_bits = edge_to_bits
        self.adj_initialize_v2c = adj_initialize_v2c
        self.adj_v2c_c2v = adj_v2c_c2v
        self.n_v2c_to_c2v_weights = int(adj_v2c_c2v.sum())
        self.adj_c2v_v2c = adj_c2v_v2c
        self.n_c2v_to_v2c_weights = int(adj_c2v_v2c.sum())
        self.adj_c2v_readout = adj_c2v_readout
        self.n_c2v_to_readout_weights = int(adj_c2v_readout.sum())
        self.non_zero_rows_v2c_c2v, self.non_zero_cols_v2c_c2v = np.nonzero(
            adj_v2c_c2v
        )
        self.non_zero_rows_c2v_v2c, self.non_zero_cols_c2v_v2c = np.nonzero(
            adj_c2v_v2c
        )
        self.non_zero_rows_c2v_readout, self.non_zero_cols_c2v_readout = np.nonzero(
            adj_c2v_readout
        )

    @property
    def nb_neurons_per_layer(self) -> int:
        """Alias mirroring the Julia field name for the number of edge nodes."""

        return self.n_edges

    @property
    def nb_weights_v2c_c2v(self) -> int:
        """Alias for the number of V2C-to-C2V edge weights."""

        return self.n_v2c_to_c2v_weights

    @property
    def nb_weights_c2v_v2c(self) -> int:
        """Alias for the number of C2V-to-V2C edge weights."""

        return self.n_c2v_to_v2c_weights

    @property
    def nb_weights_c2v_readout(self) -> int:
        """Alias for the number of C2V-to-readout edge weights."""

        return self.n_c2v_to_readout_weights

    @property
    def neuron_to_check_variable(self) -> dict[int, tuple[int, int]]:
        """Return a 1-based Julia-style edge-to-``(check, bit)`` mapping."""

        mapping: dict[int, tuple[int, int]] = {}
        for edge_index, (check_index, bit_index) in enumerate(
            self.edge_to_check_bit, start=1
        ):
            mapping[edge_index] = (int(check_index) + 1, int(bit_index) + 1)
        return mapping


def _coerce_initial_llrs(initial_llrs: ArrayLike, n_bits: int) -> FloatArray:
    """Convert a channel-LLR vector into a flat ``float32`` array."""

    llrs = np.asarray(initial_llrs, dtype=np.float32)
    if llrs.ndim != 1:
        raise ValueError("initial_llrs must be a one-dimensional vector.")
    if llrs.shape[0] != n_bits:
        raise ValueError(
            f"initial_llrs must have length {n_bits}, got {llrs.shape[0]}."
        )
    return llrs.copy()


def _coerce_connectivity(connectivity: ArrayLike | None) -> IntArray:
    """Normalize an optional connectivity array to shape ``(n_edges, 2)``."""

    if connectivity is None:
        return np.zeros((0, 2), dtype=np.int64)

    array = np.asarray(connectivity, dtype=np.int64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("connectivity must have shape (n_edges, 2).")
    return array.copy()


def _coerce_correlation_strengths(
    correlation_strengths: ArrayLike | None,
    n_edges: int,
) -> FloatArray:
    """Normalize correlation strengths and validate their edge alignment."""

    if correlation_strengths is None:
        strengths = np.zeros((0,), dtype=np.float32)
    else:
        strengths = np.asarray(correlation_strengths, dtype=np.float32).reshape(-1)

    if strengths.shape[0] != n_edges:
        raise ValueError(
            "correlation_strengths must have the same length as connectivity."
        )
    return strengths.copy()


def _build_edge_to_check_bit(parity_check_matrix: IntArray) -> IntArray:
    """Build the row-major edge list for a parity-check matrix."""

    return np.argwhere(parity_check_matrix == 1).astype(np.int64, copy=False)


def _build_initialize_adjacency(edge_to_bits: IntArray, n_bits: int) -> BoolArray:
    """Build the input-to-V2C adjacency matrix."""

    adjacency = np.zeros((edge_to_bits.shape[0], n_bits), dtype=np.bool_)
    adjacency[np.arange(edge_to_bits.shape[0]), edge_to_bits] = True
    return adjacency


def _build_v2c_to_c2v_adjacency(
    edge_to_checks: IntArray,
    edge_to_bits: IntArray,
) -> BoolArray:
    """Build the adjacency from V2C messages to C2V messages."""

    same_check = edge_to_checks[:, None] == edge_to_checks[None, :]
    different_bit = edge_to_bits[:, None] != edge_to_bits[None, :]
    return same_check & different_bit


def _build_c2v_to_v2c_adjacency(
    edge_to_checks: IntArray,
    edge_to_bits: IntArray,
) -> BoolArray:
    """Build the adjacency from C2V messages to V2C messages."""

    same_bit = edge_to_bits[:, None] == edge_to_bits[None, :]
    different_check = edge_to_checks[:, None] != edge_to_checks[None, :]
    return same_bit & different_check


def _build_readout_adjacency(edge_to_bits: IntArray, n_bits: int) -> BoolArray:
    """Build the adjacency from C2V messages to bit readout nodes."""

    adjacency = np.zeros((n_bits, edge_to_bits.shape[0]), dtype=np.bool_)
    adjacency[edge_to_bits, np.arange(edge_to_bits.shape[0])] = True
    return adjacency

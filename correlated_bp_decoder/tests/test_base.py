"""Tests for the neural BP base graph compilation."""

from __future__ import annotations

import numpy as np

from correlated_bp_decoder import NeuralBPBase


def test_neural_bp_base_builds_expected_edge_order_and_adjacencies() -> None:
    """Compile the same row-major edge ordering used by the Julia source."""

    parity_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
    parity_check_dual = np.asarray(
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        dtype=np.int64,
    )
    initial_llrs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    base = NeuralBPBase(parity_check, parity_check_dual, initial_llrs, n_layers=3)

    assert base.code_n_checks == 2
    assert base.code_n_bits == 3
    assert base.n_edges == 4
    assert base.nb_neurons_per_layer == 4
    np.testing.assert_array_equal(
        base.edge_to_check_bit,
        np.asarray([[0, 0], [0, 1], [1, 1], [1, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        base.edge_to_checks,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        base.edge_to_bits,
        np.asarray([0, 1, 1, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        base.adj_initialize_v2c.astype(np.int64),
        np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        base.adj_v2c_c2v.astype(np.int64),
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        base.adj_c2v_v2c.astype(np.int64),
        np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        base.adj_c2v_readout.astype(np.int64),
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.int64,
        ),
    )
    assert base.nb_weights_v2c_c2v == 4
    assert base.nb_weights_c2v_v2c == 2
    assert base.nb_weights_c2v_readout == 4
    assert base.neuron_to_check_variable == {
        1: (1, 1),
        2: (1, 2),
        3: (2, 2),
        4: (2, 3),
    }


def test_neural_bp_base_normalizes_empty_connectivity_inputs() -> None:
    """Treat empty correlation inputs as the non-correlated base case."""

    parity_check = np.asarray([[1, 0, 1]], dtype=np.int64)
    initial_llrs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    base = NeuralBPBase(
        parity_check,
        parity_check,
        initial_llrs,
        n_layers=1,
        connectivity=np.zeros((0, 0), dtype=np.int64),
        correlation_strengths=np.zeros((0,), dtype=np.float32),
    )

    assert base.is_correlated is False
    assert base.connectivity.shape == (0, 2)

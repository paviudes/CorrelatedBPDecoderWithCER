"""Tests for Tanner-graph construction."""

from __future__ import annotations

import numpy as np
import pytest

from correlated_bp_decoder import TannerGraph


def test_tanner_graph_builds_neighbor_lists_from_parity_check_matrix() -> None:
    """Precompute check and variable neighbors from a tiny binary matrix."""

    parity_check = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
    graph = TannerGraph(parity_check)

    np.testing.assert_array_equal(graph.c2v, parity_check)
    np.testing.assert_array_equal(graph.v2c, parity_check.T)
    assert graph.n_checks == 2
    assert graph.n_bits == 3
    assert graph.nc == 2
    assert graph.nv == 3
    assert graph.soft_constraint_start == 2
    assert graph.check_neighbors == ((0, 1), (1, 2))
    assert graph.vertex_neighbors == ((0,), (0, 1), (1,))


def test_tanner_graph_validates_soft_constraint_boundary() -> None:
    """Reject soft-constraint boundaries outside the check range."""

    parity_check = np.asarray([[1, 0], [0, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="soft_constraint_start"):
        TannerGraph(parity_check, soft_constraint_start=3)

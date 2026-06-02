"""Tanner-graph helpers for parity-check matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]


def coerce_binary_matrix(matrix: ArrayLike, *, name: str = "matrix") -> IntArray:
    """Convert an array-like input into a 2D binary integer matrix.

    Parameters
    ----------
    matrix
        Array-like object containing only zeros and ones.
    name
        Human-readable name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``int64`` array containing only ``0`` and ``1``.

    Raises
    ------
    ValueError
        If the input is not two-dimensional or contains values other than
        ``0`` and ``1``.
    """

    array = np.asarray(matrix, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {array.shape}.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values.")
    return array.copy()


class TannerGraph:
    """Tanner-graph view of a parity-check matrix.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix with shape ``(n_checks, n_bits)``.
    soft_constraint_start
        Zero-based row index where soft constraints begin. The default places
        the boundary at ``n_checks``, meaning no soft constraints are present.

    Attributes
    ----------
    n_checks
        Number of check nodes.
    n_bits
        Number of variable nodes.
    v2c
        Variable-to-check incidence matrix, stored as the transpose of the
        parity-check matrix.
    c2v
        Check-to-variable incidence matrix, stored in the original orientation.
    soft_constraint_start
        Zero-based boundary between hard and soft parity constraints.
    check_neighbors
        For each check, the zero-based indices of adjacent variable nodes.
    vertex_neighbors
        For each variable node, the zero-based indices of adjacent checks.
    """

    __slots__ = (
        "n_checks",
        "n_bits",
        "v2c",
        "c2v",
        "soft_constraint_start",
        "check_neighbors",
        "vertex_neighbors",
    )

    def __init__(
        self,
        parity_check_matrix: ArrayLike,
        soft_constraint_start: int | None = None,
    ) -> None:
        matrix = coerce_binary_matrix(parity_check_matrix, name="parity_check_matrix")
        n_checks, n_bits = matrix.shape

        if soft_constraint_start is None:
            soft_constraint_start = n_checks
        if not 0 <= soft_constraint_start <= n_checks:
            raise ValueError(
                "soft_constraint_start must lie between 0 and the number of checks."
            )

        self.n_checks = n_checks
        self.n_bits = n_bits
        self.v2c = matrix.T.copy()
        self.c2v = matrix.copy()
        self.soft_constraint_start = soft_constraint_start
        self.check_neighbors = _neighbor_tuples(self.c2v)
        self.vertex_neighbors = _neighbor_tuples(self.v2c)

    @property
    def nc(self) -> int:
        """Alias mirroring the Julia field name for the number of checks."""

        return self.n_checks

    @property
    def nv(self) -> int:
        """Alias mirroring the Julia field name for the number of bits."""

        return self.n_bits


def _neighbor_tuples(matrix: IntArray) -> tuple[tuple[int, ...], ...]:
    """Return zero-based neighbor indices for each row of a binary matrix."""

    neighbors: list[tuple[int, ...]] = []
    for row in matrix:
        indices = tuple(int(index) for index in np.flatnonzero(row))
        neighbors.append(indices)
    return tuple(neighbors)

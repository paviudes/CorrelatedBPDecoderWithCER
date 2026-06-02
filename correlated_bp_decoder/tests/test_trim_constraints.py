"""Tests for the trim-constraints helper."""

from __future__ import annotations

import numpy as np

from correlated_bp_decoder import trim_constraints


def test_trim_constraints_matches_julia_example() -> None:
    """Match the Julia example, then apply the Python column-pruning step."""

    parity_check = np.asarray(
        [
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    syndrome = np.asarray([1, 0, 1, 0, 0], dtype=np.int64)

    updated_h, updated_syndrome, fixed_bits, fixed_bit_llrs = trim_constraints(
        parity_check,
        syndrome,
        5,
        4.0,
    )

    expected_full_h = np.asarray(
        [
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    active_columns = np.flatnonzero(np.any(expected_full_h != 0, axis=0))

    np.testing.assert_array_equal(active_columns, np.asarray([0, 5], dtype=np.int64))
    np.testing.assert_array_equal(
        updated_h,
        expected_full_h[:, active_columns],
    )
    np.testing.assert_array_equal(
        updated_syndrome,
        np.asarray([1, 0, 0, 1, 0], dtype=np.int64),
    )
    assert fixed_bits == [2, 4]
    np.testing.assert_allclose(
        fixed_bit_llrs,
        np.asarray([4.0, -4.0], dtype=np.float64),
    )

"""Tests for the classical BP baseline port."""

from __future__ import annotations

import numpy as np

from correlated_bp_decoder import run_bp


def test_classical_bp_matches_julia_regression_cases() -> None:
    """Reproduce the Julia Hamming-code LLR regression checks."""

    parity_check = np.asarray(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    initial_llrs = np.log((1.0 - 0.1) / 0.1) * np.ones(7, dtype=np.float64)

    final_llrs, n_iterations = run_bp(
        parity_check,
        4,
        np.asarray([1, 0, 1], dtype=np.int64),
        initial_llrs,
        1,
    )
    np.testing.assert_allclose(
        final_llrs,
        np.asarray(
            [
                1.0663514264498881,
                3.3280977282225512,
                2.19722457733622,
                1.0663514264498881,
                -0.06452172443644333,
                2.1972245773362196,
                1.0663514264498881,
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    assert n_iterations == 1

    final_llrs, n_iterations = run_bp(
        parity_check,
        4,
        np.asarray([1, 0, 0], dtype=np.int64),
        initial_llrs,
        2,
    )
    np.testing.assert_allclose(
        final_llrs,
        np.asarray(
            [
                2.9584134938935067,
                2.9584134938935067,
                3.4891275557835617,
                -0.29005600177239454,
                1.7230087983796607,
                1.7230087983796611,
                2.011992335845365,
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    assert n_iterations == 2

    final_llrs, n_iterations = run_bp(
        parity_check,
        4,
        np.asarray([0, 1, 0], dtype=np.int64),
        initial_llrs,
        2,
    )
    np.testing.assert_allclose(
        final_llrs,
        np.asarray(
            [
                2.9584134938935067,
                -0.29005600177239454,
                1.7230087983796607,
                2.9584134938935067,
                3.4891275557835617,
                1.7230087983796611,
                2.011992335845365,
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    assert n_iterations == 2

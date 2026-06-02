"""Tests for CER parsing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from correlated_bp_decoder import parse_cer_data

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def test_parse_cer_data_reads_rates_and_pairwise_edges() -> None:
    """Parse a mixed-format CER file into structured arrays and mappings."""

    cer_data = parse_cer_data(FIXTURE_DIR / "correlated_weights.txt")

    np.testing.assert_array_equal(
        cer_data.connectivity,
        np.asarray([[1, 3], [2, 1], [3, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        cer_data.correlation_strengths,
        np.asarray(
            [
                0.005851149427861176,
                0.0063269667097774155,
                0.006110318845201688,
            ],
            dtype=np.float32,
        ),
    )
    assert cer_data.single_qubit_error_rates == pytest.approx(
        {1: 0.1, 2: 0.2, 4: 0.05}
    )
    assert cer_data.is_correlated is True


def test_parse_cer_data_rejects_unknown_line_formats(tmp_path: Path) -> None:
    """Raise a readable error when a line does not match the CER grammar."""

    invalid_file = tmp_path / "invalid_cer.txt"
    invalid_file.write_text("1 : 0.1\nthis is not valid\n")

    with pytest.raises(ValueError, match="Unrecognized CER line"):
        parse_cer_data(invalid_file)

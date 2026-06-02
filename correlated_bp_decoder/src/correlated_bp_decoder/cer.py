"""CER-style parsing helpers for the Python decoder port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]

_FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SINGLE_QUBIT_PATTERN = re.compile(
    rf"^\s*(\d+)\s*:\s*({_FLOAT_PATTERN})\s*$"
)
_PAIR_PATTERN = re.compile(
    rf"^\s*\((\d+)\s*,\s*(\d+)\)\s*:\s*({_FLOAT_PATTERN})\s*$"
)


@dataclass(frozen=True, slots=True)
class CerData:
    """Parsed CER-style calibration data.

    Parameters
    ----------
    connectivity
        Two-column array of shape ``(n_edges, 2)`` storing the correlated
        qubit pairs exactly as they appear in the CER file. The indices remain
        1-based to match the Julia source data format.
    correlation_strengths
        Edge weights aligned with ``connectivity``.
    single_qubit_error_rates
        Mapping from 1-based qubit indices to physical error probabilities.
    """

    connectivity: IntArray
    correlation_strengths: FloatArray
    single_qubit_error_rates: dict[int, float]

    @property
    def is_correlated(self) -> bool:
        """Whether the parsed data contains any pairwise correlations."""

        return self.connectivity.shape[0] > 0


def parse_cer_data(correlation_strengths_file: str | Path) -> CerData:
    """Parse CER-style qubit rates and pairwise correlations from a file.

    Parameters
    ----------
    correlation_strengths_file
        Path to a text file containing a mixture of lines of the form
        ``qubit : probability`` and ``(qubit_i, qubit_j) : weight``.

    Returns
    -------
    CerData
        Parsed single-qubit rates and pairwise correlation metadata.

    Raises
    ------
    ValueError
        If any non-empty line does not match one of the supported formats.
    """

    path = Path(correlation_strengths_file)
    connectivity: list[tuple[int, int]] = []
    correlation_strengths: list[float] = []
    single_qubit_error_rates: dict[int, float] = {}

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        single_match = _SINGLE_QUBIT_PATTERN.fullmatch(line)
        if single_match is not None:
            qubit = int(single_match.group(1))
            error_rate = float(single_match.group(2))
            single_qubit_error_rates[qubit] = error_rate
            continue

        pair_match = _PAIR_PATTERN.fullmatch(line)
        if pair_match is not None:
            qubit_i = int(pair_match.group(1))
            qubit_j = int(pair_match.group(2))
            strength = float(pair_match.group(3))
            connectivity.append((qubit_i, qubit_j))
            correlation_strengths.append(strength)
            continue

        raise ValueError(
            f"Unrecognized CER line at {path}:{line_number}: {raw_line!r}"
        )

    if connectivity:
        connectivity_array = np.asarray(connectivity, dtype=np.int64)
    else:
        connectivity_array = np.zeros((0, 2), dtype=np.int64)

    strength_array = np.asarray(correlation_strengths, dtype=np.float32)

    return CerData(
        connectivity=connectivity_array,
        correlation_strengths=strength_array,
        single_qubit_error_rates=single_qubit_error_rates,
    )

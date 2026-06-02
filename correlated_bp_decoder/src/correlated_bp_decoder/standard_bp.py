"""Classical belief-propagation baseline and trim-constraint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tanner_graph import TannerGraph, coerce_binary_matrix

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class BPSettings:
    """Summary of a classical BP decoding run.

    Parameters
    ----------
    algo
        BP update rule. Supported values are ``"SumProduct"`` and ``"MinSum"``.
    n_iterations_of_bp
        Number of outer confidence-pruning iterations requested.
    rounds_per_bp
        Maximum number of BP rounds per outer iteration.
    llr_convergence_threshold
        Maximum allowed LLR change before the inner BP loop is considered
        converged.
    llr_confidence_threshold
        Absolute LLR threshold used to freeze confident bits.
    weight_soft_constraint
        Correlation strength used when soft constraints are present.
    current_iteration
        Total number of inner BP rounds applied.
    converged
        Whether the decode terminated early because all bits became confident.
    error
        Binary error pattern that was decoded.
    syndrome
        Binary syndrome corresponding to ``error``.
    initial_probabilities
        Channel probabilities used to initialize the decoder.
    final_probabilities
        Final per-bit probabilities for the decoded state ``0``.
    recovery_hard_decision
        Hard-decision recovery derived from the final LLRs.
    runtime
        Wall-clock runtime in seconds.
    is_decoder_failure
        Whether the residual error anti-commutes with any logical operator.
    """

    algo: str
    n_iterations_of_bp: int
    rounds_per_bp: int
    llr_convergence_threshold: float
    llr_confidence_threshold: float
    weight_soft_constraint: float
    current_iteration: int
    converged: bool
    error: IntArray
    syndrome: IntArray
    initial_probabilities: FloatArray
    final_probabilities: FloatArray
    recovery_hard_decision: IntArray
    runtime: float
    is_decoder_failure: bool


def compute_llrs_from_probabilities(probabilities: ArrayLike) -> FloatArray:
    """Convert probabilities into log-likelihood ratios.

    Parameters
    ----------
    probabilities
        Probability that each bit equals ``0``.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array of LLRs.
    """

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if np.any((probs <= 0.0) | (probs >= 1.0)):
        raise ValueError("Probabilities must lie strictly between 0 and 1.")
    return np.log(probs / (1.0 - probs))


def compute_probabilities_from_llrs(llrs: ArrayLike) -> FloatArray:
    """Convert log-likelihood ratios into probabilities for bit value ``0``.

    Parameters
    ----------
    llrs
        One-dimensional LLR array.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array of probabilities.
    """

    llr_array = np.asarray(llrs, dtype=np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-llr_array))


def get_recovery_using_hard_decision(
    llrs: ArrayLike,
    *,
    confidence: float = 0.0,
) -> IntArray:
    """Recover a binary pattern by thresholding the LLRs.

    Parameters
    ----------
    llrs
        One-dimensional LLR array.
    confidence
        Threshold above which a bit is decoded as ``0``. Values below the
        threshold are decoded as ``1``.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``int64`` recovery vector in ``{0, 1}``.
    """

    llr_array = np.asarray(llrs, dtype=np.float64).reshape(-1)
    return np.where(llr_array >= confidence, 0, 1).astype(np.int64)


def get_confident_bits(llrs: ArrayLike, confidence_threshold: float) -> list[int]:
    """Return one-based indices of bits whose LLR magnitude exceeds a threshold.

    Parameters
    ----------
    llrs
        One-dimensional LLR array.
    confidence_threshold
        Absolute LLR threshold for freezing a bit.

    Returns
    -------
    list of int
        One-based bit indices, matching the Julia baseline API.
    """

    llr_array = np.asarray(llrs, dtype=np.float64).reshape(-1)
    return [int(index) + 1 for index in np.flatnonzero(np.abs(llr_array) >= confidence_threshold)]


def trim_constraints(
    parity_check_matrix: ArrayLike,
    syndrome: ArrayLike,
    soft_constraint_start: int,
    confidence_threshold: float,
) -> tuple[IntArray, IntArray, list[int], FloatArray]:
    """Iteratively remove single-bit constraints from a parity-check matrix.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix.
    syndrome
        Binary syndrome vector aligned with the matrix rows.
    soft_constraint_start
        One-based row index where soft constraints begin, matching the Julia
        public API. Use ``n_checks + 1`` when no soft constraints are present.
    confidence_threshold
        LLR magnitude assigned to bits fixed by a single hard constraint.

    Returns
    -------
    tuple
        ``(updated_h, updated_syndrome, fixed_bits, fixed_bit_llrs)`` where
        ``fixed_bits`` are one-based column positions in the input matrix.
    """

    matrix = coerce_binary_matrix(parity_check_matrix, name="parity_check_matrix")
    syndrome_array = _coerce_binary_vector(syndrome, matrix.shape[0], name="syndrome")
    soft_start_zero = _normalize_soft_constraint_start(soft_constraint_start, matrix.shape[0])

    updated_h = matrix.copy()
    updated_syndrome = syndrome_array.copy()
    fixed_bits: list[int] = []
    llrs_for_fixed_bits: list[float] = []

    for row in range(soft_start_zero, updated_h.shape[0]):
        if np.count_nonzero(updated_h[row]) == 1:
            updated_h[row, :] = 0
            updated_syndrome[row] = 0

    while True:
        single_one_rows = [
            row
            for row in range(soft_start_zero)
            if np.count_nonzero(updated_h[row]) == 1
        ]
        if not single_one_rows:
            break

        seen_bits: set[int] = set()
        row_bit_pairs: list[tuple[int, int]] = []
        for row in single_one_rows:
            bit_zero = int(np.flatnonzero(updated_h[row])[0])
            if bit_zero not in seen_bits:
                seen_bits.add(bit_zero)
                row_bit_pairs.append((row, bit_zero))

        for row, bit_zero in row_bit_pairs:
            fixed_bits.append(bit_zero + 1)
            if updated_syndrome[row] == 0:
                llrs_for_fixed_bits.append(float(confidence_threshold))
            else:
                llrs_for_fixed_bits.append(float(-confidence_threshold))

            for other_row in range(updated_h.shape[0]):
                if other_row == row or updated_h[other_row, bit_zero] == 0:
                    continue
                updated_syndrome[other_row] = (
                    updated_syndrome[other_row] + updated_syndrome[row]
                ) % 2
                if other_row >= soft_start_zero:
                    updated_h[other_row, :] = 0
                    updated_syndrome[other_row] = 0

            updated_h[:, bit_zero] = 0
            updated_syndrome[row] = 0

    non_zero_columns = np.flatnonzero(np.any(updated_h != 0, axis=0))
    updated_h = updated_h[:, non_zero_columns]
    return (
        updated_h,
        updated_syndrome,
        fixed_bits,
        np.asarray(llrs_for_fixed_bits, dtype=np.float64),
    )


def run_bp(
    parity_check_matrix: ArrayLike,
    soft_constraint_start: int,
    syndrome: ArrayLike,
    initial_llrs: ArrayLike,
    max_iterations: int,
    *,
    algo: str = "SumProduct",
    llr_convergence_threshold: float = 1e-6,
    llr_confidence_threshold: float = 2.0,
    weight_soft_constraint: float = 0.5,
) -> tuple[FloatArray, int]:
    """Run classical belief propagation on a binary parity-check matrix.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix.
    soft_constraint_start
        One-based row index where soft constraints begin, matching Julia.
    syndrome
        Binary syndrome vector.
    initial_llrs
        Initial per-bit channel LLRs.
    max_iterations
        Maximum number of BP rounds to run.
    algo
        BP update rule. Supported values are ``"SumProduct"`` and ``"MinSum"``.
    llr_convergence_threshold
        Inner-loop stopping threshold on the maximum LLR change.
    llr_confidence_threshold
        Absolute LLR threshold for declaring all bits confident.
    weight_soft_constraint
        Correlation strength for soft constraints.

    Returns
    -------
    tuple
        ``(final_llrs, n_iterations)``.
    """

    _validate_algorithm(algo)
    matrix = coerce_binary_matrix(parity_check_matrix, name="parity_check_matrix")
    syndrome_array = _coerce_binary_vector(syndrome, matrix.shape[0], name="syndrome")
    llrs = _coerce_float_vector(initial_llrs, matrix.shape[1], name="initial_llrs")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative.")
    soft_start_zero = _normalize_soft_constraint_start(soft_constraint_start, matrix.shape[0])

    non_trivial_rows = np.flatnonzero(np.count_nonzero(matrix, axis=1) > 0)
    n_hard_constraints = int(
        np.count_nonzero(np.count_nonzero(matrix[:soft_start_zero, :], axis=1) > 0)
    )
    matrix_non_trivial = matrix[non_trivial_rows, :]
    syndrome_non_trivial = syndrome_array[non_trivial_rows]

    graph = TannerGraph(matrix_non_trivial, soft_constraint_start=n_hard_constraints)
    messages_v2c = bp_initialize(graph, llrs)
    current_llrs = llrs.copy()
    current_iteration = 1
    stop = False

    while not stop and current_iteration <= max_iterations:
        messages_v2c_updated, new_llrs = bp_round(
            graph,
            llrs,
            syndrome_non_trivial,
            messages_v2c,
            weight_soft_constraint,
            algo,
        )
        stop = should_bp_stop(
            graph,
            current_llrs,
            new_llrs,
            syndrome_non_trivial,
            llr_convergence_threshold,
            llr_confidence_threshold,
        )
        current_llrs = new_llrs
        messages_v2c = messages_v2c_updated
        current_iteration += 1

    return current_llrs, current_iteration - 1


def classical_belief_propagation_decoder(
    parity_check_matrix: ArrayLike,
    logical_operators: ArrayLike,
    error: ArrayLike,
    syndrome: ArrayLike,
    initial_probabilities: ArrayLike,
    rounds_per_bp: int,
    n_iterations_of_bp: int,
    *,
    algo: str = "SumProduct",
    llr_convergence_threshold: float = 1e-6,
    llr_confidence_threshold: float = 2.0,
    weight_soft_constraint: float = 0.5,
    soft_constraint_start: int | None = None,
) -> BPSettings:
    """Run the Julia-style outer BP loop with confidence-based pruning.

    Parameters
    ----------
    parity_check_matrix
        Binary parity-check matrix.
    logical_operators
        Binary logical-operator matrix used to detect decoder failure.
    error
        Binary error pattern being decoded.
    syndrome
        Binary syndrome vector corresponding to ``error``.
    initial_probabilities
        Probability that each bit is initially ``0``.
    rounds_per_bp
        Maximum BP rounds per outer iteration.
    n_iterations_of_bp
        Number of outer confidence-pruning iterations.
    algo
        BP update rule. Supported values are ``"SumProduct"`` and ``"MinSum"``.
    llr_convergence_threshold
        Inner-loop stopping threshold on the maximum LLR change.
    llr_confidence_threshold
        Absolute LLR threshold for freezing bits.
    weight_soft_constraint
        Correlation strength for soft constraints.
    soft_constraint_start
        One-based row index where soft constraints begin. When omitted, the
        matrix is treated as hard constraints only.

    Returns
    -------
    BPSettings
        Summary of the decoding run.
    """

    _validate_algorithm(algo)
    matrix = coerce_binary_matrix(parity_check_matrix, name="parity_check_matrix")
    logicals = coerce_binary_matrix(logical_operators, name="logical_operators")
    error_array = _coerce_binary_vector(error, matrix.shape[1], name="error")
    syndrome_array = _coerce_binary_vector(syndrome, matrix.shape[0], name="syndrome")
    initial_probabilities_array = _coerce_probability_vector(
        initial_probabilities,
        matrix.shape[1],
        name="initial_probabilities",
    )
    if soft_constraint_start is None:
        soft_constraint_start = matrix.shape[0] + 1

    start_time = perf_counter()
    n_iterations_total = 0
    llrs = compute_llrs_from_probabilities(initial_probabilities_array)
    converged = False

    for _ in range(n_iterations_of_bp):
        frozen_bits = get_confident_bits(llrs, llr_confidence_threshold)
        if len(frozen_bits) == llrs.shape[0]:
            converged = True
            break

        if not frozen_bits:
            syndrome_contribution_frozen = np.zeros(matrix.shape[0], dtype=np.int64)
        else:
            frozen_zero = np.asarray(frozen_bits, dtype=np.int64) - 1
            h_f = matrix[:, frozen_zero]
            recovery_f = get_recovery_using_hard_decision(llrs[frozen_zero])
            syndrome_contribution_frozen = (h_f @ recovery_f) % 2

        uncertain_bits = [
            bit for bit in range(1, llrs.shape[0] + 1) if bit not in set(frozen_bits)
        ]
        uncertain_zero = np.asarray(uncertain_bits, dtype=np.int64) - 1
        h_u = matrix[:, uncertain_zero]
        s_u = (syndrome_array + syndrome_contribution_frozen) % 2

        updated_h_u, updated_free_syndrome, fixed_bit_locations, fixed_bit_llrs = (
            trim_constraints(
                h_u,
                s_u,
                soft_constraint_start,
                llr_confidence_threshold,
            )
        )
        if fixed_bit_locations:
            fixed_zero = np.asarray(fixed_bit_locations, dtype=np.int64) - 1
            llrs[uncertain_zero[fixed_zero]] = fixed_bit_llrs
            updated_uncertain_bits = np.delete(
                np.asarray(uncertain_bits, dtype=np.int64),
                fixed_zero,
            )
        else:
            updated_uncertain_bits = np.asarray(uncertain_bits, dtype=np.int64)

        if updated_uncertain_bits.size > 0:
            new_llrs_u, current_iteration = run_bp(
                updated_h_u,
                soft_constraint_start,
                updated_free_syndrome,
                llrs[updated_uncertain_bits - 1],
                rounds_per_bp,
                algo=algo,
                llr_convergence_threshold=llr_convergence_threshold,
                llr_confidence_threshold=llr_confidence_threshold,
                weight_soft_constraint=weight_soft_constraint,
            )
            llrs[updated_uncertain_bits - 1] = new_llrs_u
        else:
            current_iteration = 0
        n_iterations_total += current_iteration

    final_probabilities = compute_probabilities_from_llrs(llrs)
    recovery_hard_decision = get_recovery_using_hard_decision(llrs)
    runtime = perf_counter() - start_time

    return BPSettings(
        algo=algo,
        n_iterations_of_bp=n_iterations_of_bp,
        rounds_per_bp=rounds_per_bp,
        llr_convergence_threshold=llr_convergence_threshold,
        llr_confidence_threshold=llr_confidence_threshold,
        weight_soft_constraint=weight_soft_constraint,
        current_iteration=n_iterations_total,
        converged=converged,
        error=error_array,
        syndrome=syndrome_array,
        initial_probabilities=initial_probabilities_array,
        final_probabilities=final_probabilities,
        recovery_hard_decision=recovery_hard_decision,
        runtime=runtime,
        is_decoder_failure=is_decoder_failure(
            error_array,
            recovery_hard_decision,
            logicals,
        ),
    )


def is_decoder_failure(
    error: ArrayLike,
    recovery: ArrayLike,
    logical_operators: ArrayLike,
) -> bool:
    """Check whether the residual error triggers any logical operator.

    Parameters
    ----------
    error
        Binary error pattern.
    recovery
        Binary recovery pattern.
    logical_operators
        Binary logical-operator matrix.

    Returns
    -------
    bool
        ``True`` if the residual error anti-commutes with any logical operator.
    """

    logicals = coerce_binary_matrix(logical_operators, name="logical_operators")
    error_array = _coerce_binary_vector(error, logicals.shape[1], name="error")
    recovery_array = _coerce_binary_vector(recovery, logicals.shape[1], name="recovery")
    residual_error = (error_array + recovery_array) % 2
    conjugate_components = (logicals @ residual_error) % 2
    return bool(np.any(conjugate_components == 1))


def bp_initialize(graph: TannerGraph, llrs: ArrayLike) -> FloatArray:
    """Initialize V2C messages from the channel LLRs.

    Parameters
    ----------
    graph
        Tanner graph for the parity-check matrix.
    llrs
        One-dimensional array of initial channel LLRs.

    Returns
    -------
    numpy.ndarray
        Dense ``(n_bits, n_checks)`` V2C message matrix.
    """

    llr_array = _coerce_float_vector(llrs, graph.n_bits, name="llrs")
    messages_v2c = np.zeros((graph.n_bits, graph.n_checks), dtype=np.float64)
    for bit in range(graph.n_bits):
        for check in graph.vertex_neighbors[bit]:
            messages_v2c[bit, check] = llr_array[bit]
    return messages_v2c


def bp_round(
    graph: TannerGraph,
    initial_llrs: ArrayLike,
    syndrome: ArrayLike,
    messages_v2c: ArrayLike,
    weight_soft_constraint: float,
    algo: str,
) -> tuple[FloatArray, FloatArray]:
    """Perform one BP update round on a Tanner graph.

    Parameters
    ----------
    graph
        Tanner graph for the parity-check matrix.
    initial_llrs
        One-dimensional channel LLR vector.
    syndrome
        Binary syndrome vector aligned with the graph rows.
    messages_v2c
        Dense V2C message matrix.
    weight_soft_constraint
        Correlation strength used for soft constraints.
    algo
        BP update rule. Supported values are ``"SumProduct"`` and ``"MinSum"``.

    Returns
    -------
    tuple
        ``(messages_v2c_updated, updated_llrs)``.
    """

    _validate_algorithm(algo)
    llr_array = _coerce_float_vector(initial_llrs, graph.n_bits, name="initial_llrs")
    syndrome_array = _coerce_binary_vector(syndrome, graph.n_checks, name="syndrome")
    messages_v2c_array = np.asarray(messages_v2c, dtype=np.float64)
    if messages_v2c_array.shape != (graph.n_bits, graph.n_checks):
        raise ValueError(
            "messages_v2c must have shape "
            f"({graph.n_bits}, {graph.n_checks})."
        )

    messages_c2v = np.zeros((graph.n_checks, graph.n_bits), dtype=np.float64)
    for check in range(graph.n_checks):
        if check < graph.soft_constraint_start:
            for bit in graph.check_neighbors[check]:
                messages_c2v[check, bit] = get_message_from_check_to_vertex(
                    graph,
                    check,
                    bit,
                    int(syndrome_array[check]),
                    messages_v2c_array,
                    algo,
                )
        else:
            for bit in graph.check_neighbors[check]:
                messages_c2v[check, bit] = soft_message_from_check_to_vertex(
                    graph,
                    check,
                    bit,
                    messages_v2c_array,
                    weight_soft_constraint,
                    algo,
                )

    messages_v2c_updated = np.zeros((graph.n_bits, graph.n_checks), dtype=np.float64)
    for bit in range(graph.n_bits):
        for check in graph.vertex_neighbors[bit]:
            messages_v2c_updated[bit, check] = get_message_from_vertex_to_check(
                graph,
                bit,
                check,
                llr_array,
                messages_c2v,
                algo,
            )

    updated_llrs = np.zeros(graph.n_bits, dtype=np.float64)
    for bit in range(graph.n_bits):
        sum_messages = sum(messages_c2v[check, bit] for check in graph.vertex_neighbors[bit])
        updated_llrs[bit] = llr_array[bit] + sum_messages

    return messages_v2c_updated, updated_llrs


def get_message_from_check_to_vertex(
    graph: TannerGraph,
    check: int,
    bit: int,
    syndrome_bit: int,
    messages_v2c: ArrayLike,
    algo: str,
) -> float:
    """Compute one hard-constraint message from a check node to a bit node."""

    _validate_algorithm(algo)
    messages = np.asarray(messages_v2c, dtype=np.float64)
    product = 1.0
    minimum_message = np.inf
    for incident_bit in graph.check_neighbors[check]:
        if incident_bit == bit:
            continue
        value = messages[incident_bit, check]
        if algo == "SumProduct":
            product *= np.tanh(value / 2.0)
        else:
            product *= np.sign(value)
            minimum_message = min(minimum_message, abs(value))

    if algo == "SumProduct":
        message = 2.0 * np.arctanh(np.clip(product, -1.0 + 1e-12, 1.0 - 1e-12))
    else:
        if np.isinf(minimum_message):
            minimum_message = 0.0
        message = product * minimum_message
    if syndrome_bit == 1:
        message = -message
    return float(message)


def soft_message_from_check_to_vertex(
    graph: TannerGraph,
    check: int,
    bit: int,
    messages_v2c: ArrayLike,
    weight_soft_constraint: float,
    algo: str,
) -> float:
    """Compute one soft-constraint message from a check node to a bit node."""

    _validate_algorithm(algo)
    if not 0.0 < weight_soft_constraint < 1.0:
        raise ValueError("weight_soft_constraint must lie strictly between 0 and 1.")

    neighbors = [neighbor for neighbor in graph.check_neighbors[check] if neighbor != bit]
    if len(neighbors) != 1:
        raise ValueError("Soft-constraint checks must connect exactly two variables.")

    other_bit = neighbors[0]
    other_message = float(np.asarray(messages_v2c, dtype=np.float64)[other_bit, check])
    ising_coupling = 0.5 * np.log(weight_soft_constraint / (1.0 - weight_soft_constraint))

    if algo == "SumProduct":
        product = np.tanh(ising_coupling) * np.tanh(other_message / 2.0)
        return float(2.0 * np.arctanh(np.clip(product, -1.0 + 1e-12, 1.0 - 1e-12)))
    return float(
        abs(other_message + ising_coupling / 2.0)
        - abs(other_message - ising_coupling / 2.0)
    )


def get_message_from_vertex_to_check(
    graph: TannerGraph,
    bit: int,
    check: int,
    initial_llrs: ArrayLike,
    messages_c2v: ArrayLike,
    algo: str,
) -> float:
    """Compute one V2C message from a bit node to a check node."""

    _validate_algorithm(algo)
    llrs = np.asarray(initial_llrs, dtype=np.float64).reshape(-1)
    messages = np.asarray(messages_c2v, dtype=np.float64)
    sum_messages = 0.0
    for incident_check in graph.vertex_neighbors[bit]:
        if incident_check != check:
            sum_messages += messages[incident_check, bit]
    return float(llrs[bit] + sum_messages)


def should_bp_stop(
    graph: TannerGraph,
    old_llrs: ArrayLike,
    new_llrs: ArrayLike,
    syndrome: ArrayLike,
    llr_convergence_threshold: float,
    llr_confidence_threshold: float,
) -> bool:
    """Check whether the current BP run should stop."""

    old_llrs_array = _coerce_float_vector(old_llrs, graph.n_bits, name="old_llrs")
    new_llrs_array = _coerce_float_vector(new_llrs, graph.n_bits, name="new_llrs")
    syndrome_array = _coerce_binary_vector(syndrome, graph.n_checks, name="syndrome")

    hard_decision_recovery = get_recovery_using_hard_decision(new_llrs_array)
    parity_satisfied = True
    for check in range(graph.soft_constraint_start):
        parity_check = sum(hard_decision_recovery[bit] for bit in graph.check_neighbors[check]) % 2
        if parity_check != syndrome_array[check]:
            parity_satisfied = False
            break
    if not parity_satisfied:
        return False

    if len(get_confident_bits(new_llrs_array, llr_confidence_threshold)) == graph.n_bits:
        return True

    max_change = float(np.max(np.abs(new_llrs_array - old_llrs_array)))
    return max_change <= llr_convergence_threshold


def _normalize_soft_constraint_start(soft_constraint_start: int, n_checks: int) -> int:
    """Convert Julia's one-based soft-constraint boundary into zero-based form."""

    if not 1 <= soft_constraint_start <= n_checks + 1:
        raise ValueError(
            "soft_constraint_start must lie between 1 and n_checks + 1 in the "
            "Julia-style API."
        )
    return soft_constraint_start - 1


def _coerce_binary_vector(values: ArrayLike, length: int, *, name: str) -> IntArray:
    """Normalize a binary vector to a flat ``int64`` array."""

    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {array.shape[0]}.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 values.")
    return array.copy()


def _coerce_float_vector(values: ArrayLike, length: int, *, name: str) -> FloatArray:
    """Normalize a float vector to a flat ``float64`` array."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {array.shape[0]}.")
    return array.copy()


def _coerce_probability_vector(
    values: ArrayLike,
    length: int,
    *,
    name: str,
) -> FloatArray:
    """Normalize a probability vector to a flat ``float64`` array."""

    array = _coerce_float_vector(values, length, name=name)
    if np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return array


def _validate_algorithm(algo: str) -> None:
    """Validate the BP update rule name."""

    if algo not in {"SumProduct", "MinSum"}:
        raise ValueError(
            f"Unknown algorithm {algo!r}. Supported algorithms are "
            "'SumProduct' and 'MinSum'."
        )

#!/usr/bin/env python3
"""
numpy_bposd.py -- ldpc's product-sum BP (parallel schedule) and its OSD stage
(OSD-0, OSD-E, OSD-CS), in numpy, for machines where ldpc cannot be installed.
reference_decoders.py uses it with --backend numpy.

BP follows ldpc 2.x (src_cpp/bp.hpp, bp_decode_parallel) operation for operation,
in float64 and in the same order, so the two agree to the last bit on all but
rare samples:
  - initial LLR log((1 - p) / p); every bit-to-check message starts there;
  - check-to-bit: the product of tanh(m / 2) over the check's other edges, as a
    prefix product times a suffix product along the row, then
    sign * log((1 + x) / (1 - x)), sign = -1 where the syndrome bit is 1. No clamp:
    a product of +-1 gives +-inf, and inf + (-inf) gives NaN, as in ldpc;
  - posterior = initial LLR + the column's check-to-bit messages in row order;
    decision 1 where the posterior is <= 0; stop at the first iteration whose
    decision reproduces the syndrome;
  - bit-to-check: initial LLR + the column's other messages, as a prefix sum plus a
    suffix sum along the column.
All distinct syndromes are decoded together (vectorised); a syndrome leaves the
batch at its first converged iteration, as ldpc stops there.

OSD (ldpc 2.x, src_cpp/osd.hpp), on the BP posterior of a syndrome BP did not
converge on: bits sorted by posterior LLR, ascending (most likely flipped first;
ties in index order -- ldpc's std::sort is not stable, so tied orders can differ);
Gaussian elimination picks the pivot columns greedily in that order; OSD-0 sets the
non-pivot bits to 0. OSD-E of order w then tries every non-zero setting of the first
w non-pivot bits (in the order of i = 1 .. 2^w - 1, least significant bit = first
non-pivot), OSD-CS every single non-pivot bit and every pair among the first w. The
answer is the candidate of least weight sum_{i in x} log(1 / p_i) (strictly less
than every earlier candidate; OSD-0 first), summed bit by bit in index order.
Validated against ldpc's own output in reference_decoders.py's notes.
"""
from typing import List, NamedTuple, Tuple

import numpy as np

OSD_METHODS = ("osd0", "osd_e", "osd_cs")


class BatchBPResult(NamedTuple):
    decoding: np.ndarray           # (S, n) uint8, the hard decision BP returns
    converged: np.ndarray          # (S,) bool
    log_prob_ratios: np.ndarray    # (S, n) float64, the posterior at the returned iteration
    iterations: np.ndarray         # (S,) int, the iteration BP stopped at


class BatchBP:
    """ldpc's product-sum BP, parallel schedule, for many syndromes at once."""

    def __init__(self, check_matrix: np.ndarray, channel_probabilities: np.ndarray, max_iter: int,
                 chunk_size: int = 20000):
        self.check_matrix = (np.asarray(check_matrix) % 2).astype(np.uint8)
        n_checks, n_bits = self.check_matrix.shape
        self.n_checks, self.n_bits = n_checks, n_bits
        self.max_iter = int(max_iter)
        self.chunk_size = int(chunk_size)
        probabilities = np.asarray(channel_probabilities, dtype=np.float64)
        if probabilities.shape != (n_bits,):
            raise ValueError(f"{probabilities.size} channel probabilities for {n_bits} bits")
        self.initial_llr = np.log((1 - probabilities) / probabilities)
        # Edges in row-major order (row, then column): np.nonzero gives exactly that.
        edge_rows, edge_cols = np.nonzero(self.check_matrix)
        self.n_edges = edge_rows.size
        self.edge_cols = edge_cols
        dummy = self.n_edges                       # a padding edge: tanh 1, message 0
        row_degree = np.bincount(edge_rows, minlength=n_checks)
        col_degree = np.bincount(edge_cols, minlength=n_bits)
        self.row_edges = np.full((n_checks, row_degree.max()), dummy, dtype=np.int64)
        for check in range(n_checks):
            edges = np.flatnonzero(edge_rows == check)                 # increasing column
            self.row_edges[check, :edges.size] = edges
        self.col_edges = np.full((n_bits, col_degree.max()), dummy, dtype=np.int64)
        for bit in range(n_bits):
            edges = np.flatnonzero(edge_cols == bit)                   # increasing row
            self.col_edges[bit, :edges.size] = edges

    def decode(self, syndromes: np.ndarray) -> BatchBPResult:
        syndromes = (np.asarray(syndromes) % 2).astype(np.uint8)
        if syndromes.ndim != 2 or syndromes.shape[1] != self.n_checks:
            raise ValueError(f"syndromes must be (S, {self.n_checks})")
        n_syndromes = syndromes.shape[0]
        decoding = np.zeros((n_syndromes, self.n_bits), dtype=np.uint8)
        converged = np.zeros(n_syndromes, dtype=bool)
        log_prob_ratios = np.zeros((n_syndromes, self.n_bits), dtype=np.float64)
        iterations = np.zeros(n_syndromes, dtype=np.int64)
        for start in range(0, n_syndromes, self.chunk_size):
            stop = min(start + self.chunk_size, n_syndromes)
            self._decode_chunk(syndromes[start:stop], decoding[start:stop], converged[start:stop],
                               log_prob_ratios[start:stop], iterations[start:stop])
        result = BatchBPResult(decoding, converged, log_prob_ratios, iterations)
        return result

    def _decode_chunk(self, syndromes: np.ndarray, decoding: np.ndarray, converged: np.ndarray,
                      log_prob_ratios: np.ndarray, iterations: np.ndarray) -> None:
        n_edges, dummy = self.n_edges, self.n_edges
        check_matrix_transposed = self.check_matrix.T.astype(np.int64)
        active = np.arange(syndromes.shape[0])
        bit_to_check = np.zeros((active.size, n_edges + 1), dtype=np.float64)
        bit_to_check[:, :n_edges] = self.initial_llr[self.edge_cols]
        signs = np.where(syndromes == 1, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for iteration in range(1, self.max_iter + 1):
                # check to bit: prefix product x suffix product of tanh(m / 2) along each row
                tanh_half = np.tanh(bit_to_check / 2)
                tanh_half[:, dummy] = 1.0
                along_rows = tanh_half[:, self.row_edges]                       # (A, m, dr)
                prefix = np.ones_like(along_rows)
                prefix[:, :, 1:] = np.cumprod(along_rows[:, :, :-1], axis=2)
                suffix = np.ones_like(along_rows)
                suffix[:, :, :-1] = np.cumprod(along_rows[:, :, :0:-1], axis=2)[:, :, ::-1]
                product = prefix * suffix
                messages = signs[active][:, :, None] * np.log((1 + product) / (1 - product))
                check_to_bit = np.zeros((active.size, n_edges + 1), dtype=np.float64)
                check_to_bit[:, self.row_edges] = messages
                check_to_bit[:, dummy] = 0.0
                # posterior: initial LLR, then the column's messages in row order
                along_cols = check_to_bit[:, self.col_edges]                    # (A, n, dc)
                posterior = np.broadcast_to(self.initial_llr, (active.size, self.n_bits)).copy()
                for position in range(along_cols.shape[2]):
                    posterior = posterior + along_cols[:, :, position]
                decision = (posterior <= 0).astype(np.uint8)
                candidate_syndrome = (decision.astype(np.int64) @ check_matrix_transposed) % 2
                done_now = np.all(candidate_syndrome == syndromes[active], axis=1)
                if iteration == self.max_iter:
                    finished = np.ones(active.size, dtype=bool)
                else:
                    finished = done_now
                if finished.any():
                    rows = active[finished]
                    decoding[rows] = decision[finished]
                    converged[rows] = done_now[finished]
                    log_prob_ratios[rows] = posterior[finished]
                    iterations[rows] = iteration
                if iteration == self.max_iter:
                    break
                keep = ~done_now
                active = active[keep]
                if active.size == 0:
                    break
                along_cols = along_cols[keep]
                # bit to check: prefix sum (from the initial LLR) plus suffix sum along each column
                degree = along_cols.shape[2]
                prefix_sum = np.empty_like(along_cols)
                running = np.broadcast_to(self.initial_llr, (active.size, self.n_bits)).copy()
                for position in range(degree):
                    prefix_sum[:, :, position] = running
                    running = running + along_cols[:, :, position]
                suffix_sum = np.empty_like(along_cols)
                running = np.zeros((active.size, self.n_bits), dtype=np.float64)
                for position in range(degree - 1, -1, -1):
                    suffix_sum[:, :, position] = running
                    running = running + along_cols[:, :, position]
                bit_to_check = np.zeros((active.size, n_edges + 1), dtype=np.float64)
                bit_to_check[:, self.col_edges] = prefix_sum + suffix_sum
                bit_to_check[:, dummy] = 0.0


def osd_correction(check_matrix: np.ndarray, syndrome: np.ndarray, log_prob_ratios: np.ndarray,
                   bit_weights: np.ndarray, method: str, order: int) -> np.ndarray:
    """The OSD answer for one syndrome from BP's posterior (see the module notes)."""
    if method not in OSD_METHODS:
        raise ValueError(f"OSD method {method!r}: one of {OSD_METHODS}")
    n_checks, n_bits = check_matrix.shape
    bit_order = np.argsort(log_prob_ratios, kind="stable")
    augmented = np.concatenate([check_matrix[:, bit_order], syndrome[:, None]], axis=1).astype(np.uint8)
    pivots: List[int] = []
    rank = 0
    for column in range(n_bits):
        candidates_rows = np.flatnonzero(augmented[rank:, column])
        if candidates_rows.size == 0:
            continue
        pivot_row = int(candidates_rows[0]) + rank
        if pivot_row != rank:
            augmented[[rank, pivot_row]] = augmented[[pivot_row, rank]]
        mask = augmented[:, column].astype(bool)
        mask[rank] = False
        augmented[mask] ^= augmented[rank]
        pivots.append(column)
        rank += 1
        if rank == n_checks:
            break
    pivot_set = set(pivots)
    non_pivots = [column for column in range(n_bits) if column not in pivot_set]
    reduced = augmented[:rank, :n_bits].astype(np.int64)
    reduced_syndrome = augmented[:rank, n_bits].astype(np.int64)
    head = min(order, len(non_pivots))
    flips: List[Tuple[int, ...]] = [()]
    if method == "osd_e":
        for setting in range(1, 2 ** head):
            flips.append(tuple(non_pivots[bit] for bit in range(head) if (setting >> bit) & 1))
    elif method == "osd_cs":
        flips += [(column,) for column in non_pivots]
        flips += [(non_pivots[first], non_pivots[second])
                  for first in range(head) for second in range(first + 1, head)]
    candidates = np.zeros((len(flips), n_bits), dtype=np.uint8)
    for index, flip in enumerate(flips):
        sorted_solution = np.zeros(n_bits, dtype=np.int64)
        sorted_solution[list(flip)] = 1
        sorted_solution[pivots] = (reduced_syndrome + reduced[:, non_pivots] @ sorted_solution[non_pivots]) % 2
        candidates[index, bit_order] = sorted_solution
    # weight summed bit by bit in index order (cumsum is sequential), first strict minimum
    weights = np.cumsum(candidates * bit_weights[None, :], axis=1)[:, -1]
    best = 0
    for index in range(1, len(flips)):
        if weights[index] < weights[best]:
            best = index
    correction = candidates[best]
    return correction


def bposd_decode(check_matrix: np.ndarray, syndromes: np.ndarray, channel_probabilities: np.ndarray,
                 max_iter: int, osd_method: str = "", osd_order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(corrections (S, n) uint8, BP converged (S,) bool): BP, then OSD where BP did not
    converge when osd_method is given (as ldpc's BpOsdDecoder)."""
    check_matrix = (np.asarray(check_matrix) % 2).astype(np.uint8)
    probabilities = np.asarray(channel_probabilities, dtype=np.float64)
    bp_result = BatchBP(check_matrix, probabilities, max_iter).decode(syndromes)
    corrections = bp_result.decoding.copy()
    if osd_method:
        bit_weights = np.log(1 / probabilities)
        for row in np.flatnonzero(~bp_result.converged):
            corrections[row] = osd_correction(check_matrix, np.asarray(syndromes[row], dtype=np.uint8),
                                              bp_result.log_prob_ratios[row], bit_weights, osd_method, osd_order)
    return corrections, bp_result.converged

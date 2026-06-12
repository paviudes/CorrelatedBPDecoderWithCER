#!/usr/bin/env python3
"""Generate a parameterized 72q ballistic dataset under Standard_BP_OSD.

This is a small, scriptable replacement for the original collaborator
``72q_BB_code_data_gen.py`` workflow when we want to generate one or a few
explicit samples at different `(p, q)` settings without depending on the
`ldpc`/`bposd` imports that are irrelevant for data generation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np


R, C = 2, 36
N_QUBITS = R * C
Q_STD_DEFAULT = 0.1
REFERENCE_CODE_DIR = (
    Path("pavi/Standard_BP_OSD/72q_BB_p_0.006_q_0.1_std_0.1_data/72q_BB_code")
)


def idx(r: int, c: int) -> int:
    """Return the one-based qubit label for a ladder-grid coordinate."""

    return r * C + c + 1


def neighboring_qubits() -> dict[int, set[int]]:
    """Return the nearest-neighbour graph on the 2x36 ladder."""

    neighbors = {idx(r, c): set() for r in range(R) for c in range(C)}
    for r in range(R):
        for c in range(C):
            q = idx(r, c)
            if c == 0:
                neighbors[q].add(idx(r, C - 1))
            if c == C - 1:
                neighbors[q].add(idx(r, 0))
            if c > 0:
                neighbors[q].add(idx(r, c - 1))
            if c < C - 1:
                neighbors[q].add(idx(r, c + 1))
            if r > 0:
                neighbors[q].add(idx(r - 1, c))
            if r < R - 1:
                neighbors[q].add(idx(r + 1, c))
    return neighbors


NEIGHBORS = neighboring_qubits()


def neighboring_pairs() -> dict[int, set[tuple[int, int]]]:
    """Return the directed nearest-neighbour pairs for each qubit."""

    pairs = {q: set() for q in NEIGHBORS}
    for q, nbs in NEIGHBORS.items():
        for nb in nbs:
            pairs[q].add((q, nb))
    return pairs


PAIRS = neighboring_pairs()


def neighbors_of_macro_nodes() -> dict[tuple[int, int], tuple[set[int], set[int]]]:
    """Return the neighbour sets used by the collaborator CER construction."""

    out: dict[tuple[int, int], tuple[set[int], set[int]]] = {}
    for q, qubit_pairs in PAIRS.items():
        for pair in qubit_pairs:
            q_i, q_j = pair
            out[pair] = (NEIGHBORS[q_i] - {q_j}, NEIGHBORS[q_j] - {q_i})
    return out


NEIGHBORS_OF_MACRO_NODES = neighbors_of_macro_nodes()


def format_component(value: float) -> str:
    """Match the collaborator's compact float naming convention."""

    return f"{value:.3g}"


def assign_error_probabilities(
    p_value: float,
    q_mean: float,
    *,
    q_std: float,
    rng: np.random.Generator,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Assign the single-qubit and conditional neighbour-flip probabilities."""

    independent_error_prob: dict[int, float] = {}
    conditional_nb_flip_prob: dict[tuple[int, int], float] = {}
    undirected_pair_q: dict[tuple[int, int], float] = {}

    for qubit in range(1, N_QUBITS + 1):
        independent_error_prob[qubit] = p_value
        for neighbor in NEIGHBORS[qubit]:
            undirected_key = (min(qubit, neighbor), max(qubit, neighbor))
            if undirected_key not in undirected_pair_q:
                q_val = rng.normal(loc=q_mean, scale=q_std)
                undirected_pair_q[undirected_key] = float(np.clip(q_val, 0.0, 1.0))
            conditional_nb_flip_prob[(qubit, neighbor)] = undirected_pair_q[undirected_key]

    return independent_error_prob, conditional_nb_flip_prob


def compute_cer(
    independent_error_prob: dict[int, float],
    conditional_nb_flip_prob: dict[tuple[int, int], float],
) -> dict[tuple[int, int], float]:
    """Compute the collaborator's normalized CER weights for each directed pair."""

    cer: dict[tuple[int, int], float] = {}
    for pair, (neighbors_i, neighbors_j) in NEIGHBORS_OF_MACRO_NODES.items():
        q_i, q_j = pair
        p_i = independent_error_prob[q_i]
        p_j = independent_error_prob[q_j]
        p_00 = (1 - p_i) * (1 - p_j)
        p_01 = (1 - p_i) * p_j
        p_10 = p_i * (1 - p_j)
        p_11 = p_i * p_j

        q_k, q_l = tuple(neighbors_i)
        p_k = independent_error_prob[q_k]
        p_l = independent_error_prob[q_l]
        q_ik = conditional_nb_flip_prob.get((q_i, q_k), 0.0)
        q_il = conditional_nb_flip_prob.get((q_i, q_l), 0.0)
        kl = {
            "KL_01": (1 - p_k) * p_l * q_il,
            "KL_10": p_k * (1 - p_l) * q_ik,
            "KL_11": p_k * p_l * ((1 - q_il) * q_ik + (1 - q_ik) * q_il),
        }

        q_m, q_n = tuple(neighbors_j)
        p_m = independent_error_prob[q_m]
        p_n = independent_error_prob[q_n]
        q_jm = conditional_nb_flip_prob.get((q_j, q_m), 0.0)
        q_jn = conditional_nb_flip_prob.get((q_j, q_n), 0.0)
        mn = {
            "MN_01": (1 - p_m) * p_n * q_jn,
            "MN_10": p_m * (1 - p_n) * q_jm,
            "MN_11": p_m * p_n * ((1 - q_jn) * q_jm + (1 - q_jm) * q_jn),
        }

        klmn = {f"{k}_{m}": kl[k] * mn[m] for k in kl for m in mn}
        cer[pair] = p_00 * sum(klmn.values()) + p_01 * sum(kl.values()) + p_10 * sum(mn.values()) + p_11

    total = sum(cer.values())
    return {key: value / total for key, value in cer.items()}


def generate_error_patterns(
    num_err: int,
    independent_error_prob: dict[int, float],
    conditional_nb_flip_prob: dict[tuple[int, int], float],
    *,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate the explicit ballistic error patterns."""

    error_patterns: list[np.ndarray] = []
    for _ in range(num_err):
        pattern = np.zeros(N_QUBITS + 1, dtype=int)
        for qubit in range(1, N_QUBITS + 1):
            if rng.random() < independent_error_prob[qubit]:
                pattern[qubit] = 1
        orig_pattern = pattern.copy()
        for qubit in range(1, N_QUBITS + 1):
            for neighbor in NEIGHBORS[qubit]:
                if (
                    orig_pattern[qubit] == 1
                    and rng.random() < conditional_nb_flip_prob.get((qubit, neighbor), 0.0)
                    and pattern[neighbor] == 0
                ):
                    pattern[neighbor] = 1
        error_patterns.append(pattern[1:])
    return error_patterns


def build_training_patterns(
    test_data: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the collaborator-style 10k training file from a test matrix."""

    weights = np.sum(test_data, axis=0)
    filtered = test_data[:, weights >= 2]
    if filtered.shape[1] < 9927:
        raise ValueError(
            "Not enough weight>=2 patterns to build the 10k training subset."
        )
    indices = rng.choice(filtered.shape[1], 9927, replace=False)
    sampled_patterns = filtered[:, indices]
    basis_errors = np.eye(N_QUBITS, dtype=int)
    zero_pattern = np.zeros((N_QUBITS, 1), dtype=int)
    return np.hstack((zero_pattern, basis_errors, sampled_patterns))


def ensure_code_dir(dataset_root: Path) -> None:
    """Copy the existing 72q code matrices into the new dataset root if needed."""

    target = dataset_root / "72q_BB_code"
    if target.is_dir():
        return
    if not REFERENCE_CODE_DIR.is_dir():
        raise FileNotFoundError(
            f"Reference code directory not found: {REFERENCE_CODE_DIR}"
        )
    shutil.copytree(REFERENCE_CODE_DIR, target)


def build_dataset_root(p_value: float, q_mean: float, q_std: float) -> Path:
    """Return the Standard_BP_OSD dataset root for the chosen parameters."""

    p_str = format_component(p_value)
    q_str = format_component(q_mean)
    std_str = format_component(q_std)
    return Path("pavi/Standard_BP_OSD") / f"72q_BB_p_{p_str}_q_{q_str}_std_{std_str}_data"


def main(argv: list[str] | None = None) -> int:
    """Generate one or more 72q ballistic explicit samples."""

    parser = argparse.ArgumentParser(
        description="Generate a parameterized 72q ballistic Standard_BP_OSD dataset."
    )
    parser.add_argument("--p-value", type=float, required=True)
    parser.add_argument("--q-mean", type=float, required=True)
    parser.add_argument("--q-std", type=float, default=Q_STD_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--sample-start", type=int, default=1)
    parser.add_argument("--num-patterns", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset_root = build_dataset_root(args.p_value, args.q_mean, args.q_std)
    testing_dir = dataset_root / "testing_data"
    training_dir = dataset_root / "training_data"
    cer_dir = dataset_root / "correlated_weights"
    prob_dir = dataset_root / "assigned_probabilities"
    results_dir = dataset_root / "results"
    models_dir = dataset_root / "models"

    for directory in (
        dataset_root,
        testing_dir,
        training_dir,
        cer_dir,
        prob_dir,
        results_dir,
        models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_code_dir(dataset_root)

    p_str = format_component(args.p_value)
    q_str = format_component(args.q_mean)
    base_rng = np.random.default_rng(args.seed)

    for sample in range(args.sample_start, args.sample_start + args.n_samples):
        sample_seed = int(base_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(sample_seed)

        independent_error_prob, conditional_nb_flip_prob = assign_error_probabilities(
            args.p_value,
            args.q_mean,
            q_std=args.q_std,
            rng=rng,
        )
        cer_normalized = compute_cer(
            independent_error_prob,
            conditional_nb_flip_prob,
        )
        error_patterns = generate_error_patterns(
            args.num_patterns,
            independent_error_prob,
            conditional_nb_flip_prob,
            rng=rng,
        )
        test_matrix = np.asarray(error_patterns, dtype=int).T
        train_matrix = build_training_patterns(test_matrix, rng=rng)

        prob_file = prob_dir / f"assigned_probabilities_p_{p_str}_q_{q_str}_s_{sample}.txt"
        with prob_file.open("w") as handle:
            handle.write("Independent Error Probabilities:\n")
            for key, val in independent_error_prob.items():
                handle.write(f"Qubit {key} : {val}\n")
            handle.write("\nConditional Neighbor Flip Probabilities:\n")
            for key, val in conditional_nb_flip_prob.items():
                handle.write(f"Pair {key} : {val}\n")

        cer_file = cer_dir / f"correlated_weights_p_{p_str}_q_{q_str}_s_{sample}.txt"
        with cer_file.open("w") as handle:
            for key, val in cer_normalized.items():
                handle.write(f"{key} : {val}\n")

        test_file = testing_dir / f"test_ballistic_p_{p_str}_q_{q_str}_s_{sample}.txt"
        train_file = training_dir / f"train_ballistic_p_{p_str}_q_{q_str}_s_{sample}.txt"
        np.savetxt(test_file, test_matrix, fmt="%d")
        np.savetxt(train_file, train_matrix, fmt="%d")
        print(
            f"Generated sample {sample} in {dataset_root} "
            f"(seed={sample_seed}, train={train_matrix.shape[1]}, test={test_matrix.shape[1]})."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

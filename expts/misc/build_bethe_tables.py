#!/usr/bin/env python3
"""
build_bethe_tables.py -- the Bethe log-weight table of every check, from data.

    python3 misc/build_bethe_tables.py --workdir 72q_BB_cycles_1_trainable_alpha \
        --key p_0.0015_sig_0.0015_s_1

reads  <data>/<workdir>/code/HZ.txt
       <data>/<workdir>/training_data/train_<key>.txt      (72 rows x N samples)
writes <data>/<workdir>/correlated_weights/bethe_table_<key>.txt

one line per check c (1-based row of HZ):

    c : T_c(1) T_c(2) ... T_c(2^d)

with T_c(k) = ln P_c(k) - sum_u ln p_u(b_u(k)): the log of the check's joint
probability of configuration k over the product of its single-qubit marginals,
the factor the enriched check node carries under `enriched_table = "bethe"`
(src/soft_constraints.jl). Configuration k (1-based) has, in slot j (the
check's support in increasing qubit order), the bit ((k - 1) >> (j - 1)) & 1,
the order `configuration_table` uses on the Julia side.

Estimates: P_c(k) = count(k) / N, p_u = the column mean. A configuration never
seen gets the detection floor 0.5 / N, and then at most T = 0: the data cannot
say it is MORE likely than independence, so it is not promoted above it. A
configuration independence expects but the data never shows (count 0, P_ind
above the floor) keeps the negative value: that is the data saying "rarer than
independence", which is the information the table exists to carry.

The file is keyed like the CER file (`correlated_weights_<key>.txt`) and lives
next to it, so `bethe_table_path` finds a test distribution's table next to its
priors. Build one for every key a sweep tests on.
"""
import argparse
import os
import sys

import numpy as np


def read_sample_matrix(path):
    """The 0/1 matrix of a training or test file: rows = qubits, columns = samples."""
    raw = np.fromfile(path, dtype=np.uint8)
    newline_positions = np.flatnonzero(raw == 10)
    n_rows = newline_positions.size
    if n_rows == 0:
        raise ValueError(f"{path}: no newline found")
    line_lengths = np.diff(np.concatenate(([-1], newline_positions)))
    if np.all(line_lengths == line_lengths[0]) and raw.size == n_rows * line_lengths[0]:
        line_length = int(line_lengths[0])
        matrix = raw.reshape(n_rows, line_length)[:, :-1:2] - 48
        if matrix.min() < 0 or matrix.max() > 1:
            raise ValueError(f"{path}: entries other than 0/1 (not a single-digit, space-separated file)")
        return matrix.astype(np.uint8)
    matrix = np.loadtxt(path, dtype=np.int64)
    if matrix.min() < 0 or matrix.max() > 1:
        raise ValueError(f"{path}: entries other than 0/1")
    return matrix.astype(np.uint8)


def bethe_table(samples, support):
    """(table of 2^d log-weights, count vector, independent-model probabilities) for one check."""
    degree = len(support)
    n_configurations = 1 << degree
    n_samples = samples.shape[1]
    bits = ((np.arange(n_configurations)[:, None] >> np.arange(degree)) & 1).astype(np.float64)
    codes = samples[support].T.astype(np.int64) @ (1 << np.arange(degree))
    counts = np.bincount(codes, minlength=n_configurations).astype(np.float64)
    floor = 0.5 / n_samples
    marginals = np.clip(samples[support].mean(axis=1), floor, 1.0 - floor)
    log_p_one = np.log(marginals)
    log_p_zero = np.log1p(-marginals)
    log_independent = bits @ log_p_one + (1.0 - bits) @ log_p_zero
    joint = np.maximum(counts, 0.5) / n_samples
    table = np.log(joint) - log_independent
    unseen = counts == 0
    table[unseen] = np.minimum(table[unseen], 0.0)
    return table, counts, np.exp(log_independent)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, help="run directory name under --data (the codename)")
    parser.add_argument("--key", required=True, help="dataset key, e.g. p_0.0015_sig_0.0015_s_1")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"),
                        help="the data directory (default: the repository's data/)")
    parser.add_argument("--samples", default="", help="sample file to use instead of training_data/train_<key>.txt")
    parser.add_argument("--output", default="", help="output path instead of correlated_weights/bethe_table_<key>.txt")
    arguments = parser.parse_args()

    run_directory = os.path.join(arguments.data, arguments.workdir)
    parity_check_path = os.path.join(run_directory, "code", "HZ.txt")
    samples_path = arguments.samples or os.path.join(run_directory, "training_data", f"train_{arguments.key}.txt")
    output_path = arguments.output or os.path.join(run_directory, "correlated_weights", f"bethe_table_{arguments.key}.txt")
    for path in (parity_check_path, samples_path):
        if not os.path.isfile(path):
            sys.exit(f"missing: {path}")

    parity_check_matrix = np.loadtxt(parity_check_path, dtype=np.int64)
    samples = read_sample_matrix(samples_path)
    if samples.shape[0] != parity_check_matrix.shape[1]:
        sys.exit(f"{samples_path} has {samples.shape[0]} rows but HZ has {parity_check_matrix.shape[1]} columns")
    n_checks, n_qubits = parity_check_matrix.shape
    n_samples = samples.shape[1]
    print(f"{n_checks} checks, {n_qubits} qubits, {n_samples} samples from {samples_path}")

    lines = []
    total_unseen = 0
    total_unseen_independent_mass = 0.0
    kl_values = []
    for check in range(n_checks):
        support = np.flatnonzero(parity_check_matrix[check])
        table, counts, independent = bethe_table(samples, support)
        empirical = counts / n_samples
        seen = counts > 0
        kl_values.append(float(np.sum(empirical[seen] * (np.log(empirical[seen]) - np.log(independent[seen])))))
        total_unseen += int(np.sum(~seen))
        total_unseen_independent_mass += float(independent[~seen].sum())
        lines.append(f"{check + 1} : " + " ".join(f"{value:.7g}" for value in table))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(f"# Bethe log-weight tables ln P_c(k) - sum_u ln p_u(b_u(k)) from {os.path.basename(samples_path)}, "
                     f"{n_samples} samples; see misc/build_bethe_tables.py\n")
        handle.write("\n".join(lines) + "\n")
    kl = np.array(kl_values)
    print(f"wrote {output_path}")
    print(f"KL(joint || independent) per check: mean {kl.mean():.4f}, min {kl.min():.4f}, max {kl.max():.4f} nats")
    print(f"configurations never seen: {total_unseen} of {n_checks * (1 << len(support))} "
          f"(independence gives them {total_unseen_independent_mass / n_checks:.2e} of a check's mass on average)")


if __name__ == "__main__":
    main()

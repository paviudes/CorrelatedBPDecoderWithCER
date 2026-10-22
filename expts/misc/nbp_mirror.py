#!/usr/bin/env python3
"""
nbp_mirror.py -- decode a test set with a TRAINED Neural BP model in numpy, so a
test-time change (above all the enriched node's alpha) can be tried on the laptop
without Julia or a GPU.

    # the enriched model as tested, then with alpha rescaled to the test set
    python3 misc/nbp_mirror.py --workdir 72q_BB_cycles_1_sweep \
        --testing-dir ../data/72q_BB_cycles_1_trainable_alpha/testing_data \
        --key p_0.0005_sig_0.0_s_1 --arm cer_enr --seeds 1 2 3 4 5 --alpha learned rescaled

It mirrors the TEST forward of src/forward_pass_weights.jl (tanh rule with the
same clamps, the enriched rule of soft_constraints.jl, the readout) and the
commit-at-last-layer scoring of src/predict.jl. Priors: the test key's CER rates
rescaled to median --single-qubit-rescale (log 9 for nocer). Each distinct
syndrome is decoded once.

--alpha    learned   the model's alpha (what the test job uses)
           rescaled  learned x alpha*(test) / alpha*(train), alpha* = ln 9 / ln((1-m)/m),
                     m the median raw single-qubit rate of the key's CER file
           a number  that alpha
--couplings test (default) | train: the J of the test key's CER file, or of the
           training key's (the couplings the model was trained with)
--weights  all, or error weights to decode, e.g. 3 or 4,5,6 (others are skipped
           and count as successes; no model here fails below weight 3)

FIDELITY (checked 2026-10-22 against the GPU test files): the tanh arms reproduce
the GPU counts (cer seed 1 on p_0.0005_sig_0.0_s_1: 217 = 217, 63 coset + 154
convergence and 198 distinct failing patterns on both; first 250k of
p_0.0015_sig_0.0_s_1: 599 vs 595). The enriched arm agrees where
the decode is stable (weights >= 4 on p_0.0005_sig_0.0: 201 vs 204; first 250k
of p_0.0015_sig_0.0: 445 vs 446, coset 286 vs 285) but NOT on the frequent
weight-3 "block" patterns at low p, which sit on a knife edge: the GPU run itself
fails (10,30,40) on 225 of its 229 occurrences, and the mirror and the GPU fail
different blocks (seed 1, p_0.0005_sig_0.0: 671 vs 964). Compare mirror with
mirror; confirm a final number with the Julia test.
"""
import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EPS32 = np.finfo(np.float32).eps
MESSAGE_CAP = np.float32(2.0) * np.arctanh(np.float32(1.0) - EPS32)   # ENRICHED_MESSAGE_CAP
RUN_TAGS = {"nocer": "_no_cer_nocer", "cer": "_cer", "cer_enr": "_cer_enr"}


def read_sample_matrix(path: str) -> np.ndarray:
    """The 0/1 matrix of a test file: rows = qubits, columns = samples."""
    raw = np.fromfile(path, dtype=np.uint8)
    line_length = int(np.flatnonzero(raw == 10)[0]) + 1
    matrix = (raw.reshape(-1, line_length)[:, :-1:2] - 48).astype(np.uint8)
    return matrix


def parse_cer(path: str) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """(rates, pairs, J) of a CER file, 0-based qubits, as parse_cer_data reads it."""
    rates: Dict[int, np.float32] = {}
    pairs: List[Tuple[int, int]] = []
    couplings: List[np.float32] = []
    for line in open(path):
        line = line.strip()
        pair_match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*:\s*(\S+)", line)
        if pair_match:
            pairs.append((int(pair_match[1]) - 1, int(pair_match[2]) - 1))
            couplings.append(np.float32(float(pair_match[3])))
            continue
        rate_match = re.fullmatch(r"(\d+)\s*:\s*(\S+)", line)
        if rate_match:
            rates[int(rate_match[1]) - 1] = max(np.float32(float(rate_match[2])), np.float32(1e-6))
    rate_vector = np.array([rates[qubit] for qubit in range(len(rates))], dtype=np.float32)
    return rate_vector, pairs, np.array(couplings, dtype=np.float32)


def alpha_star(rates: np.ndarray) -> float:
    median_rate = float(np.median(rates))
    return float(np.log(9.0) / np.log((1.0 - median_rate) / median_rate))


def initial_llrs(rates: np.ndarray, single_qubit_rescale: float) -> np.ndarray:
    """rescale_single_qubit_error_rates, then log((1-p)/p), in Float32."""
    scaled = rates.astype(np.float32)
    if single_qubit_rescale > 0 and scaled.max() < np.float32(single_qubit_rescale):
        scaled = scaled * (np.float32(single_qubit_rescale) / np.float32(np.median(scaled)))
    return np.log((np.float32(1) - scaled) / scaled).astype(np.float32)


class TannerGraph:
    """Neuron order, adjacencies and (optionally) the pairwise enriched tables of NeuralBPBase."""

    def __init__(self, parity_check_matrix: np.ndarray, pairs: Optional[List[Tuple[int, int]]] = None,
                 couplings: Optional[np.ndarray] = None):
        self.H = parity_check_matrix.astype(np.int64)
        n_checks, n_bits = self.H.shape
        self.check_of, self.bit_of = np.nonzero(self.H)            # check-major, variables inside
        self.n_neurons = self.check_of.size
        same_bit = (self.bit_of[:, None] == self.bit_of[None, :]) & (self.check_of[:, None] != self.check_of[None, :])
        self.weight_cols, self.weight_rows = np.nonzero(same_bit.T)   # column-major, as Julia's findall
        same_check = (self.check_of[:, None] == self.check_of[None, :]) & (self.bit_of[:, None] != self.bit_of[None, :])
        self.same_check = same_check.astype(np.float32)
        self.tables: Optional[Dict[str, np.ndarray]] = None
        if pairs is None:
            return
        checks_of_qubit = [set(np.flatnonzero(self.H[:, qubit])) for qubit in range(n_bits)]
        pairs_of_check: List[List[int]] = [[] for _ in range(n_checks)]
        for pair_index, (qubit_a, qubit_b) in enumerate(pairs):
            shared = sorted(checks_of_qubit[qubit_a] & checks_of_qubit[qubit_b])
            if shared:
                pairs_of_check[shared[0]].append(pair_index)      # the first shared check, as in Julia
        enriched = [check for check in range(n_checks) if pairs_of_check[check]]
        degree = int(self.H[enriched[0]].sum())
        configurations = np.array([[(code >> slot) & 1 for slot in range(degree)] for code in range(2 ** degree)],
                                  dtype=np.float64)
        neuron_of = {(int(check), int(bit)): neuron for neuron, (check, bit) in enumerate(zip(self.check_of, self.bit_of))}
        slots = np.zeros((len(enriched), degree), dtype=int)
        table = np.zeros((len(enriched), 2 ** degree))
        for enriched_index, check in enumerate(enriched):
            support = list(np.flatnonzero(self.H[check]))
            slot_of = {qubit: slot for slot, qubit in enumerate(support)}
            slots[enriched_index] = [neuron_of[(check, qubit)] for qubit in support]
            for pair_index in pairs_of_check[check]:
                qubit_a, qubit_b = pairs[pair_index]
                table[enriched_index] += float(couplings[pair_index]) * \
                    configurations[:, slot_of[qubit_a]] * configurations[:, slot_of[qubit_b]]
        self.tables = dict(checks=np.array(enriched), slots=slots, bits=configurations,
                           parity=configurations.sum(1) % 2, log_weights=table)


def load_model(path: str, graph: TannerGraph) -> Dict[str, object]:
    weights = __import__("json").load(open(path))
    message_weights = np.array(weights["weights_c2v_v2c"], dtype=np.float32)
    llr_weights = np.array(weights["weights_llrs"], dtype=np.float32)
    n_layers = llr_weights.size // graph.H.shape[1]
    message_weights = message_weights.reshape(n_layers, -1)
    dense_weights = np.zeros((n_layers, graph.n_neurons, graph.n_neurons), dtype=np.float32)
    for layer in range(n_layers):
        dense_weights[layer, graph.weight_rows, graph.weight_cols] = message_weights[layer]
    readout = np.zeros((graph.H.shape[1], graph.n_neurons), dtype=np.float32)
    readout[graph.bit_of, np.arange(graph.n_neurons)] = np.array(weights["weights_c2v_readout"], dtype=np.float32)
    theta = np.float32(weights["coupling_logit"][0])
    model = dict(W=dense_weights, b=llr_weights.reshape(n_layers, -1), readout=readout,
                 alpha=float(np.float32(1) / (np.float32(1) + np.exp(-theta))),
                 check_node=weights.get("check_node", "tanh"), table=weights.get("enriched_table", "pairwise"))
    return model


def enriched_messages(c2v: np.ndarray, v2c: np.ndarray, syndromes: np.ndarray, alpha: float,
                      tables: Dict[str, np.ndarray]) -> np.ndarray:
    """m_{c->v} = S_v^0 - S_v^1 - m_v over {parity = s_c}, LW = alpha T - b.m (float64, exact here)."""
    incoming = v2c[tables["slots"]].astype(np.float64)                          # (E, d, N)
    log_weights = alpha * tables["log_weights"][:, :, None] - np.matmul(tables["bits"][None], incoming)
    valid = tables["parity"][None, :, None] == syndromes[tables["checks"]].astype(np.float64)[:, None, :]
    log_weights = np.where(valid, log_weights, -np.inf)
    stabilised = np.exp(log_weights - log_weights.max(1, keepdims=True))
    class_one = np.matmul(tables["bits"].T[None], stabilised)
    class_zero = np.matmul((1 - tables["bits"]).T[None], stabilised)
    messages = np.clip(np.log(class_zero) - np.log(class_one) - incoming, -MESSAGE_CAP, MESSAGE_CAP)
    c2v[tables["slots"].reshape(-1)] = messages.astype(np.float32).reshape(-1, messages.shape[2])
    return c2v


def last_layer_decisions(model: Dict[str, object], graph: TannerGraph, llrs: np.ndarray, syndromes: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Hard decisions (posterior < 0) of the last layer, (n_bits x N)."""
    c2v = np.zeros((graph.n_neurons, syndromes.shape[1]), dtype=np.float32)
    routed_syndromes = syndromes[graph.check_of].astype(np.int64)
    posterior = None
    for layer in range(model["b"].shape[0]):
        channel = (model["b"][layer] * llrs).astype(np.float32)
        v2c = model["W"][layer] @ c2v + channel[graph.bit_of][:, None]
        magnitudes = np.log(np.clip(np.tanh(np.abs(v2c) * np.float32(0.5)), EPS32, np.float32(1) - EPS32))
        signs = (np.mod(graph.same_check @ (v2c < 0).astype(np.float32), 2).astype(np.int64) + routed_syndromes) % 2
        exponentials = np.clip(np.exp(graph.same_check @ magnitudes), EPS32, np.float32(1) - EPS32)
        c2v = (np.float32(2) * np.arctanh(np.where(signs == 1, -exponentials, exponentials))).astype(np.float32)
        if model["check_node"] == "enriched":
            c2v = enriched_messages(c2v, v2c, syndromes, alpha, graph.tables)
        posterior = model["readout"] @ c2v + channel[:, None]
    decisions = posterior < 0
    return decisions


def score(model, graph, logicals, patterns, counts, llrs, alpha, chunk=2048) -> Dict[str, int]:
    syndromes = ((graph.H @ patterns.astype(np.int64)) % 2).astype(np.uint8)
    packed = np.packbits(syndromes, axis=0).T.copy()
    _, first, inverse = np.unique(packed.view(np.dtype((np.void, packed.shape[1]))).ravel(),
                                  return_index=True, return_inverse=True)
    distinct = syndromes[:, first]
    decisions = np.zeros((patterns.shape[0], distinct.shape[1]), dtype=bool)
    for start in range(0, distinct.shape[1], chunk):
        decisions[:, start:start + chunk] = last_layer_decisions(model, graph, llrs, distinct[:, start:start + chunk], alpha)
    residual = patterns ^ decisions[:, inverse.ravel()]
    syndrome_weight = ((graph.H @ residual) % 2).sum(0)
    logical_weight = ((logicals.astype(np.int64) @ residual) % 2).sum(0)
    failed = ~((syndrome_weight == 0) & (logical_weight == 0))
    coset = (syndrome_weight == 0) & (logical_weight > 0)
    error_weight = patterns.sum(0)
    result = dict(total=int(counts[failed].sum()), coset=int(counts[coset].sum()),
                  convergence=int(counts[failed & ~coset].sum()),
                  weight3_convergence=int(counts[failed & ~coset & (error_weight == 3)].sum()),
                  distinct_failing=int(failed.sum()))
    return result


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, help="run directory: a path, or a codename under data/")
    parser.add_argument("--testing-dir", default="", help="where test_<key>.txt is (default: <run>/testing_data)")
    parser.add_argument("--key", required=True, help="test key, p_<p>_sig_<sigma>_s_<sample>")
    parser.add_argument("--train-key", default="p_0.0015_sig_0.003_s_1")
    parser.add_argument("--arm", required=True, choices=sorted(RUN_TAGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--alpha", nargs="+", default=["learned"], help="learned, rescaled, or numbers")
    parser.add_argument("--couplings", default="test", choices=["test", "train"])
    parser.add_argument("--weights", default="all", help="all, or comma-separated error weights")
    parser.add_argument("--samples", type=int, default=0, help="only the first N samples (0 = all)")
    parser.add_argument("--single-qubit-rescale", type=float, default=0.1)
    arguments = parser.parse_args(argv)

    run_directory = arguments.workdir if os.path.isdir(arguments.workdir) else os.path.join(REPO_ROOT, "data", arguments.workdir)
    testing_directory = arguments.testing_dir or os.path.join(run_directory, "testing_data")
    parity_check_matrix = np.loadtxt(os.path.join(run_directory, "code", "HZ.txt"), dtype=int)
    logicals = np.loadtxt(os.path.join(run_directory, "code", "LZ.txt"), dtype=int)
    samples = read_sample_matrix(os.path.join(testing_directory, f"test_{arguments.key}.txt"))
    if arguments.samples > 0:
        samples = samples[:, :arguments.samples]
    packed = np.packbits(samples, axis=0).T.copy()
    _, first, counts = np.unique(packed.view(np.dtype((np.void, packed.shape[1]))).ravel(),
                                 return_index=True, return_counts=True)
    patterns = samples[:, first].astype(bool)
    if arguments.weights != "all":
        kept = np.isin(patterns.sum(0), [int(weight) for weight in arguments.weights.split(",")])
        patterns, counts = patterns[:, kept], counts[kept]

    def cer_file(key: str) -> str:
        return os.path.join(run_directory, "correlated_weights", f"correlated_weights_{key}.txt")

    rates, pairs, couplings = parse_cer(cer_file(arguments.key))
    train_rates, train_pairs, train_couplings = parse_cer(cer_file(arguments.train_key))
    if arguments.couplings == "train":
        if train_pairs != pairs:
            raise SystemExit("the training key's CER file lists other pairs than the test key's")
        couplings = train_couplings
    llrs = np.full(rates.size, np.log(np.float32(9)), dtype=np.float32)
    graph = TannerGraph(parity_check_matrix)
    if arguments.arm != "nocer":
        llrs = initial_llrs(rates, arguments.single_qubit_rescale)
    if arguments.arm == "cer_enr":
        graph = TannerGraph(parity_check_matrix, pairs, couplings)
    print(f"{arguments.key}: {samples.shape[1]} samples, {patterns.shape[1]} distinct patterns decoded "
          f"(weights {arguments.weights}); alpha*(test) {alpha_star(rates):.3f}, alpha*(train) {alpha_star(train_rates):.3f}")
    for seed in arguments.seeds:
        model_pattern = os.path.join(run_directory, "models", f"neuralbp_weights_*_trained_using_train_"
                                     f"{arguments.train_key}{RUN_TAGS[arguments.arm]}_seed_{seed}.json")
        model_files = sorted(glob.glob(model_pattern))
        if len(model_files) != 1:
            raise SystemExit(f"{len(model_files)} models match {model_pattern}")
        model = load_model(model_files[0], graph)
        if model["check_node"] == "enriched" and model["table"] != "pairwise":
            raise SystemExit(f"{model_files[0]}: only the pairwise table is mirrored")
        for alpha_spec in arguments.alpha:
            alpha = model["alpha"]
            if alpha_spec == "rescaled":
                alpha = model["alpha"] * alpha_star(rates) / alpha_star(train_rates)
            elif alpha_spec != "learned":
                alpha = float(alpha_spec)
            result = score(model, graph, logicals, patterns, counts, llrs, alpha)
            print(f"  {arguments.arm} seed {seed} alpha {alpha:.4f} ({alpha_spec}, {arguments.couplings} J): "
                  f"{result['total']} failures = {result['coset']} coset + {result['convergence']} convergence "
                  f"({result['weight3_convergence']} at weight 3); {result['distinct_failing']} distinct patterns",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

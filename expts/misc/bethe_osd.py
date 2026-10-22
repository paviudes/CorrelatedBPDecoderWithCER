#!/usr/bin/env python3
"""
bethe_osd.py -- BP+OSD-CS whose OSD stage can choose its answer with the Bethe model.

    python3 misc/bethe_osd.py --workdir 72q_BB_cycles_1_sweep \
        --samples-dir ../data/72q_BB_cycles_1_trainable_alpha --key p_0.0015_sig_0.0_s_1

BP is ldpc's (product-sum, parallel, --max-iter, priors = the key's CER single-qubit
rates rescaled to median --rescale, the neural decoder's priors). Where BP does not
converge, the OSD-CS stage is written out here: bits ordered by BP's posterior,
Gaussian elimination, then OSD-0 plus every single flip of the non-pivot bits and
every pair among the first --order of them. Among those candidates it reports three
choices:
  independent  the lowest weight under the priors (what ldpc's OSD picks; printed
               beside ldpc's own BP+OSD as a check of this implementation)
  pairwise     the highest log-probability under independent raw rates + the CER
               file's raw pair couplings J (alpha = 1)
  bethe        the highest log-probability under the Bethe model of the key's
               TRAINING data: sum_c T_c(x_c) + sum_u log p_u(x_u), with T_c the
               check's table from misc/build_bethe_tables.py (bethe_table)
--always also runs the OSD search where BP converged, with BP's answer as one more
candidate (measured 2026-10-22: worse for every choice; the Bethe model then
over-rates correlated alternatives on easy patterns).

reads  <run>/code/HZ.txt, LZ.txt, <run>/correlated_weights/correlated_weights_<key>.txt,
       <samples-dir>/training_data/train_<key>.txt, <samples-dir>/testing_data/test_<key>.txt
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bethe_tables import bethe_table, read_sample_matrix  # noqa: E402
from reference_decoders import load_ldpc_decoders, single_qubit_rates  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_pairs(path):
    pairs, couplings = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("("):
            left, value = line.split(":")
            a, b = left.strip("() ").split(",")
            pairs.append((int(a) - 1, int(b) - 1))
            couplings.append(float(value))
    return np.array(pairs, dtype=int), np.array(couplings)


def osd_candidates(check_matrix, syndrome, llr, order):
    """OSD-CS candidates (rows, 0/1) for one syndrome, from BP's posterior LLRs."""
    n_bits = check_matrix.shape[1]
    bit_order = np.argsort(llr, kind="stable")              # most likely in error first
    augmented = np.concatenate([check_matrix[:, bit_order], syndrome[:, None]], axis=1).astype(np.uint8)
    pivots, rank = [], 0
    for column in range(n_bits):
        rows = np.flatnonzero(augmented[rank:, column])
        if rows.size == 0:
            continue
        pivot_row = rows[0] + rank
        if pivot_row != rank:
            augmented[[rank, pivot_row]] = augmented[[pivot_row, rank]]
        mask = augmented[:, column].astype(bool)
        mask[rank] = False
        augmented[mask] ^= augmented[rank]
        pivots.append(column)
        rank += 1
        if rank == check_matrix.shape[0]:
            break
    pivot_set = set(pivots)
    non_pivots = [column for column in range(n_bits) if column not in pivot_set]
    reduced, reduced_syndrome = augmented[:rank, :n_bits], augmented[:rank, n_bits]
    head = min(order, len(non_pivots))
    flips = [()] + [(j,) for j in non_pivots] + [(non_pivots[a], non_pivots[b])
                                                  for a in range(head) for b in range(a + 1, head)]
    candidates = np.zeros((len(flips), n_bits), dtype=np.uint8)
    for index, flip in enumerate(flips):
        sorted_solution = np.zeros(n_bits, dtype=np.int64)
        sorted_solution[list(flip)] = 1
        sorted_solution[pivots] = (reduced_syndrome + reduced[:, non_pivots].astype(np.int64)
                                   @ sorted_solution[non_pivots]) % 2
        candidates[index, bit_order] = sorted_solution
    return candidates


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", required=True, help="run directory: a path, or a codename under data/")
    parser.add_argument("--samples-dir", default="", help="directory with training_data/ and testing_data/ (default: the run)")
    parser.add_argument("--key", required=True)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=90)
    parser.add_argument("--rescale", type=float, default=0.1, help="median the priors are rescaled to (0 = raw rates)")
    parser.add_argument("--always", action="store_true")
    arguments = parser.parse_args(argv)
    run = arguments.workdir if os.path.isdir(arguments.workdir) else os.path.join(REPO_ROOT, "data", arguments.workdir)
    samples_dir = arguments.samples_dir or run
    check_matrix = np.loadtxt(os.path.join(run, "code", "HZ.txt"), dtype=np.uint8)
    logicals = np.loadtxt(os.path.join(run, "code", "LZ.txt"), dtype=np.int64)
    checks_and_logicals = np.vstack([check_matrix.astype(np.int64), logicals])
    train = read_sample_matrix(os.path.join(samples_dir, "training_data", f"train_{arguments.key}.txt"))
    test = read_sample_matrix(os.path.join(samples_dir, "testing_data", f"test_{arguments.key}.txt"))
    rates = single_qubit_rates(run, arguments.key, check_matrix.shape[1])
    prior = rates
    if arguments.rescale > 0:
        prior = rates * (arguments.rescale / np.median(rates))
    pairs, couplings = parse_pairs(os.path.join(run, "correlated_weights", f"correlated_weights_{arguments.key}.txt"))
    supports = [np.flatnonzero(check_matrix[c]) for c in range(check_matrix.shape[0])]
    tables = [bethe_table(train, support)[0] for support in supports]
    raw = np.clip(train.mean(axis=1), 0.5 / train.shape[1], 1 - 0.5 / train.shape[1])
    raw_weight, prior_weight = np.log((1 - raw) / raw), np.log((1 - prior) / prior)

    def choose(candidates):
        x = candidates.astype(np.int64)
        independent = x @ prior_weight
        base = -(x @ raw_weight)
        pairwise = base + (x[:, pairs[:, 0]] * x[:, pairs[:, 1]]) @ couplings
        bethe = base.astype(np.float64)
        for table, support in zip(tables, supports):
            bethe = bethe + table[x[:, support] @ (1 << np.arange(len(support)))]
        return dict(independent=x[np.argmin(independent)], pairwise=x[np.argmax(pairwise)], bethe=x[np.argmax(bethe)])

    BpDecoder, BpOsdDecoder = load_ldpc_decoders()
    settings = dict(error_channel=list(prior), max_iter=arguments.max_iter, bp_method="product_sum", schedule="parallel")
    bp = BpDecoder(check_matrix, **settings)
    reference = BpOsdDecoder(check_matrix, osd_method="osd_cs", osd_order=arguments.order, **settings)
    packed = np.packbits(test, axis=0).T.copy()
    _, first, counts = np.unique(packed.view(np.dtype((np.void, packed.shape[1]))).ravel(),
                                 return_index=True, return_counts=True)
    names = ("independent", "pairwise", "bethe")
    failures = dict.fromkeys(("ldpc",) + names, 0)
    converged_wrong, reached_osd, cache = 0, 0, {}
    started = time.time()
    for column, count in zip(first, counts):
        error = test[:, column].astype(np.int64)
        syndrome = ((check_matrix.astype(np.int64) @ error) % 2).astype(np.uint8)
        cache_key = syndrome.tobytes()
        if cache_key not in cache:
            bp_output = np.asarray(bp.decode(syndrome), dtype=np.int64)
            entry = dict(ldpc=np.asarray(reference.decode(syndrome), dtype=np.int64), converged=bool(bp.converge))
            if entry["converged"] and not arguments.always:
                entry.update({name: bp_output for name in names})
            else:
                candidates = osd_candidates(check_matrix, syndrome, np.asarray(bp.log_prob_ratios), arguments.order)
                if entry["converged"]:
                    candidates = np.vstack([bp_output.astype(np.uint8)[None, :], candidates])
                entry.update(choose(candidates))
            cache[cache_key] = entry
        entry = cache[cache_key]
        for name in ("ldpc",) + names:
            failures[name] += int(count) * bool(((checks_and_logicals @ ((error + entry[name]) % 2)) % 2).any())
        if entry["converged"]:
            converged_wrong += int(count) * bool(((checks_and_logicals @ ((error + entry["independent"]) % 2)) % 2).any()) \
                if not arguments.always else 0
        else:
            reached_osd += int(count)
    print(f"{arguments.key}: {test.shape[1]} samples, {reached_osd} reach OSD ({time.time() - started:.0f} s)")
    print(f"  ldpc BP+OSD-CS{arguments.order}: {failures['ldpc']}")
    print(f"  this OSD, choosing by  independent weight {failures['independent']}  |  pairwise J {failures['pairwise']}"
          f"  |  Bethe {failures['bethe']}")
    if not arguments.always:
        print(f"  of which BP converged to a wrong answer (OSD never runs): {converged_wrong}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

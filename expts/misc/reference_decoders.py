#!/usr/bin/env python3
"""
reference_decoders.py -- re-run the reference decoders (ldpc BP and BP+OSD) on
test sets, with a configuration you choose.

    python3 misc/reference_decoders.py --workdir 72q_BB_cycles_1_sweep \
        --keys p_0.0005_sig_0.0005_s_1 p_0.0015_sig_0.0015_s_1 \
        --configs bp:cer:500 osd_cs10:cer:500 osd_cs10:mean:500

    # every test set in <run>/testing_data:
    python3 misc/reference_decoders.py --workdir 72q_BB_cycles_1_sweep --all-keys

    # the same samples kept in another run directory:
    python3 misc/reference_decoders.py --workdir 72q_BB_cycles_1_sweep \
        --testing-dir ../data/72q_BB_cycles_1_trainable_alpha/testing_data --keys ...

reads  <run>/code/HZ.txt, <run>/code/LZ.txt
       <run>/testing_data/test_<key>.txt                 (72 rows x N samples; or --testing-dir)
       <run>/correlated_weights/correlated_weights_<key>.txt   (priors `cer` and `mean`)
writes <run>/results/reference_decoders_rerun.csv: one row per (key, config); a
       re-run of the same (key, config) replaces its row. compare_decoders.py
       draws these next to the shipped tables.

A config is <decoder>:<prior>:<max_iter>:
  decoder   bp, osd0, osd_e<order> (e.g. osd_e2) or osd_cs<order> (e.g. osd_cs10)
  prior     cer  -- each qubit at its own single-qubit rate from the CER file
            mean -- every qubit at the mean of those rates
            a number -- every qubit at that rate
  max_iter  BP iterations (BP stops early once its output matches the syndrome)
BP is product-sum with the parallel schedule unless --bp-method / --schedule say
otherwise.

THE SHIPPED TABLES (results/72q_BB_<key>_BP_failure_rates.txt and
72q_BB_<key>BP+OSD_failure_rates_OSD_E_order_2.txt) are `bp:0.1:500` and
`osd_e2:0.1:500`: reproduced to the failure on the five sweep test sets that were
on the laptop (2026-10-21). That is a uniform prior of 0.1 per qubit, 10-200x the
actual single-qubit rates of these sets, and OSD-E of order 2, which searches 2^2
candidates and so is essentially OSD-0.

A failure is a residual (error + correction) that violates any row of [HZ; LZ],
as in refs/bposd.py: a BP output that does not match the syndrome counts. The
decoders are deterministic functions of the syndrome, so each distinct syndrome
is decoded once.

When results/ holds per_sample_failures files for a key, the error weight of
every sample they list is checked against the test file first, so a
--testing-dir holding different samples is refused rather than compared.
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import sys
import time
import types
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RERUN_FILE_NAME = "reference_decoders_rerun.csv"
RERUN_COLUMNS = ["test_key", "test_p", "test_sigma", "test_sample", "config", "decoder", "prior", "max_iter",
                 "osd_method", "osd_order", "bp_method", "schedule", "num_failures", "num_not_converged",
                 "num_samples", "average_logical_error_rate", "label"]
KEY_PATTERN = re.compile(r"^p_(?P<p>[^_]+)_sig_(?P<sigma>[^_]+)_s_(?P<sample>\d+)$")


def load_ldpc_decoders():
    """BpDecoder and BpOsdDecoder without running ldpc/__init__.py, which imports
    its sinter integration and so needs stim even when nothing here uses it."""
    package_spec = importlib.util.find_spec("ldpc")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise SystemExit("ldpc is not installed: pip install ldpc")
    if "ldpc" not in sys.modules:
        package = types.ModuleType("ldpc")
        package.__path__ = list(package_spec.submodule_search_locations)
        sys.modules["ldpc"] = package
    from ldpc.bp_decoder import BpDecoder
    from ldpc.bposd_decoder import BpOsdDecoder
    return BpDecoder, BpOsdDecoder


def read_sample_matrix(path: str) -> np.ndarray:
    """The 0/1 matrix of a test file: rows = qubits, columns = samples."""
    raw = np.fromfile(path, dtype=np.uint8)
    newline_positions = np.flatnonzero(raw == 10)
    if newline_positions.size == 0:
        raise ValueError(f"{path}: no newline found")
    line_length = int(newline_positions[0]) + 1
    if raw.size == newline_positions.size * line_length:
        matrix = raw.reshape(-1, line_length)[:, :-1:2] - 48
        if matrix.min() >= 0 and matrix.max() <= 1:
            return matrix.astype(np.uint8)
    matrix = np.loadtxt(path, dtype=np.int64)
    if matrix.min() < 0 or matrix.max() > 1:
        raise ValueError(f"{path}: entries other than 0/1")
    return matrix.astype(np.uint8)


def single_qubit_rates(run_directory: str, key: str, n_qubits: int) -> np.ndarray:
    """The per-qubit rates of a key's CER file: the lines `i : rate` (pairs are `(i, k) : J`)."""
    path = os.path.join(run_directory, "correlated_weights", f"correlated_weights_{key}.txt")
    rates: Dict[int, float] = {}
    with open(path) as handle:
        for line in handle:
            line_match = re.fullmatch(r"\s*(\d+)\s*:\s*([0-9.eE+-]+)\s*", line)
            if line_match:
                rates[int(line_match[1])] = float(line_match[2])
    missing_qubits = [qubit for qubit in range(1, n_qubits + 1) if qubit not in rates]
    if missing_qubits:
        raise ValueError(f"{path}: no single-qubit rate for qubit(s) {missing_qubits[:5]}")
    rate_vector = np.array([rates[qubit] for qubit in range(1, n_qubits + 1)])
    return rate_vector


def parse_config(config: str) -> Tuple[str, str, int, str, int]:
    """(decoder kind, prior, max_iter, osd_method, osd_order) of `<decoder>:<prior>:<max_iter>`."""
    parts = config.split(":")
    if len(parts) != 3:
        raise SystemExit(f"config {config!r}: write <decoder>:<prior>:<max_iter>, e.g. osd_cs10:cer:500")
    decoder_text, prior, max_iter_text = parts
    if prior not in ("cer", "mean"):
        try:
            prior_value = float(prior)
        except ValueError:
            raise SystemExit(f"config {config!r}: prior is cer, mean or a number")
        if not 0.0 < prior_value < 0.5:
            raise SystemExit(f"config {config!r}: a numeric prior must lie in (0, 0.5)")
    if not max_iter_text.isdigit() or int(max_iter_text) < 1:
        raise SystemExit(f"config {config!r}: max_iter must be a positive integer")
    osd_method, osd_order = "", 0
    decoder_kind = "BP"
    if decoder_text == "bp":
        pass
    elif decoder_text == "osd0":
        decoder_kind, osd_method = "BP+OSD", "osd0"
    else:
        osd_match = re.fullmatch(r"osd_(e|cs)(\d+)", decoder_text)
        if osd_match is None:
            raise SystemExit(f"config {config!r}: decoder is bp, osd0, osd_e<order> or osd_cs<order>")
        decoder_kind, osd_method, osd_order = "BP+OSD", f"osd_{osd_match[1]}", int(osd_match[2])
    return decoder_kind, prior, int(max_iter_text), osd_method, osd_order


def config_label(decoder_kind: str, prior: str, max_iter: int, osd_method: str, osd_order: int) -> str:
    """A legend label, e.g. 'BP+OSD-CS10, CER rates, 500 it'."""
    decoder_text = "BP"
    if decoder_kind == "BP+OSD":
        decoder_text = "BP+OSD-0"
        if osd_method != "osd0":
            decoder_text = f"BP+OSD-{osd_method.split('_')[1].upper()}{osd_order}"
    prior_text = {"cer": "CER rates", "mean": "mean rate"}.get(prior, f"prior {prior}")
    label = f"{decoder_text}, {prior_text}, {max_iter} it"
    return label


def check_same_samples(run_directory: str, key: str, matrix: np.ndarray) -> int:
    """Compare the error weight of every sample a per_sample_failures file lists
    for this key with the test file's column; returns how many were checked."""
    weights = matrix.sum(axis=0)
    n_checked = 0
    for path in sorted(glob.glob(os.path.join(run_directory, "results", f"per_sample_failures_test_{key}_*.csv"))):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                sample_index = int(row["sample_index"])          # 1-based, as Julia writes it
                if sample_index > weights.size or int(weights[sample_index - 1]) != int(row["error_weight"]):
                    raise SystemExit(f"{key}: sample {sample_index} has weight {row['error_weight']} in "
                                     f"{os.path.basename(path)} but not in the test file: these are different "
                                     "samples.")
                n_checked += 1
    return n_checked


def decode_failures(decoder, patterns: np.ndarray, counts: np.ndarray, check_matrix: np.ndarray,
                    checks_and_logicals: np.ndarray) -> Tuple[int, int, int]:
    """(failures, failures where BP did not converge, distinct syndromes), each
    distinct pattern weighted by how often it occurs."""
    syndromes = (patterns.astype(np.int64) @ check_matrix.T.astype(np.int64)) % 2
    corrections: Dict[bytes, Tuple[np.ndarray, bool]] = {}
    n_failures, n_not_converged = 0, 0
    for pattern, syndrome, count in zip(patterns, syndromes, counts):
        syndrome_key = syndrome.astype(np.uint8).tobytes()
        if syndrome_key not in corrections:
            correction = np.asarray(decoder.decode(syndrome.astype(np.uint8)), dtype=np.int64)
            corrections[syndrome_key] = (correction, bool(getattr(decoder, "converge", True)))
        correction, converged = corrections[syndrome_key]
        residual = (pattern.astype(np.int64) + correction) % 2
        if ((checks_and_logicals @ residual) % 2).any():
            n_failures += int(count)
            if not converged:
                n_not_converged += int(count)
    return n_failures, n_not_converged, len(corrections)


def read_rerun_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Re-run ldpc BP / BP+OSD on test sets.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--workdir", required=True, help="run directory: a path, or a codename under data/")
    parser.add_argument("--keys", nargs="+", default=[], help="test keys, p_<p>_sig_<sigma>_s_<sample>")
    parser.add_argument("--all-keys", action="store_true", help="every test_<key>.txt in the testing directory")
    parser.add_argument("--configs", nargs="+", default=["bp:cer:500", "osd_cs10:cer:500", "osd_cs10:mean:500"],
                        help="<decoder>:<prior>:<max_iter> (default: %(default)s)")
    parser.add_argument("--testing-dir", default="", help="where the test files are (default: <run>/testing_data)")
    parser.add_argument("--bp-method", default="product_sum", choices=["product_sum", "minimum_sum"])
    parser.add_argument("--schedule", default="parallel", choices=["parallel", "serial"])
    arguments = parser.parse_args(argv)

    run_directory: str = arguments.workdir
    if not os.path.isdir(run_directory):
        run_directory = os.path.join(REPO_ROOT, "data", arguments.workdir)
    if not os.path.isdir(run_directory):
        print(f"no run directory: {arguments.workdir}", file=sys.stderr)
        return 1
    testing_directory: str = arguments.testing_dir or os.path.join(run_directory, "testing_data")
    keys: List[str] = list(arguments.keys)
    if arguments.all_keys:
        keys += [re.sub(r"^test_|\.txt$", "", os.path.basename(path))
                 for path in sorted(glob.glob(os.path.join(testing_directory, "test_p_*.txt")))]
    keys = sorted(set(keys))
    if not keys:
        print(f"no test keys: give --keys, or --all-keys with test files in {testing_directory}", file=sys.stderr)
        return 1
    for key in keys:
        if KEY_PATTERN.match(key) is None:
            print(f"{key!r} is not a key of the form p_<p>_sig_<sigma>_s_<sample>", file=sys.stderr)
            return 1
        if not os.path.isfile(os.path.join(testing_directory, f"test_{key}.txt")):
            print(f"no test file test_{key}.txt in {testing_directory}", file=sys.stderr)
            return 1
    configs = [(config, *parse_config(config)) for config in arguments.configs]

    BpDecoder, BpOsdDecoder = load_ldpc_decoders()
    check_matrix = np.loadtxt(os.path.join(run_directory, "code", "HZ.txt"), dtype=np.uint8)
    logicals = np.loadtxt(os.path.join(run_directory, "code", "LZ.txt"), dtype=np.uint8)
    checks_and_logicals = np.vstack([check_matrix, logicals]).astype(np.int64)
    n_qubits = check_matrix.shape[1]

    rerun_path = os.path.join(run_directory, "results", RERUN_FILE_NAME)
    new_rows: List[Dict[str, object]] = []
    for key in keys:
        start = time.time()
        matrix = read_sample_matrix(os.path.join(testing_directory, f"test_{key}.txt"))
        if matrix.shape[0] != n_qubits:
            raise SystemExit(f"{key}: {matrix.shape[0]} rows in the test file, {n_qubits} qubits in HZ")
        n_checked = check_same_samples(run_directory, key, matrix)
        packed = np.ascontiguousarray(np.packbits(matrix, axis=0).T)
        packed_view = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
        _, first_columns, counts = np.unique(packed_view, return_index=True, return_counts=True)
        patterns = matrix[:, first_columns].T
        n_samples = int(matrix.shape[1])
        rates = single_qubit_rates(run_directory, key, n_qubits)
        check_text = ""
        if n_checked:
            check_text = f", same samples as the Neural BP tests ({n_checked} listed failures checked)"
        print(f"{key}: {n_samples} samples, {len(patterns)} distinct{check_text} [{time.time() - start:.1f} s]",
              flush=True)
        key_match = KEY_PATTERN.match(key)
        for config, decoder_kind, prior, max_iter, osd_method, osd_order in configs:
            channel = rates.copy()
            if prior == "mean":
                channel = np.full(n_qubits, rates.mean())
            elif prior != "cer":
                channel = np.full(n_qubits, float(prior))
            settings = dict(error_channel=list(channel), max_iter=max_iter, bp_method=arguments.bp_method,
                            schedule=arguments.schedule)
            decoder = BpDecoder(check_matrix, **settings)
            if decoder_kind == "BP+OSD":
                decoder = BpOsdDecoder(check_matrix, osd_method=osd_method, osd_order=osd_order, **settings)
            start = time.time()
            n_failures, n_not_converged, n_syndromes = decode_failures(decoder, patterns, counts, check_matrix,
                                                                       checks_and_logicals)
            label = config_label(decoder_kind, prior, max_iter, osd_method, osd_order)
            print(f"   {config:<20} {label:<34} failures {n_failures:>7}  (BP not converged on {n_not_converged}; "
                  f"{n_syndromes} syndromes, {time.time() - start:.1f} s)", flush=True)
            new_rows.append(dict(test_key=key, test_p=key_match["p"], test_sigma=key_match["sigma"],
                                 test_sample=key_match["sample"], config=config, decoder=decoder_kind, prior=prior,
                                 max_iter=max_iter, osd_method=osd_method, osd_order=osd_order,
                                 bp_method=arguments.bp_method, schedule=arguments.schedule,
                                 num_failures=n_failures, num_not_converged=n_not_converged, num_samples=n_samples,
                                 average_logical_error_rate=n_failures / n_samples, label=label))

    replaced = {(row["test_key"], row["config"], row["bp_method"], row["schedule"]) for row in new_rows}
    kept_rows = [row for row in read_rerun_rows(rerun_path)
                 if (row["test_key"], row["config"], row["bp_method"], row["schedule"]) not in replaced]
    all_rows = kept_rows + new_rows
    all_rows.sort(key=lambda row: (float(row["test_sigma"]) / float(row["test_p"]), float(row["test_p"]),
                                   row["test_key"], row["config"]))
    os.makedirs(os.path.dirname(rerun_path), exist_ok=True)
    with open(rerun_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RERUN_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({column: row[column] for column in RERUN_COLUMNS})
    print(f"wrote {len(new_rows)} row(s), {len(all_rows)} in all -> {rerun_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

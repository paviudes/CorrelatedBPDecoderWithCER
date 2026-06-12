#!/usr/bin/env python3
"""Run CER-vs-no-CER neural-BP ablations on explicit Standard_BP_OSD data.

This script is designed for the checked-in 72q/90q explicit dataset trees under
``pavi/Standard_BP_OSD``. It focuses on a practical question:

    how much of the logical-performance gain comes from the neural decoder
    itself, and how much comes from adding CER-informed pairwise loss terms?

The current collaborator data format stores pairwise CER weights in
``correlated_weights/`` and the single-qubit probabilities separately in
``assigned_probabilities/``. The live Julia/Python file-backed path currently
uses the pairwise CER weights in the loss but falls back to a uniform ``p=0.1``
channel prior. To make that distinction explicit, this script supports multiple
matched neural modes:

- ``no_cer``:
  uniform ``p=0.1`` prior, no CER connectivity, no correlation penalty
- ``cer_loss``:
  uniform ``p=0.1`` prior, CER connectivity and correlation penalty
- ``priors_only``:
  per-qubit prior from ``assigned_probabilities/``, no CER connectivity
- ``priors_plus_cer_loss``:
  per-qubit prior from ``assigned_probabilities/`` plus CER correlation loss

It also evaluates a standard-BP logical-failure baseline using the translated
Python implementation and, when available, loads the stored BP+OSD result for
the same sample as a reference point.

The current CLI defaults are tuned for the more stable follow-up experiment we
want after the initial disappointing run:

- compare ``priors_only`` against ``priors_plus_cer_loss``
- train for longer (`20` epochs)
- use a lower learning rate (`1e-2`)
- reduce the CER correlation-loss schedule by roughly `10x`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PAVI_ROOT = THIS_FILE.parents[2]
PYTHON_WORKSPACE_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PYTHON_WORKSPACE_ROOT / "src"))

from correlated_bp_decoder import (  # noqa: E402
    AnnealingSchedule,
    NachmaniNeuralBP,
    NeuralBPBase,
    TrainingConfig,
    build_initial_llrs,
    check_bp_solutions,
    classical_belief_propagation_decoder,
    load_binary_matrix,
    parse_cer_data,
    random_values_around_one,
    save_trained_neuralbp_model,
    train_nachmani_neuralbp,
)


DATASETS: dict[str, dict[str, object]] = {
    "72q": {
        "code_dir": "72q_BB_code",
        "default_sample": 52,
        "default_p": 0.006,
        "default_q": 0.1,
        "default_q_std": 0.1,
    },
    "90q": {
        "code_dir": "90q_BB_code",
        "default_sample": 36,
        "default_p": 0.006,
        "default_q": 0.1,
        "default_q_std": 0.1,
    },
}

NEURAL_MODES = (
    "no_cer",
    "cer_loss",
    "priors_only",
    "priors_plus_cer_loss",
)
ASSIGNED_QUBIT_PATTERN = re.compile(
    r"^\s*Qubit\s+(\d+)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


@dataclass(slots=True)
class DatasetPaths:
    """Resolved dataset-relative paths for one explicit sample."""

    dataset_root: Path
    parity_check: Path
    logicals: Path
    train_errors: Path
    test_errors: Path
    correlated_weights: Path
    assigned_probabilities: Path
    bp_osd_results: Path


def main(argv: list[str] | None = None) -> int:
    """Run the CER ablation experiment on one explicit sample."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    dataset_spec = DATASETS[args.dataset]
    if args.n_layers < 2:
        raise ValueError("n_layers must be at least 2 so at least one loss layer remains.")

    p_value = float(dataset_spec["default_p"]) if args.p_value is None else args.p_value
    q_mean = float(dataset_spec["default_q"]) if args.q_mean is None else args.q_mean
    q_std = float(dataset_spec["default_q_std"]) if args.q_std is None else args.q_std
    preferred_sample = (
        int(dataset_spec["default_sample"])
        if args.sample <= 0
        else args.sample
    )
    sample = preferred_sample
    paths = _build_paths(args.dataset, dataset_spec, sample, p_value, q_mean, q_std)
    if args.sample <= 0 and not paths.train_errors.is_file():
        available_samples = _discover_available_samples(paths.dataset_root, p_value, q_mean)
        if available_samples:
            sample = available_samples[0]
            paths = _build_paths(args.dataset, dataset_spec, sample, p_value, q_mean, q_std)

    parity_check = load_binary_matrix(paths.parity_check)
    logicals = load_binary_matrix(paths.logicals)
    train_errors_full = load_binary_matrix(paths.train_errors).astype(np.int64, copy=False)
    test_errors_full = load_binary_matrix(paths.test_errors).astype(np.int64, copy=False)

    train_errors, train_subset_info = _select_training_subset(
        train_errors_full,
        requested_samples=args.train_samples,
        seed=args.seed,
        preserved_prefix_size=int(parity_check.shape[1]) + 1,
    )
    test_errors, total_test_samples = _truncate_samples(
        test_errors_full,
        max_samples=args.test_samples,
    )

    assigned_single_qubit_rates = _parse_assigned_single_qubit_rates(
        paths.assigned_probabilities
    )
    cer_data = parse_cer_data(paths.correlated_weights)

    run_tag = (
        f"{args.dataset}_sample_{sample}_nlayers_{args.n_layers}_epochs_{args.n_epochs}"
        f"_train_{train_errors.shape[1]}_test_{test_errors.shape[1]}"
    )
    if args.run_label:
        run_tag = f"{run_tag}_{args.run_label}"

    models_dir = paths.dataset_root / "models"
    results_dir = paths.dataset_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    neural_results: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        neural_results[mode] = _run_neural_mode(
            mode=mode,
            args=args,
            run_tag=run_tag,
            paths=paths,
            parity_check=parity_check,
            logicals=logicals,
            train_errors=train_errors,
            test_errors=test_errors,
            assigned_single_qubit_rates=assigned_single_qubit_rates,
            cer_data=cer_data,
            models_dir=models_dir,
        )

    standard_bp_result = None
    if not args.skip_standard_bp:
        standard_bp_result = _evaluate_standard_bp(
            parity_check,
            logicals,
            test_errors,
            algo=args.standard_bp_algo,
            rounds_per_bp=args.standard_bp_rounds_per_iteration,
            n_iterations_of_bp=args.standard_bp_iterations,
            llr_convergence_threshold=args.standard_bp_llr_convergence_threshold,
            llr_confidence_threshold=args.standard_bp_llr_confidence_threshold,
            weight_soft_constraint=args.standard_bp_weight_soft_constraint,
            initial_error_rate=args.standard_bp_initial_error_rate,
        )

    bp_osd_reference = _load_bp_osd_reference(
        paths.bp_osd_results,
        sample=sample,
        test_samples_used=int(test_errors.shape[1]),
        full_test_samples=int(total_test_samples),
    )

    comparisons = _build_comparisons(
        neural_results,
        standard_bp_result=standard_bp_result,
        bp_osd_reference=bp_osd_reference,
    )

    summary = {
        "implementation": "python",
        "script": str(THIS_FILE),
        "experiment": "cer_ablation_explicit",
        "dataset": args.dataset,
        "dataset_root": str(paths.dataset_root),
        "p_value": p_value,
        "q_mean": q_mean,
        "q_std": q_std,
        "sample": sample,
        "seed": args.seed,
        "n_layers": args.n_layers,
        "n_epochs": args.n_epochs,
        "run_label": args.run_label,
        "device": args.device,
        "modes": list(args.modes),
        "train_samples_requested": args.train_samples,
        "train_samples_used": int(train_errors.shape[1]),
        "train_samples_available": int(train_errors_full.shape[1]),
        "test_samples_requested": args.test_samples,
        "test_samples_used": int(test_errors.shape[1]),
        "test_samples_available": int(total_test_samples),
        "training_subset": train_subset_info,
        "neural_results": neural_results,
        "standard_bp": standard_bp_result,
        "bp_osd_reference": bp_osd_reference,
        "comparisons": comparisons,
    }

    summary_path = results_dir / f"cer_ablation_{run_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the CER ablation runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Train matched neural-BP CER ablations on one explicit Standard_BP_OSD "
            "sample and compare them against a standard-BP logical baseline."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="72q")
    parser.add_argument(
        "--p-value",
        type=float,
        default=None,
        help="Per-qubit ballistic error probability. Defaults to the dataset alias default.",
    )
    parser.add_argument(
        "--q-mean",
        type=float,
        default=None,
        help="Mean conditional neighbour-flip probability. Defaults to the dataset alias default.",
    )
    parser.add_argument(
        "--q-std",
        type=float,
        default=None,
        help="Standard deviation used when drawing neighbour-flip probabilities.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help=(
            "Sample index within the dataset. Use 0 to pick the dataset default, "
            "or the first available sample when the default file is absent."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=NEURAL_MODES,
        default=["priors_only", "priors_plus_cer_loss"],
        help=(
            "Which neural ablation modes to train and evaluate. Defaults to the "
            "more targeted priors-vs-priors+CER follow-up comparison."
        ),
    )
    parser.add_argument("--n-layers", type=int, default=100)
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=20,
        help=(
            "Training epochs. Defaults to the longer follow-up run after the "
            "undertrained 5-epoch comparison."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="How many training patterns to use. Defaults to the requested 10^3 setup.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=100000,
        help="How many test patterns to use. Defaults to the requested 10^5 setup.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--initial-conditions-scale", type=float, default=0.3)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Optimizer learning rate. Lowered from 1e-1 for better conditioning.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--nan-skip-count", type=int, default=5)
    parser.add_argument("--warmup-layers", type=int, default=10)
    parser.add_argument("--loss-layer-temperature-min", type=float, default=0.1)
    parser.add_argument("--loss-layer-temperature-max", type=float, default=5.0)
    parser.add_argument("--loss-layer-temperature-decay", type=float, default=0.9)
    parser.add_argument(
        "--correlation-importance-min",
        type=float,
        default=0.01,
        help="CER loss weight floor. Reduced by 10x from the initial probe.",
    )
    parser.add_argument(
        "--correlation-importance-max",
        type=float,
        default=0.1,
        help="CER loss weight start. Reduced by 10x from the initial probe.",
    )
    parser.add_argument("--correlation-importance-decay", type=float, default=0.1)
    parser.add_argument("--llr-certainty-importance-min", type=float, default=0.001)
    parser.add_argument("--llr-certainty-importance-max", type=float, default=0.01)
    parser.add_argument("--llr-certainty-importance-decay", type=float, default=0.1)
    parser.add_argument("--sparsity-importance-min", type=float, default=0.0)
    parser.add_argument("--sparsity-importance-max", type=float, default=0.01)
    parser.add_argument("--sparsity-importance-decay", type=float, default=0.5)
    parser.add_argument(
        "--skip-standard-bp",
        action="store_true",
        help="Skip the classical logical-failure baseline.",
    )
    parser.add_argument("--standard-bp-algo", choices=("SumProduct", "MinSum"), default="MinSum")
    parser.add_argument(
        "--standard-bp-iterations",
        type=int,
        default=5,
        help="Outer confidence-pruning iterations for the translated BP baseline.",
    )
    parser.add_argument(
        "--standard-bp-rounds-per-iteration",
        type=int,
        default=500,
        help="Inner BP rounds per outer iteration for the translated BP baseline.",
    )
    parser.add_argument("--standard-bp-llr-convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--standard-bp-llr-confidence-threshold", type=float, default=4.0)
    parser.add_argument("--standard-bp-weight-soft-constraint", type=float, default=0.75)
    parser.add_argument(
        "--standard-bp-initial-error-rate",
        type=float,
        default=0.1,
        help="Uniform initial channel error rate used by the translated standard BP baseline.",
    )
    return parser


def _build_paths(
    dataset_key: str,
    dataset_spec: dict[str, object],
    sample: int,
    p_value: float,
    q_mean: float,
    q_std: float,
) -> DatasetPaths:
    """Resolve all dataset-relative paths for one sample."""

    p_str = _format_float_component(p_value)
    q_str = _format_float_component(q_mean)
    q_std_str = _format_float_component(q_std)
    dataset_root = (
        PAVI_ROOT
        / "Standard_BP_OSD"
        / f"{dataset_key}_BB_p_{p_str}_q_{q_str}_std_{q_std_str}_data"
    )
    code_dir = dataset_root / str(dataset_spec["code_dir"])
    return DatasetPaths(
        dataset_root=dataset_root,
        parity_check=code_dir / "HZ.txt",
        logicals=code_dir / "LZ.txt",
        train_errors=dataset_root / "training_data" / f"train_ballistic_p_{p_str}_q_{q_str}_s_{sample}.txt",
        test_errors=dataset_root / "testing_data" / f"test_ballistic_p_{p_str}_q_{q_str}_s_{sample}.txt",
        correlated_weights=dataset_root / "correlated_weights" / f"correlated_weights_p_{p_str}_q_{q_str}_s_{sample}.txt",
        assigned_probabilities=dataset_root / "assigned_probabilities" / f"assigned_probabilities_p_{p_str}_q_{q_str}_s_{sample}.txt",
        bp_osd_results=dataset_root / f"{dataset_key}_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    )


def _format_float_component(value: float) -> str:
    """Format a float in the same compact style used by the collaborator files."""

    return f"{value:.3g}"


def _discover_available_samples(
    dataset_root: Path,
    p_value: float,
    q_mean: float,
) -> list[int]:
    """Return sorted sample indices present in a dataset training directory."""

    training_dir = dataset_root / "training_data"
    if not training_dir.is_dir():
        return []
    p_str = _format_float_component(p_value)
    q_str = _format_float_component(q_mean)
    pattern = re.compile(
        rf"^train_ballistic_p_{re.escape(p_str)}_q_{re.escape(q_str)}_s_(\d+)\.txt$"
    )
    samples: list[int] = []
    for path in training_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            samples.append(int(match.group(1)))
    return sorted(samples)


def _truncate_samples(
    matrix: np.ndarray,
    *,
    max_samples: int,
) -> tuple[np.ndarray, int]:
    """Optionally truncate the sample axis of a binary matrix."""

    total_samples = int(matrix.shape[1])
    if max_samples > 0:
        matrix = matrix[:, : min(max_samples, total_samples)]
    return matrix.astype(np.int64, copy=False), total_samples


def _select_training_subset(
    matrix: np.ndarray,
    *,
    requested_samples: int,
    seed: int,
    preserved_prefix_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subsample a training file while preserving the deterministic prefix.

    The checked-in ballistic training files begin with:

    - one all-zero pattern
    - all weight-1 basis patterns

    When the requested subset is smaller than the full file, we keep that
    prefix intact and sample the remaining columns uniformly without
    replacement from the higher-weight tail.
    """

    total_samples = int(matrix.shape[1])
    if requested_samples <= 0 or requested_samples >= total_samples:
        return matrix.astype(np.int64, copy=False), {
            "strategy": "full_file",
            "preserved_prefix_size": min(preserved_prefix_size, total_samples),
            "sampled_tail_count": max(total_samples - preserved_prefix_size, 0),
            "seed": seed,
        }

    if requested_samples <= preserved_prefix_size:
        subset = matrix[:, :requested_samples]
        return subset.astype(np.int64, copy=False), {
            "strategy": "prefix_only",
            "preserved_prefix_size": requested_samples,
            "sampled_tail_count": 0,
            "seed": seed,
        }

    rng = np.random.default_rng(seed)
    prefix = matrix[:, :preserved_prefix_size]
    tail = matrix[:, preserved_prefix_size:]
    sampled_tail_count = requested_samples - preserved_prefix_size
    tail_indices = np.sort(
        rng.choice(tail.shape[1], size=sampled_tail_count, replace=False)
    )
    subset = np.hstack((prefix, tail[:, tail_indices]))
    return subset.astype(np.int64, copy=False), {
        "strategy": "preserve_prefix_then_sample_tail",
        "preserved_prefix_size": preserved_prefix_size,
        "sampled_tail_count": sampled_tail_count,
        "seed": seed,
    }


def _parse_assigned_single_qubit_rates(path: Path) -> dict[int, float]:
    """Extract the single-qubit error rates from an assigned-probabilities file."""

    single_qubit_error_rates: dict[int, float] = {}
    for raw_line in path.read_text().splitlines():
        match = ASSIGNED_QUBIT_PATTERN.fullmatch(raw_line.strip())
        if match is None:
            continue
        single_qubit_error_rates[int(match.group(1))] = float(match.group(2))
    if not single_qubit_error_rates:
        raise ValueError(f"No single-qubit rates found in {path}")
    return single_qubit_error_rates


def _build_base_for_mode(
    *,
    mode: str,
    parity_check: np.ndarray,
    logicals: np.ndarray,
    n_layers: int,
    assigned_single_qubit_rates: dict[int, float],
    cer_connectivity: np.ndarray,
    cer_strengths: np.ndarray,
    default_error_rate: float = 0.1,
) -> NeuralBPBase:
    """Construct the neural base for one CER ablation mode."""

    dual_matrix = np.vstack((parity_check, logicals))
    n_bits = int(parity_check.shape[1])

    if mode in {"priors_only", "priors_plus_cer_loss"}:
        initial_llrs = build_initial_llrs(
            n_bits,
            assigned_single_qubit_rates,
            default_error_rate=default_error_rate,
        )
    else:
        initial_llrs = np.full(
            n_bits,
            np.float32(np.log((1.0 - default_error_rate) / default_error_rate)),
            dtype=np.float32,
        )

    if mode in {"cer_loss", "priors_plus_cer_loss"}:
        connectivity = cer_connectivity
        strengths = cer_strengths
    else:
        connectivity = np.zeros((0, 2), dtype=np.int64)
        strengths = np.zeros((0,), dtype=np.float32)

    return NeuralBPBase(
        parity_check,
        dual_matrix,
        initial_llrs,
        n_layers,
        connectivity=connectivity,
        correlation_strengths=strengths,
    )


def _run_neural_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    run_tag: str,
    paths: DatasetPaths,
    parity_check: np.ndarray,
    logicals: np.ndarray,
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    assigned_single_qubit_rates: dict[int, float],
    cer_data: Any,
    models_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one neural ablation mode."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base = _build_base_for_mode(
        mode=mode,
        parity_check=parity_check,
        logicals=logicals,
        n_layers=args.n_layers,
        assigned_single_qubit_rates=assigned_single_qubit_rates,
        cer_connectivity=cer_data.connectivity,
        cer_strengths=cer_data.correlation_strengths,
        default_error_rate=args.standard_bp_initial_error_rate,
    )
    device = torch.device(args.device)
    model = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=random_values_around_one(
            (base.nb_weights_c2v_v2c * base.n_layers,),
            scale=args.initial_conditions_scale,
            device=device,
        ),
        weights_llrs=random_values_around_one(
            (base.code_n_bits * base.n_layers,),
            scale=args.initial_conditions_scale,
            device=device,
        ),
        weights_c2v_readout=random_values_around_one(
            (base.nb_weights_c2v_readout,),
            scale=args.initial_conditions_scale,
            device=device,
        ),
    ).to(device)

    train_syndromes = np.mod(parity_check @ train_errors, 2)
    test_syndromes = np.mod(parity_check @ test_errors, 2)
    effective_warmup = min(args.warmup_layers, args.n_layers - 1)
    config = TrainingConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        adam_eps=args.adam_eps,
        max_nan_skips_per_epoch=args.nan_skip_count,
        warmup_loss_layers=effective_warmup,
        julia_loss_compat=True,
        batching_mode="julia_batch_local",
        loss_layer_temperature=AnnealingSchedule(
            maximum=args.loss_layer_temperature_max,
            minimum=args.loss_layer_temperature_min,
            decay=args.loss_layer_temperature_decay,
            direction="down",
        ),
        correlation_importance=AnnealingSchedule(
            maximum=args.correlation_importance_max,
            minimum=args.correlation_importance_min,
            decay=args.correlation_importance_decay,
            direction="down",
        ),
        llr_certainty_importance=AnnealingSchedule(
            maximum=args.llr_certainty_importance_max,
            minimum=args.llr_certainty_importance_min,
            decay=args.llr_certainty_importance_decay,
            direction="down",
        ),
        sparsity_importance=AnnealingSchedule(
            maximum=args.sparsity_importance_max,
            minimum=args.sparsity_importance_min,
            decay=args.sparsity_importance_decay,
            direction="up",
        ),
    )

    train_start = time.perf_counter()
    training_summary = train_nachmani_neuralbp(
        model,
        torch.as_tensor(train_syndromes, dtype=torch.bool),
        torch.as_tensor(train_errors, dtype=torch.bool),
        config,
    )
    training_time_s = time.perf_counter() - train_start

    weights_path = models_dir / f"cer_ablation_{mode}_{run_tag}.json"
    save_trained_neuralbp_model(
        weights_path,
        model,
        metadata={
            "script": THIS_FILE.name,
            "mode": mode,
            "dataset_root": str(paths.dataset_root),
            "seed": args.seed,
            "initial_conditions_scale": args.initial_conditions_scale,
        },
    )

    eval_start = time.perf_counter()
    evaluation = _evaluate_model(
        model,
        parity_check=parity_check,
        dual_matrix=np.vstack((parity_check, logicals)),
        syndromes=test_syndromes,
        errors=test_errors,
        batch_size=args.eval_batch_size,
    )
    evaluation_time_s = time.perf_counter() - eval_start

    logical_error_rate = 1.0 - evaluation["dual_success_rate"]
    return {
        "mode": mode,
        "uses_pairwise_cer_loss": mode in {"cer_loss", "priors_plus_cer_loss"},
        "uses_assigned_single_qubit_priors": mode in {"priors_only", "priors_plus_cer_loss"},
        "weights_file": str(weights_path),
        "training_time_s": training_time_s,
        "evaluation_time_s": evaluation_time_s,
        "logical_success_rate": evaluation["dual_success_rate"],
        "logical_error_rate": logical_error_rate,
        "parity_success_rate": evaluation["parity_success_rate"],
        "dual_success_rate": evaluation["dual_success_rate"],
        "parity_n_correct": evaluation["parity_n_correct"],
        "dual_n_correct": evaluation["dual_n_correct"],
        "epoch_summaries": [
            {
                "epoch": epoch.epoch,
                "mean_loss": epoch.mean_loss,
                "applied_batches": epoch.applied_batches,
                "nan_skip_count": epoch.nan_skip_count,
                "rolled_back": epoch.rolled_back,
                "hyperparameters": asdict(epoch.hyperparameters),
            }
            for epoch in training_summary.epochs
        ],
    }


def _evaluate_model(
    model: NachmaniNeuralBP,
    *,
    parity_check: np.ndarray,
    dual_matrix: np.ndarray,
    syndromes: np.ndarray,
    errors: np.ndarray,
    batch_size: int,
) -> dict[str, int | float]:
    """Evaluate one trained model with both parity and logical validation."""

    parity_correct = 0
    dual_correct = 0
    n_samples = int(errors.shape[1])

    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            chunk_syndromes = torch.as_tensor(
                syndromes[:, start:stop],
                dtype=torch.bool,
                device=model.device,
            )
            chunk_llrs = model.expand_initial_llrs(stop - start)
            chunk_posteriors = model(chunk_llrs, chunk_syndromes)
            chunk_recoveries = chunk_posteriors < 0

            parity = check_bp_solutions(
                parity_check,
                errors[:, start:stop],
                chunk_recoveries,
            )
            dual = check_bp_solutions(
                dual_matrix,
                errors[:, start:stop],
                chunk_recoveries,
            )
            parity_correct += int(parity.sum().item())
            dual_correct += int(dual.sum().item())

    return {
        "parity_n_correct": parity_correct,
        "parity_success_rate": parity_correct / n_samples,
        "dual_n_correct": dual_correct,
        "dual_success_rate": dual_correct / n_samples,
    }


def _evaluate_standard_bp(
    parity_check: np.ndarray,
    logicals: np.ndarray,
    test_errors: np.ndarray,
    *,
    algo: str,
    rounds_per_bp: int,
    n_iterations_of_bp: int,
    llr_convergence_threshold: float,
    llr_confidence_threshold: float,
    weight_soft_constraint: float,
    initial_error_rate: float,
) -> dict[str, Any]:
    """Evaluate the translated classical BP baseline on an explicit test file."""

    start_time = time.perf_counter()
    initial_probabilities = np.full(
        parity_check.shape[1],
        1.0 - initial_error_rate,
        dtype=np.float64,
    )
    n_failures = 0
    total_iterations = 0
    max_iterations_seen = 0

    for sample_index in range(test_errors.shape[1]):
        error = test_errors[:, sample_index]
        syndrome = np.mod(parity_check @ error, 2)
        bp_result = classical_belief_propagation_decoder(
            parity_check,
            logicals,
            error,
            syndrome,
            initial_probabilities,
            rounds_per_bp,
            n_iterations_of_bp,
            algo=algo,
            llr_convergence_threshold=llr_convergence_threshold,
            llr_confidence_threshold=llr_confidence_threshold,
            weight_soft_constraint=weight_soft_constraint,
        )
        n_failures += int(bp_result.is_decoder_failure)
        total_iterations += int(bp_result.current_iteration)
        max_iterations_seen = max(max_iterations_seen, int(bp_result.current_iteration))

    n_samples = int(test_errors.shape[1])
    runtime = time.perf_counter() - start_time
    logical_error_rate = n_failures / n_samples
    return {
        "algo": algo,
        "n_iterations_of_bp": n_iterations_of_bp,
        "rounds_per_bp": rounds_per_bp,
        "llr_convergence_threshold": llr_convergence_threshold,
        "llr_confidence_threshold": llr_confidence_threshold,
        "weight_soft_constraint": weight_soft_constraint,
        "initial_error_rate": initial_error_rate,
        "logical_failures": n_failures,
        "logical_successes": n_samples - n_failures,
        "logical_success_rate": 1.0 - logical_error_rate,
        "logical_error_rate": logical_error_rate,
        "runtime_s": runtime,
        "mean_inner_iterations": total_iterations / n_samples,
        "max_inner_iterations": max_iterations_seen,
    }


def _load_bp_osd_reference(
    path: Path,
    *,
    sample: int,
    test_samples_used: int,
    full_test_samples: int,
) -> dict[str, int | float | bool] | None:
    """Load the stored BP+OSD summary for the chosen sample when present."""

    if not path.is_file():
        return None

    table = np.loadtxt(path, comments="#", ndmin=2)
    matches = table[table[:, 2].astype(np.int64) == sample]
    if matches.shape[0] == 0:
        return None

    failures = int(matches[0, 3])
    return {
        "results_file": str(path),
        "sample": sample,
        "full_test_samples": full_test_samples,
        "logical_failures": failures,
        "logical_success_rate": 1.0 - failures / full_test_samples,
        "logical_error_rate": failures / full_test_samples,
        "comparable_to_full_test_run": test_samples_used == full_test_samples,
    }


def _build_comparisons(
    neural_results: dict[str, dict[str, Any]],
    *,
    standard_bp_result: dict[str, Any] | None,
    bp_osd_reference: dict[str, Any] | None,
) -> dict[str, dict[str, float | None]]:
    """Compute logical-error improvement factors for each neural mode."""

    comparisons: dict[str, dict[str, float | None]] = {}
    standard_error = (
        None if standard_bp_result is None else float(standard_bp_result["logical_error_rate"])
    )
    bp_osd_error = (
        None if bp_osd_reference is None else float(bp_osd_reference["logical_error_rate"])
    )

    for mode, result in neural_results.items():
        mode_error = float(result["logical_error_rate"])
        comparisons[mode] = {
            "improvement_factor_over_standard_bp": _safe_improvement_factor(
                baseline_error=standard_error,
                improved_error=mode_error,
            ),
            "improvement_factor_over_bp_osd": _safe_improvement_factor(
                baseline_error=bp_osd_error,
                improved_error=mode_error,
            ),
        }
    return comparisons


def _safe_improvement_factor(
    *,
    baseline_error: float | None,
    improved_error: float,
) -> float | None:
    """Return baseline/improved when both are meaningful."""

    if baseline_error is None:
        return None
    if improved_error == 0.0:
        return float("inf")
    return baseline_error / improved_error


if __name__ == "__main__":
    raise SystemExit(main())

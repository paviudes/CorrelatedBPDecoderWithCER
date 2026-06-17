#!/usr/bin/env python3
"""Head-to-head Python runner for explicit Standard_BP_OSD datasets.

This script trains the Python neural-BP translation on one explicit error-model
sample from ``pavi/Standard_BP_OSD/`` and evaluates it on the matching test
file. It is intentionally paired with
``pavi/expts/head_to_head_explicit_compare.jl`` so both implementations can be
run on the same dataset, sample index, and hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import tomllib

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PAVI_ROOT = THIS_FILE.parents[2]
PYTHON_WORKSPACE_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PYTHON_WORKSPACE_ROOT / "src"))

from correlated_bp_decoder import (  # noqa: E402
    AnnealingSchedule,
    NachmaniNeuralBP,
    TrainingConfig,
    check_bp_solutions,
    describe_torch_runtime,
    load_base_bp_model,
    load_binary_matrix,
    load_trained_neuralbp_model,
    random_values_around_one,
    resolve_torch_device,
    save_trained_neuralbp_model,
    synchronize_torch_device,
    train_nachmani_neuralbp,
)


DATASETS: dict[str, dict[str, object]] = {
    "72q": {
        "subdir": "72q_BB_p_0.006_q_0.1_std_0.1_data",
        "code_dir": "72q_BB_code",
        "default_sample": 52,
        "bp_osd_results_file": "72q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    },
    "90q": {
        "subdir": "90q_BB_p_0.006_q_0.1_std_0.1_data",
        "code_dir": "90q_BB_code",
        "default_sample": 36,
        "bp_osd_results_file": "90q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    },
}


def main(argv: list[str] | None = None) -> int:
    """Train and evaluate the Python decoder on one explicit dataset sample."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_hyperparams_file(args)
    dataset_spec = DATASETS[args.dataset]
    sample = (
        int(dataset_spec["default_sample"])
        if args.sample <= 0
        else args.sample
    )
    if args.n_layers < 2:
        raise ValueError("n_layers must be at least 2 so at least one loss layer remains.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = _build_paths(args, args.dataset, dataset_spec, sample)
    train_errors, total_train_samples = _load_binary_subset(
        paths["train_errors"],
        args.max_train_samples,
    )
    test_errors, total_test_samples = _load_binary_subset(
        paths["test_errors"],
        args.max_test_samples,
    )
    parity_check = load_binary_matrix(paths["parity_check"])

    base = load_base_bp_model(
        paths["parity_check"],
        paths["logicals"],
        args.n_layers,
        correlation_strengths_file=paths["correlated_weights"],
    )
    device = resolve_torch_device(args.device)
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
    if args.initial_weights_file:
        model = load_trained_neuralbp_model(
            args.initial_weights_file,
            model,
            device=device,
        )
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    train_syndromes = np.mod(parity_check @ train_errors, 2)
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
            direction=args.loss_layer_temperature_direction,
        ),
        correlation_importance=AnnealingSchedule(
            maximum=args.correlation_importance_max,
            minimum=args.correlation_importance_min,
            decay=args.correlation_importance_decay,
            direction=args.correlation_importance_direction,
        ),
        llr_certainty_importance=AnnealingSchedule(
            maximum=args.llr_certainty_importance_max,
            minimum=args.llr_certainty_importance_min,
            decay=args.llr_certainty_importance_decay,
            direction=args.llr_certainty_importance_direction,
        ),
        sparsity_importance=AnnealingSchedule(
            maximum=args.sparsity_importance_max,
            minimum=args.sparsity_importance_min,
            decay=args.sparsity_importance_decay,
            direction=args.sparsity_importance_direction,
        ),
    )

    run_tag = (
        f"{args.dataset}_sample_{sample}_nlayers_{args.n_layers}_epochs_{args.n_epochs}"
    )
    if args.run_label:
        run_tag = f"{run_tag}_{args.run_label}"
    models_dir = paths["dataset_root"] / "models"
    results_dir = paths["dataset_root"] / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_path = models_dir / f"head_to_head_python_{run_tag}.json"
    summary_path = results_dir / f"head_to_head_python_{run_tag}.json"

    synchronize_torch_device(device)
    train_start = time.perf_counter()
    training_summary = train_nachmani_neuralbp(
        model,
        torch.as_tensor(train_syndromes, dtype=torch.bool),
        torch.as_tensor(train_errors, dtype=torch.bool),
        config,
    )
    synchronize_torch_device(device)
    training_time_s = time.perf_counter() - train_start

    save_trained_neuralbp_model(
        weights_path,
        model,
        metadata={
            "script": THIS_FILE.name,
            "dataset": args.dataset,
            "sample": sample,
            "seed": args.seed,
            "initial_conditions_scale": args.initial_conditions_scale,
        },
    )

    test_syndromes = np.mod(parity_check @ test_errors, 2)
    synchronize_torch_device(device)
    eval_start = time.perf_counter()
    evaluation = _evaluate_model(
        model,
        test_syndromes,
        test_errors,
        batch_size=args.eval_batch_size,
    )
    synchronize_torch_device(device)
    evaluation_time_s = time.perf_counter() - eval_start

    bp_osd_reference = _load_bp_osd_reference(
        paths["bp_osd_results"],
        sample,
        int(test_errors.shape[1]),
        total_test_samples,
    )

    payload = {
        "implementation": "python",
        "script": str(THIS_FILE),
        "dataset": args.dataset,
        "dataset_root": str(paths["dataset_root"]),
        "sample": sample,
        "seed": args.seed,
        "n_layers": args.n_layers,
        "n_epochs": args.n_epochs,
        "run_label": args.run_label,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "requested_device": args.device,
        "resolved_device": str(device),
        "torch_runtime": describe_torch_runtime(),
        "initial_conditions_scale": args.initial_conditions_scale,
        "initial_weights_file": (
            None if args.initial_weights_file is None else str(args.initial_weights_file)
        ),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "adam_eps": args.adam_eps,
        "warmup_layers_requested": args.warmup_layers,
        "warmup_layers_used": effective_warmup,
        "train_samples_used": int(train_errors.shape[1]),
        "train_samples_available": int(total_train_samples),
        "test_samples_used": int(test_errors.shape[1]),
        "test_samples_available": int(total_test_samples),
        "training_time_s": training_time_s,
        "evaluation_time_s": evaluation_time_s,
        "weights_file": str(weights_path),
        "parity_n_correct": evaluation["parity_n_correct"],
        "parity_success_rate": evaluation["parity_success_rate"],
        "parity_logical_error_rate": evaluation["parity_logical_error_rate"],
        "dual_n_correct": evaluation["dual_n_correct"],
        "dual_success_rate": evaluation["dual_success_rate"],
        "dual_logical_error_rate": evaluation["dual_logical_error_rate"],
        "epoch_summaries": [
            {
                "epoch": epoch.epoch,
                "mean_loss": epoch.mean_loss,
                "applied_batches": epoch.applied_batches,
                "nan_skip_count": epoch.nan_skip_count,
                "rolled_back": epoch.rolled_back,
            }
            for epoch in training_summary.epochs
        ],
        "bp_osd_reference": bp_osd_reference,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Python head-to-head runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the Python neural-BP translation on one explicit "
            "Standard_BP_OSD sample and save a comparable JSON summary."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="72q")
    parser.add_argument(
        "--codename",
        type=str,
        default="",
        help=(
            "Optional explicit Standard_BP_OSD dataset directory name. "
            "When provided, it overrides the built-in dataset shortcut."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index within the dataset. Use 0 to pick the dataset default.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="",
        help="Optional explicit training-data filename inside training_data/.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="",
        help="Optional explicit testing-data filename inside testing_data/.",
    )
    parser.add_argument(
        "--correlated-weights-file",
        type=str,
        default="",
        help="Optional explicit correlated-weights filename inside correlated_weights/.",
    )
    parser.add_argument(
        "--hyperparams-file",
        type=Path,
        help=(
            "Optional TOML hyperparameter file. Relative paths are resolved "
            "inside the dataset's models/ directory."
        ),
    )
    parser.add_argument("--n-layers", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
        help=(
            "Torch device for training/evaluation. Use 'mps' on Apple Silicon, "
            "'cpu' for the baseline path, or 'auto' to pick the best available "
            "accelerator."
        ),
    )
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--initial-weights-file", type=Path)
    parser.add_argument("--initial-conditions-scale", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--nan-skip-count", type=int, default=5)
    parser.add_argument("--warmup-layers", type=int, default=10)
    parser.add_argument("--loss-layer-temperature-min", type=float, default=0.1)
    parser.add_argument("--loss-layer-temperature-max", type=float, default=5.0)
    parser.add_argument("--loss-layer-temperature-decay", type=float, default=0.9)
    parser.add_argument(
        "--loss-layer-temperature-direction",
        choices=("down", "up"),
        default="down",
    )
    parser.add_argument("--correlation-importance-min", type=float, default=0.1)
    parser.add_argument("--correlation-importance-max", type=float, default=1.0)
    parser.add_argument("--correlation-importance-decay", type=float, default=0.1)
    parser.add_argument(
        "--correlation-importance-direction",
        choices=("down", "up"),
        default="down",
    )
    parser.add_argument("--llr-certainty-importance-min", type=float, default=0.001)
    parser.add_argument("--llr-certainty-importance-max", type=float, default=0.01)
    parser.add_argument("--llr-certainty-importance-decay", type=float, default=0.1)
    parser.add_argument(
        "--llr-certainty-importance-direction",
        choices=("down", "up"),
        default="down",
    )
    parser.add_argument("--sparsity-importance-min", type=float, default=0.0)
    parser.add_argument("--sparsity-importance-max", type=float, default=0.01)
    parser.add_argument("--sparsity-importance-decay", type=float, default=0.5)
    parser.add_argument(
        "--sparsity-importance-direction",
        choices=("down", "up"),
        default="up",
    )
    return parser


def _build_paths(
    args: argparse.Namespace,
    dataset_key: str,
    dataset_spec: dict[str, object],
    sample: int,
) -> dict[str, Path]:
    """Resolve all dataset-relative file paths for one sample."""

    codename = args.codename.strip()
    if codename:
        dataset_root = PAVI_ROOT / "Standard_BP_OSD" / codename
    else:
        dataset_root = PAVI_ROOT / "Standard_BP_OSD" / str(dataset_spec["subdir"])

    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    default_code_dir = dataset_root / "code"
    fallback_code_dir = dataset_root / str(dataset_spec["code_dir"])
    code_dir = default_code_dir if default_code_dir.is_dir() else fallback_code_dir

    if args.train_file:
        train_file = dataset_root / "training_data" / args.train_file
    else:
        train_file = dataset_root / "training_data" / (
            f"train_ballistic_p_0.006_q_0.1_s_{sample}.txt"
        )

    if args.test_file:
        test_file = dataset_root / "testing_data" / args.test_file
    else:
        test_file = dataset_root / "testing_data" / (
            f"test_ballistic_p_0.006_q_0.1_s_{sample}.txt"
        )

    if args.correlated_weights_file:
        cer_file = dataset_root / "correlated_weights" / args.correlated_weights_file
    else:
        cer_file = dataset_root / "correlated_weights" / (
            f"correlated_weights_p_0.006_q_0.1_s_{sample}.txt"
        )
    return {
        "dataset_root": dataset_root,
        "parity_check": code_dir / "HZ.txt",
        "logicals": code_dir / "LZ.txt",
        "train_errors": train_file,
        "test_errors": test_file,
        "correlated_weights": cer_file,
        "bp_osd_results": dataset_root / str(dataset_spec["bp_osd_results_file"]),
    }


def _load_binary_subset(path: Path, max_samples: int) -> tuple[np.ndarray, int]:
    """Load a binary matrix and optionally truncate its sample axis."""

    matrix = load_binary_matrix(path)
    total_samples = int(matrix.shape[1])
    if max_samples > 0:
        matrix = matrix[:, : min(max_samples, total_samples)]
    return matrix.astype(np.int64, copy=False), total_samples


def _evaluate_model(
    model: NachmaniNeuralBP,
    syndromes: np.ndarray,
    errors: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, int | float]:
    """Evaluate one trained model with both parity and dual validation."""

    parity_correct = 0
    dual_correct = 0
    n_samples = errors.shape[1]

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
                model.base.parity_check_matrix,
                errors[:, start:stop],
                chunk_recoveries,
            )
            dual = check_bp_solutions(
                model.base.parity_check_matrix_dual,
                errors[:, start:stop],
                chunk_recoveries,
            )
            parity_correct += int(parity.sum().item())
            dual_correct += int(dual.sum().item())

    return {
        "parity_n_correct": parity_correct,
        "parity_success_rate": parity_correct / n_samples,
        "parity_logical_error_rate": 1.0 - parity_correct / n_samples,
        "dual_n_correct": dual_correct,
        "dual_success_rate": dual_correct / n_samples,
        "dual_logical_error_rate": 1.0 - dual_correct / n_samples,
    }


def _load_bp_osd_reference(
    path: Path,
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
        "success_rate": 1.0 - failures / full_test_samples,
        "logical_error_rate": failures / full_test_samples,
        "comparable_to_full_test_run": test_samples_used == full_test_samples,
    }


def _apply_hyperparams_file(args: argparse.Namespace) -> None:
    """Override CLI defaults from a Julia-style dataset TOML file when given."""

    if args.hyperparams_file is None:
        return

    hyperparams_path = args.hyperparams_file
    if not hyperparams_path.is_absolute():
        codename = args.codename.strip()
        if codename:
            dataset_root = PAVI_ROOT / "Standard_BP_OSD" / codename
        else:
            dataset_root = (
                PAVI_ROOT
                / "Standard_BP_OSD"
                / str(DATASETS[args.dataset]["subdir"])
            )
        hyperparams_path = dataset_root / "models" / hyperparams_path

    payload = tomllib.loads(hyperparams_path.read_text())
    scalar_overrides = {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "max_grad_norm": "max_grad_norm",
        "adam_eps": "adam_eps",
        "batch_size": "batch_size",
        "n_epochs": "n_epochs",
        "warmup_layers": "warmup_layers",
        "initial_conditions_scale": "initial_conditions_scale",
        "nanskip": "nan_skip_count",
    }
    for toml_key, arg_name in scalar_overrides.items():
        if toml_key in payload:
            setattr(args, arg_name, payload[toml_key])

    schedule_overrides = {
        "loss_layer_temperature": "loss_layer_temperature",
        "correlation_importance": "correlation_importance",
        "llr_certainty_importance": "llr_certainty_importance",
        "sparsity_importance": "sparsity_importance",
    }
    for toml_key, prefix in schedule_overrides.items():
        if toml_key not in payload:
            continue
        minimum, maximum, decay, direction = payload[toml_key].split(",")
        if direction not in {"down", "up"}:
            raise ValueError(
                f"Unsupported annealing direction {direction!r} in {hyperparams_path}"
            )
        setattr(args, f"{prefix}_min", float(minimum))
        setattr(args, f"{prefix}_max", float(maximum))
        setattr(args, f"{prefix}_decay", float(decay))
        setattr(args, f"{prefix}_direction", direction)


if __name__ == "__main__":
    raise SystemExit(main())

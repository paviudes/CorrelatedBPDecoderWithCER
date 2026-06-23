"""Thin CLI wrapper for training and evaluating the neural BP decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from ..io import (
    load_base_bp_model,
    load_trained_neuralbp_model,
    save_trained_neuralbp_model,
)
from ..devices import (
    maybe_compile_torch_module,
    resolve_torch_device,
    synchronize_torch_device,
)
from ..neural.nachmani import NachmaniNeuralBP
from ..neural.predict import neuralbp_test_predictions
from ..neural.training import (
    AnnealingSchedule,
    TrainingConfig,
    generate_training_data,
    train_nachmani_neuralbp,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the neural-BP experiment CLI.

    Parameters
    ----------
    argv
        Optional command-line arguments. When omitted, ``sys.argv`` is used.

    Returns
    -------
    int
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the experiment driver."""

    parser = argparse.ArgumentParser(
        description="Train or evaluate the Python neural BP decoder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-synthetic",
        help="Train on synthetic Bernoulli errors generated from the parity-check matrix.",
    )
    _add_shared_model_arguments(train_parser)
    train_parser.add_argument("--n-samples", type=int, required=True)
    train_parser.add_argument("--error-rate", type=float, required=True)
    train_parser.add_argument("--n-epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--max-grad-norm", type=float, default=5.0)
    train_parser.add_argument("--llr-certainty-importance", type=float, default=0.0)
    train_parser.add_argument("--correlation-importance", type=float, default=0.0)
    train_parser.add_argument("--sparsity-importance", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
    )
    _add_torch_compile_arguments(train_parser)
    train_parser.add_argument("--weights-out", type=Path)
    train_parser.set_defaults(handler=_run_train_synthetic)

    evaluate_parser = subparsers.add_parser(
        "evaluate-errors",
        help="Evaluate a model on a file of explicit error patterns.",
    )
    _add_shared_model_arguments(evaluate_parser)
    evaluate_parser.add_argument("--weights-file", type=Path)
    evaluate_parser.add_argument("--test-errors-file", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=4096)
    evaluate_parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default="cpu",
    )
    _add_torch_compile_arguments(evaluate_parser)
    evaluate_parser.set_defaults(handler=_run_evaluate_errors)

    return parser


def _add_shared_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by the train and evaluate commands."""

    parser.add_argument("--parity-check-file", type=Path, required=True)
    parser.add_argument("--logicals-file", type=Path, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--correlation-strengths-file", type=Path)


def _add_torch_compile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared torch.compile toggles to a CLI parser."""

    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Wrap the model with torch.compile before use.",
    )
    parser.add_argument(
        "--torch-compile-backend",
        type=str,
        default="",
        help="Optional torch.compile backend override.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
        help="Optional torch.compile mode.",
    )
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        help="Request fullgraph=True for torch.compile.",
    )
    parser.add_argument(
        "--torch-compile-dynamic",
        action="store_true",
        help="Request dynamic=True for torch.compile.",
    )


def _run_train_synthetic(args: argparse.Namespace) -> None:
    """Train the neural-BP model on synthetic data and print a JSON summary."""

    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    model = _build_model(args).to(device)
    runtime_model = maybe_compile_torch_module(
        model,
        enabled=args.torch_compile,
        backend=args.torch_compile_backend,
        mode=args.torch_compile_mode,
        fullgraph=args.torch_compile_fullgraph,
        dynamic=args.torch_compile_dynamic,
    )
    parity_check = torch.tensor(
        model.base.parity_check_matrix.astype("float32"),
        device=model.device,
    )
    syndromes, expected_recoveries = generate_training_data(
        parity_check,
        args.n_samples,
        args.error_rate,
    )
    synchronize_torch_device(device)
    summary = train_nachmani_neuralbp(
        runtime_model,
        syndromes,
        expected_recoveries,
        TrainingConfig(
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            correlation_importance=AnnealingSchedule(
                args.correlation_importance,
                args.correlation_importance,
            ),
            llr_certainty_importance=AnnealingSchedule(
                args.llr_certainty_importance,
                args.llr_certainty_importance,
            ),
            sparsity_importance=AnnealingSchedule(
                args.sparsity_importance,
                args.sparsity_importance,
            ),
        ),
    )
    synchronize_torch_device(device)
    if args.weights_out is not None:
        save_trained_neuralbp_model(
            args.weights_out,
            model,
            metadata={
                "command": "train-synthetic",
                "n_samples": args.n_samples,
                "error_rate": args.error_rate,
                "n_epochs": args.n_epochs,
            },
        )

    final_epoch = summary.epochs[-1]
    print(
        json.dumps(
            {
                "command": "train-synthetic",
                "n_epochs": args.n_epochs,
                "applied_batches": final_epoch.applied_batches,
                "final_mean_loss": final_epoch.mean_loss,
                "nan_skip_count": final_epoch.nan_skip_count,
                "rolled_back": final_epoch.rolled_back,
                "weights_file": None if args.weights_out is None else str(args.weights_out),
            }
        )
    )


def _run_evaluate_errors(args: argparse.Namespace) -> None:
    """Evaluate the neural-BP model on a file of explicit test errors."""

    device = resolve_torch_device(args.device)
    model = _build_model(args).to(device)
    if args.weights_file is not None:
        model = load_trained_neuralbp_model(
            args.weights_file,
            model,
            device=model.device,
        )
    runtime_model = maybe_compile_torch_module(
        model,
        enabled=args.torch_compile,
        backend=args.torch_compile_backend,
        mode=args.torch_compile_mode,
        fullgraph=args.torch_compile_fullgraph,
        dynamic=args.torch_compile_dynamic,
    )
    synchronize_torch_device(device)
    is_correct = neuralbp_test_predictions(
        runtime_model,
        args.test_errors_file,
        batch_size=args.batch_size,
    )
    synchronize_torch_device(device)
    n_samples = int(is_correct.shape[0])
    n_correct = int(is_correct.sum().item())
    print(
        json.dumps(
            {
                "command": "evaluate-errors",
                "n_samples": n_samples,
                "n_correct": n_correct,
                "success_rate": 0.0 if n_samples == 0 else n_correct / n_samples,
            }
        )
    )


def _build_model(args: argparse.Namespace) -> NachmaniNeuralBP:
    """Build an untrained model from the shared CLI arguments."""

    base = load_base_bp_model(
        args.parity_check_file,
        args.logicals_file,
        args.n_layers,
        correlation_strengths_file=args.correlation_strengths_file,
    )
    return NachmaniNeuralBP(base)


if __name__ == "__main__":
    raise SystemExit(main())

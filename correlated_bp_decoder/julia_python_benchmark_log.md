# Julia / Python Benchmark Log

## Purpose

This note tracks a controlled set of repeated runs on the same explicit
`72q_BB_p_0.010_q_0.001_std_0.01_data` dataset so we can compare:

- Julia neural-BP with `USE_GPU=1`
- Julia neural-BP with `USE_GPU=0`
- the Python translation on the same files

The goal is to separate three questions:

1. Does Metal-backed prediction change Julia wall-clock time materially?
2. Do the Julia GPU and CPU runs produce the same training/evaluation outcome
   when started from the same random seed?
3. How closely does the Python implementation match the Julia reference once
   the dataset, seed, and evaluation files are fixed?

## Fixed Benchmark Setup

- Dataset codename: `72q_BB_p_0.010_q_0.001_std_0.01_data`
- Correlation file: `correlated_weights_p_0.01_q_0.001_s_1.txt`
- Training file: `train_ballistic_p_0.01_q_0.001_s_1.txt`
- Testing file: `test_ballistic_p_0.01_q_0.001_s_1.txt`
- Hidden layers: `100`
- Hyperparameter file: `default_hyperparams.toml`
- Seed: `0`

To keep repeated runs distinct, the Julia CLI now accepts:

- `--seed` for deterministic weight initialization
- `--run_label` to append a suffix to saved model/result filenames

## Julia Commands

GPU-enabled evaluation path:

```bash
cd /Users/quantum/Documents/workspace/projects/research/ML\ for\ decoding/pavi/expts

USE_GPU=1 /Users/quantum/.juliaup/bin/julia --compiled-modules=no --project="./../" neural_bp_experiments.jl \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 100 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test test_ballistic_p_0.01_q_0.001_s_1.txt \
  --seed 0 \
  --run_label julia_gpu_seed0
```

CPU-only reference path:

```bash
cd /Users/quantum/Documents/workspace/projects/research/ML\ for\ decoding/pavi/expts

USE_GPU=0 /Users/quantum/.juliaup/bin/julia --compiled-modules=no --project="./../" neural_bp_experiments.jl \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 100 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test test_ballistic_p_0.01_q_0.001_s_1.txt \
  --seed 0 \
  --run_label julia_cpu_seed0
```

## Python Commands

These Python commands were run from a normal macOS Terminal outside the Codex
sandbox, using the Python `3.13` virtualenv at
`/Users/quantum/.venvs/correlated-bp-mps`.

MPS-enabled path:

```bash
cd /Users/quantum/Documents/workspace/projects/research/ML\ for\ decoding

/Users/quantum/.venvs/correlated-bp-mps/bin/python \
  pavi/correlated_bp_decoder/scripts/head_to_head_explicit_compare.py \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --hyperparams-file default_hyperparams.toml \
  --correlated-weights-file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train-file train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test-file test_ballistic_p_0.01_q_0.001_s_1.txt \
  --sample 1 \
  --n-layers 100 \
  --seed 0 \
  --device mps \
  --run-label mps_full_seed0_outside_sandbox
```

CPU-only reference path:

```bash
cd /Users/quantum/Documents/workspace/projects/research/ML\ for\ decoding

/Users/quantum/.venvs/correlated-bp-mps/bin/python \
  pavi/correlated_bp_decoder/scripts/head_to_head_explicit_compare.py \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --hyperparams-file default_hyperparams.toml \
  --correlated-weights-file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train-file train_ballistic_p_0.01_q_0.001_s_1.txt \
  --test-file test_ballistic_p_0.01_q_0.001_s_1.txt \
  --sample 1 \
  --n-layers 100 \
  --seed 0 \
  --device cpu \
  --run-label cpu_full_seed0_outside_sandbox
```

## Results Log

| Run label | Impl | USE_GPU | Seed | External wall time | Script runtime | Correct / total | Output CSV | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `julia_gpu_seed0` | Julia | `1` | `0` | `193.66s` | n/a | n/a | none | crashed in `Enzyme` / `GPUCompiler` with `signal 11` before training completed |
| `julia_cpu_seed0` | Julia | `0` | `0` | n/a | n/a | n/a | none | abandoned because the shell defaulted to `JULIA_NUM_THREADS=1`, which is not a representative CPU baseline |
| `julia_cpu_seed0_threads18` | Julia | `0` | `0` | `2558.11s` | `2550.29s` | `99566 / 100000` | `simulation_results_test_ballistic_p_0.01_q_0.001_s_1_nlayers_100_epochs_10_trained_using_train_ballistic_p_0.01_q_0.001_s_1_julia_cpu_seed0_threads18.csv` | finished cleanly; logical error rate `0.00434`; much slower than Pavi's saved `378.62s` result on the same dataset |
| `python_seed0` | Python | n/a | `0` | `~547s` | `546.51s` (`367.51s` train + `179.00s` eval) | `98341 / 100000` (dual), `99622 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_julia_dataset_seed0.json` | same sample/files and TOML as Julia; dual logical error rate `0.01659`, parity logical error rate `0.00378` |
| `python_mps_full_seed0_outside_sandbox` | Python | `mps` | `0` | n/a | `472.93s` (`466.16s` train + `6.77s` eval) | `98386 / 100000` (dual), `99587 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_mps_full_seed0_outside_sandbox.json` | verified outside Codex sandbox with `resolved_device = mps`; evaluation became dramatically faster, while training was slower than CPU and parity accuracy drifted slightly |
| `python_cpu_full_seed0_outside_sandbox` | Python | `cpu` | `0` | n/a | `551.96s` (`362.89s` train + `189.08s` eval) | `98341 / 100000` (dual), `99622 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_cpu_full_seed0_outside_sandbox.json` | matched the earlier CPU benchmark closely; this is the clean outside-sandbox CPU reference for the Python comparison |

Important comparison note:

- `neural_bp_experiments.jl` reports success using the plain parity-check matrix, not the dual matrix.
- The Python head-to-head runner records both:
  - `parity_*` for the apples-to-apples Julia comparison
  - `dual_*` as a stricter diagnostic that catches logical-operator mismatches

## Julia GPU Diagnosis

- Outside the Codex sandbox, Julia can see Metal correctly when launched with
  `USE_GPU=1` and `--compiled-modules=no`: `CorrelatedBPDecoderWithCER` reports
  `(USE_GPU, METAL_LOADED) = (true, true)`, and `Metal.functional()` returns
  `true`.
- Inside the Codex sandbox, bypassing `juliaup` and calling the Julia binary
  directly still fails during GPU initialization because `GPUCompiler` / `Scratch`
  cannot open `~/.julia/logs/scratch_usage.toml`. So the sandbox does interfere
  with Julia's GPU startup path.
- However, the sandbox is not the whole story. A small outside-sandbox Julia GPU
  prediction smoke test on `256` samples reaches:

  ```text
  Using GPU for predictions with batch size = 64. Total samples = 256.
  ```

  and then fails later in the Metal codegen path with an `InvalidIRError` from
  `GPUArrays.gpu_getindex_kernel`, including unsupported calls such as
  `gpu_gc_pool_alloc`.
- Conclusion: the Julia GPU problem is only partly a sandbox issue. The sandbox
  blocks some Julia GPU startup behavior, but even outside the sandbox the
  current Julia/Metal forward path still hits a real `Metal` / `GPUCompiler` /
  `GPUArrays` compilation error.

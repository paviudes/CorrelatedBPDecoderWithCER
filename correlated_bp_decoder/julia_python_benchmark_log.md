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

Current status:

- the vectorized-loss path is the current default Python implementation
- the `torch.compile` path is optional and opt-in only
- enabling `torch.compile` requires passing `--torch-compile` explicitly

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

MPS-enabled path with vectorized loss
(current default Python path; no extra flag is required):

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
  --run-label mps_full_seed0_vectorized_loss
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
| `python_cpu_full_seed0_vectorized_loss` | Python | `cpu` | `0` | n/a | `311.32s` (`179.95s` train + `131.37s` eval) | `98341 / 100000` (dual), `99622 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_cpu_full_seed0_vectorized_loss.json` | after vectorizing the layer-wise loss computation; same decode metrics as the earlier CPU run, with a large training speedup |
| `python_mps_full_seed0_vectorized_loss` | Python | `mps` | `0` | n/a | `152.78s` (`149.53s` train + `3.25s` eval) | `98419 / 100000` (dual), `99629 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_mps_full_seed0_vectorized_loss.json` | after vectorizing the layer-wise loss computation; large speedup on both training and evaluation, and slightly better decode metrics on this run |
| `python_cpu_full_seed0_vectorized_loss_compile_default` | Python | `cpu` | `0` | n/a | `445.15s` (`247.78s` train + `197.37s` eval) | `98349 / 100000` (dual), `99610 / 100000` (parity) | `head_to_head_python_72q_sample_1_nlayers_100_epochs_10_cpu_full_seed0_vectorized_loss_compile_default.json` | `torch.compile` enabled with default mode; slower than the uncompiled vectorized baseline end-to-end |
| `python_mps_full_seed0_vectorized_loss_compile_default` | Python | `mps` | `0` | n/a | n/a | n/a | none | `torch.compile` enabled with default mode; failed during compiled backward on the MPS/Metal path |

Important comparison note:

- `neural_bp_experiments.jl` reports success using the plain parity-check matrix, not the dual matrix.
- The Python head-to-head runner records both:
  - `parity_*` for the apples-to-apples Julia comparison
  - `dual_*` as a stricter diagnostic that catches logical-operator mismatches

Vectorized-loss summary:

- CPU improved from `551.96s` to `311.32s` total.
- MPS improved from `472.93s` to `152.78s` total.
- The speedup came mostly from training, which is consistent with removing the
  Python-layer loss loop that used to run once per unfolded BP layer.

`torch.compile` summary:

- On CPU, default `torch.compile` was a regression for this workload:
  `311.32s` uncompiled vectorized -> `445.15s` compiled.
- On MPS, default `torch.compile` did not complete. The run failed in the
  compiled backward path with a Metal compiler error during autograd codegen.
- Recommendation for now: keep `torch.compile` disabled by default.

## Proposed Next Steps

1. Keep the vectorized-loss implementation as the default Python path, since it
   produced the largest confirmed speedup without harming decoding quality.
2. Leave `torch.compile` as an experimental opt-in only. Do not enable it in
   default scripts or collaborator-facing commands unless a specific mode is
   shown to help.
3. If we revisit `torch.compile`, test CPU-only `--torch-compile-mode
   reduce-overhead` first on the same `72q` benchmark, since the default mode
   was a regression.
4. Do not prioritize `torch.compile` on MPS until the current compiled-backward
   Metal failure is better understood or fixed upstream.
5. Next optimization target after the loss vectorization should be the unfolded
   forward pass itself, especially reducing Python-side layer orchestration and
   buffer allocation overhead in `nachmani.py`.
6. If more performance is needed after that, benchmark alternative sparse
   kernels or preallocated message/readout buffers before attempting larger
   architectural changes.

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

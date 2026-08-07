# CorrelatedBPDecoderWithCER

## Introduction and Background

This repository contains the implementation of a generalized Belief Propagation (BP) decoder by training a Neural Network. It includes new ideas that will eventually be part of a paper, aimed at enhancing decoder performance under spatially correlated errors using [Cycle Error Reconstruction](https://www.nature.com/articles/s41467-019-13068-7). The transformation from standard belief propagation to a Neural Network is based on ideas discussed in:

- [Learning to Decode Linear Codes Using Deep Learning (arXiv:1607.04793)](https://arxiv.org/abs/1607.04793)
- [Neural Belief-Propagation Decoders for Quantum Error-Correcting Codes (PRL 122, 200501)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.200501)

We use a correlated error model based on the Ballistic noise model discussed in [Quantum 3, 131 (2019)](https://quantum-journal.org/papers/q-2019-04-08-131/).

---

## Installation

The package is built in Julia. [Install Julia](https://julialang.org/downloads/) before proceeding.

When using the package for the first time:

```sh
git clone git@github.com:paviudes/CorrelatedBPDecoderWithCER.git
cd CorrelatedBPDecoderWithCER
julia
```

Then, inside the Julia REPL:

```
]                    # enter package mode
activate .           # activate the environment
instantiate          # install all dependencies
resolve              # clean up
<backspace>          # exit package manager
exit(0)              # exit the REPL
```

---

## Running

### Single Execution

Navigate to the `expts` directory and run:

```sh
julia --project="./../" neural_bp_experiments.jl \
  --codename <codename> \
  --n_hidden_layers <n_hidden_layers> \
  --n_epochs <n_epochs> \
  --batch_size <batch_size> \
  --correlation_strengths_file <correlation_strengths_file> \
  --train <training_errors_file> \
  --test <testing_errors_file> \
  --retrain <retrain> \
  --learning_rate <learning_rate> \
  --max_grad_norm <max_grad_norm>
```

| Argument | Type | Description |
|---|---|---|
| `codename` | String | Codename for the experiment, used to locate input data and save outputs. |
| `n_hidden_layers` | Int | Number of hidden layers in the Neural BP model. |
| `n_epochs` | Int | Number of training epochs. |
| `batch_size` | Int | Batch size for training. |
| `correlation_strengths_file` | String | File containing correlation strengths (located in `correlated_weights/`). |
| `use_CER` | Bool | Use correlated-error-rate priors (hyperparameters TOML key, default `true`). Set `false` to ignore `correlated_weights/` even if present: single-qubit priors become p=0.1 and the correlation loss term is dropped. No-CER outputs (results CSVs and model weights) are tagged `_no_cer` so they never overwrite the CER runs. With `use_CER = true`, submission errors out if `correlated_weights/` is missing. |
| `training_errors_file` | String | File containing training errors (located in `training_data/`). |
| `testing_errors_file` | String | File containing testing errors (located in `testing_data/`). |
| `retrain` | Bool | Whether to retrain the model (`true`/`false`). |
| `seed` | Int | **Optional.** RNG seed for a reproducible training run (hyperparameters TOML key, or `--seed <n>` which overrides it). Seeds the global RNG before the initial weights are drawn and before any batch is sampled, and appends `_seed_<n>` to the weights JSON, the results CSV and the debug CSVs so runs differing only in seed never collide. **Unset by default**, in which case the RNG is not touched and behaviour is exactly as before. Two unseeded runs of an identical configuration measured 309 vs 620 logical failures out of 10^6 — a 2× spread from RNG alone — so fix a set of seeds and pool across them rather than comparing single runs. Reproducibility assumes single-threaded BLAS: `src/CorrelatedBPDecoderWithCER.jl` calls `BLAS.set_num_threads(Threads.nthreads())`, so keep `JULIA_NUM_THREADS=1` (as the sweep scripts do) if you need bit-identical results. |
| `learning_rate` | Float | Learning rate for training. |
| `max_grad_norm` | Float | Maximum gradient norm for clipping during training. |

#### Required Directory Structure

Before running, ensure that `./../data/<codename>/` exists and contains the following subdirectories:

```
<codename>/
├── code/                    # HX.txt, HZ.txt, LX.txt, LZ.txt
├── correlated_weights/      # correlation strengths file (required when use_CER = true; ignored when false)
├── training_data/           # training errors file
└── testing_data/            # testing errors file
```

For example:

```
pavi@Pavithrans-MacBook-Air 72q_BB_p_0.006_q_0.1_std_0.1 % tree -L 1
.
├── assigned_probabilities
├── code
├── commands.txt
├── correlated_weights
├── models
├── results
├── run_2026-05-05_20-58-59.sh
├── testing_data
└── training_data
```

Outputs are written to:
- `./../data/<codename>/models/` — trained Neural BP model
- `./../data/<codename>/results/` — experiment results

---

### Batch Execution

For running simulations locally or on a cluster over a range of Ballistic error model parameters, edit `main()` in `expts/batch_run.jl`, then `cd` into `expts` and run:

```
julia
```

Inside the Julia REPL:

```
]activate ./../      # activate the environment
<backspace>          # exit package manager
using CorrelatedBPDecoderWithCER
include("batch_run.jl")
main()
```

This generates a `commands.txt` file and a run script under `./../data/<codename>/`, and prints a command to run them:

```
Run simulations with:
bash ./../data/<codename>/<name of the script>
```

For example:

```julia
julia> include("batch_run.jl")
main (generic function with 1 method)

julia> main()
56 commands written to: ./../data/72q_BB_p_0.006_q_0.1_std_0.1/commands.txt

Run simulations with:
bash ./../data/72q_BB_p_0.006_q_0.1_std_0.1/run_2026-05-05_20-46-54.sh
```

Exit the REPL and run the printed command on the shell:

```sh
bash ./../data/72q_BB_p_0.006_q_0.1_std_0.1/run_2026-05-05_20-46-54.sh
```

#### Cluster Backends

The `cluster_backend` option in `main()` of `batch_run.jl` controls how jobs are submitted:

| Backend | Description |
|---|---|
| `local` | Runs jobs sequentially on your local machine. |
| `Google_VM` | Same as `local` — run the `bash` command inside a `screen` session to keep it running after logout. |
| `SLURM` | For Slurm-based clusters (e.g., Compute Canada). Prints an `sbatch` command to submit the job array, e.g. `sbatch ./../data/72q_BB_p_0.006_q_0.1_std_0.1/run_2026-05-05_20-46-54.sh`. |

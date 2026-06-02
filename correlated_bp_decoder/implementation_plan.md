# Correlated BP Decoder Python Translation Plan

## Goal

Create a self-contained Python package workspace that ports the active neural belief-propagation path from Pavi's Julia repository into a concise, testable, and idiomatic codebase. The workspace root will be `correlated_bp_decoder/`, which is a shortened Pythonic form of the Julia package name `CorrelatedBPDecoderWithCER`.

Within that workspace, use a `src/` layout for importable code and a sibling `tests/` directory for `pytest`, so all Python translation work stays contained under `pavi/correlated_bp_decoder/`.

The first milestone should cover the live neural decoder path:

- parity-check and logical-operator loading
- CER / correlated-weight parsing
- Tanner-graph-derived neural graph construction
- Nachmani-style neural BP forward pass
- loss computation
- training loop
- prediction / evaluation helpers

The first milestone should explicitly defer or treat as reference-only:

- `src/neuralbp.jl` (archival alternative implementation)
- `src/legacy.jl` except where it clarifies behavior
- `src/printfuns.jl` (stale with respect to current model fields)
- `src/forward_gpu.jl` (Metal-specific inference path)
- plotting, cluster helpers, and one-off analysis scripts

## Current Status

As of this branch state:

- Phase 1 is implemented:
  - CER parsing
  - parity-check and logical-operator loading
  - Tanner-graph construction
  - neural graph compilation
- Phase 2 is implemented:
  - stable BP math helpers
  - `torch.nn.Module` forward path
  - batch-first convenience APIs
  - device-aware regression coverage
- Phase 3 is implemented:
  - loss functions
  - annealing schedules
  - training loop
  - optimizer-step coverage on toy fixtures
- Phase 4 is implemented:
  - prediction helpers
  - structured JSON checkpoint save/load support
  - legacy Julia weight-file compatibility
  - thin training/evaluation CLI wrapper
- Phase 5 is implemented:
  - classical BP baseline
  - trim-constraint simplification path
  - Julia regression coverage for the Hamming-code baseline
- Validation note as of June 2, 2026:
  - phases 1-5 can be treated as complete for the scoped Python translation milestone
  - `pytest -q pavi/correlated_bp_decoder/tests` passed with `36 passed, 1 skipped`
  - the Python tests now track the Julia suite more closely, including:
    - functional vs in-place message-update parity
    - all-ones neural-BP parity with standard BP at the final layer
    - Julia-aligned Hamming-code baseline regressions
    - loss fixtures checked against the live Julia formulas
- Remaining work is optional extension territory:
  - exact parity for older file-backed Julia training workflows and dataset layouts
  - CSS wrappers and broader experiment parity
  - any additional archival Julia helpers we decide are still worth porting

## Head-to-Head Comparison Scripts

To compare the translated Python implementation against the Julia reference on
the same explicit error-model data, this workspace now includes a paired set of
single-sample runners:

- `pavi/expts/head_to_head_explicit_compare.jl`
  - trains the Julia `NachmaniNeuralBP` implementation directly from one
    `Standard_BP_OSD` training file
  - evaluates the matching test file in batches
  - saves trained weights and a JSON run summary under the selected dataset's
    `models/` and `results/` directories
- `pavi/correlated_bp_decoder/scripts/head_to_head_explicit_compare.py`
  - performs the same direct file-backed workflow using the Python translation
  - uses Julia-style random initialization around `1.0` so the starting point is
    closer to the reference implementation
  - saves trained weights and a JSON run summary alongside the Julia outputs

These scripts are meant to bypass the older CLI-layout assumptions and act as a
reproducible bridge between the two implementations:

- both target the same `Standard_BP_OSD/<dataset>/` tree
- both default to the same representative sample for each dataset
- both expose the same main hyperparameters:
  - layer count
  - epoch count
  - batch size
  - optional caps on the number of training and testing samples
  - annealing schedule values
- both report two evaluation views:
  - parity-check success rate, matching the current Julia prediction helper
  - dual-matrix success rate, which is stricter and useful for debugging logical
    discrepancies
- both now also support:
  - `--run-label` to keep multiple comparison runs distinct
  - `--initial-weights-file` to force Python and Julia to start from the exact
    same serialized weights rather than only sharing a nominal RNG seed

### Shared-Initialization Rerun

As of June 2, 2026, the latest head-to-head rerun used:

- dataset `72q`
- sample `52`
- `50` layers
- `5` epochs
- shared initial weights file:
  - `pavi/Standard_BP_OSD/72q_BB_p_0.006_q_0.1_std_0.1_data/models/head_to_head_initial_72q_sample_52_nlayers_50_seed_0_legacy.json`
- run label:
  - `rerun_shared_init`

Result summaries:

- Python:
  - parity success rate `0.99633`
  - dual success rate `0.99452`
  - training time about `71.3s`
  - evaluation time about `109.6s`
  - one unstable epoch at the end:
    - epoch 5 rolled back after `6` skipped batches
- Julia:
  - parity success rate `0.99901`
  - dual success rate `0.99631`
  - training time about `184.9s`
  - evaluation time about `50.9s`
  - repeated instability across later epochs:
    - epochs 2-5 each hit the `6` skipped-batch rollback threshold

Interpretation:

- forcing exact shared initialization narrows the comparison more meaningfully
  than seed-matching alone
- the Python translation is now close to the Julia reference on this benchmark,
  but still slightly below it
- the remaining discrepancy appears to be smaller than the original loss-path
  mismatch and is likely tied to residual optimizer / training-dynamics
  differences rather than a large structural translation error

## Translation Principles

1. Keep the algorithmic structure close to the live Julia files when that improves traceability and validation.
2. Deviate from the Julia layout when a Python split reduces duplication or keeps each file focused on one responsibility.
3. Use standard Python naming:
   - `snake_case` for functions, methods, variables, and modules
   - `PascalCase` for classes
   - uppercase names only for true constants
4. Use NumPy-style docstrings for every public class, function, and method.
5. Prefer shared helpers over copy-pasted logic, especially for:
   - edge-index sparse accumulation
   - stable BP nonlinearities
   - tensor shape normalization
6. Keep public APIs small and explicit; avoid "god objects" that mix data loading, model definition, training, and evaluation.
7. Favor typed dataclasses and small configuration objects over large unstructured dictionaries where practical.

## Proposed Python Stack

- `numpy` for file-backed arrays, preprocessing, and deterministic fixtures
- `torch` for model parameters, forward passes, autodiff, and optimization
- `pathlib` for paths
- `dataclasses` and type hints for structural clarity

Rationale:

- PyTorch is the cleanest replacement for the Julia `Flux` + `Optimisers` + `Enzyme` stack.
- The neural model is fundamentally a trainable message-passing network, so `torch.nn.Module` is a natural fit.
- The Julia code already uses explicit edge lists for efficient sparse-style multiplies; we can mirror that with `torch` tensors and `index_add_` rather than rebuilding dense adjacency matrices during every forward pass.

## Package Layout

The package should grow into the following structure:

```text
correlated_bp_decoder/
    implementation_plan.md
    pyproject.toml
    src/
        correlated_bp_decoder/
            __init__.py
            cer.py
            io.py
            math_utils.py
            tanner_graph.py
            codes.py
            standard_bp.py
            neural/
                __init__.py
                base.py
                nachmani.py
                losses.py
                training.py
                predict.py
            experiments/
                __init__.py
                run_neural_bp.py
    tests/
        README.md
        test_forward.py
        test_bp_algo.py
        test_bp_utils.py
        test_loss.py
        test_predict.py
        test_experiments.py
        test_classical_bp.py
        test_trim_constraints.py
        test_cer.py
        test_tanner_graph.py
        test_base.py
        test_io.py
```

### File Responsibilities

- `src/correlated_bp_decoder/cer.py`
  - Port `parse_cer_data(...)`.
  - Hold CER-specific parsing and normalization helpers only.

- `src/correlated_bp_decoder/io.py`
  - Load parity-check matrices, logical operators, and saved model weights.
  - Provide the Python equivalent of `load_base_BP_model(...)`.
  - Keep filesystem and serialization concerns out of model classes.

- `src/correlated_bp_decoder/math_utils.py`
  - Port numerically stable helpers from `utils.jl`.
  - Centralize the edge-wise sparse accumulation kernel used by the neural model.

- `src/correlated_bp_decoder/tanner_graph.py`
  - Define `TannerGraph`.
  - Build reusable graph views from a parity-check matrix.

- `src/correlated_bp_decoder/codes.py`
  - Define `ClassicalCode` and `QuantumCode` only if they are needed by the translated workflows.
  - Keep CSS-specific helper logic separate from neural model internals.

- `src/correlated_bp_decoder/standard_bp.py`
  - Optional first-pass scope.
  - Port the plain BP decoder only after the neural path is stable, unless we need it earlier for regression checks.

- `src/correlated_bp_decoder/neural/base.py`
  - Define `NeuralBPBase`.
  - Build the edge-index representation derived from the parity-check matrix.

- `src/correlated_bp_decoder/neural/nachmani.py`
  - Define `NachmaniNeuralBP`.
  - Hold the trainable forward-pass implementation and layer helper methods.

- `src/correlated_bp_decoder/neural/losses.py`
  - Port the syndrome, certainty, sparsity, and correlation penalties.
  - Keep aggregation across layers here rather than inside the model class.

- `src/correlated_bp_decoder/neural/training.py`
  - Port the training loop, annealing schedule, gradient clipping, and checkpoint rollback logic.

- `src/correlated_bp_decoder/neural/predict.py`
  - Port posterior thresholding, recovery checking, and test-set evaluation helpers.

- `src/correlated_bp_decoder/experiments/run_neural_bp.py`
  - Replace the Julia experiment script with a thin Python entry point once the library API is stable.

- `tests/`
  - Hold the Python test suite for the translation.
  - Reproduce the existing Julia test suite file-for-file wherever feasible.
  - Live inside the same Python workspace so the translation is self-contained.

## Julia-to-Python Mapping

Use the live Julia files as the primary translation reference:

- `src/neuralbase.jl` -> `src/correlated_bp_decoder/neural/base.py`
- `src/nachmani.jl` -> `src/correlated_bp_decoder/neural/nachmani.py`
- `src/utils.jl` -> `src/correlated_bp_decoder/math_utils.py`
- `src/loss.jl` -> `src/correlated_bp_decoder/neural/losses.py`
- `src/train.jl` -> `src/correlated_bp_decoder/neural/training.py`
- `src/predict.jl` -> `src/correlated_bp_decoder/neural/predict.py`
- `src/log_model.jl` -> `src/correlated_bp_decoder/io.py`
- `src/tanner_graph.jl` -> `src/correlated_bp_decoder/tanner_graph.py`
- `src/quantum_code.jl` -> `src/correlated_bp_decoder/codes.py`
- `src/bp_algo.jl` -> `src/correlated_bp_decoder/standard_bp.py` when we decide to port the baseline decoder

## Shape and Data Conventions

Use the following conventions from the start so the package stays consistent:

- Public batch-facing arrays should be batch-first where practical:
  - bit LLRs: `(n_samples, n_bits)`
  - syndromes: `(n_samples, n_checks)`
- Internal message-passing helpers may transpose once into Julia-like `(n_bits, n_samples)` or `(n_edges, n_samples)` layouts if that keeps the port faithful and simpler.
- Boolean parity-check style data should stay boolean or integer-valued until moved into differentiable `torch.float32` tensors.
- The edge ordering used to build the neural graph must be deterministic and documented, because it defines weight ordering and checkpoint compatibility.

## Implementation Phases

### Phase 1: Scaffold and Core Data Model

Deliverables:

- self-contained package skeleton under `pavi/correlated_bp_decoder/`
- `pyproject.toml`
- `src/correlated_bp_decoder/cer.py`
- `src/correlated_bp_decoder/tanner_graph.py`
- `src/correlated_bp_decoder/neural/base.py`
- `src/correlated_bp_decoder/io.py`
- `tests/test_cer.py`
- `tests/test_tanner_graph.py`
- `tests/test_base.py`

Tasks:

1. Define dataclasses for graph and model metadata.
2. Port `parse_cer_data(...)`.
3. Port `TannerGraph`.
4. Port the `NeuralBPBase` constructor logic from `neuralbase.jl`.
5. Replace Julia dictionaries and matrix scans with clear Python helpers that build:
   - edge-to-`(check, bit)` maps
   - per-edge check indices
   - per-edge bit indices
   - edge lists for the three logical adjacency patterns
6. Add tests that verify adjacency sizes and edge ordering on a tiny parity-check matrix.

### Phase 2: Numerical Kernels and Forward Pass

Deliverables:

- `src/correlated_bp_decoder/math_utils.py`
- `src/correlated_bp_decoder/neural/nachmani.py`
- `tests/test_bp_utils.py`
- `tests/test_forward.py`

Tasks:

1. Port the stable nonlinearities:
   - `safe_log_tanh_split`
   - `safe_atanh_exp_signed`
   - `binary_entropy_of_sigmoid`
2. Implement one reusable edge-accumulation helper instead of duplicating separate sparse-multiply code paths.
3. Define `NachmaniNeuralBP` as a `torch.nn.Module`.
4. Store trainable weights in shapes that are easy to inspect and slice by layer, even if we flatten them for checkpoint compatibility.
5. Port the layer update flow:
   - C2V -> V2C
   - V2C -> C2V
   - readout
6. Make the forward pass return per-layer posterior LLRs, matching the Julia behavior needed by the loss and evaluation code.

### Phase 3: Losses and Training

Deliverables:

- `src/correlated_bp_decoder/neural/losses.py`
- `src/correlated_bp_decoder/neural/training.py`
- `tests/test_loss.py`
- `tests/test_bp_algo.py`

Tasks:

1. Port the active loss path from `loss.jl`, starting with the current sine-residue implementation.
2. Keep the quadratic-residue alternative available behind a separate function for experiments.
3. Port:
   - certainty regularizer
   - sparsity penalty
   - Ising-style correlation penalty
   - layer aggregation
4. Replace Julia hyperparameter dictionaries with a typed training config object.
5. Port annealing schedules, gradient clipping, unstable-batch skipping, and epoch rollback.
6. Use PyTorch optimizers in a way that preserves the current training semantics while removing Julia-specific complexity.

### Phase 4: Prediction, Model I/O, and Experiment Driver

Deliverables:

- `src/correlated_bp_decoder/neural/predict.py`
- `src/correlated_bp_decoder/io.py` weight save/load support
- `src/correlated_bp_decoder/experiments/run_neural_bp.py`
- `tests/test_predict.py`
- `tests/test_experiments.py`

Tasks:

1. Port recovery thresholding and correctness checks.
2. Save and load weights in a stable format that records:
   - package version or schema version
   - edge ordering assumptions
   - tensor shapes
3. Recreate the minimal experiment flow of `expts/neural_bp_experiments.jl` with a thinner Python CLI.
4. Keep plotting and large sweep orchestration out of the first port.

### Phase 5: Standard BP Baseline and Optional Extensions

Deliverables:

- `src/correlated_bp_decoder/standard_bp.py`
- optional CSS wrapper support
- `tests/test_trim_constraints.py`

Tasks:

1. Port the plain BP baseline only after the neural path is validated.
2. Use the same Tanner-graph and code structures as the neural stack to avoid duplicated graph logic.
3. Add the CSS split wrapper only if we need parity with the Julia baseline experiments.

## Design Decisions To Lock In Early

1. Use a shortened package name: `correlated_bp_decoder`.
2. Keep CER parsing independent from the model class.
3. Keep graph construction independent from training code.
4. Use one numerical helper module for stable transforms and edge accumulation.
5. Treat experiment scripts as thin wrappers around importable library functions.
6. Defer archival Julia files unless they clarify a discrepancy in the live implementation.

## Testing Strategy

Add tests alongside implementation, not afterward.

Test location:

- Keep the Python tests under `pavi/correlated_bp_decoder/tests/`.
- Start by mirroring Pavi's Julia test filenames so it is obvious which Python test corresponds to which Julia source:
  - `test_forward.jl` -> `test_forward.py`
  - `test_bp_algo.jl` -> `test_bp_algo.py`
  - `test_bp_utils.jl` -> `test_bp_utils.py`
  - `test_loss.jl` -> `test_loss.py`
  - `test_classical_bp.jl` -> `test_classical_bp.py`
  - `trim_constraints_test.jl` -> `test_trim_constraints.py`
- Use `pytest` as the default runner.
- If we later add Python packaging metadata, put that config at the `pavi/correlated_bp_decoder/` workspace root.

Primary rule:

- Reproduce Pavi's existing tests as the first validation target for the Python port.
- Preserve the same fixtures, syndromes, expected LLR values, and behavioral comparisons wherever possible.
- Upgrade print-only Julia checks into real Python assertions so failures are machine-detectable.

Canonical Julia sources to mirror first:

1. `pavi/tests/run_tests.jl`
2. `pavi/tests/test_forward.jl`
3. `pavi/tests/test_bp_algo.jl`
4. `pavi/tests/test_bp_utils.jl`
5. `pavi/tests/test_loss.jl`
6. `pavi/tests/test_classical_bp.jl`
7. `pavi/tests/trim_constraints_test.jl`

Reference-only Julia source:

- `pavi/tests/neural_test_original.jl`
  - This appears to be an older combined test script with overlap and stale references.
  - Use it only to recover fixtures or behavioral intent that are missing from the canonical files above.

Known stale test references to normalize during the port:

- `compute_loss_error_from_llrs`
  - This name is referenced in the Julia tests but is not present in the current live source.
  - In the Python port, bind these tests to the active loss implementation they are actually exercising, and document that mapping in the test file.

- `parse_correlation_strengths_connectivity`
  - The live source exports `parse_cer_data(...)` instead.
  - Python tests should follow the live implementation name while preserving the same test fixture inputs and expected outputs.

- `weights_loss_layers`
  - Referenced in `neural_test_original.jl`, but not part of the current active `NachmaniNeuralBP`.
  - Do not make this part of the required Python public API unless a live Julia path depends on it.

Minimum test layers:

1. Parsing tests
  - CER file parsing
   - parity-check and logical-operator loading

2. Structural tests
   - Tanner graph neighbor lists
   - neural edge ordering
   - adjacency index construction

3. Numerical tests
   - stable transforms on edge cases
   - sparse accumulation helper against a small dense reference implementation

4. Model tests
   - forward-pass shapes
   - deterministic tiny-graph forward pass against hand-computed expectations
   - weight slicing per layer

5. Loss and training tests
   - loss terms are finite on small fixtures
   - one optimizer step changes weights in the expected direction
   - rollback logic triggers on injected non-finite gradients

6. Integration tests
   - small synthetic training run on a toy code
   - prediction path returns the expected output shape and correctness mask

7. Cross-language validation
   - once the Python implementation is runnable, compare fixtures against Julia outputs for:
     - `parse_cer_data`
     - `NeuralBPBase` edge counts
     - one forward pass
     - one loss evaluation
   - use the existing Julia tests as the default source of those fixtures before inventing new ones

## Documentation and Style Requirements

- Every public class, function, and method must include a NumPy-style docstring.
- Each module should begin with a short module docstring.
- Public APIs should be type hinted.
- Complex tensor-shape assumptions should be documented directly where they are introduced.
- Keep helper functions small and single-purpose; if a function starts managing parsing, graph building, and numerical transforms at once, split it.

## Recommended First Implementation Slice

Start with the smallest end-to-end path that exercises the live neural stack:

1. `parse_cer_data`
2. `load_base_bp_model`
3. `NeuralBPBase`
4. stable math helpers
5. `NachmaniNeuralBP.forward`
6. one loss function
7. one smoke test on a tiny parity-check matrix

This slice will validate the hardest architectural choices before we spend time on full training and experiment plumbing.

## Current Assumptions

- PyTorch is an acceptable dependency for the translation.
- The first useful Python milestone is the neural decoder path, not the entire Julia repository.
- GPU-specific acceleration can wait until the CPU implementation is correct and tested.
- We should preserve the Julia algorithm's semantics before trying to optimize or generalize the architecture.

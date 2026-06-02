# Python Test Port Plan

This directory is reserved for the `pytest` reproduction of Pavi's Julia test suite.

The package is intentionally self-contained:

- source lives under `pavi/correlated_bp_decoder/src/correlated_bp_decoder/`
- tests live under `pavi/correlated_bp_decoder/tests/`

Initial one-to-one mapping:

- `pavi/tests/test_forward.jl` -> `pavi/correlated_bp_decoder/tests/test_forward.py`
- `pavi/tests/test_bp_algo.jl` -> `pavi/correlated_bp_decoder/tests/test_bp_algo.py`
- `pavi/tests/test_bp_utils.jl` -> `pavi/correlated_bp_decoder/tests/test_bp_utils.py`
- `pavi/tests/test_loss.jl` -> `pavi/correlated_bp_decoder/tests/test_loss.py`
- `pavi/tests/test_classical_bp.jl` -> `pavi/correlated_bp_decoder/tests/test_classical_bp.py`
- `pavi/tests/trim_constraints_test.jl` -> `pavi/correlated_bp_decoder/tests/test_trim_constraints.py`

First-slice infrastructure tests:

- `pavi/correlated_bp_decoder/tests/test_cer.py`
- `pavi/correlated_bp_decoder/tests/test_tanner_graph.py`
- `pavi/correlated_bp_decoder/tests/test_base.py`
- `pavi/correlated_bp_decoder/tests/test_io.py`

Phase 2 neural-path tests:

- `pavi/correlated_bp_decoder/tests/test_bp_utils.py`
- `pavi/correlated_bp_decoder/tests/test_forward.py`
  - includes batch-first input/output checks
  - includes a deterministic forward regression fixture
  - includes CPU vs accelerator parity when CUDA or MPS is available

Phase 4 prediction and checkpoint tests:

- `pavi/correlated_bp_decoder/tests/test_predict.py`
- `pavi/correlated_bp_decoder/tests/test_experiments.py`
- `pavi/correlated_bp_decoder/tests/test_io.py`
  - includes structured checkpoint round-trips
  - includes legacy Julia checkpoint compatibility

Phase 5 classical-baseline tests:

- `pavi/correlated_bp_decoder/tests/test_classical_bp.py`
- `pavi/correlated_bp_decoder/tests/test_trim_constraints.py`

Ground rules for the Python port:

- Preserve the same fixtures and expected values whenever the Julia tests are still aligned with the live code.
- Convert print-only checks into real `pytest` assertions.
- When a Julia test references an outdated symbol, keep the same behavioral target but bind it to the current live Python API and document the mapping inside the test.
- Treat `pavi/tests/neural_test_original.jl` as a historical reference, not the canonical source of required test coverage.

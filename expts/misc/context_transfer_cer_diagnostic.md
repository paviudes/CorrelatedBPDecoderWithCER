# Context transfer — CER results + syndrome-satisfaction diagnostic

*Two turns happened in the wrong thread. This is the catch-up so work can continue here.*

## Setup

- Full circuit-level simulation of a BB code, syndrome extraction included.
- X errors only, tracked through the circuit, decoded against `HZ`.
- NBP loss = syndrome term + correlation term (one-qubit and two-qubit marginals from
  1M training shots).
- Two-qubit marginals live on "clique edges": pairs of data qubits co-supported on the
  same Z check (`get_clique_edges` in `cer_tools.py`). Note these are *check*-sharing
  pairs, not CNOT-sharing pairs — a CNOT touches one data qubit and one ancilla.
- Results below: `72q_BB_cycles_1`, p = 5e-4, 1 cycle, 30 layers, 5 epochs,
  1M test shots, sparsity parameter = 0.0.

## The result

| run | failures / 1M | LER | vs no-CER |
|---|---|---|---|
| no-CER | 335 | 3.35e-4 | — |
| single-qubit priors only | 322 | 3.22e-4 | 0.961 |
| + two-qubit correlations | 489 | 4.89e-4 | 1.460 |

- single-qubit vs no-CER: z = −0.51, 95% CI on ratio [0.825, 1.120]. **No evidence of
  any effect** — the interval covers real improvement, nothing, and real degradation.
  Resolving the observed 3.9% gap unpaired at 3σ would take ~36M shots per arm.
- include_corr vs no-CER: z = +5.36, 95% CI [1.270, 1.677]. **Solid degradation**,
  27–68% more logical errors. Consistent with the earlier 20×/15× observation (1.33
  there, 1.46 here).

Net: **one-body neutral, two-body harmful.** The ablation localizes the problem to the
clique-edge terms specifically, not to the CER pipeline as a whole.

## Leading explanation

Two compounding mechanisms, both pushing the same direction.

**Systematic — cycle-space alignment.** Clique edges are derived from co-support in
`HZ`, so every pairwise potential points along the code's cycle space, which is spanned
by stabilizers *and* logicals. For a check-sharing pair, flipping both contributes zero
to the shared check, so "prefer a correlated error over an uncorrelated one at fixed
syndrome" is structurally a preference for adding check-canceling pairs to the
correction. The stabilizer component is free (same coset); the logical component is
exactly the failure being scored. The thing that makes these edges look physically
motivated — hook errors from shared ancillas — is the same thing that makes them
collinear with the logicals.

**Statistical — noise-dominated pair estimates.** `p_ij = np.mean(E_train[:,q1] *
E_train[:,q2])` is a raw joint, not a connected correlation (no `p_i·p_j` subtraction).
For pairs with no genuine hook correlation the expected coincidence count is
`N·p_i·p_j` ≈ single to low double digits at 1M shots. A weight-6 `HZ` on 72 qubits
gives ~540 clique pairs; only the genuine hook pairs are actually measured. With
sparsity at 0.0, every noise-dominated edge is kept.

**Side note on why single-qubit is only neutral.** BB codes are 2BGA constructions with
a group acting transitively on data qubits, so the true `p_i` should be near-constant
across all 72. Worth checking `np.std(one_body)/np.mean(one_body)` against the sampling
floor `sqrt((1-p)/(N·p))` (~2% at p = 2e-3, N = 1e6). If the spread doesn't clear the
floor, the "rescaled priors" are a uniform prior plus estimation noise, and neutral is
the correct outcome.

## Decided next step — check (1)

Split failures into **coset failures** (recovery cleared the syndrome but carries a
logical) vs **convergence failures** (no layer ever cleared the syndrome).
`check_bp_solutions` in `src/predict.jl` already computes both facts — it finds the
first syndrome-clearing layer, then checks logical triviality — but collapses both
outcomes into the same `false`. A spec for a new additive `count_syndrome_satisfactions`
is in `handoff_count_syndrome_satisfactions.md`, intended for Cowork.

Discriminator:

| observation | reading |
|---|---|
| convergence failures flat, coset failures up | cycle-space tipping → prune edges by raw coincidence count, drop edges overlapping low-weight logical support |
| convergence failures up | correlation term fighting the syndrome term in training → sweep λ, re-check sign/normalization of the two-body term |

## Still open

- **check (2)**: paired McNemar comparison on per-sample success vectors — gets power on
  the single-qubit question without 36M shots/arm. Requires all three configs run on the
  same test file with the same sample ordering.
- **check (3)**: histogram raw coincidence counts `N_ij` against the independence
  expectation `N·p_i·p_j`, to set a principled sparsity threshold.
- Seed reporting fix (train and test both labelled `s_1`).
- `18q` vs `72q`: the earlier pipeline files (`cer_tools.py`, `BP_OSD_decoding.py`) were
  configured for `18q_BB_p_0.0005_cycles_1`, but the result CSVs are `72q_BB_cycles_1`.

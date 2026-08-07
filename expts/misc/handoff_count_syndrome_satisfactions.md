# Handoff: add `count_syndrome_satisfactions` to `src/predict.jl`

## Why

Adding two-qubit correlation weights to the NBP loss makes the logical error rate
*worse*, not better:

| run | failures / 1M | LER | vs no-CER |
|---|---|---|---|
| no-CER | 335 | 3.35e-4 | — |
| single-qubit priors only | 322 | 3.22e-4 | 0.96 (z = −0.51, not significant) |
| + two-qubit correlations | 489 | 4.89e-4 | 1.46 (z = +5.4, significant) |

We need to know **which kind of failure** the extra ~154 failures are. There are two
candidate mechanisms and they call for opposite fixes:

- **H1 — coset tipping.** The recovery still clears the syndrome, but lands in the
  wrong logical coset. The correlation edges are derived from co-support in `HZ`, so
  every pairwise potential points along the code's cycle space, which is spanned by
  stabilizers *and logicals*. Fix would be to prune/reweight those edges.
- **H2 — objective competition.** The correlation term steals gradient from the
  syndrome term during training, so BP simply converges worse and fails to clear the
  syndrome at all. Fix would be to rebalance the loss weight λ.

`check_bp_solutions` **already computes both facts** — it finds the first
syndrome-clearing layer, then checks logical triviality — but it collapses "never
cleared the syndrome" and "cleared it but wrong coset" into the same `false`. We just
want that distinction surfaced.

## What to build

A new, **additive** function. Do not change the semantics or return type of
`check_bp_solutions` or `predict_and_check_neuralbp` — existing runs must stay
reproducible.

### 1. `count_syndrome_satisfactions`

Same commit rule as `check_bp_solutions` (first layer whose residual has zero
syndrome weight), but returns the decomposition instead of a single Bool.

```julia
function count_syndrome_satisfactions(
    parity_check_matrix::Matrix{Int},
    logicals::Matrix{Int},
    errors::BitMatrix,
    proposed_recoveries::Array{Bool, 3},
)
    """
    Decompose decoder outcomes into three mutually exclusive buckets, using the
    same commit rule as `check_bp_solutions`:

      1. SUCCESS              - some layer cleared the syndrome, and the first such
                                layer is also logically trivial.
      2. COSET FAILURE        - some layer cleared the syndrome, but the first such
                                layer carries a logical. (Evidence for H1.)
      3. CONVERGENCE FAILURE  - no layer ever cleared the syndrome. (Evidence for H2.)

    Also records which layer was committed to, and - for samples that never cleared -
    the smallest residual syndrome weight reached across all layers, which says
    whether BP was close or wildly off.
    """
    n_samples = size(errors, 2)

    syndrome_cleared    = falses(n_samples)
    is_correct          = falses(n_samples)
    committed_layer     = zeros(Int, n_samples)   # 0 = no layer ever cleared
    min_syndrome_weight = zeros(Int, n_samples)

    for i in 1:n_samples
        residuals = errors[:, i] .⊻ proposed_recoveries[:, i, :]
        layer_syndrome_weight = vec(sum(mod.(parity_check_matrix * residuals, 2), dims = 1))
        min_syndrome_weight[i] = minimum(layer_syndrome_weight)

        layer = findfirst(==(0), layer_syndrome_weight)
        if layer === nothing
            continue                      # convergence failure
        end

        syndrome_cleared[i] = true
        committed_layer[i]  = layer

        committed_residual = residuals[:, layer]
        if all(mod.(logicals * committed_residual, 2) .== 0)
            is_correct[i] = true
        end
    end

    n_cleared = count(syndrome_cleared)
    n_correct = count(is_correct)

    return (
        # per-sample (keep these - needed for the paired McNemar test later)
        syndrome_cleared       = syndrome_cleared,
        is_correct             = is_correct,
        committed_layer        = committed_layer,
        min_syndrome_weight    = min_syndrome_weight,
        # aggregate
        n_samples              = n_samples,
        n_syndrome_cleared     = n_cleared,
        n_correct              = n_correct,
        n_coset_failures       = n_cleared - n_correct,
        n_convergence_failures = n_samples - n_cleared,
    )
end
```

### 2. `predict_and_diagnose_neuralbp`

Mirror `predict_and_check_neuralbp` exactly — same `gpu_active()` branch, same
batching, same `parity_check_matrix_dual` tail-slice to get the logicals — but call
`count_syndrome_satisfactions` per chunk and concatenate the per-sample vectors.

**Do not add a second forward pass.** Either reuse the existing one or, if simpler,
have `predict_and_check_neuralbp` delegate to the diagnostic version and return just
`.is_correct`. The latter is cleaner as long as the returned `is_correct` is proven
identical (see acceptance test).

### 3. Thread it through `neuralbp_test_predictions`

Add a keyword like `diagnose::Bool = false`. When true, return the full NamedTuple
and write the extra columns into the results CSV alongside the existing ones:

```
num_syndrome_cleared, num_coset_failures, num_convergence_failures, mean_committed_layer
```

### 4. Persist the per-sample vectors

Write `is_correct` and `syndrome_cleared` to disk per run, keyed by the same run name
already used for the CSV. These are 1M-element BitVectors — negligible on disk, and
they are what enables the paired (McNemar) comparison that gets us statistical power
on the single-qubit question without needing ~36M shots per arm.

**Critical:** all three configs must run on the *same* test file with the *same*
sample ordering, or the vectors won't align and the paired test is invalid.

## Acceptance tests

1. **Backward compatibility.** On at least one real run, elementwise:
   `diag.is_correct == check_bp_solutions(parity_check_matrix, logicals, errors, proposed_recoveries)`.
   Assert this, don't eyeball it.
2. **Partition.** `n_correct + n_coset_failures + n_convergence_failures == n_samples`.
3. **Consistency with existing reporting.** `n_samples - n_correct` equals the
   `num_failures` already written to the CSV.
4. `committed_layer[i] == 0` if and only if `syndrome_cleared[i] == false`.

## How to read the result

Run all three configs (`no_cer`, `only_single_qubit`, `include_corr`) and compare:

| observation | reading | next move |
|---|---|---|
| `n_convergence_failures` roughly flat across runs, `n_coset_failures` clearly up in `include_corr` | **H1 confirmed.** Corrections are valid but tipped into wrong cosets by the cycle-space-aligned prior. | Prune correlation edges overlapping low-weight logical support; threshold edges by raw coincidence count, not by `p_ij`. |
| `n_convergence_failures` up in `include_corr` | **H2 confirmed.** The correlation term is fighting the syndrome term in training. | Sweep λ / `correlation_strength` → 0 and check smooth recovery; re-check sign and normalization of the two-body term in the loss. |
| both up | Both mechanisms active. | Do the λ sweep first to separate them. |
| `mean_committed_layer` noticeably higher in `include_corr` | BP is converging later even when it does converge — a slowdown signal worth reporting either way. | — |

The extra 154 failures are being split into two buckets, so even a lopsided split like
70/30 is comfortably resolvable at these counts.

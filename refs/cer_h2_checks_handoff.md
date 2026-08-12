# Handoff: diagnosing why the two-body CER term degrades decoding (H2 follow-up)

## Where we are

The `count_syndrome_satisfactions` diagnostic has been run at three physical error
rates, 90 layers, 5 epochs, 1M test shots each, `sparsity_importance = 0.0`.
CER = single-qubit priors + two-body correlation term; no-CER = preset `p = 0.1`
flat priors, correlation term dropped. `seed = 1` on the no-CER arm; the CER arm
predates the seeding change and carries no seed tag.

| p | arm | total | coset | convergence | mean committed layer | runtime (s) |
|---|---|---|---|---|---|---|
| 5e-4 | CER | 294 | 88 | 206 | 1.333 | 4229 |
| 5e-4 | no-CER | 329 | 64 | 265 | 1.210 | 631 |
| 7e-4 | CER | 934 | 199 | 735 | 1.429 | 14302 |
| 7e-4 | no-CER | 551 | 177 | 374 | 1.328 | 583 |
| 1.9e-3 | CER | 6255 | 2097 | 4158 | 2.199 | 29598 |
| 1.9e-3 | no-CER | 5050 | 2225 | 2825 | 2.516 | 1818 |

**The diagnostic came back H2, not H1.** Coset failures are flat or slightly down
with CER (z = +1.95, +1.13, −1.95 across the three p — no consistent sign).
Convergence failures are up 1.97x and 1.47x at the two higher p (z = +10.8, +16.0).
So the correlation term is stopping BP from clearing the syndrome at all, rather
than tipping otherwise-valid corrections into the wrong logical coset. The
cycle-space/edge-pruning fix is therefore NOT indicated; the loss-balance fix is.

Three supporting observations, all pointing the same way:

- **Layer profile.** At p = 7e-4 the fraction of samples clearing at layer 1 is
  0.947 (no-CER) vs 0.827 (CER), with the deficit appearing at layer 2
  (0.036 -> 0.152). Same pattern at every p. BP converges measurably slower with
  the correlation term on.
- **Failing samples are easier.** Mean true error weight among failures is 4.78
  (CER) vs 5.73 (no-CER) at p = 7e-4, and 5.84 vs 6.28 at p = 1.9e-3. The CER
  decoder fails on lower-weight errors the baseline handles.
- **Paired McNemar** on the per-sample failure vectors (same test file, same
  ordering, verified aligned): z = −2.20, +14.5, +19.3 at the three p. Large
  discordance (e.g. 2545 CER-only vs 1340 no-CER-only failures at p = 1.9e-3), so
  this is not two near-identical decoders separated by counting noise.

Caveat that the checks below are designed to address: these are single training
runs per arm. A previously measured training-seed spread on *provably identical*
configurations was 309 vs 620 failures (2.0x), so single-run comparisons at this
scale are not yet conclusive. Note also the sign flip: CER is slightly *better*
at p = 5e-4 and clearly worse at the two higher p.

---

## Check A — confirm the correlation term is live and at J scale

Cheapest check, and it gates everything else: if the two-body values are on a
probability scale (~1e-4) rather than a log-odds scale (~2-4), the term is
numerically inert and the whole effect is something else. I believe this is
already correct, but please verify rather than assume.

1. Parse the correlation weights file used by the CER runs (under
   `data/72q_BB_cycles_1/correlated_weights/`). Report, for the two-qubit
   entries: count, min, max, mean, median, fraction negative, and a coarse
   histogram. Expect ~540 edges with values roughly in [−1, 4] and a bimodal
   shape (a near-zero cluster and a cluster around 2-4). Flag it if instead
   everything sits in (0, 0.01).
2. In the training debug CSV for a CER run
   (`debugging_train_*_individual_losses.csv`), report the min / max / mean of
   `correlation_penalty` across all rows and layers, and the count of exactly
   zero entries. **A previous run had this identically zero across all 50,000
   layer-entries** — i.e. the arm labelled "CER" had no correlation term at all.
   Confirm that is not the case here.
3. Report `correlation_weight` per epoch from the same run's hyperparameter log,
   so we know the annealed lambda actually reached.

## Check B — lambda sweep (the main experiment)

Goal: establish whether convergence failures fall smoothly back to the no-CER
level as the correlation weight goes to zero. That is what confirms H2 and
locates a usable operating point.

- Fix p = **7e-4** (largest effect, cheapest runs), 90 layers, 5 epochs.
- Sweep `correlation_weight` over **{0, 0.01, 0.03, 0.1, 0.3, 1.0}**. The
  hyperparameter is an annealing spec string, so hold it constant rather than
  ramping — e.g. `"0.1,0.1,0.7,up"` for the 0.1 point. Confirm from the debug log
  that the value is actually constant across epochs.
- **Same seed for every point in the sweep**, so points differ only in lambda.
- Run the `count_syndrome_satisfactions` diagnostic on each and report the table:
  lambda, total failures, coset, convergence, mean committed layer, and the
  layer-1 clearing fraction from the layer profile.

Reading: convergence failures declining monotonically to the no-CER value as
lambda -> 0 confirms H2 and gives the operating point. Convergence failures
staying elevated even at lambda = 0.01 means the problem is in the *form* of the
term rather than its weight, and we would look at sign and normalization next.

## Check C — the sparsity confound

`sparsity_importance = 0.0` in these runs. The Ising correlation term is a reward
that argues monotonically for *more* predicted flips and has no internal
counterweight — the codebase docstring notes that `sparsity_penalty` is what
normally plays that role. With sparsity at zero, the reward is unopposed, which
would produce exactly the observed picture (BP pushed off the sparse
syndrome-clearing solution, converging later or not at all).

This is currently indistinguishable from "lambda too high", and Check B alone
will conflate them. So at the single best lambda from Check B, add two arms:

- `sparsity_importance` constant at **0.05**
- `sparsity_importance` constant at **0.15**

(Keep these constant, not annealed. Earlier runs annealed sparsity up to 0.3-0.5
and that was independently harmful — dropping it took the no-CER arm from 653 to
309 failures — so stay well below that range.)

If a small sparsity weight restores convergence at a lambda that was otherwise
harmful, the mechanism is the missing counterweight, not the correlation
information itself.

## Check D — seed replication of the headline comparison

Needed before any of this goes in a writeup, and it also resolves the p = 5e-4
sign flip.

- At p = **7e-4**: run CER and no-CER for seeds **1-5**, same seed set for both
  arms, everything else fixed.
- Report per-seed totals and the pooled comparison (sum failures per arm, then
  the two-proportion z and the ratio CI).
- Also report the per-seed spread within each arm — that is the error bar that
  every other comparison in this project has to clear.

## Check E — runtime

CER runs are 7-16x slower than no-CER (4229 vs 631 s; 14302 vs 583; 29598 vs
1818). Testing should be identical work in both arms — the correlation term lives
in the loss, which is training-only — so this needs an explanation before the
sweep, since it sets the cost of everything above.

Please determine whether the difference is in training or in testing, and whether
the correlation term (or the CER code path generally) has leaked into the forward
pass or the prediction path. If it is genuinely in the loss only, check whether
`is_correlated = true` is triggering a slower branch somewhere unintended. Report
the wall-clock split between train and test phases for one CER run and one
no-CER run.

---

## Reporting

For each check, please give the numbers in a table plus a one-line reading, and
say explicitly which files and code paths you used. Do not change loss functions,
annealing behaviour, or defaults as part of this — these are measurement runs. If
something looks like a bug (especially in Check A or E), report it and stop rather
than fixing it inline, so we can decide whether the fix invalidates the runs above.

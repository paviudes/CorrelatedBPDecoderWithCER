# Handoff: λ sweep — testing the correlation term at a weight where it can matter

Follow-on to the 2x2 gate x CER sweep (`handoff_cer_gate_science.md`). Read that
first for the system description; this document assumes it.

---

## 1. Why this sweep

**The 2x2 came back null on every contrast.** All four paired-by-seed t values
were |t| <= 1.59. Under the protocol fixed in §4.4 of the previous handoff, that
is the "nothing beyond the seed spread" row: the gate is not the explanation.

More importantly, the contrast that motivated the whole line of work did not
replicate. At the same p and configuration, the single-seed diagnostic had read
convergence failures 735 (CER) vs 374 (no-CER), z = +10.8, which was read as
"H2 confirmed". Across 5 paired seeds:

```
cer_ungated - nocer_ungated   convergence   +59.8 +- 257.1   t = +0.52
```

The point estimate collapsed from +361 to +60. Given the observed paired sd, a
true +361 effect would have produced t ~ 3.1, so the original effect size is
rejected at roughly p = 0.03. **The "CER worsens convergence" result was seed
noise.** This agrees with the independent magnitude analysis (§3.2 of the
previous handoff): the correlation term contributes 0.05% of the total loss and
mechanically cannot move failure rates by 1.7x.

So the correlation term is not harmful. It is **inert**, and it has never been
tested at a weight where it could do anything. That is what this sweep fixes.

### Why it is inert — the mechanism, which matters for the design

The term is `-(1/|C|) * Σ_(i,k in C) J_ik * σ_i * σ_k`, quadratic in σ. At a
converged solution σ is near-binary, so `σ_i σ_k = 1` only for pairs that
genuinely co-flipped. Expected co-flipped pairs per sample is
`|C| * P11 ~ 540 * 1e-4 ~ 0.05`. So

```
per-sample corr ~ -J_bar * 0.05 / 540 ~ -1.6e-4
```

against a measured `correlation_penalty` mean of **-3.1e-4** — agreement to
within a factor of two, so the mechanism is confirmed rather than conjectured.

Two consequences drive the design below:

1. The weakness is **structural, not a tuning oversight**: the `1/|C|` divides by
   540 total edges when only ~0.05 are active per sample.
2. The co-firing rate scales as **p^2**, so the term gets weaker at lower p. A
   single λ will not be correct across the p range.

---

## 2. What to run

### 2.1 Primary sweep

- p = **7e-4**, 90 layers, 5 epochs, `use_CER = true`, **gate ON**
  (`syndrome_gate_threshold = 0.5`) for every point.
- λ (`correlation_weight`) held **constant**, not annealed:
  **{0, 1, 10, 100}**.
- **5 seeds per λ, the same seed set at every λ** (reuse seeds 1-5 so the λ=0
  point can be checked against the existing `cer_gated` arm).
- Everything else identical to the `cer_gated` arm of the 2x2.

Use a constant annealing spec, e.g. `correlation_weight = "10,10,0.7,up"`, and
**verify from the debug log that the value really is flat across all 5 epochs**
before trusting the run. The existing schedule `"1e-2,1,0.7,up"` only reaches
0.762 by epoch 5 and confounds "how strong" with "when" — do not simply raise its
ceiling.

`run_tag` must encode λ (e.g. `_lam10`), or runs collide and `retrain = false`
silently loads the wrong weights (§6.4 of the previous handoff).

### 2.2 Keep the gate on throughout

The gate confines the correlation reward to samples that already (softly) clear
the syndrome, where it is ordering-safe by construction. **Ungated at λ = 10 or
100 would likely be destructive**, since the reward would then compete with the
syndrome objective at full strength on failing samples. If you want to confirm
that is what the gate buys, run `λ = 10, gate OFF, 5 seeds` as a **separate
deliberate probe** — but expect it to be bad, and do not put it in the main
sweep.

### 2.3 Sanity gate before spending the full budget

Run **λ = 100, one seed** first. If it produces something obviously pathological
(e.g. >5000 failures, or training divergence — check `min_weight_c2v_v2c` going
negative in the debug log), stop and report before running the remaining 19.

---

## 3. What to measure, and the reading protocol

Fixed in advance, as before.

Report per λ: mean +- sd across seeds of **total failures, coset failures,
convergence failures**, plus `layer1_clearing_fraction` and
`mean_committed_layer`. Contrasts paired by seed against the λ=0 point, quoting
**t**, not z.

| observation | reading |
|---|---|
| total failures fall with λ, coset flat or down | **The correlation prior helps.** Find the optimum, then check it holds at other p. |
| **coset** failures climb with λ, convergence flat | **Ceiling found, H1 becomes live.** The prior is adding cycle-space elements — valid syndrome, wrong coset. Expected at high λ; the λ where it starts is the useful number. |
| convergence climbs with λ | The gate is not confining the term as intended. Check `gate_open_fraction` per run; investigate before interpreting anything else. |
| nothing moves even at λ=100 | The term cannot be made to matter in this form. Go to §5 (normalization) rather than pushing λ higher. |

**Coset failures are the primary endpoint here**, which is a change from previous
sweeps. The gate protects convergence by construction; what it does *not* prevent
is the network hallucinating co-activated pairs that still satisfy the syndrome.
So high λ is expected to convert the failure mode rather than remove it.

Note the previously estimated "parity at λ ~ 168" is an **upper bound** — it
assumes `corr_pen` stays at -3.1e-4 as λ grows, but more co-activation makes the
reward more negative, so the ratio grows superlinearly. Expect real effects well
below 168.

### Power

From the existing `cer_gated` arm: totals 567 +- 49, coset 172 +- 13,
convergence 395 +- 60 (n=5). With paired seeds and n=5, a coset shift of ~25
failures (15%) is detectable at t ~ 2.8 if the paired sd resembles the 2x2's
coset sd (~35). Convergence is noisier and will need a larger effect. If λ=100
produces only a marginal signal, the answer is more seeds, not more λ points.

---

## 4. Data-integrity issue to resolve first

**The collector's train-side columns are internally inconsistent and must be
fixed before any log-derived number from this sweep is quoted.**

In `gate_cer_per_run.csv`, seeds *within the same arm* report different
hyperparameters:

| arm | `lam_fin=0.762 / spars_fin=0.2952` | `lam_fin=0 / spars_fin=0` |
|---|---|---|
| cer_gated | seeds 4, 5 | seeds 1-3 |
| nocer_gated | seeds 2, 3 | seeds 1, 4, 5 |
| nocer_ungated | seeds 1, 3 | seeds 2, 4, 5 |

Seeds within one arm cannot have trained under different schedules if the sweep
ran one TOML per arm. Two candidate causes:

- **(A) Stale log association.** The collector reported **42** debug files where
  20 runs x 2 should give 40, so logs from earlier experiments are in the
  directory. The two profiles are exactly the two known schedule sets — 0.762301
  and 0.2952 are what the old `hyperparams_epochs_5` annealing gives at epoch 5,
  to the digit. This is the likely explanation.
- **(B) Some runs genuinely used the wrong TOML**, in which case the arms are
  heterogeneous and the 2x2 pairing is broken.

**Discriminating check:** the hp log carries `syndrome_gate_threshold`. A log
matched to a gated run must show τ = 0.5; a log matched to an ungated run must
show -1.0 (or absent). Any mismatch proves (A). Please run this check, report
which it is, and fix the log-to-run matching (match on the full run tag including
seed, and assert one log per run) before the new sweep writes more logs into the
same directory.

Test-side files (failure counts, layer profiles) are keyed by run/seed tags and
appear clean — the 2x2's headline conclusions above rest only on those.

### Also close the silent-fallback hole first

§6.3 of the previous handoff: `use_CER = true` with a missing or pair-free CER
file falls through to the `log(9)` preset and an empty connectivity, producing an
arm labelled "CER" that is bit-identical to no-CER. Make this a **hard error**
before running a sweep whose entire content is the correlation term. Otherwise a
λ=100 arm with a silently-missing file is indistinguishable from a null result.

Per-run assertion for this sweep: `correlation_penalty_frac_zero` should be
~0.003, not 1.0, and `correlation_penalty_mean` should be negative and should
scale roughly with λ.

---

## 5. If the sweep comes back null even at λ = 100

Do not push λ higher. The better fix is the normalization, and it is a small
change to `compute_additional_loss_from_ising_correlations` in `src/loss.jl`:

Dividing by `|C|` = 540 total edges when only ~0.05 are active per sample is what
crushes the term. Normalizing by the **expected number of active edges**
(`|C| * mean(P11)`), or simply dropping the `1/|C|`, gives an O(1)-scale term.
Mathematically this is a relabeled λ, but it makes the hyperparameter mean the
same thing at every error rate — which matters because the co-firing rate scales
as p^2, so a λ tuned at 7e-4 will be wrong at 5e-4 and 1.9e-3.

Report this as a proposal with the arithmetic, not as an implemented change,
unless the sweep result specifically calls for it.

---

## 6. Reporting

Per λ: the table from §3, the paired-t contrasts against λ=0, and the per-seed
raw numbers (the 2x2's most informative structure — the ungated tail blowups —
was only visible per-seed, not in the arm means).

Do not change loss functions, annealing behaviour, or defaults beyond what §4
requires. If something looks like a bug, report it and stop rather than fixing it
inline, so we can decide whether the fix invalidates runs already made.

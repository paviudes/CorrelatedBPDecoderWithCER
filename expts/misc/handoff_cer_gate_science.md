# Handoff: does CER correlation data help a Neural BP decoder, and why not so far

Self-contained brief for a new conversation. Everything below is either a
measurement (with the file it came from) or a clearly-labelled inference.

**Status:** the 2x2 gate x CER sweep has run (20 runs, 5 seeds). Its numbers are
not yet in this document — see [Results to fill in](#results-to-fill-in). The
reading protocol is fixed in advance, on purpose.

---

## 1. The system

Nachmani-style unrolled Neural BP for quantum LDPC Bivariate Bicycle codes,
Julia, Enzyme reverse-mode AD.

- Code under study: **[[72,12,6]]**, `HZ` is 36x72, row weight 6, **column weight 3**,
  216 Tanner edges. Z-errors only (`HZ`/`LZ`).
- 90 unrolled layers, 5 epochs, batch 20, 500 gradient updates/epoch,
  online sampling from a 10^6-sample pool. 10^6 test shots.
- Loss = sine-residue syndrome loss + certainty + sparsity + Ising correlation,
  combined across layers by a **softmin**.
- Decoding rule: commit to the **first layer whose residual clears the syndrome**
  (`HZ`), then score that layer against the logicals (`LZ`). No OSD backstop.

**CER data** = per-qubit error rates plus pairwise couplings
`J_ik = log[P00*P11 / (P01*P10)]`, from `data/<codename>/correlated_weights/`.

**Training is CPU-only.** Enzyme cannot differentiate through GPU array
allocation, so the GPU is used only by the test-time forward pass. This governs
every scheduling decision.

---

## 2. The question and the two hypotheses

Adding the two-body correlation term to the loss made the logical error rate
**worse**, repeatedly. `count_syndrome_satisfactions` (in `src/predict.jl`) splits
every failure into two mutually exclusive kinds, which call for opposite fixes:

- **H1 — coset tipping.** The recovery clears the syndrome but lands in the wrong
  logical coset. The correlation edges come from co-support in `HZ`, so the
  pairwise potentials point along the code's cycle space, which is spanned by
  stabilizers *and* logicals. Fix: prune/reweight edges overlapping low-weight
  logical support.
- **H2 — objective competition.** BP never clears the syndrome at all. The
  correlation term steals gradient from the syndrome term. Fix: rebalance the loss.

**The diagnostic came back H2.** Across three error rates, coset failures were
flat or slightly down with CER (no consistent sign), while convergence failures
rose 1.97x and 1.47x at the two higher p. So the cycle-space fix is not indicated.

Supporting, same runs: BP converges measurably *slower* with CER (layer-1
clearing fraction 0.947 -> 0.827 at p=7e-4), and the CER decoder fails on
*lower*-weight errors than the baseline (mean failing error weight 4.78 vs 5.73).

---

## 3. What is established

### 3.1 The correlation term is live and correctly scaled

From `data/72q_BB_cycles_1/correlated_weights/*.txt`:

| p | edges | min J | max J | mean | median | negative |
|---|---|---|---|---|---|---|
| 5e-4 | 540 | −1.03 | +3.97 | +1.81 | +2.06 | 22.6% |
| 7e-4 | 540 | −0.69 | +3.67 | +1.64 | +1.81 | 22.0% |
| 1.9e-3 | 540 | −0.15 | +2.69 | +1.13 | +1.06 | 19.1% |

Log-odds scale, bimodal, ~20% anti-correlated. Not on a probability scale.

From `logs/debugging_*_individual_losses.csv` (187,520 layer-entries):
`correlation_penalty` min −0.0373, max +6.1e-5, mean −3.05e-4, **0.32% exactly
zero**. Sign negative, i.e. a reward, as the docstring specifies. The failure mode
where an arm labelled "CER" carries an identically-zero correlation term **did not
happen here** — but note it *has* happened before (§6.3).

### 3.2 The correlation term is numerically tiny

Loss terms as they enter the total, at λ=0.1, p=7e-4:

| term | mean x weight | share |
|---|---|---|
| base_loss (syndrome) | 0.051178 | **75.7%** |
| sparsity x 0.05 | 0.015863 | **23.5%** |
| syndrome_regularizer x 0.001 | 0.000542 | 0.8% |
| **correlation x 0.1** | **−0.000031** | **0.05%** |

The correlation term is **0.06% of the syndrome term**; linear extrapolation puts
parity at λ ≈ 168. It is live but small. **Sparsity is ~400x larger** and is the
auxiliary term with real leverage.

### 3.3 The correlation term does not change layer selection

The softmin combines per-layer losses, so an auxiliary term could in principle
decide *which* layer the decoder commits to. Measured directly: restricted to the
473 batches where `base_loss` actually varies across layers, adding the
correlation term moved the argmin **0 times out of 473**. Sparsity moved it twice
(0.4%); all aux terms together, 52 times (11.0%).

Across-layer spread within a batch: base_loss 1.255, sparsity x0.05 3.6e-2,
correlation x0.1 **7.7e-4**.

> Caveat on an earlier version of this number: a first pass reported "correlation
> flips the argmin 15.7% of the time". That was an artifact — **79.8% of batches
> have `base_loss` identical on every layer**, so `argmin` returned the first tied
> index and float noise reshuffled it. Restricting to non-degenerate batches gives
> 0%.

### 3.4 Four-fifths of the training signal is degenerate

At p=7e-4, **79.8%** of logged batches have `base_loss` identical on every layer.
The error-weight distribution explains it: at p=5e-4, **82.9% of samples carry no
error at all**, so every layer clears trivially.

The distribution is **not** binomial — i.i.d. at p=5e-4 would give 96.5% zeros and
mean weight 0.036; the data has 82.9% and 0.307, with a fat tail at weight 3–4.
That is the intended clustered/correlated structure. Effective batch is ~8.6
informative samples out of 50, a 5.8x dilution.

This is arguably a bigger lever than λ or the gate: importance-sampling toward
weight >= 1 recovers most of it, at the cost of changing the training
distribution (applied identically to both arms it remains a fair comparison).

### 3.5 All failures sit beyond the guaranteed correction radius

At p=5e-4, both arms are **perfect at error weights 0, 1, 2** — exactly the
guaranteed radius of a d=6 code. Every failure is at weight >= 3.

Convergence failures are mostly **near-misses**: the modal minimum residual
syndrome weight is **3**, and since `HZ` has column weight 3 that means exactly
one uncorrected qubit. 397/567 (no-CER) and 232/428 (CER) at p=5e-4. An OSD step
would mop these up — which is the mechanistic reason BP-OSD exists.

### 3.6 The prior magnitude, not the correlation information, was the original problem

Raw CER single-qubit rates are **correctly calibrated**: median 0.004239 against
an empirical per-qubit rate of 0.004259 in the test data (0.5%). Rescaling them so
the median is 0.1 makes the prior **23.6x wrong** — and that is the version that
decodes better, by 12.5x:

| arm (p=5e-4, 30 layers, 5 epochs, seed 1) | failures | convergence | coset |
|---|---|---|---|
| `corr` raw priors + J (LLR 5.46) | **3788** | 3767 (99.4%) | 21 |
| `corr_rescaled` rescaled + J (LLR 2.20) | **303** | 243 (80.2%) | 60 |
| `sq_priors` rescaled singles only | 483 | 428 | 55 |
| `no_cer` flat p=0.1 | 620 | 567 | 53 |

Same couplings, same seed; only the single-qubit prior scale differs. At LLR 5.46
the initial BP messages sit where tanh' = 0.018 versus 0.36 at LLR 2.20, so BP
starts too confident to iterate and 99.4% of failures are convergence failures.

**So the rescaling is an inference temperature, not a calibration fix.** The data
is right; BP cannot use a prior that sharp. There is no privileged value — 0.1 was
a guess that worked, and the optimum should be swept.

**Corollary — `prior_llr_clip` was the wrong operation.** Every raw CER LLR is in
5.33–5.58, so *any* clip below 5.33 collapses all 72 qubits to one constant
(spread 0.0000) and destroys exactly the per-qubit information under test.
Median rescaling is approximately an additive LLR shift and preserves the spread
(0.2512 -> 0.2782). Results from a clip sweep bound the magnitude effect only.

**Confound to keep in view:** `J` is a log-odds coupling and is *not* rescaled, so
softening the field changes the field/coupling ratio — median h/|J| goes 2.66 ->
1.07. Part of `corr_rescaled`'s advantage may be that the correlation term became
~2.5x stronger relative to the field, which is confounded with λ.

### 3.7 Neural BP vs the classical baselines

19 error rates, `data/update_23July2026/`, 100 layers, 10 epochs, 10^6 shots:

- Neural BP (either arm) beats plain BP everywhere.
- Neural BP beats **BP-OSD (order 2)** below p ≈ 0.0024, best **24.4x at
  p = 3e-4**; BP-OSD overtakes it from p ≈ 0.0028 up (1.36x at p=5e-3).
- CER vs no-CER across those 19 points: significantly better at 5, worse at 7,
  indistinguishable at 7 — sign flipping with p, i.e. consistent with noise.

The crossover is a stronger result than anything in the CER comparison, and it is
robust to the seed caveat (a 24x gap is far outside a 2x noise band). **Before
relying on it, confirm BP-OSD was run on the same Z-only decoding problem and the
same test samples** — a 24x gap invites that question first.

---

## 4. The current experiment: the syndrome gate

### 4.1 The defect

`syndrome_gate_threshold` was **absent from `default_hyperparams`** in
`src/command_line.jl`, so the −1.0 fallback lived only in a `get(...)` at
`src/train.jl:353`, and no TOML anywhere set it. **Every run in the project's
history took the ungated path** (`src/loss.jl:400`), silently.

### 4.2 The mechanism (this is the hypothesis under test)

- **Ungated:** the auxiliary terms — certainty, sparsity, and the CER correlation
  reward — apply to **every sample**, including samples whose residual is nowhere
  near clearing. For those, the correlation reward *competes with* the syndrome
  objective instead of choosing among solutions that already satisfy it.
- **Gated (τ > 0):** `syndrome_gate_per_sample` opens the aux terms only where the
  soft syndrome weight is below τ ("τ = 0.5 => open below half a broken check"),
  and layer selection sees `base_loss` alone. This is the constrained form:
  satisfy the syndrome first, then let the prior choose among the solutions that do.

This predicts **convergence** failures specifically, which is what was observed.
Note it is a **per-sample** mechanism and therefore *not* addressed by §3.3, which
only rules out the correlation term changing layer *selection*.

### 4.3 The design

`expts/misc/sweep_gate_cer.sh`, 2 x 2 x 5 seeds = 20 runs, p=7e-4, 90 layers:

| run_tag | use_CER | τ | correlation term |
|---|---|---|---|
| `_gc_cer_ungated` | true | −1.0 | active (λ 0.01 -> 0.76) |
| `_gc_cer_gated` | true | 0.5 | active, gated |
| `_gc_nocer_ungated` | false | −1.0 | inactive |
| `_gc_nocer_gated` | false | 0.5 | inactive |

Same seed set in every arm; both annealing schedules exactly as in the runs being
explained, so the only differences across arms are `use_CER` and the gate.

**The no-CER gated arm is load-bearing, not padding.** The gate also governs
sparsity and certainty, which carry ~400x the weight of the correlation term
(§3.2). Without that arm, a convergence improvement in `cer_gated` could not be
attributed to the correlation term rather than to gating in general.

### 4.4 Reading protocol (fixed before seeing the numbers)

| observation | reading |
|---|---|
| convergence down in `cer_gated`, **flat** in `nocer_gated` | **H2 confirmed via the ungated correlation reward.** The fix is the gate. |
| convergence down in **both** | Gating helps generally, most plausibly through sparsity. Not a CER story. |
| coset moves instead | H1 after all; the gate is not the relevant knob. |
| nothing beyond the seed spread | The gate is not the explanation. Look at §3.4 (degenerate batches) or §3.6 (prior scale). |

**Check `gate_open_fraction` first.** If the gate is almost always closed at
p=7e-4, the "gated" arms measured *"aux terms off"* rather than *"aux terms
applied conditionally"*, and every contrast means something cruder than intended.
It is in `logs/debugging_*_individual_losses.csv`, surfaced by the collector.

### 4.5 The statistic that counts

`expts/misc/collect_gate_cer.jl` (via `bash misc/sweep_gate_cer.sh --collect`)
emits per-run, per-arm and contrast CSVs. For each contrast it reports **both**:

- a **paired-by-seed** difference with `t` — blocks on the seed, so it measures
  the configuration effect against the seed-to-seed spread. **This is the honest one.**
- a **pooled two-proportion z** — test-set uncertainty only.

They diverge badly. On a synthetic set with a known 5% effect the pair read
t=+3.36 / z=+2.45; with a known 25% effect, t=+6.59 / **z=+13.19**. Quote `t`.

---

## 5. Results to fill in

> The 20 runs have completed but their numbers are not in this document.
> Paste `gate_cer_per_arm.csv` and `gate_cer_contrasts.csv` here.

```
PER ARM (mean +- sd across 5 seeds)
arm              seeds   failures        coset          convergence      gate open
cer_ungated          5   ...             ...            ...              ...
cer_gated            5   ...             ...            ...              ...
nocer_ungated        5   ...             ...            ...              ...
nocer_gated          5   ...             ...            ...              ...

CONTRASTS (paired by seed)
cer_ungated  - nocer_ungated     convergence  ... +- ...   t = ...
cer_gated    - nocer_gated       convergence  ... +- ...   t = ...
cer_ungated  - cer_gated         convergence  ... +- ...   t = ...     <- the test
nocer_ungated- nocer_gated       convergence  ... +- ...   t = ...     <- the control
```

---

## 6. Methodological traps already paid for

Each of these produced a wrong or unusable result at least once.

### 6.1 Test-set significance is not configuration significance

**The recurring error.** With 10^6 shots, a two-proportion z on failure counts is
enormously powerful — and answers *"do these two trained networks differ?"*, to
which the answer is always yes. The question of interest is *"does the
configuration differ?"*, and the relevant noise is the **training-seed spread**.

Measured spread on **provably identical** configurations: **309 vs 620 failures
(2.0x)**. Any single-seed ratio inside that band is not evidence. Several earlier
"significant" findings (z up to −26.6) were single-seed and remain uninterpreted.

Power: to resolve a 137-failure difference needs ~6 seeds/arm if σ=80, ~21 if
σ=155, ~41 if σ=220. σ was estimated from n=2, so **measure σ first**.

### 6.2 Baseline trial counts differ between files

`72q_BB_cycles_1_BP+OSD_failure_rates_OSD_E_order_2.txt` (aggregate) is **10^5
trials**; the per-p files are **10^6**. They agree on *rate* (2.98e-3 vs 3.08e-3
at p=5e-4). Comparing raw counts across them produced a factor-of-10 error and a
briefly-believed false claim that Neural BP "matched BP-OSD". **Always compare
rates.**

### 6.3 `use_CER = true` with unusable CER data is bit-identical to `use_CER = false`

`src/log_model.jl:48-58`: with `use_cer = true` but an empty marginals dict, every
qubit falls through to `log(9)` — exactly the `use_cer = false` preset — and the
connectivity comes back empty, so the correlation term is inactive too. And
`parse_cer_data` is called with `verbose = false`, so the "file not present"
warning never prints.

This produced an arm labelled `corr` whose **weights JSON was byte-identical
(same SHA-256) to the `no_cer` arm**, from a genuinely separate 26-minute run. It
was caught only because seeding made the collision exact rather than merely
similar. Triggered by either omitting `--cer_data` (default `unspecified.txt`
does not exist) or pointing it at a flat-priors file.

**Recommended:** make `use_CER = true` with a missing or pair-free CER file a hard
error. Not yet implemented.

### 6.4 Every varied hyperparameter must appear in the filename, or runs collide

Filenames are `..._trained_using_<source><cer_tag><run_tag><seed_tag>`. With
`retrain = false`, two configurations sharing a name mean the second **silently
loads the first's weights**. Seed and run_tag are in; `single_qubit_rescale`
deliberately is not (put it in `run_tag` when sweeping it).

Related: the debug logfile path carried **no** `run_tag`, so a four-point sweep
wrote one log and lost three. Fixed in `src/train.jl`.

### 6.5 The test-side preflight must reconstruct exactly what training wrote

`_expected_weights_path` in `expts/submission/batch_commands.jl` omitted
`run_tag` and `seed_tag`, so it searched for names that could never exist and
dropped **all 19** test cases with "nothing to run" while the models sat on disk.
Fixed, and the warning now prints the path it looked for.

### 6.6 `runtime` in the results CSV spans training **and** testing

`start = time()` is set before `train_Nachmani_neuralbp` and `runtime` computed
after testing. A train/test split cannot be recovered from it.

Separately: the correlation term costs ~864,000 scalar loop iterations per forward
pass (80 scored layers x 20 samples x 540 edges, 2 sigmoids each) — **2.2x the
entire BP forward pass** — inside the Enzyme tape. That is a real 7–25x training
slowdown for the CER arm, independent of any wall-clock confound.

### 6.7 Cluster specifics that cost hours

- **Compute nodes have no internet.** `Pkg.instantiate()` blocks on TCP until
  timeout. Warm the depot on a **login** node (`misc/precompile_depot.sh`) and run
  the job with `JULIA_PKG_OFFLINE=true`. `LocalPreferences.toml` must keep
  `[CUDA_Runtime_jll] local_toolkit = true` — artifact mode tries to download.
  The CUDA "could not find an appropriate runtime" message on a login node is
  **benign** and its advice (`set_runtime_version!`) is wrong here.
- **Julia's GC sizes its heap against total physical memory, not the cgroup
  share.** 42 workers on one node each parked at ~5.4 GB (live set ~108 MB),
  exhausted a 226 GB pool, collapsed the page cache, and ran at **1% CPU**.
  Always pass `--heap-size-hint`.
- **`--mem-per-cpu` is a pooled limit** (`mem_per_cpu x cpus_per_task`) for the
  whole task, not per process.
- **MIG allows one instance per job** and has no MPS. Four concurrent workers each
  opening a CUDA context on one 20 GB MIG killed two at
  `cuDevicePrimaryCtxRetain`. Train at full CPU concurrency, then test
  **serially**, in the same job.
- Narval bundle ratio: 1 RGU = 3.00 cores = 31.1 GB; billing is
  `max(RGU, cores/3.00, memory/31.1)`. At the recommended ratio
  (`a100_3g.20gb` = 6 cores/62 GB = 2.0 RGU; whole `a100` = 12 cores/124 GB =
  4.0 RGU) the GPU costs nothing beyond the cores. The GPU idles ~90% of a
  train+test run, which is the honest cost of doing both in one job.

---

## 7. What I would do next, in order

1. **Read `gate_open_fraction`** before anything else (§4.4).
2. **Apply §4.4** to the contrasts, quoting the paired `t`, not `z`.
3. **Sweep `single_qubit_rescale`** — {raw, 0.01, 0.03, 0.1, 0.3} x {with J,
   without J} x the same seed set. §3.6 makes this the largest known effect
   (12.5x), and 0.1 was never optimised. Tune on a different test seed to avoid
   selecting on the reported sample.
4. **Re-scope any λ sweep upward.** {0.01 … 1.0} spans 0.006%–0.6% of the syndrome
   term (§3.2) and will measure nothing. Try {1, 10, 100}.
5. **Address the degenerate-batch problem** (§3.4). Four-fifths of training
   batches cannot inform layer selection.
6. **Close the silent-fallback hole** (§6.3) before the next campaign.

## 8. Where things live

```
src/loss.jl                 gated / ungated paths (line ~400); correlation term (~104)
src/train.jl                syndrome_gate_threshold read (~353); weights filename (~653)
src/command_line.jl         parse_hyper_parameters, defaults, seed helpers
src/predict.jl              count_syndrome_satisfactions, predict_and_diagnose_neuralbp
src/log_model.jl            load_base_BP_model, the p=0.1 fallback (48-58)
src/neuralbase.jl           parse_cer_data, rescale_single_qubit_error_rates

expts/misc/sweep_gate_cer.sh        the 2x2 sweep; --setup, --collect
expts/misc/collect_gate_cer.jl      all diagnostics -> 3 CSVs + contrast table
expts/misc/cer_vs_no_cer.jl         CER vs no-CER across p, with BP / BP-OSD
expts/misc/test_syndrome_diagnosis.jl   acceptance tests for the diagnostic
expts/misc/precompile_depot.sh      login-node depot warm-up

data/72q_BB_cycles_1_debug/results/  test-side diagnostics
data/72q_BB_cycles_1_debug/logs/     train-side (gate_open_fraction et al.)
```

Conventions for any code changes are in `CLAUDE.md` at the repo root.

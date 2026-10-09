# CorrelatedBPDecoderWithCER — working notes

## Julia coding conventions

These are standing requirements for all Julia code in this repository. Apply them
to new code, and when editing an existing function bring the parts you touch into
line.

### 1. Names state what the value is

Single letters and abbreviations are not acceptable for anything with meaning.

```julia
# no
s = uppercase(strip(input))
raw = floor(usable * 1024 * 1024 / bps)

# yes
normalised_memory::String = uppercase(strip(input))
unrounded_batch_size::Int = floor(Int, usable_bytes / bytes_per_sample)
```

Loop indices (`i`, `j`, `k`) and conventional mathematical symbols in a formula
that the surrounding comment defines (`J`, `σ`, `α₄`) are fine — the rule is about
values whose meaning is not otherwise stated.

### 2. Annotate variable types

Every local gets a type unless the type genuinely cannot be asserted.

```julia
# no
value = parse(Float64, captured_digits)

# yes
numeric_value::Float64 = parse(Float64, captured_digits)
```

"Cannot be asserted" means the value is legitimately a union or is inferred from a
generic argument — **not** that the type is inconvenient to write. When a function
can return nothing, say so:

```julia
# WRONG — `match` returns Union{RegexMatch, Nothing}; a RegexMatch is not
# convertible to String, so this throws on the first *successful* match.
memory_match::String = match(pattern, text)

# right
memory_match::Union{RegexMatch, Nothing} = match(pattern, text)
```

### 3. No guard clauses that hide an action behind `&&` or `||`

```julia
# no
memory_in_mb <= 0 && throw(ArgumentError("..."))
isempty(specification) || return default_batch_size

# yes
if memory_in_mb <= 0
    throw(ArgumentError("..."))
end
```

Applies to `throw`, `@warn`, `return`, and any other side effect.

### 4. No `variable = if ... end`

Declare the variable, then assign in the branches.

```julia
# no
cer_tag = if use_CER "" else "_no_cer" end

# yes
cer_tag::String = ""
if !use_CER
    cer_tag = "_no_cer"
end
```

### 5. Ternaries only for short values

`condition ? value_a : value_b` is fine when both arms are simple values. As soon
as an arm contains a call chain, an interpolation, or anything with side effects,
use `if`/`else`.

### 6. No `return <long expression>`

Bind the result to a named, typed variable and return that. It gives the value a
name, gives the reader the type, and makes the return breakpoint-able.

```julia
# no
return max(1, round(Int, numeric_value * megabytes_per_unit[unit_symbol]))

# yes
megabytes::Int = max(1, round(Int, numeric_value * megabytes_per_unit[unit_symbol]))
return megabytes
```

### 7. Every function declares its return type

Including functions that return nothing.

```julia
function print_console_rule(io::IO = Base.stdout)::Nothing
    println(io, repeat('-', CONSOLE_RULE_WIDTH))
    return nothing
end
```

---

## Cluster notes (Narval)

- **Warm the Julia depot on a LOGIN node** before `sbatch`: `bash misc/precompile_depot.sh`.
  Compute nodes have no internet, and array tasks share one Lustre depot — in-job
  `Pkg.instantiate()`/`Pkg.precompile()` can stall for the whole walltime.
- **`LocalPreferences.toml` must keep `local_toolkit = true`** for `CUDA_Runtime_jll`.
  Switching to artifact mode (`version = "..."`) makes CUDA.jl try to download on a
  node with no network, which hangs until timeout.
- **Always pass `--heap-size-hint`** when running many Julia processes on one node.
  The GC sizes its heap against total physical memory, not the cgroup share, so N
  workers each balloon to their high-water mark and collapse the page cache.
- `--mem-per-cpu` is a **pooled** cgroup limit (`mem_per_cpu × cpus_per_task`) shared
  by every process in the task — not a per-process cap.
- **Run `parallel` in the BACKGROUND and `wait` on it.** Bash runs a trap only
  between *foreground* commands, so with `parallel` in the foreground the TERM that
  `--signal=B:TERM@600` promises sits queued behind it, the wall's hard kill arrives
  first, the handler never runs and **every completed point in the slice is lost**.
  Measured: foreground staged out 0/8 points, background + `wait` staged out 3/8.
  `sweep_{train,test,h2_checks}.sh` always did this; `sweep_{hyperparams,transfer}.sh`
  had regressed to foreground and were fixed on 2026-10-05.
- **Serialise the `CUDA_Runtime_jll` rebuild across array tasks.** Tasks that lose the
  race read a half-written `.ji` and die with `ArgumentError: No value arguments
  present` after "CUDA Being precompiled by another machine". That, not walltime, is
  what cost 105 of 280 results on 2026-10-01 (3 of 8 tasks died at the gate; the 5
  survivors each did 35 tests in 75 min, 37% of a 4 h budget). The test job now takes
  an `mkdir` lock, one task rebuilds and touches a `.done` sentinel, the rest poll for
  it (1800 s), and `CUDA.functional()` is re-checked once after 60 s.
- `sweep_hyperparams.sh` **refuses to write a job that cannot finish**: a preflight
  multiplies `seconds_per_test` / `seconds_per_train_point` / `job_startup_seconds`
  (measured, in the settings file) against the array width and exits 1 with the
  minimum `*_array_tasks` that would fit. Raise those constants if the code or the
  sample count grows.
- **`--dependency=afterok` needs the training job to actually report failure.** The
  train script used to end on an `echo`, so a task exited 0 even when every point
  had crashed — chaining on `afterok` was decorative. It now exits 1 when any point
  has a non-zero exit in the joblog, when `parallel` itself fails, or when the
  walltime handler fires (a wall-killed task leaves a partial model set; the points
  it finished are still staged out first, so nothing is lost). The generator prints:

  ```
  TRAIN_ID=$(sbatch --parsable <train>.sh)
  sbatch --dependency=afterok:$TRAIN_ID --kill-on-invalid-dep=yes <test>.sh
  ```

  `afterok` on an array is **all-or-nothing** — every task must exit 0, so one bad
  point stops every test — and `--kill-on-invalid-dep=yes` cancels the test job
  rather than parking it as `DependencyNeverSatisfied`. A NaN-**rolled-back** epoch
  is not a failure: the point exits 0 and writes weights, so rollbacks do not block
  testing. Behind the dependency the test job also **refuses a partial model set**
  (counts staged-in `*.json` against the expected point count and exits 1), which
  covers what the dependency cannot see: training never submitted, a stage-out that
  did not land, a lone test resubmission, or a `cleanup.py` between the two.
- **`bash -n` on a generated job script proves nothing about `set -u`.** On
  2026-10-05 all 8 test tasks died in under a second — `line 75: TASK: unbound
  variable` — because the depot-lock block echoes `[test task $TASK]` and sat *above*
  `TASK=${SLURM_ARRAY_TASK_ID:-0}`. 0 of 160 results, 4 h of training wasted, and it
  looked exactly like a walltime kill. An unbound variable is a RUNTIME error, so the
  verification is: generate, then **execute** the job script with stub `julia` /
  `parallel` on `PATH` and fake `SLURM_*` env vars, and check it exits 0. That
  reproduces the failure on the old script and passes on the fixed one.

---

## The loss (`src/loss.jl`)

- `compute_loss` = softmin over scored layers of the base loss, the residue of
  e + σ(μ) against [H; L]. **That is the whole loss.** The certainty (L2), sparsity
  and correlation (L3) terms, their gates, and every hyperparameter that fed them
  were removed on 2026-09-16: six L3 forms over ~1500 runs never produced a
  coupling effect through the loss, while the couplings inside the check node
  (below) halve the classical failure rate with nothing trained.
- The only annealed hyperparameter is `loss_layer_temperature`; `warmup_layers`
  drops the first layers from scoring. TOML keys for the removed terms are ignored
  if present in an old file.
- Sweep tags are now `_hp<cer|nocer>[_cnenr<alpha><F|L>]`; the collector still
  parses the legacy `_sp/_lam/_tau/...` scheme for old results directories.
- `expts/misc/{sweep_correlation_weight,sweep_gate_cer,sweep_lambda,sweep_h2_checks}.sh`,
  `collect_{lambda,gate_cer}.jl`, `summarize_cer_sweep.jl` drove the removed terms
  and are obsolete.

## Enriched check node (`src/soft_constraints.jl`, branch `correlation-adapted-messages`)

- `check_node = "tanh" | "enriched"` in the hyperparameters TOML selects the
  check-to-variable rule of the **forward pass** (training and testing alike).
  "enriched" puts the CER couplings inside each check factor and sends the exact
  marginal; derivation in `refs/soft_check_nbp.tex`. Reduces to "tanh" at α = 0.
- α is `coupling_scale` (a length-1 weight vector on `NachmaniNeuralBP`), initialised
  from `coupling_scale_init` (1 = Bayesian), learned unless
  `coupling_scale_learnable = false`, saved in the weights JSON and the results CSV.
- **An enriched run needs a non-empty `run_tag`** (the scripts refuse otherwise): no
  filename carries a check-node tag, so it would load/overwrite the tanh run's files.
- Same kernel for classical BP: `standard_bp_experiments.jl` with `check_node = "enriched"`
  and a fixed `coupling_scale_init` is the training-free α = 0 vs α = 1 comparison.
- Tests: `tests/test_soft_constraints.jl` (tables, α = 0 ≡ tanh, brute-force posterior,
  Enzyme gradients, CPU ≡ GPU, tanh path unchanged, weights-file round trip).

## Layer schedule on α (`coupling_schedule`, added 2026-09-22)

- **α is a unit-consistency factor, not a fitted number.** `single_qubit_rescale` maps
  the median single-qubit rate to 0.1 and leaves J untouched, so the pair term must be
  softened by the same ratio: `α* = log(9) / log((1−m)/m)`, m the median raw rate.
  That is 0.426 on `p_0.0005_sig_0.001` (where "0.42" came from) and **0.503 on
  `p_0.0015_sig_0.0015`**. Every run on the new dataset before this date used 0.42.
- `coupling_schedule = "constant" | "step"` in the TOML. "step" uses
  `α_t = α · d(t)`, `d(t) = 1/(1 + exp((t − T₀)/w))`: full couplings early, rolled
  off around layer T₀ over ~4w layers. Rationale: loopy overcounting compounds with
  iteration, and the per-weight failure analysis showed the enriched decoder's late
  commits (median layer 8–16) are the ones that land in the wrong coset.
- **Two trainable parameters, not one per layer.** Stored as `coupling_schedule =
  [T₀, log(w − 0.5)]` on `NachmaniNeuralBP`; init from `coupling_schedule_layer_init`
  / `coupling_schedule_width_init` (layers), learned unless
  `coupling_schedule_learnable = false`. Only ~2% of samples are active past layer 10
  and `warmup_layers` hides the first layers from the loss, so 90 free α_t would fit
  noise. The width has a 0.5-layer floor (its gradient is d(1−d)/w).
- **d(t) is computed as `(1 − tanh(u/2))/2`, never `1/(1+exp(u))`.** Same function;
  the exp form overflows to Inf at 88.7·w layers past T₀ and Enzyme's reverse pass
  returns Inf·0 = NaN there, NaN-skipping the batch — reachable at the width floor.
- Needs `check_node = "enriched"` (`NeuralBPBase` refuses it on tanh) and, like the
  check node, **a non-empty `run_tag`** — no filename carries a schedule tag.
- Sweep spec: `enriched:<α>:<fixed|learn>[:step:<T₀>:<w>:<fixed|learn>]`, tag
  `_cnenr<α>F/L[_sch<T₀>w<w>F/L]`. `standard_bp_experiments.jl` honours the same keys
  with everything fixed: that is the classical (T₀, w) scan.
- `check_training_health.sh <codename> [tag]` reports completed epochs per arm (an
  epoch with 6 non-finite gradients is **rolled back**, weights and Adam state) and
  weight sd (~0.058 = never moved). Arm comparisons are only meaningful when
  completed epochs are equal across arms.
- **The NaN gradients are the enriched check node, not the layer design** (measured
  2026-10-05, 40 points, `initial_conditions_scale = 0.3`, lr 0.001):

  | split | damaged points |
  |---|---|
  | `check_node = "enriched"` | **8/20** |
  | `check_node = "tanh"` | **0/20** |
  | `loss/commit = last/last` (design A) | 4/20 |
  | `loss/commit = softmin/first` | 4/20 |

  Two of five seeds in *every* enriched arm, none in any tanh arm, and dead even
  across the layer designs. 6 of the 8 lost **5 of 5** epochs, 2 lost 4 of 5 — those
  models are the initialisation with a few hundred surviving updates on top.
  The schedule (`_sch12w3L`) neither causes nor prevents it.
- **Reading rollbacks out of the logs.** `logs/debugging_*.csv` is the reliable
  source: a damaged point has `max(nan_skip_count) = 5` (the 6th breaks out before
  the row is written) and carries an extra `epoch 0` block. The `.out` files show
  only the *first* failing point's stderr, so their rollback count is a floor, and
  `cluster/logs/<task>/<slot>/<cmd>/stderr` cannot be joined to an arm — GNU
  `parallel --results` keys on **argument slot, not seq** (everything lands under
  `1/`) and truncates the command-derived directory name mid-tag, dropping the seed.
## Which layer is scored vs which layer commits (added 2026-10-01)

- The training loss and the test-time readout were looking at **different layers**.
  Measured on the best config (α = 0.42 learned, `initial_conditions_scale = 0.3`):
  98.8% of test-time commits land in **layers 1–10**, which `warmup_layers = 10`
  excluded from the loss entirely, while the loss's own per-batch argmin sat at
  **layer 11** — the first layer it was allowed to see — in 87% of error-free
  batches, and at a late layer (median 20) in the ones carrying gradient.
- Worse, `softmin` is **anti-aligned** with first-to-clear testing: it is satisfied
  as soon as ONE layer is good, while first-to-clear needs EVERY layer to be
  trustworthy because any of them might commit. A layer that clears `H` in the
  wrong coset has base loss ≈ 1; `argmin` skips it, testing commits to it. That is
  the mechanism behind 1745 coset failures under a base loss of 0.046.
- Two new TOML keys make the pairing explicit. **They must agree:**

  | `loss_layer_selection` | `commit_layer_rule` | |
  |---|---|---|
  | `"last"` | `"last"` | design A — same layer both sides, no mismatch; gives up early stopping |
  | `"mean"` | `"first"` | design B — all layers trained, so first-to-clear is safe |
  | `"softmin"` | `"first"` | the historical pair, which cannot work |

  Both default to the historical values (`softmin`, `first`), so every earlier run
  reproduces. Under `"last"` the `warmup_layers` value is irrelevant to training.
- **So `warmup_layers` is a knob on the SOFTMIN arm only.** `combine_layer_losses`
  reads `losses_per_layer[end]` under `"last"` and nothing else (`loss.jl:239` says
  so outright), so changing warmup leaves design-A training bit-identical. Setting
  it to 0 is the fix for the softmin arm, which was scoring layers 11–90 while
  first-to-clear commits at layer 1.2–3.0.
- **Sizing the design A vs historical question.** The gap measured −5.4% on the CER
  arms and +6.5% on the no-CER arms against a seed sd of ~10%, so a 2σ answer needs
  about (2 × 10 / 5)² ≈ **16 seeds per arm** — 64 trainings. Two or four seeds will
  keep returning "1–2σ, sign depends which arm you look at". That is a cluster
  question. Layer-count effects are a different matter: anything above ~10% shows
  up at 2 seeds.
- `commit_layer_rule = "last"` reclassifies samples that cleared early and drifted
  back out as **convergence** failures, not coset failures. Recorded in the
  results CSV alongside `loss_layer_selection`.
- Also measured: only **3.8%** of the base loss is coset signal
  (1745/10⁶ × ≥1 per violated logical row); the other 96.2% is `σ(μ)` saturation on
  already-correct decodes. Weighting the 12 logical rows up is the lever for that,
  and is not yet implemented.

### Where the per-layer minimum actually sits (`misc/analyse_layer_losses.py`, 2026-10-08)

Per-layer logging **already exists** and is already on: `--isdebug true` (the sweep
generator always passes it) makes `train.jl` write
`logs/debugging_*_individual_losses.csv`, whose `base_loss` column is the FULL
per-layer vector and whose `total_loss` is the reduced number. No code change was
needed. Run `python3 misc/analyse_layer_losses.py --workdir <codename>`.

Measured on the 50-layer / warmup-0 run, 2500 batches per point:

- **`last == min` on 94–98% of batches.** The per-layer loss decays to a floor and
  sits there, so the last layer ties the minimum almost always. Design A's loss and
  the softmin's target are *the same number* on nearly every batch — which is why
  the two designs train such similar models.
- **Only 5.6–12.2% of batches carry gradient** (per-batch minimum above 1e-3). The
  median per-batch minimum is **6.4e-6** (CER) and **3.3e-7** (no-CER). So ~90% of
  gradient updates are on already-solved batches. *This* is the binding constraint,
  not the layer rule — it is the same story as weights moving only +3–4%, and no
  choice of which layer to read can fix a loss that is zero nine batches in ten.
- On the batches that DO carry gradient the argmin is genuinely far from the end:
  median layer **20–27** (CER) and **32–41** (no-CER), p10 ≈ 8–10. In roughly 40% of
  them the last layer sits **1.3–1.8×** above the minimum. So design A is mildly
  mis-targeted exactly where it matters — real, but small, and consistent with the
  ambiguous failure-count difference.
- **Only 50–58% of batches are monotone non-increasing.** The per-layer loss rises
  again about half the time; that is the loss-space signature of the same drift that
  costs design A its cleared decodes at test time.
- **The 50-layer argmin distribution is CENSORED**: 17.9% of informative batches
  (up to 36.8% on a no-CER arm) have their argmin at layer 50 exactly, i.e. the loss
  was still falling when the layers ran out. Any "where is the minimum" statement
  from a 50-layer run is a lower bound; re-run the analysis at 90.

### Why the last-layer loss under-constrains, and what weighting fixes it (2026-10-08)

- **Mechanism.** Every weight is on the backprop path to layer 90, but once a sample
  has converged (layer 5–10 on most batches) the remaining layers sit at a BP fixed
  point whose backward Jacobian is contractive, so ∂L₉₀/∂(layer-t weights) decays
  like ρ^(90−t). Early weights are not unconstrained in principle; they are
  unconstrained in practice. A per-layer weighted loss gives each layer an
  un-attenuated gradient from its own term.
- **The signal a failure produces is tiny by construction**: one coset failure in a
  batch of 20 violates ≥1 of 12 logical rows among 72, so its batch loss sits
  ~1/(20×72) ≈ 7×10⁻⁴ above the floor. That is the 10⁻³ "informative" threshold,
  derived independently. No layer weighting changes this dilution — weighting the
  logical rows up, or smaller batches, is the complementary lever.
- **A flat mean over layers ("mean", what was called design B) is a BAD loss.**
  Measured on real per-layer profiles: it separates solved from informative batches
  by only **2×**, with 100% of a solved batch's loss coming from layers 1–4. It would
  spend its gradient teaching 1-iteration BP to decode. Retracted as a recommendation.
- **The first ~5 layers are the BP transient and no weighting can fix them**:
  median per-layer loss on solved batches is L1=35, L2=7.5, L3=1.1, L4=0.24,
  L5=0.12, then L6=5.6e-5 and the floor from L7. Any weighting that gives them
  non-negligible weight is dominated by them (linear 18×, quadratic 333×, cubic
  5,797×, a sigmoid at the natural midpoint 123× — all mostly layers 1–4). The
  transient is a property of BP convergence, not of the layer count.
- **The weighting that works: zero weight on layers 1–5, then a linear ramp to 1 at
  the last layer.** Separation **108,000×** — better than design A's 70,000× — while
  41 of 50 layers carry weight > 0.1 and 66% of an informative batch's loss comes
  from the last 40% of layers (design A: 100% from one layer). Once the transient is
  excluded the power barely matters (quadratic 106,000×); linear constrains more
  layers. Warmup 3 is not enough (1,500×: L4, L5 leak in); 8 gains nothing over 5.
- Commit at the last layer with this loss, since it is what the ramp emphasises and
  it directly penalises the drift that the last-layer readout suffers from.
- **Implemented as `loss_layer_selection = "ramp"`** (`LOSS_LAYERS_RAMP = 3`):

  ```
  w_t = tanh(k·u) / tanh(k),   u = (t − warmup_layers) / (n_layers − warmup_layers)
  loss = Σ w_t L_t / Σ w_t     over the scored layers t = warmup+1 … n_layers
  ```

  Zero through `warmup_layers`, exactly 1 at the last layer, a weighted MEAN so the
  scale matches `last`. `k` is `loss_layer_ramp_sharpness` (default 3.0,
  `DEFAULT_LOSS_LAYER_RAMP_SHARPNESS`): k → 0 is linear, k large is a step to
  uniform-after-warmup. At 90 layers / warmup 5 / k = 3, w crosses 0.5 at layer 21
  and 0.9 at layer 47 (at 50 layers: 14 and 27). Once the transient is excluded
  every k separates ~110,000×, so k is a choice about emphasis, not scale.
  - `ramp_layer_weight` is written against the SCORED index (1 = first layer after
    warmup), so it needs neither `warmup` nor `n_layers` — `u` is the same number.
  - `warmup_layers` is part of the loss definition under ramp (it is where the
    weights start from zero). The generator warns below 3; the measured knee is 5.
  - Threaded as one new positional argument `loss_layer_ramp_sharpness` after
    `loss_layer_selection` in `get_loss_value` / `get_individual_loss_values`
    (`Enzyme.Const`), with defaults on `combine_layer_losses` / `compute_loss` so
    older call sites still work. **Every direct `get_loss_value` call in `tests/`
    must list all four `Const`s before `base`** — three of them had been missing
    `loss_layer_selection` since it was added and would have failed on positional
    binding; fixed 2026-10-08.
  - Recorded in the results CSV as `loss_layer_ramp_sharpness` and `warmup_layers`
    for every mode (the column set must not depend on the mode).
  - Sweep: `loss_layer_ramp_sharpness=<k>` is a sweepable override; the filename
    tag is `_lslramp<k>_clr<rule>` with `.` → `p`, so two sharpnesses never collide.
    `run_local.sh --design ramp|all` adds the arm with its own `--ramp-warmup`
    (5) and `--ramp-sharpness` (3), independent of `--warmup`.
  - Tests in `tests/test_loss.jl`: weight limits (both k extremes), the exact
    0.5/0.9 crossings at the production geometry, a hand-computed reduction, the
    transient/plateau/rise profile where `last` sees only the rise, `mean` is
    swamped, and `ramp` scores the rise; Enzyme gradients finite and non-zero.
    Every assertion was checked against a Float32 Python reference, since Julia
    cannot run in this environment. **The first version of the crossing test was
    wrong** — it carried the 50-layer numbers into the 90-layer claim. Check the
    geometry before trusting a "0.5 by layer N" statement.
- Separately: **coset and convergence failures trade off almost perfectly** along
  every axis swept so far. Nothing in the training budget reduces coset failures —
  more layers *increase* them (a coset failure has already cleared `H`, so later
  layers can only create more), and more epochs do too (904 → ~1100 from 1 to 5
  effective epochs). Which column to minimise depends on whether OSD is downstream.

### Measured 2026-10-07 (local run: 4 arms × 2 seeds × 4 test sets, 200k each)

Pooled over all four test sets and both seeds, 800,000 samples per arm:

| arm | total | coset | convergence |
|---|---|---|---|
| no-CER, design A | 2331 | 557 | 1774 |
| CER-enriched, design A | **1481** | 963 | 518 |
| no-CER, historical | 2188 | 626 | 1562 |
| CER-enriched, historical | 1566 | 990 | 576 |

- **CER + enriched is settled**: −36.5% under design A (13.8σ), −28.4% under the
  historical pair (10.2σ). This arm set conflates the priors with the check node —
  there is no CER-tanh arm here — so it does not re-separate the −20% / −10%.
- **Design A vs the historical pair is NOT settled**: −5.4% (1.5σ) on the CER arms,
  **+6.5% (2.1σ) on the no-CER arms** — opposite signs, neither significant. Two
  seeds cannot resolve a ~5% effect against ~10% seed spread.
- **THE CER GAIN IS ENTIRELY CONVERGENCE FAILURES.** Convergence 1774 → 518
  (−71%); coset 557 → **963 (+73%, 10.4σ)**. OSD repairs non-convergence and is
  blind to a coset failure, so **under a perfect OSD this data says no-CER wins**
  (557 vs 963). The −36% headline holds only if BP is the final decoder. Decide
  which column to minimise before quoting either number.
- **Design A's own cost is now measured.** 14–16% of its convergence failures have
  `min_syndrome_weight == 0` — the sample *did* clear at some layer and drifted back
  out before layer 90. That is 70 samples for CER and 279 for no-CER (12% of all its
  failures). `commit_layer_rule = "first"` has **exactly 0** such samples in all four
  arm-seeds, which is the right consistency check on the implementation.
- So the two designs cancel: design A removes the train/test layer mismatch but
  throws away cleared decodes; the historical pair keeps every cleared decode but
  trains on a loss dominated by `−T·log(n)`. **Design B (`"mean"` + `"first"`) is
  the combination neither has**, and this is the first concrete evidence for it.
- `mean_committed_layer` (over successes): design A is 90.0 by construction, the
  historical pair is **1.2–3.0**. Design A therefore buys an insignificant ~5% for
  roughly 30–70× the inference work.
- Learned α moved 0.42 → 0.413–0.437. Still inert.
- The `p=.0005 sig=.0005` test set is **seed-unstable** (cer/designA 106 vs 44) and
  is the one set where CER shows no advantage (+6.4%). The data is not at fault —
  identity fractions order cleanly at 83.9 / 79.8 / 76.6 / 61.1% across the four
  sets. Needs more seeds before it means anything.
- Metal cost: **143 s per test at 200k**, vs 129 s at 1e6 on an A100 — about 5.5×
  slower per sample, not the 6–12× estimated. 32 tests took 1.27 h.

### 50 layers does NOT work (measured 2026-10-08, same 4 arms × 2 seeds, warmup 0)

**Do not cut the layer budget.** Every arm got worse, and the decomposition says why.

| arm | total 90L → 50L | coset | convergence |
|---|---|---|---|
| no-CER, design A | 2331 → 3645 (+56%) | 557 → 502 (−10%) | 1774 → 3143 (+77%) |
| CER, design A | 1481 → **2703 (+83%)** | 963 → 920 (−4%) | 518 → **1783 (+244%)** |
| no-CER, historical | 2188 → 2421 (+11%) | 626 → 589 (−6%) | 1562 → 1832 (+17%) |
| CER, historical | 1566 → 2051 (+31%) | 990 → 899 (−9%) | 576 → 1152 (+100%) |

- **Coset failures are nearly layer-independent** (−4% to −10% for a 44% cut). They
  are a property of the model and the priors, not of the iteration budget. This
  sharpens the older note that "more layers increase coset failures" — the effect is
  real but small, and nothing like the convergence penalty.
- **Convergence failures are the whole story**: +17% to +244%. Cutting iterations
  strands every sample that needed more than 50 of them. The design-A comparison is
  clean (warmup cannot touch design A), so +83% for CER design A is a real
  90→50 effect, not a warmup artefact.
- **Design A is hit far harder than first-to-clear** (+83%/+56% vs +31%/+11%),
  because it pays twice: the truncation, and its inability to bank an early clear.
  Cleared-then-drifted rose 70 → **321** (CER) and 279 → **464** (no-CER). At 50
  layers design A loses to the historical pair by +32% (9.5σ, CER) and +51%
  (15.7σ, no-CER) — but that cross-design gap is confounded, because the historical
  arm also moved from warmup 10 to warmup 0 in the same run.
- The CER advantage **shrinks** at 50 layers: −25.8% (was −36.5%) under design A,
  −15.3% (was −28.4%) under the historical pair. The coset penalty worsens to +83%
  and +53%, so **the OSD inversion holds and gets stronger** — 920 vs 502 coset.
- Seed noise is visibly larger at 50 layers (e.g. CER design A on `p=.0007 sig=.001`:
  293 vs 176 across two seeds), as expected when more samples sit near the
  convergence boundary.
- Runtime at 50 layers averaged 109 s/test vs 143 s at 90 — roughly linear in layers,
  so the budget is not where the saving is worth taking.
- 0 of 8 points rolled back, so none of this is a training-health artefact.

### The ramp, measured (2026-10-08, 90 layers, `--design all`: 6 arms × 2 seeds × 4 sets)

Pooled, 800,000 samples per arm. Design A and softmin at warmup 0; ramp at warmup 5, k = 3.

| arm | total | coset | conv | vs 2026-10-07 (warmup 10) |
|---|---|---|---|---|
| CER, design A | 1481 | 963 | 518 | **identical** |
| CER, softmin + first | **1413** | 996 | 417 | 1566 → 1413 (−9.8%) |
| CER, ramp + last | 1599 | 964 | 635 | new |
| no-CER, design A | 2331 | 557 | 1774 | **identical** |
| no-CER, softmin + first | **1898** | 657 | 1241 | 2188 → 1898 (−13.3%) |
| no-CER, ramp + last | 2135 | 620 | 1515 | new |

- **Design A reproduced bit for bit** across two days — same seed, same machine. So the
  pipeline is deterministic here and cross-day comparisons are valid; and warmup is
  confirmed irrelevant to design A.
- **The warmup fix is the real win.** With softmin finally scoring the layers that
  commit (1–10), convergence failures fell 576 → 417 and 1562 → 1241 while coset
  stayed flat. Softmin + first + warmup 0 is now the best arm on both sides, and
  design A loses to it by +4.8% (1.3σ, CER) and **+22.8% (6.7σ, no-CER)** in the
  same run. The design-A question is settled in softmin's favour.
- **The ramp did not help**: +8.0% / +13.2% vs design A / softmin on CER, −8.4% /
  +12.5% on no-CER (2–4σ). Its coset count equals design A's exactly (964 vs 963);
  the deficit is convergence. It also has a **35% seed spread** on the training
  distribution (649 vs 479) where the others sit at 1–2%.
- **Why — the premise was wrong under Adam.** The per-layer RMS difference between
  any two losses trained from the same init is ~0.05–0.07 per weight, *flat across
  all 90 layers*, and the same for every pair (ramp–A, ramp–softmin, A–softmin).
  Three things follow. (1) Weights move ~0.05 RMS each — about 30% of their 0.173
  spread — so **"weights barely moved" was wrong**; the sd only grows 3–4% because
  independent per-weight moves add in quadrature. (2) Design A moves layers 1–5 as
  much as layers 86–90: **Adam's per-parameter normalisation removes the ρ^(90−t)
  attenuation entirely**, so early layers were never under-constrained in step
  size — they take full-sized steps in a direction set by a vanishing, noise-
  dominated gradient. The ramp cannot "give them gradient"; they already get
  full steps. (3) The three losses' updates are nearly uncorrelated at every
  layer, including the last, where all three agree on the objective — the
  update *direction* is noise-dominated. Corroborated by design A following the
  identical loss trajectory at 50 and 90 layers (a per-parameter constant
  rescaling of the gradient is invisible to Adam).
- So the binding constraint is signal-to-noise in the gradient direction, not which
  layers are weighted: ~94% of batches carry no signal and a failure contributes
  1/(20×72) when it appears. The levers are the ones that raise the signal —
  **logical-row weighting**, gradient accumulation / larger effective batches — or an
  optimiser that does not normalise magnitude away. Not another layer weighting.
- **Correction**: first-to-clear is NOT cheaper at inference in this codebase. The
  forward pass always runs all 90 layers and the commit rule is a post-hoc readout:
  measured 177 vs 183 s/test. The 30–70× figure applies only to a deployed decoder
  with an early exit.
- CER effect: −36.5% / −25.6% / −25.1% under design A / softmin / ramp. Coset still
  +50–73% worse with CER (963–996 vs 557–657). The OSD inversion stands.
- 0 of 12 points rolled back. Epoch-mean losses rise at seed 1 in epochs 4–5 for
  *every* arm because seed 1's batch sequence draws more hard batches there (35 vs
  22 informative) — not divergence. The batch sequence is shared across arms at a
  given seed, which is also what makes same-seed cross-arm weight differences a
  clean probe.
- **The per-layer profile is identical under all three losses, and identical at
  epoch 1 and epoch 5.** Median over informative batches (CER arms, layers 6–90):
  ~1.1 at layer 6, 0.8 at 10, 0.65 at 15, 0.35 at 25, then **flat at 0.30 from layer
  40 to 90**. Monotone on 52–62% of batches, last layer above the minimum on 16–33%
  of informative batches, mean rise 0.1–0.3 — same numbers for design A, softmin and
  ramp, before and after training. **No loss reshaped the trajectory.**
- **The 0.30 plateau is six violated checks per batch** (residue summed over rows,
  divided by 20 samples): BP converging to a *stable* fixed point with the syndrome
  unsatisfied — median syndrome weight 6.0 / 6.5 / 6.0 under the three losses, p25–p75
  of 4–10. Flat from layer 40 because a stable fixed point does not move. The typical
  informative batch therefore has **no rise** — its last layer IS its minimum — so the
  ramp's mechanism (penalise the rise) had nothing to act on. The rise is a 16–33%
  minority of informative batches (1–2% of all), and the ramp did not tame it there
  either (p90 at layer 90: 1.07 ramp vs 0.83 design A, within noise).
- So the shape of the per-layer loss was never the problem; the LEVEL of the plateau
  is. Lowering it means teaching the decoder to leave stuck fixed points of syndrome
  weight ~6, and 2,500 noise-directed updates do not. Same conclusion as the weight
  analysis, from the loss side.

### The residue itself: `base_loss = "sin_residue" | "smooth_loss"` (added 2026-10-08)

**Why the stuck checks are stuck — the per-check residue has zero gradient there.**
The base loss sums a residue `g(x)` over the rows of `[H; L]`, with
`x = Σ_k H⊥_ik (e_k + σ(μ_k))` the real-valued syndrome bit. Both candidate `g`
are 0 at even `x` and 1 at odd `x`; they differ in where their *cusp* is, and
that is everything:

| | at a VIOLATED check (x ≈ 1) | at a SATISFIED check (x ≈ 0) |
|---|---|---|
| `sin_residue` = \|sin(πx/2)\| | smooth, **g′ = (π/2)cos(π/2) = 0** | cusp, slope π/2 |
| `smooth_loss` = x² / (2−x)² | cusp, **g′ = ±2** toward the nearest even integer | smooth, g′ = 2x ≈ 0 |

So the sine spends its gradient on the already-satisfied checks (pushing correct
bits to saturate harder, which shrinks σ′(μ) and entrenches the stuck ones) and has
nothing left for the violated ones — exactly the plateau at six violated checks.
Quantified at a stuck check with six bits saturated to |μ|, `dL/dμ = g′(x)·σ′(μ)`:

| \|μ\| | σ′(μ) | sine dL/dμ | smooth dL/dμ | ratio |
|---|---|---|---|---|
| 5 | 6.7e-3 | 6.6e-4 | 1.3e-2 | 19× |
| 8 | 3.4e-4 | **1.7e-6** | 6.7e-4 | 400× |
| 10 | 4.5e-5 | **3.1e-8** | 9.1e-5 | 3,000× |
| 12 | 6.1e-6 | 5.6e-10 | 1.2e-5 | 22,000× |

Float32 noise on a 0.3 loss is ~2e-8, so the sine is at or below noise for any
|μ| > 8 — in a Float32 finite difference at |μ| = 8 it is *exactly* zero — while the
quadratic stays above noise until |μ| ≈ 12. Adam normalises magnitude, so a
consistent sign above noise is precisely what the sine failed to supply. The
quadratic was already in `loss.jl` with a comment saying all this ("subgradient ±2
at the odd integers, so descent is defined everywhere including at the maxima")
and was commented out in favour of the sine; it is now a switch.

- `base_loss = "sin_residue"` is the default and reproduces every earlier run;
  `"smooth_loss"` selects the quadratic. `BASE_LOSS_SIN_RESIDUE = 0`,
  `BASE_LOSS_SMOOTH = 1`, `base_loss_code` / `base_loss_name`.
- Threaded as one more positional argument after `loss_layer_ramp_sharpness` in
  `compute_smooth_loss_from_llrs` → `base_loss_per_layer` → `compute_loss` (all
  with defaults) and in `get_loss_value` / `get_individual_loss_values` (no
  default; `Enzyme.Const`). **Every direct `get_loss_value` call now lists five
  `Const`s before `base`**: temperature, warmup, selection, sharpness, base_loss.
- Recorded in the results CSV as `base_loss`. Sweepable as an optimizer-arm
  override `base_loss=<name>`; only the non-default is tagged, `_blsmooth`, so
  historical filenames are untouched.
- `run_local.sh --design baseloss` is the A/B as specified: {softmin, ramp} ×
  {sin_residue, smooth_loss} × {no-CER, CER} × 2 seeds = 16 trainings / 64 tests,
  **both losses committing at the last layer** so the comparison is between losses
  (`--commit first` changes that; `--base-loss sin|smooth` restricts the residue).
  Arm tags `_optsm<sin|smooth><commit>` and `_optrp<...>`, so nothing collides with
  the `pair` / `all` presets in the same directory.
- Tests (`tests/test_loss.jl`): values at integers and between, the default
  reproduces, both residues on the standard posteriors, and **the gradient claim as
  a test** — on a stuck check at |μ| = 8, `|∂L/∂μ|` for the bit in violated checks
  is > 100× larger under the quadratic, and for the bit in the satisfied check it
  is > 100× larger under the sine. Analytic ratio ≈ 1,200 both ways. Plus Enzyme
  through the full training loss under `smooth_loss` for softmin / last / ramp.
- `--settings` edge case fixed: re-running within the same second on the file the
  generator itself wrote made `cp` refuse (same path) and `set -e` exit; it is now
  used in place.

#### Measured 2026-10-09: `--design baseloss`, 50 layers, commit = LAST, 8 arms × 2 seeds

Pooled over 4 test sets × 2 seeds, 800k per arm. 0 of 16 rolled back.

| arm | sin_residue | smooth_loss | smooth vs sin |
|---|---|---|---|
| CER, softmin | 2076 (885 \| 1191) | **2788** (885 \| 1903) | **+34.3% (10.2σ)** |
| CER, ramp | 2111 (872 \| 1239) | 2247 (883 \| 1364) | +6.4% (2.1σ) |
| no-CER, softmin | 2713 (534 \| 2179) | 2597 (515 \| 2082) | −4.3% (1.6σ) |
| no-CER, ramp | 2685 (506 \| 2179) | **2444** (502 \| 1942) | **−9.0% (3.4σ)** |

- **`smooth_loss` is not the lever.** It helps no-CER modestly and hurts CER, badly
  under softmin. The effect is **entirely convergence failures** — coset is within
  ±4% in all four comparisons (885 vs 885, 883 vs 872, 515 vs 534, 502 vs 506).
- **The per-layer profiles, read correctly.** The two residues take different
  VALUES between integers (the quadratic is smaller everywhere except at 0 and 1),
  so a sine curve and a quadratic curve are only comparable where they are FLAT at a
  multiple of 0.05 — a converged state with an integer number of violated checks.
  An earlier draft of this note compared a non-flat sine curve to a flat quadratic
  one ("0.97 → 0.40"); that was not a like-for-like reading. What the profiles
  (median over informative batches, epoch 5, layers 6–50) actually show:
  - **CER**: every arm settles to a flat 0.30–0.35 by layer ~40 under either residue
    — six to seven violated checks, bits saturated. The quadratic changed nothing
    here, and test failures rose.
  - **no-CER, sine**: no plateau at all by layer 50 — the curve hovers at 1.0–1.2
    for the whole trajectory and only starts falling after layer 45 (0.72 / 0.86 at
    layer 50, not multiples of 0.05). The decoder is still oscillating when the
    layers run out, which is the 50-layer convergence penalty seen directly.
  - **no-CER, quadratic**: settles to a flat 0.35 = seven violated checks by layer
    ~48. So the quadratic made the no-CER decoder *converge* within 50 layers, to a
    stuck state — and test failures fell modestly. Settling at 7 beats oscillating.
  - Epoch 1 → 5 moved the CER sine plateau by about one check (0.43 → 0.35) and the
    quadratic plateaus not at all; training reshapes none of these curves.
- Hypothesis for the CER/no-CER split: the enriched check node plus rescaled priors
  leave the stuck bits harder-saturated, where even the quadratic's `2·σ′(μ)` is dead
  (9e-5 at |μ| = 10, 1e-5 at 12). Nothing caps the posterior μ — `prior_llr_clip`
  only touches the initial LLRs — so a saturation cap on the readout before σ(μ) in
  the loss would be the mechanistic test. Not built.
- The collapsed arm trained normally: 0 NaN skips, ordinary epoch trajectory,
  ordinary weight movement, α 0.41–0.42. Both seeds are worse (671 / 955 vs 576 / 593
  on the training distribution); it is systematic, not an accident.
- **The robust surprise is the LOSS, not the residue.** At the same last-layer commit
  and the same sine residue, 50 layers:

  | loss | CER | no-CER |
  |---|---|---|
  | last layer only (design A, 2026-10-08) | 2703 | 3645 |
  | softmin + last | **2076 (−23%)** | **2713 (−26%)** |
  | ramp + last | 2111 (−22%) | 2685 (−26%) |

  Scoring more than one layer beats scoring the commit layer alone by a quarter, and
  which multi-layer weighting (softmin vs ramp: +1.7% / −1.0%, < 1σ) is irrelevant.
  This is the resolution of "losses are interchangeable": **at 90 layers the commit
  layer is converged for nearly every sample, so the loss choice does not matter; at
  50 it is not, and a single-layer loss trains on an unconverged, noisy target.**
- softmin + last ≈ softmin + first for CER at 50 layers (2076 vs 2051); worse for
  no-CER (2713 vs 2421, +12%). First-to-clear is still the readout to use.

## Which samples to train on: `training_samples` / `failure_weight_boundary` (`src/sample_selection.jl`, added 2026-10-09)

- **The lever this targets is the gradient signal, not the loss shape.** ~94% of
  batches carry no failure, and a uniform draw from the file lands on a failing
  sample 0.29% of the time (measured 2026-10-09, CER, 50 layers: weights 0–2
  never fail and are 87.5% of the pool; weight 3 fails 0.3%; 4–6 fail 1.7–2.7%;
  7+ fail 7–22%). `failure_weight_boundary = λ` draws the training set so its
  error weights follow **Poisson(λ)** instead of the file's own distribution
  (mean weight 0.91): Poisson(5) puts 3.4% of draws on failures, 12× uniform.
- `training_samples = N` is the size of the set the batches are drawn from
  (0 = the whole file, every earlier run); `failure_weight_boundary = λ` is the
  Poisson centre (0 = the file's own weights). λ > 0 needs N > 0 — a weighted
  draw has no natural size — and both the generator and Julia refuse otherwise.
- **How the draw works** (`filter_training_samples`): bin the pool's columns by
  Hamming weight, draw a weight from Pois(λ) *renormalised over the weights the
  pool contains*, then a uniform column from that bin, **with replacement**, N
  times. So the set always has exactly N entries (no bin can be empty by
  construction), the samples are still the error model's own with their CER
  correlations intact, and it equals rejection with acceptance ∝ Pois(λ; w) /
  p_pool(w) without the rejections. The naive "accept with Pois(λ; w)" gives
  Pois × pool and lands at mean weight 2.9 for λ = 5.
- **The price of exactly N is repeats where the pool is thin.** On the 200k cut
  at λ = 5, N = 10⁴: 8,276 distinct of 10,000 — weights 8 and 9 are
  oversubscribed (653 draws from 536 columns, 363 from 326) and weights 3–5 (~6,300
  columns each) repeat lightly. On the full 1e6 file no bin is oversubscribed
  (~9,600 distinct). A without-replacement draw would guarantee distinctness but
  then either falls short of N or shifts mass away from exactly the heavy weights
  the Poisson is there to supply; repeats were the smaller distortion, and the
  batch draw (`rand(1:N, batch_size)`, 50,000 draws from N = 10⁴) revisits every
  sample ~5× anyway.
- **Where the randomness lives.** The draw is taken in `train_Nachmani_neuralbp`
  right after `readdlm`, from the global RNG, AFTER `apply_training_seed!` and
  after the initial weights are drawn — so at one seed every arm sees the same
  initial weights, the same set, and (the RNG state being identical after the
  draw) the same batch sequence. Nothing is stored: a set is a function of
  (file, λ, N, seed). Under `--isdebug true` (the generator always passes it) the
  set is written to `logs/training_selection_<tag>.csv` (pool column, weight) and
  one summary line goes to stdout (draws, distinct, mean weight vs pool, share at
  weight ≥ 4). Pre-filtering to a stored file is the alternative — it would fix
  one set across seeds and let each process read 10⁴ columns instead of 2e5–1e6 —
  but it changes the dataset key and so every filename; not built.
- Filename tags `_ts<N>` and `_fwb<λ>` (`.` → `p`, trailing `.0` dropped), only
  when non-zero, so every earlier filename is reproduced. Both are sweepable
  optimizer-arm overrides and both are recorded in the results CSV.
- `run_local.sh --design sampling --training-samples 10000 --failure-weight-boundary 5`
  runs softmin + first on (a) a uniform set of N and (b) a Poisson(λ) set of N,
  × {no-CER, CER} × 2 seeds (8 trainings / 32 tests). The whole-file arm is the
  deterministic `histpair` point of `--design all` at the same layer count. With
  any other `--design` the two flags apply to every arm.
- `hyperparams_baseline.toml` was cleaned on 2026-10-09: the removed loss terms'
  keys (`correlation_importance`, `correlation_weight`, `llr_certainty_importance`,
  `sparsity_importance`) are gone, and `loss_layer_temperature` is marked
  softmin-only.
- `run_local.sh --design icscale --layers 50 --jobs 8`: ramp + last + smooth_loss
  with `initial_conditions_scale` over `--scales` (0.1,0.3), and everything else —
  warmup, k, training set — from the baseline TOML. The ramp-at-warmup-0 note
  prints once, not per point.

## Run tags (changed 2026-10-09)

- **The run tag is `_<arm>_<optimizer tag>` and nothing else** — `_cer_init_0p1`,
  `_nocer_histpair` — or `_<arm>` alone for a lone, untagged `base` arm. Every
  `_hp<arm>`, `_cnenr<α>`, `_sch…`, `_opt…`, `_lsl…_clr…`, `_blsmooth`, `_fwb…`,
  `_ts…` tag mentioned above describes filenames written BEFORE this date; the
  settings a sweep holds fixed now live in the TOML and the results CSV, not the
  name. Consequently two points that would share a filename (two check-node
  arms in one sweep, say) are refused by the generator rather than merged.
- Not part of the tag, and unchanged: the code's own `_no_cer` infix on a no-CER
  run (so `..._200000_no_cer_nocer_init_0p1_seed_1.json`), and the weights
  filename's `nlayers_<N>_epochs_<n>_trained_using_<training file>` fields.
- The sweep TOML keeps its `hyperparams_hp_` prefix (`hyperparams_hp_cer_init_0p1_<key>_seed1.toml`):
  cleanup.py and run_local find sweep TOMLs by it. Weights and results names carry
  no `_hp`, so every glob that used to match on `_hp` now matches on `_seed_`
  (run_local's model gate and results clearing, the generator's collect and
  stage-in steps).

## Running locally on Apple Silicon (`misc/run_local.sh`, added 2026-10-07)

- **Metal does nothing for training.** Enzyme cannot differentiate through
  device-array allocation, so the training forward pass is always the CPU one
  (`src/train.jl:20`). Only testing uses the GPU. A local run is therefore CPU
  training (`USE_GPU=0`) plus Metal testing (`USE_GPU=1`).
- **Testing is the cost, not training.** A test point is ~129 s on an A100; an
  M-series GPU has roughly 1/15th the memory bandwidth for a workload bound by it.
  The lever is the sample count, and there is no CLI flag for it — `--n_samples` is
  **training-only**. Cut the file instead: the data is 72 rows × n_samples
  space-separated columns, so a sample is a COLUMN and `cut -d' ' -f1-N` takes the
  first N. At 200k the failure counts land near 480–630 per arm (Poisson ~4%),
  which still separates a 20% effect. Training is unaffected: `online_training`
  draws `n_epochs × n_gradient_updates_per_epoch × batch_size` = 50,000 samples, so
  a 200k pool is already 4× oversubscribed.
- **4 concurrent trainings, not 8**, on a 16 GB machine. Each process holds the
  sample matrix (n_samples × 72 × 8 bytes) plus several times that in `readdlm`
  intermediates — ~2 GB each at 1e6 samples. An M-series chip is also 4 performance
  + 4 efficiency cores and the E-cores run this ~3× slower, so 8 at once finishes
  *later* than two waves of 4 as well as risking swap.
- The cut datasets get their own `_<N>` key, so every weights file and results CSV
  is named differently from the cluster run and **nothing collides**.
- **A no-CER arm puts `_no_cer` BETWEEN the dataset key and the run tag**:

  ```
  ..._trained_using_train_<key>_cer_init_0p1_seed_1.json
  ..._trained_using_train_<key>_no_cer_nocer_init_0p1_seed_1.json
  ```

  So any glob that joins the key straight onto the tag silently drops every no-CER
  arm. `*<key>_hp*.json` found 4 of 8 models after a clean 8-point run and refused
  to test; a wildcard between key and tag finds all 8. The same infix is why an
  arm-attribution regex of the form `_s_1_(.+?)_seed_` must allow for it.
- `sweep_hyperparams.sh --settings <file> --no-edit` drives the generator from an
  existing settings TOML, which is how `run_local.sh` narrows the sweep without
  editing the cluster defaults. All arm/tag/TOML logic stays in one place.
- The no-CER arm is emitted **independently of `check_node_arms`** (it has no
  couplings to enrich), so one enriched arm + `include_nocer` + 2 optimizer arms
  gives 4 arms, not 2.
- Two steps that are easy to miss when running the command lists by hand, and that
  `run_local.sh` and the SLURM test job both do: the generator writes
  `retrain = true` (training needs it) so **testing retrains from scratch unless it
  is flipped to false**, and `neural_bp_experiments.jl` **skips testing when the
  results CSV already exists** and reports the stale numbers as if fresh.
- **Testing locally runs as a plain sequential loop, not GNU parallel.**
  `parallel --jobs 1` also runs one at a time but captures stdout into its
  `--results` tree, so you watch nothing for hours. The loop prints
  `[n/32] <arm> -> <test key>`, then the point's own output, then its duration and a
  mean-so-far ETA. The first point is the number worth having: nothing extrapolated
  from an A100 timing tells you what this machine costs.
  - `--quiet` only gates the **training** progress bars (they all live in
    `train.jl`); the test path prints nothing at all until "Test sample results
    saved to file", so the per-point timing is the only progress signal there is.
  - The command list is read on **fd 3** (`done 3< "$LIST"`), not stdin. A point
    that reads stdin would otherwise swallow the rest of the list and the loop
    would stop after one.
  - Only training needs GNU parallel, so `--test-only` works without it installed.

- `python3 misc/cleanup.py --workdir <codename> [--dry-run|--yes]` resets a run dir
  to its inputs. Rules are one table (`RULES`): logs/ cluster/ results/ emptied;
  models/ loses `*.json` and sweep TOMLs (`hyperparams_hp_*`, `hyperparams_xf_*`),
  keeps every other `*.toml` as the base config. Refuses `/`, `$HOME`, the repo,
  `data/` itself, anything outside `data/`, and any run whose models/ would be left
  with no `.toml`. Replaces `clean_up_data.sh`.

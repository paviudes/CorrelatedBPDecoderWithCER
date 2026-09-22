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
- `python3 misc/cleanup.py --workdir <codename> [--dry-run|--yes]` resets a run dir
  to its inputs. Rules are one table (`RULES`): logs/ cluster/ results/ emptied;
  models/ loses `*.json` and sweep TOMLs (`hyperparams_hp_*`, `hyperparams_xf_*`),
  keeps every other `*.toml` as the base config. Refuses `/`, `$HOME`, the repo,
  `data/` itself, anything outside `data/`, and any run whose models/ would be left
  with no `.toml`. Replaces `clean_up_data.sh`.

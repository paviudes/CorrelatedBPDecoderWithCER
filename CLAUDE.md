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

- **Training runs on GPU nodes (`sweep_hyperparams.sh`, 2026-10-15).** Both jobs are
  GPU array jobs on `account_gpu`, one card per task (`gpu_type`, `n_gpus_per_node`,
  `mem_per_gpu`), chained with afterok: training (`train_wall_time = "1:00:00"`),
  then testing (`test_wall_time = "4:00:00"`, unchanged).
  - Training uses the hand-written backward pass with USE_GPU=1. The base TOML
    **must** say `manual_backprop = true` (the generator refuses otherwise: Enzyme
    cannot use the GPU, and ~6600 s/point on the allocation's few cores would blow
    the 1 h wall), and every point's TOML states it. `check_node_arms` takes `tanh`,
    `enriched:<α>:<fixed|learn>` and `bethe:<α>:<fixed|learn>` (since 2026-10-19; the
    hand-written gradient covers the enriched node). A non-tanh arm's run tag is
    `_<arm>_<enr|bethe>_<opt>`, so it never overwrites the tanh arm's files; a bethe
    arm needs `correlated_weights/bethe_table_<key>.txt` for the training key and
    every test key (the preflight lists the missing ones with the command).
  - `train_jobs = 4` trainings share the task's card (GNU parallel, the test job's
    card-assignment line). Safe for TRAINING — a CUDA context plus batches of 20,
    well under 1 GB of VRAM each — unlike testing, which sizes its batch to the card
    and stays at one process per card. Host RAM is the limit: the generator refuses
    `train_jobs × heap_size_hint > n_gpus_per_node × mem_per_gpu` (4 × 4G ≤ 32G),
    `train_cpus < train_jobs`, and `train_cpus > test_cpus × n_gpus_per_node` (an old
    CPU-node value like 54). `account_cpu`, `train_mem_per_cpu`, `cpu_node_memory_mb`
    are ignored, with a note.
  - **Measured with nvtop (2026-10-19, enriched_vs_nocer task 0, ~21 min in):** two
    trainings on the card, each ~700 MiB of VRAM, one core at 100%, 3.7–4.7 GB host
    RSS (above the 4G heap hint); the card 16–82% busy, ~40% mean. So a training uses
    ~20% of an A100: it is HOST-bound at a batch of 20 (tiny kernels, plus the per-step
    rebuild and upload of the dense per-layer weights — PCIe RX spikes to 1.1 GiB/s),
    and ~5 trainings saturate one card. Each task holds ONE A100 + 12 cores + 32 GB (a
    quarter of a Narval GPU node); Slurm puts the tasks on whatever nodes have a free
    card, and the trainings never talk to each other, so spreading over nodes costs
    nothing. A task mixing fast (tanh) and slow (enriched) points holds its card at
    half occupancy once the fast ones finish. (The two missing processes had
    finished, not crashed: the afterok test job started, so all 20 points exited 0.
    The whole training job took roughly 25 min.)
  - **Measured with nvtop, testing (same sweep, one test task, 4 min recorded): the
    card is busy ~24% of the time.** Each test is a fresh Julia process, ~2 min:
    ~55–65 s with no CUDA context at all (Julia start, package load, `readdlm` of the
    144 MB text test file, its syndromes — all CPU); for an enriched model ~25 s more
    with the context up and the card idle (GPU kernel compilation); then the decode,
    in chunks of 65,536 samples (enriched) or 131,072 (tanh) — what
    `resolve_prediction_batch_size` gives at GPU_MEMORY=34816M, so 16 or 8 chunks
    per 10⁶ — each chunk a ~2 s burst at 100% followed by ~2 s at 0% while the host
    runs `count_syndrome_satisfactions` on it (`--diagnose true`: 90 layers of XOR,
    SGEMM and bookkeeping): 40–60 s at ~45%; then exit (tests run without
    `--isdebug`, so the file is read once). The 38 of 40 GB nvtop shows is CUDA.jl's
    caching pool, not the working set (~5.5 GB per enriched chunk), which is also
    why two tests cannot share a card. Levers, by payoff: score on the device inside
    the layer loop (removes the ~2 s gaps and the 0.4–0.85 GB copy per chunk); one
    Julia process per test task looping over its points (startup, package load and
    kernel compilation once, each test set parsed once); a binary copy of each test
    set.
  - **The network is fixed at 90 layers** in the generator (`NLAYERS=90`); a settings
    file naming another `n_hidden_layers` is refused. `run_local.sh` lost `--layers`.
  - `seconds_per_train_point = 720` is the MEASURED wall time of one wave of
    `train_jobs = 4` points sharing an A100 (2026-10-15, tanh: ~12 min per task of 4);
    the test job used 21–27 min of its 4 h. Each point prints `Training: N gradients
    (...) in X s ...` to its stdout under `cluster/logs/hp_<ts>_train_task<t>/`. The
    enriched node does ~10× the check-node arithmetic: a wave of 2 enriched + 2 tanh
    points took ~25 min (measured 2026-10-19), so use 1500 for any wave holding
    enriched points.
  - **Walltimes actually used (2026-10-19 sweep):** training job ~31 min of its 3 h (6 min
    CUDA rebuild + one 25 min wave), test job ~20–27 min of its 4 h (10 tests per task,
    ~2.3 min each with Julia startup). Usage is charged for time used, and ≤ 3 h is the
    shortest Alliance walltime class (most nodes), so the oversizing cost nothing but
    backfill chances (and up to the full wall if a job hangs). One wave per task fits a
    1 h wall for both jobs; the preflight scales it when tasks get more waves.
  - **Trillium (SciNet, assessed 2026-10-19; decided the same day to stay on Narval,
    GPU for training and testing).** GPU subcluster: 63 nodes ×
    4 H100 SXM 80 GB (252 GPUs, 96 Zen 4 cores per node, scheduled by whole GPU); CPU
    subcluster: 1224 nodes × 192 Zen 5 cores, 749 GB, scheduled by WHOLE NODE; jobs
    15 min–24 h. H100s do not pay here: training is host-bound (~20% of an A100) and
    ~3/4 of a test is CPU work, so expect ~10–20% per point, at 12.2 RGU per H100 vs 4.0
    per A100-40GB. The CPU nodes are the possible win for a BIG sweep — not per-point
    speed but ~100+ concurrent CPU trainings (USE_GPU=0, manual backprop) on one node —
    and need a benchmark first: CPU time per 90-layer point is unmeasured, the enriched
    node's CPU kernel is the heavy part, and one process per core needs BLAS threads
    pinned to 1. Site differences the scripts would need: $HOME and $PROJECT are
    read-only from jobs (repo, data and depot on $SCRATCH), no local disk
    ($SLURM_TMPDIR is a RAM disk, so staging the run directory costs RAM), no internet
    on compute nodes.
  - Verified 2026-10-15 by executing both generated scripts with stub
    `julia`/`parallel`/`module` and fake `SLURM_*`: all 5 train and 8 test tasks exit
    0 (lock owner + waiting siblings, 20 models and 80 results staged out); a failing
    point exits 1; the TERM handler stages out the finished point and exits 1; a CUDA
    gate failing twice exits 1. Re-run after the gate became `CUDA.functional(true)`:
    same results, with the reason in the .err file. Not yet run on Narval.
- **Warm the Julia depot on a LOGIN node** before `sbatch`: `bash misc/precompile_depot.sh`.
  Compute nodes have no internet, and array tasks share one Lustre depot — in-job
  `Pkg.instantiate()`/`Pkg.precompile()` can stall for the whole walltime. Since
  2026-10-15 it also refuses (exit 1, printing the commands) a manifest resolved by
  another Julia minor version, and prints the CUDA toolkit the login node selected,
  refusing `none`.
  - **What a cold depot looks like (2026-10-22, `hp_2026-10-22_16-25-01`, the
    enriched_ramp_commit sweep): nothing was trained.** The lock owner's .out ends at
    "rebuilding CUDA_Runtime_jll" (no "done" an hour later); the other 4 tasks print
    "no rebuild sentinel after 1800s; proceeding to the gate anyway", pass the CUDA
    gate (it loads CUDA, not the package) and start their points, and every point's
    stderr in `cluster/logs/` is "signal 15: Terminated in expression starting at
    neural_bp_experiments.jl:4" — `using CorrelatedBPDecoderWithCER` — with an idle
    event-loop backtrace (epoll_pwait, uv_run) and an empty stdout: 20–29 min waiting
    on its precompilation. src/ had changed (2f31ddd, the α transfer) since the depot
    was last compiled; in the job the owner compiles the package twice (instantiate's
    auto-precompile, then again after the CUDA_Runtime_jll rebuild) while 8 points and
    4 gates wait on the same cache files. The array was then cancelled at 19:05 EDT,
    all 5 tasks within 30 s ("CANCELLED … DUE to SIGNAL Terminated"): a job cancel,
    not the wall (which says DUE TO TIME LIMIT and would have hit task 3 nine minutes
    later than the rest); the TERM handler staged out 0 points and afterok dropped
    the test job. A warm depot makes this step ~6 min (2026-10-19).
  - Since then `sweep_hyperparams.sh` **warns at generation time** when the depot the
    jobs will use (first entry of JULIA_DEPOT_PATH, else $SCRATCH/.julia; nothing to
    check on a laptop) has no cache of CorrelatedBPDecoderWithCER for the job's Julia
    (`julia_module` → v1.12), or its newest `.ji` is older than the newest of
    src/*.jl, Project.toml and the manifest that Julia picks — modification times, so
    a warning, not a refusal; and the owner's rebuild steps and the CUDA gate are
    time-stamped in the .out. Checked with a fake depot (stale, fresh, missing, no
    variables, $SCRATCH only) and stub-executed jobs (5/5 train, 8/8 test exit 0).
- **One manifest per Julia minor version (2026-10-15).** The laptop runs 1.13.1 and
  Narval `julia/1.12.5` (Narval has no 1.13 module, checked 2026-10-15), and a
  manifest only works with the minor version that resolved it. Julia X.Y reads `Manifest-vX.Y.toml` before `Manifest.toml` (Julia
  ≥ 1.10.8), so `Manifest.toml` is the laptop's and `Manifest-v1.12.toml` the
  cluster's. Commit b207a50 re-resolved `Manifest.toml` under 1.13.1 (Enzyme
  0.13.138 → 0.13.215, CUDA 5.11.2 → 6.4.1, Metal 1.9.3 → 1.11.1) with no 1.12
  manifest beside it, and `precompile_depot.sh` on Narval failed while precompiling
  Metal (through LLVMDowngrader_jll 0.11): `Package Zstd_jll [3161d3a3-…] is required
  but does not seem to be installed`. 1.13 made Zstd_jll a standard library, so a 1.13
  manifest records it with no `git-tree-sha1`, and 1.12, where it is an ordinary
  package, has nothing to install. Create the cluster's manifest with the cluster's
  Julia — `cp Manifest.toml Manifest-v1.12.toml`, then
  `julia --project=. -e 'using Pkg; Pkg.resolve()'` (resolve keeps every version it
  can) — and commit it. Redo both after any `Pkg.add`/`Pkg.update` on the laptop, so
  the cluster follows the laptop's versions.
- **CUDA.jl 6 (which came with that manifest) removed `CUDA.available_memory()`.**
  `forward_pass_gpu` caught the `UndefVarError` and would have sized every test chunk
  for 1 GB without a word; it now calls `CUDA.free_memory()` (there since 5.4) and
  warns if the query fails. The other CUDA names the code uses (`functional`,
  `CuArray`, `unsafe_free!`; `name`, `device`, `runtime_version` in
  `submission/slurm.jl`) all still resolve in 6.4.1.
- **`local_toolkit = true` in `LocalPreferences.toml` is not a key CUDA_Runtime_jll
  reads** (checked in the source of 0.21.0+1 and 0.25.0+2: it reads `local` and
  `version`; `CUDA.set_runtime_version!(local_toolkit=true)` writes `local = "true"`).
  So the cluster has always used the CUDA runtime ARTIFACT, chosen from the driver
  Julia can see and downloaded lazily on first load — on the login node, online,
  during `precompile_depot.sh`. An offline GPU node can only use an artifact already
  in the depot, and 0.25's artifacts are new downloads (different tree hashes from
  0.21's). If the login node selects `none`, pin the toolkit:
  `julia --project=. -e 'using CUDA; CUDA.set_runtime_version!(v"12.9")'` (writes
  `version` for CUDA_Runtime_jll and CUDA_Compiler_jll; CUDA.jl rejects a runtime
  only when its MAJOR version exceeds the driver's). `local = "true"` would switch
  to the module's toolkit instead — untested here. The jobs' CUDA gate now runs
  `CUDA.functional(true)`, which throws with the reason rather than returning false.
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
  survivors each did 35 tests in 75 min, 37% of a 4 h budget). Both jobs now include
  ONE generated preamble (modules, depot, rebuild, gate) that takes an `mkdir` lock:
  one task rebuilds and touches a `.done` sentinel, the rest poll for it (1800 s). The
  lock is per generation and shared by the two jobs, so the training job rebuilds and
  the test job finds the sentinel and goes straight to the gate. The rebuild is still
  needed because the LOGIN-node precompile has no driver. `CUDA.functional()` runs
  once, and a second time only after a failure (60 s later).
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
  (exits 1), which covers what the dependency cannot see: training never submitted,
  a stage-out that did not land, a lone test resubmission, or a `cleanup.py` between
  the two. **Since 2026-10-22 it checks by NAME**: the generator writes the weights
  file of every point to `cluster/hp_sweep_models_<ts>.txt` and the test job lists any
  that is not staged in. It used to count `*.json`, which a run directory reused by
  several sweeps defeats (72q_BB_cycles_1_trainable_alpha held 20 old models, so the
  count passed with all 10 of a new sweep's missing), and a missing weights file does
  not fail on its own: the test then TRAINS a model under `retrain = false`. The
  generator also warns, listing them, when models of the sweep already exist (they
  will be retrained and replaced, with their results).
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
- α is stored as its logit, `coupling_logit` (a length-1 weight vector on
  `NachmaniNeuralBP`; α = 1/(1 + e^(−θ))), initialised from `coupling_scale_init`
  (1 = Bayesian), learned unless
  `coupling_scale_learnable = false`, saved in the weights JSON and the results CSV.
- **An enriched run needs a non-empty `run_tag`** (the scripts refuse otherwise): no
  filename carries a check-node tag, so it would load/overwrite the tanh run's files.
- Same kernel for classical BP: `standard_bp_experiments.jl` with `check_node = "enriched"`
  and a fixed `coupling_scale_init` is the training-free α = 0 vs α = 1 comparison.
- Tests: `tests/test_soft_constraints.jl` (tables, α = 0 ≡ tanh, brute-force posterior,
  Enzyme gradients, CPU ≡ GPU, tanh path unchanged, weights-file round trip, refusal
  of the removed layer schedule).
- **Known limit of the GPU kernel** (`apply_enriched_checks_gpu`, testing only; noted
  2026-10-13, not fixed): it stabilises with ONE maximum per (check, sample), and
  each target's two class sums still contain the target's own factor. A huge
  incoming LLR on the target (|m_v2c| > ~87) can push one sum below Float32's range,
  and the message then saturates at ±16.6 even when the true message is small
  (Float32 emulation, incoming (−110, 1, 1): GPU −16.6, CPU 0.30; the CPU kernel
  stabilises each class separately). Exact while every |m_v2c| < ~70; at unit
  weights a BB v2c message is at most 2 × 16.6 + |l|, so it needs weights above ~2.

### The pairwise factor is mis-specified on this data (measured 2026-10-15)

Checked on `p_0.0015_sig_0.0015_s_1`: H_Z, the CER file, and the 1e6-sample training file.

- The 540 CER pairs are exactly the 15 pairs inside each of the 36 checks (each pair
  shares one check, none dropped). 287 are strong (|J| > 1, max 4.43); 225 are noise
  (|J| < 0.2). Inside a check the strong pairs are triangle-free and most qubits have
  3 strong partners; across its 3 checks a qubit has 6–9. The file's J and rates
  equal the training data's MARGINAL pair log-odds ratios and flip rates exactly.
- The check's local model built from them (α = 1, raw rates) against the true
  distribution of the check's 6 bits: 0 flips 80.6% vs 94.1%; 3 flips 1.15% vs 0.049%
  (26× on triples where one qubit has two strong partners); 4–6 flips 9.7% vs 0.005%.
  Marginal J's compound in the product (e^{J_uv + J_vw}); they are not Ising couplings.
- First-layer messages (incoming = priors) against the true joint's: satisfied check
  true +0.76 (median), tanh +2.75, α = 0.5 +1.71, α = 0.75 +0.79, α = 1 **−1.47**
  (wrong sign); violated check true −2.71, tanh −2.75, α = 1 −4.12. So tanh overstates
  a satisfied check by ~2 nats (correlated pairs flip without violating it), and α = 1
  overcorrects past zero. One α scales every coupling, so it can fix the average, not
  the shape.
- Also: 61% (median) of a qubit's flips come with a strong partner, so the marginal-rate
  fields double count the pair events that e^J adds again; the data has 3-qubit
  mechanisms (the CER "block" patterns: (3,23,69) all flip ~3,000 times per 1e6,
  (2,24,54) ~4,000), whose pairs straddle two checks; and 185 of the 472 strongly
  correlated pairs in the data share no check, so no check factor can carry them
  ((3,23) has J = 3.8 and is absent from the file).
- The general replacement within a check is the true local joint over the product of
  marginals, f_c(x) = 1[parity] · P_c(x) / Π_u p_u(x_u): one log-weight per
  configuration, now the `"bethe"` table (below), exact at the first layer by construction.
  Cross-check correlations need mechanism variables (DEM-style columns) instead.
- **Per-coupling α does not recover it** (each check's factor fitted by maximum
  likelihood to its true 6-bit joint; message errors RMS over the 216 slots, first layer):

  | factor | KL (nats) | msg err, satisfied | violated |
  |---|---|---|---|
  | α = 1 | 0.126 | 2.36 | 1.73 |
  | one α = 0.75 | 0.013 | 0.09 | 0.49 |
  | per-coupling α ∈ [0, 1] | 0.012 | 0.25 | 0.45 |
  | per-coupling, unbounded | 0.009 | 0.03 | 0.41 |
  | couplings + per-(check, qubit) field corrections | 0.0008 | 0.07 | 0.04 |

  The fitted per-coupling α's cluster at 0.59–0.76 (median 0.70, none at a bound), so
  they buy almost nothing over one α. The residual is the FIELDS: the single-qubit
  factors come from the marginal rates, which include the pair events, and α cannot
  touch them. The last row is the max-ent pairwise model (matches every single rate and
  pair table inside the check), computable from the CER file alone (p_i, p_k and J_ik
  fix each 2×2 table): a 21-parameter convex fit per check over 64 configurations, and
  it fits the existing per-configuration table with the field corrections folded in.
  Within a check, three-body terms are then negligible (KL 0.0008). Raw rates here;
  the runs rescale the priors, so 0.70 is not directly the α to use there.
- **Where the correlations come from** (`refs/72q_BB_code_data_gen.py`): a cascade on
  a ladder lattice, not on the checks. Each qubit flips independently with p; then
  each lattice neighbour of a flipped qubit flips with a per-pair q ~ N(q_mean, 0.2)
  (one step, no chain reactions; OR, not XOR). So the strong pairs are lattice
  neighbours and "siblings" (two neighbours of one seed), and a seed with two
  neighbours is the 3-qubit block pattern. Their overlap with check supports is
  incidental, which is why 185 of 472 strong pairs sit in no check. The exact factor
  graph for this model has pure-parity checks plus mechanism variables (seed z_i,
  induced y_{i→j}, both columns of H on the qubit they flip) tied by degree-2
  factors P(y_{i→j} = 1 | z_i) = q_{ij} z_i; a check factor can only ever see the
  within-check part. (The repo's `p_0.0015_sig_0.0015` filenames do not match this
  script's p/q naming, so the exact generating version is unconfirmed.)
### Enriched check node: the table-driven implementation (2026-10-19)

- **One node, two tables.** `ψ_c(x) = 1[⊕x = s_c] · exp(α T_c(x))` with
  `SoftCheckTables.log_weight_table` (was `pair_energy`) holding T_c and
  `table_kind` saying which: `enriched_table = "pairwise"` (default, T = Σ J x x,
  every earlier enriched run) or `"bethe"` (T = ln P_c − Σ ln p_u, read from
  `correlated_weights/bethe_table_<key>.txt` next to the CER file, the path
  `bethe_table_path` derives). `check_node` stays `"tanh" | "enriched"`; the table
  kind is a second TOML key, recorded in the weights JSON (`enriched_table`; loading a
  file trained with the other table warns) and the results CSV. α multiplies either
  table; α = 1 is the Bayesian value of both.
- **Forward** (`message_check_to_vertex_enriched!`, one check and one sample, on an
  `EnrichedKernelWorkspace`; `apply_enriched_checks!` loops it): the softplus terms
  of ln q_u cancel, so `LW(k) = α T_c(k) − b(k)·m`, `S_v^e` = LSE of LW over the class
  {parity = s_c, b_v = e}, stabilised by the class's own maximum, and
  `m_{c→v} = S_v^0 − S_v^1 − m_v`, capped. Same message as the old kernel, fewer
  operations (so enriched runs before this date are not bit-reproducible, only
  equal to Float32 round-off); Enzyme still differentiates it (the oracle in the
  tests). The dense GPU forward (`apply_enriched_checks_gpu`) is unchanged apart
  from the rename.
- **Adjoint** (`backward_message_check_to_vertex_enriched!`, `backward_enriched_checks!`;
  `backward_enriched_checks_gpu` with a `GPUEnrichedBackwardState` sized for the
  training batch, built and freed per gradient call): recompute LW and the class
  sums, then with π_v^e the class softmax and a_v = ∂L/∂m_{c→v} (0 where the
  forward's message sits on the cap, read off the tape's outgoing messages):
  ω(k) = Σ_v a_v [π_v^0(k) − π_v^1(k)], ∂L/∂m_u = −Σ_k ω(k) b_u(k) − a_u (exactly 0
  for u = v), ∂L/∂(αT)(k) = ω(k) (accumulated in `adjoint_log_weight_table`, not a
  parameter yet), ∂L/∂α = Σ_k ω(k) T_c(k), and dL/dθ = dL/dα · α(1−α) through the
  logistic link. The device version stabilises each class by its own maximum, so it
  has no "exact while |m| < ~70" limit. Prototyped against central finite
  differences in float64 and Float32 (outputs/proto/enriched_table_node.py).
- **Where it plugs in.** In both reverse sweeps the tanh chain runs on every row as
  before, and the enriched checks' v2c edges are then OVERWRITTEN with the table
  node's adjoint (CPU) or masked with `keep_mask` and added (GPU): those edges feed
  no other check, so the chain's share there is wrong and discarded whole. The GPU
  tape copies each layer's outgoing c2v messages under the enriched node (the kernel
  returns a buffer it rewrites). `manual_backprop = true` now runs with
  `check_node = "enriched"`, on the CPU and under USE_GPU=1.
- **Building a Bethe table**: `python3 misc/build_bethe_tables.py --workdir <codename>
  --key <key>` reads `code/HZ.txt` and `training_data/train_<key>.txt` (0.6 s per
  1e6 samples) and writes one line per check, `c : T(1) … T(2^d)`, in
  `configuration_table`'s order. P_c = counts/N; a configuration never seen gets the
  floor 0.5/N and then at most T = 0, so it is never promoted above independence,
  while one that independence expects and the data never shows keeps its negative
  value. On `p_0.0015_sig_0.0015_s_1`: KL(joint‖independent) 0.030–0.050 nats per
  check, 446 of 2304 configurations unseen (2e-6 of a check's independent mass).
  `run_local.sh --check-node bethe:1:learn` builds the tables for the training and
  test keys from the FULL files and copies them under the cut keys.
- **Sweep settings files live in `expts/sweeps/`** (tracked; `expts/scripts/` is the
  generator's per-run copy and is ignored). `sweeps/enriched_vs_nocer.toml`
  (2026-10-19): `["enriched:0.503:learn"]` + no-CER × `softmin_first` (softmin +
  first, warmup 0; the 2026-10-15 sweep's histpair) / `ramp_last` (ramp + last,
  warmup 5, k = 3) × 5 seeds, 20 points, 80 tests. Every arm names its whole layer
  pair, warmup and sharpness in `optimizer_arms`, so the file sets no top-level
  `loss_layer_selection` / `commit_layer_rule` (those would only be the default
  for an arm that names neither; the generator's own default is softmin + first).
  `train_wall_time = "3:00:00"` against a guessed `seconds_per_train_point = 1500`
  (one wave of 4 per task, est 40 min, 22%): the enriched node's cost on an A100
  is unmeasured (~2× the kernel launches of tanh), so the wall carries the margin.
  Run with `bash sweep_hyperparams.sh --settings sweeps/enriched_vs_nocer.toml
  --no-edit` (verified with stubs: 5 train + 8 test tasks exit 0, 20 models, 80
  results); the report's weights-spread check names the first arm's file with its
  `_enr`/`_bethe` token.
- **Tests** (not yet run — Julia is unavailable here; syntax checked with tree-sitter):
  `test_soft_constraints.jl` — the per-check kernel equals the batch kernel, a zero
  table is the tanh rule, the adjoint equals Enzyme on the probe loss (messages and
  α) and finite differences on the table, a clamped slot has a zero adjoint, the
  Bethe table gives the exact first-layer posterior of a random 3-bit joint, file
  round trip and refusals, the dense adjoint equals the loop one on plain Arrays, the
  JSON carries `enriched_table`; `test_gradient.jl` — manual vs Enzyme on the enriched
  two-check graph (every mode, θ's gradient non-zero and compared) and the enriched BB
  code at the CER priors, GPU vs CPU for both, and training with Enzyme / manual CPU /
  manual GPU agreeing on θ as well as the weights. The two "refuses the enriched
  node" testsets became these.

## α: one value for every layer (the layer schedule was removed 2026-10-11)

- **α is a unit-consistency factor, not a fitted number.** `single_qubit_rescale` maps
  the median single-qubit rate to 0.1 and leaves J untouched, so the pair term must be
  softened by the same ratio: `α* = log(9) / log((1−m)/m)`, m the median raw rate.
  That is 0.426 on `p_0.0005_sig_0.001` (where "0.42" came from) and **0.503 on
  `p_0.0015_sig_0.0015`**. Every run on the new dataset before 2026-09-22 used 0.42.
- **In the neural decoder (and the unit-weight standard BP built on it) the enriched
  check node is the only correlation-aware message-passing rule, and its α is the same
  in every layer.** From 2026-09-22 to 2026-10-11 a separate key,
  `coupling_schedule = "step"`, let α vary by layer as `α_t = α · d(t)` (a logistic
  step with two trainable parameters, T₀ and w). It was removed to keep one rule with
  one α. Nothing measured depended on it: every enriched result in `data/` (395 rows)
  and every local run from 2026-10-07 had a constant α; the cluster step arm
  (`_sch12w3L`, 2026-10-05) is recorded only as neither causing nor preventing the NaN
  rollbacks; and its learned T₀ moved 12 → 12.0006 ± 0.039 in 2,500 updates
  (`check_parameter_movement.py`), so the init was the schedule.
  - The forward pass still allocates α per layer (`Float32[α]` inside the layer loop)
    so that Enzyme sums α's gradient over layers in the same order as every earlier run.
  - Signatures: `get_loss_value`, `get_individual_loss_values`,
    `compute_loss_and_gradients`, `forward_pass_with_weights`, `forward_pass_with_tape`,
    `forward_pass_gpu` and `build_gpu_state` lost their `coupling_schedule` argument —
    **four `Duplicated` arguments, then the five `Const`s**. `NachmaniNeuralBP` lost the
    field (positional constructor: base, three weight vectors, `coupling_logit`);
    `LossAndGradients` lost it too; `GPUState.coupling_scale` is a scalar.
  - Old inputs: a TOML with `coupling_schedule = "step"` is refused by
    `parse_hyper_parameters`; `"constant"` and the `coupling_schedule_*` keys are
    ignored (the sweep generator strips them). A weights JSON with
    `coupling_schedule_kind = "step"` is refused by `load_trained_neuralbp_model`; one
    with `"constant"` loads as before. New weights files, results CSVs and debug logs
    carry no schedule fields (the collector stacks CSVs with `cols = :union`).
  - Sweep spec: `enriched:<α>:<fixed|learn>`; a spec with more fields is refused.
  - **Left as is (decided 2026-10-12):** the old Float64 classical-BP stack
    (`bp_algo.jl`, `tanner_graph.jl`, `quantum_code.jl`; drivers `quantum_BP_test.jl`,
    `ballistic_errors.jl`, `misc/explicit_errors.jl`) has its own correlation rule:
    one degree-2 soft check per correlated pair, appended to H, message
    `2 atanh(tanh(J) tanh(m/2))` with `J = ½ log(w/(1−w))`, `w = weight_soft_constraint`.
    Its plain path, `run_bp(…, n_checks + 1, …)`, is the standard-BP reference in
    `test_forward.jl` and `test_bp_algo.jl`.
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
  `BASE_LOSS_SMOOTH = 1`, `base_loss_code` / `base_loss_name`. All of these, the
  residues and their derivatives now live in `src/base_loss.jl` (2026-10-09).
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
- `run_local.sh --design icscale --jobs 8` (run at `--layers 50` on 2026-10-09; the
  option was removed 2026-10-15, 90 is fixed): ramp + last + smooth_loss
  with `initial_conditions_scale` over `--scales` (0.1,0.3), and everything else —
  warmup, k, training set — from the baseline TOML. The ramp-at-warmup-0 note
  prints once, not per point.

### Measured 2026-10-09: `--design icscale`, 50 layers, ramp + last + smooth_loss, Poisson(3) 10k set, warmup 0

Pooled over 4 test sets × 2 seeds, 800k per arm. 0 of 8 rolled back. Baseline = the
2026-10-09 ramp + smooth arms (whole file, warmup 5, init 0.3).

| arm | total | coset | conv | TRAIN set (s1/s2) | baseline (init 0.3) |
|---|---|---|---|---|---|
| CER, init 0.1 | 2305 | 930 | 1375 | 538 / 562 | — |
| CER, init 0.3 | **2014** | 863 | 1151 | 575 / 550 | 2247 (883 \| 1364), TRAIN 620 / 568 |
| no-CER, init 0.1 | 2482 | 533 | 1949 | 807 / 811 | — |
| no-CER, init 0.3 | **2314** | 544 | 1770 | 774 / 768 | 2444 (502 \| 1942), TRAIN 767 / 845 |

- **The sampler did its job on the input side and the decoder did not respond.**
  The set was exactly as designed (Poisson(3): 42% / 23% / 35% at w ≤ 2 / 3 / ≥ 4,
  ~9,050 distinct of 10,000, identical across the four arms at a seed). Informative
  batches rose from 5.6–12% to **15–17% (CER) and 21–29% (no-CER)** — exactly
  20 × Σ_w Pois(3;w)·P(fail|w) ≈ 0.19 failures per batch, as the arithmetic said.
  Training was healthy and the epoch-mean loss fell 30%. And the **per-weight
  failure rates at weights ≥ 5 on the training distribution are unchanged to the
  third digit** (CER: 1.69 / 2.72 / 6.6 / 12.3% at w = 5 / 6 / 7 / 8 vs 1.72 / 2.72 /
  7.05 / 12.5% before; no-CER likewise). Totals moved −5% (no-CER) and −10% (CER),
  1–2σ, and the CER number is mostly the transferred-set blocks below.
- **Init scale: 0.3 vs 0.1 is nothing on the training distribution** (CER 1125 vs
  1100, no-CER 1542 vs 1618). The pooled +13% for CER at 0.1 is one seed's block
  on one transferred set, not the init.
- **Warmup 0 under the ramp costs what was predicted.** On solved batches (85% of
  all) the layers 1–5 transient is **46–70% of the ramp loss** for CER (11–28% for
  no-CER, whose smooth-loss transient is smaller), and training spent itself there:
  the median layer-1 loss on solved batches fell 18.7 → 12.2 while the informative
  plateau stayed at 0.30–0.35. Half the gradient taught layer 1 to decode.

#### The failures are PATTERNS, and the sampler could not reach them (2026-10-09)

The per-sample files make the structure explicit, and it changes what "a failure"
means here:

- **BP's verdict is a function of the error pattern** (same pattern ⇒ same
  syndrome ⇒ same decode): every failing pattern fails on **every one** of its
  occurrences, in all 32 arm-seed-set files. A failure count is a sum over distinct
  patterns of their multiplicities, not a Poisson count of samples.
- **The error model is a few thousand patterns with heavy repetition.** The TRAIN
  test set's 200k samples are 10,848 distinct patterns; at p = .0005 only 2,600–3,600.
  Within one weight the probability is extremely non-uniform: at p = .0005 sig = .001,
  82% of the weight-3 samples are patterns that occur ≥ 20× (top pattern 129×), while
  C(72,3) = 59,640 weight-3 patterns exist.
- **On the training distribution the failures are the RARE patterns**: 807 failures =
  727 distinct patterns (multiplicity 1–3). Only **2–4%** of a model's failing test
  patterns were in its 10k training set and only **~11%** exist anywhere in the 200k
  training file. The Poisson draw is uniform *within* a weight, i.e. proportional to
  frequency, so it over-samples precisely the frequent cliques the decoder already
  handles (2,325 weight-3 draws covered 676 distinct patterns) and never sees the
  tail where the failures are. That is why tripling the informative share moved
  nothing: the signal came from the wrong patterns.
- **The CER seed "blocks" on the transferred sets are 1–4 FREQUENT patterns each**:
  cer_init_0p1 s1 at p = .0005 sig = .001 has 425 failures of which 334 are four
  patterns — (3,19,49) 129/129, (6,22,52) 100/100, (9,29,39) 77/77, (2,24,54) 28/28 —
  all weight 3, all containing a coupled pair, all convergence failures (either never
  cleared, or cleared and drifted out under commit = last: 159 of 162 at p = .0007).
  Nearly disjoint between seeds (overlap 0–2 patterns). This pre-existed (the
  baseline's 297/140, 98/224, 82/218 swings) and is specific to CER under a change of
  test-time priors/couplings; the no-CER arms, whose inputs do not change between
  test sets, have no blocks. **Counted per distinct pattern, CER beats no-CER on
  every set** (TRAIN 511–544 vs 711–744, −27%; transferred sets −25 to −35%); the
  sample-weighted number flips sign in the seeds that hit a frequent clique.
- **Half of a model's failing patterns are flippable.** Jaccard overlap of the
  failing-pattern sets between seeds is 0.5–0.7 (CER vs no-CER: 0.3–0.44), so about
  half the failures are structural (every init fails them) and half are decided by
  the init — the training target, if training could reach them.
- **Consequences.** (1) The statistical unit is the pattern: report distinct failing
  patterns alongside sample counts, and do not trust √N error bars on the
  transferred CER sets. (2) Weight-balancing cannot target failures; the lever is
  *pattern* coverage — synthetic errors drawn uniformly over patterns of a given
  weight (the `online_training` stub, which today only resamples the file), or
  hard-example mining of the decoder's own failing patterns (a forward pass over the
  pool, ~80 s on Metal, yields a few hundred patterns). The file's own patterns
  cannot cover the tail: 89% of the failing test patterns are not in it. (3) For
  the CER arms, the four-pattern blocks are the cheapest large win: frequent cliques
  the decoder fails deterministically, fixable by including the most frequent
  patterns of each prior in the training set, or masked by first-to-clear where they
  are drift.

## Hand-written backward pass (`src/backward_pass_with_weights.jl`, `src/backward_pass_gpu.jl`, started 2026-10-09)

- Built one forward rule at a time, CPU and GPU together, each function the
  ADJOINT of one forward step (∂L/∂outputs in, ∂L/∂inputs out). Step one of each
  rule is the reversed data flow — the forward rule's sum over the same
  neighbourhood, outputs back to inputs, constants (syndrome, channel LLR)
  dropped; step two, the derivative factors between raw and activated adjoints
  and the weight gradients, comes separately. Notation: m raw message, A = a(m)
  activated, bar = ∂L/∂.
- **Done: `backward_v2c_to_c2v!`** (rule 1̄, activated domain, within a layer):
  Ā_{v′→c} = Σ_{v∈N(c)−v′} Ā_{c→v}, i.e. `transpose(adj_V2C_C2V) * Ā_c2v`.
  Derived by differentiating rule 1 w.r.t. one input (Jacobian entries 1[v ≠ v′])
  and the chain rule. `adj_V2C_C2V` is symmetric, so the transpose is the
  forward's own matrix — asserted in a test, not relied on. GPU mirror
  `backward_v2c_to_c2v_gpu` with a `GPUBackwardState` holding the dense
  transpose (built once per model, `release_gpu_backward_state!`).
- **Done: `backward_c2v_to_v2c!`** (rule 2̄, raw domain, layer t → t−1):
  m̄^{t−1}_{c′→v} = Σ_{c∈N(v)−c′} W^{t−1}_{(v,c;c′,v)} m̄^t_{v→c} — `sparse_multiply!`
  with `rows`/`cols` swapped, the weight vector unchanged (the weight of a pair,
  read from its other end). Two kinds of derivative come out of one rule and they
  are different in kind: the adjoints w.r.t. the NEURONS (m̄) are the carrier that
  propagates to the next layer down; the derivatives w.r.t. W and b are the
  gradient the optimiser wants and are read off locally, since each weight
  appears in exactly one place — `weight_gradients_c2v_to_v2c!`:
  W̄_{(v,c;c′,v)} = Σ_samples m̄^t_{v→c}·m^{t−1}_{c′→v} (outer product at the
  nonzeros) and b̄^t_v += Σ_samples l_v·Σ_{c∈N(v)} m̄^t_{v→c} (`adj_initialize_V2C'`
  then the LLR). Both accumulate; b^t_v is also used by the readout, whose share
  comes with the readout adjoint. GPU: `backward_c2v_to_v2c_gpu(m̄_v2c, state,
  layer)` with the transposed, masked per-layer weight matrices in
  `GPUBackwardState` (now built from the model, `build_gpu_backward_state(bpnn)`,
  so it is rebuilt per training step like the forward state);
  `weight_gradients_c2v_to_v2c_gpu` returns the FLAT (n_pairs,) weight gradient
  (one-hot selectors, since 2026-10-12; it returned the dense (nb × nb) form
  before) and the (n_bits,) channel gradient.
- Tests `tests/test_backward_pass.jl`: the two-check graph by hand (both rules),
  the BB code against pair-by-pair reference loops, the adjoint identities, finite
  differences of L = Σ G ⊙ (rule output) through the production forward
  functions for W̄ and b̄, accumulation, CPU ≡ GPU (plain Arrays when no
  backend). Rule 1̄'s CPU tests pass (2026-10-09).
- Two test lessons. `GPUBackwardState` is built from a MODEL (it carries the
  transposed per-layer weights); rule 1̄'s GPU test uses `unit_weight_neuralbp(base)`.
  A finite difference against the Float32 forward must sum the loss in Float64 and
  use a large step — the rules are linear in every parameter, so any step is exact
  — because a 648-term Float32 sum over a 1e-2 step is mostly rounding (simulated:
  19 of 30 seeds over 2e-3; Float64 sum at step 0.5: max error 6e-6).
- **Done (2026-10-09): the whole CPU gradient.** `src/gradient.jl`:
  `compute_loss_and_gradients` takes `get_loss_value`'s arguments in the same
  order and returns a `LossAndGradients` (loss + the four gradients Enzyme writes
  into the `Duplicated` shadows). Three stages: `forward_pass_with_tape` (the same
  `compute_layer_with_weights!` calls as `forward_pass_with_weights`, so the
  posteriors are bit-identical; keeps incoming/outgoing c2v, raw v2c, check-node
  magnitudes and signs per layer), the seed c_t · ∂L_t/∂μᵗ, and the reverse sweep.
  New rule adjoints in backward_pass_with_weights.jl: `backward_readout!`,
  `weight_gradients_readout!` (W^ro is shared across layers, so W̄^ro sums over
  them; bᵗ gets the readout's share AND rule 2's), `backward_atanh_exp_signed!`
  ((a⁻¹)′ out of the check node), `backward_log_tanh_split!` (a′ into it).
  `combine_layer_losses_gradient` (src/loss.jl) is c_t: softmax weights, 1 at the
  last layer, 1/n, or w_t/Σw.
- **`src/base_loss.jl`** holds the residue options as (g, g′) pairs —
  `sine_residue_loss`/`sine_residue_derivative`, `smooth_loss`/`smooth_loss_derivative`
  — plus `compute_base_loss` (moved verbatim from loss.jl; `compute_smooth_loss_from_llrs`
  is now its alias, so the Enzyme path is the identical computation) and
  `compute_base_loss_gradient` = (1/N) σ′(μ) ⊙ H⊥ᵀ g′(x). Adding an option = a g, a
  g′, a code, and one branch in each of the two wrappers.
- **Numerical forms, chosen to match Enzyme where it matters.** The forward's
  clamps are replicated (derivative 0 where tanh(|m|/2) or exp(M) is clamped), and
  the factors use Enzyme's forms in Enzyme's order: tanh′ as 1/cosh², atanh′ as
  1/(1 − y²). On a Float32 emulation the "same" derivative written as 1 − tanh²
  differs by up to 25% next to the clamp, and the gradient spread between the two
  forms reaches 2e-4 (LLR 5.5, 3 layers) to 1e-2 (LLR 8, 5 layers). σ′ is
  −σ(x)σ(−x): −σ(1 − σ) loses all accuracy for x ≲ −10, and Enzyme's
  −eˣ/(1 + eˣ)² is Inf/Inf beyond 88.7.
- Validated before transcription by a float64 numpy prototype against central
  finite differences: every layer rule × residue × warmup × temperature, ≤ 1e-6
  relative (Richardson), including saturated regimes.
- **The enriched check node is covered too (2026-10-19; it was tanh-only from
  2026-10-13 to then).** `backward_enriched_checks!` (CPU) and
  `backward_enriched_checks_gpu` (device) are the adjoint of the table node; see
  "Enriched check node: the table-driven implementation" below for the formulas
  and where they plug into the sweep. Under tanh α is never read and its gradient
  is exactly zero, as before.
- **The GPU gradient, `compute_loss_and_gradients_gpu` (`src/gradient_gpu.jl`,
  2026-10-12; both nodes since 2026-10-19)**: same arguments and result as the CPU wrapper, message
  passing in both directions on `ArrayT` (Metal on a Mac **whether or not
  `USE_GPU` is set**, CUDA on Linux, plain Arrays otherwise).
  - **The posteriors are not enough for the gradient.** `forward_pass_gpu` already
    returns μᵗ for every layer, which is all the LOSS needs, and the wrapper uses
    exactly that (host `base_loss_per_layer`, `combine_layer_losses`,
    `compute_base_loss_gradient` on the GPU's posteriors; only the per-layer seed
    goes back up). But the reverse sweep evaluates a′, (a⁻¹)′ and the weight
    read-offs at each layer's MESSAGES, and μᵗ = R m_c2vᵗ + bᵗ ⊙ l is 72 numbers per
    sample made from 216 messages — they cannot be recovered from it. So
    `forward_pass_gpu_with_tape` keeps raw v2c, check-node magnitudes and 0/1 signs,
    and outgoing c2v per layer on the device. It runs the testing forward's own
    layer function (`_compute_layer_gpu` now wraps `_compute_layer_gpu_values`), so
    its posteriors equal `forward_pass_gpu`'s bit for bit.
  - New GPU rule adjoints (`backward_pass_gpu.jl`): `backward_readout_gpu`,
    `weight_gradients_readout_gpu`, `backward_atanh_exp_signed_gpu`,
    `backward_log_tanh_split_gpu` (the CPU's scalar factors `atanh_exp_signed_adjoint`
    / `log_tanh_split_adjoint`, broadcast — one definition for both paths; the CPU
    loops now call them, same operations).
  - **Weight gradients are FLAT on the device**: one-hot selectors per weighted pair
    (`pair_output_selector`, `pair_input_selector`, and the readout's two) turn the
    read-off into matrix products, W̄_k = Σ_samples (S_out m̄)_k (S_in m)_k. Replaced
    the dense (nb × nb) gradient and `gather_weight_gradient_entries` (removed).
    `GPUBackwardState` now also holds the transposed readout matrix; it is built
    from (base, message weights, readout weights).
  - The wrapper builds and frees its forward state, backward state and tape per
    call (Julia's GC does not feel device memory). It rebuilds and re-uploads the
    dense per-layer weights every call — fine for correctness, the first thing to
    optimise for GPU training. At the training batch of 20 a GPU step is likely
    launch-bound; GPU training pays off with much larger batches.
- Tests: `tests/test_base_loss.jl` (residue and σ derivatives vs Enzyme on
  scalars, `compute_base_loss_gradient` vs Enzyme on the toy and a random 18×40
  dual) and `tests/test_gradient.jl` (tape ≡ forward bit for bit; manual vs Enzyme
  for 4 rules × 2 residues × warmups × {T = 1, 0.2} on the toy graph (1e-4), the BB
  code at LLR log 9 (1e-3), and at the CER priors (5e-3 norm); enriched refused).
  Each comparison prints its largest relative difference. **Measured 2026-10-11
  (Julia 1.13.1, updated Enzyme): largest relative gradient difference 5.3e-7 on
  the toy graph, 3.1e-7 on the BB code at LLR log 9, 1.6e-5 at the CER priors** —
  far inside tolerance, and inside the 2e-4 spread a mismatched derivative form
  would give, so the factor forms do match Enzyme's. The one failure was the LOSS
  against Enzyme's primal at rtol 1e-6 (1.7e-6 apart): Enzyme returns the value of
  its own compiled copy, whose @simd sums associate differently. The test now
  demands the manual loss EQUAL a plain `get_loss_value` call (same functions,
  bit-identical posteriors) and allows Enzyme's primal 1e-5.
- **GPU tests, measured 2026-10-13** (`test_gradient.jl`, Julia 1.13.1, Metal): the
  tape's posteriors `==` `forward_pass_gpu`'s; GPU wrapper vs CPU wrapper on the BB
  code at LLR log 9, 1.6e-6. At the CER priors the gradients agreed to 7.7e-4 but
  the LOSSES were 2–3e-4 apart, failing a loss tolerance of 1e-4. Not a bug: near
  saturation the tanh rule is ill-conditioned in Float32 (2 atanh(exp(M)) with
  exp(M) a few ulp below 1), so the device's own rounding of exp/log/tanh moves the
  posteriors by ~1e-3 relative — which the GPU forward test in
  `test_soft_constraints.jl` already allows. That test's loss tolerance is now 1e-3
  (not yet re-run), and a two-check-graph GPU test and the GPU refusal of an
  enriched base were added. `test_backward_pass.jl` — readout and activation
  adjoints GPU vs CPU, the flat rule-2 weight gradient.
- **Training with it: `manual_backprop = true` (TOML, added 2026-10-15).** Default
  `false` = Enzyme, so every earlier run reproduces bit for bit. `true` calls
  `compute_loss_and_gradients` in `train_neuralbp_enzyme!` in place of
  `Enzyme.autodiff` — or `compute_loss_and_gradients_gpu` when `gpu_active()`
  (USE_GPU=1, the same switch testing reads); the optimiser, clipping, NaN guard
  and rollback are unchanged. `manual_backprop_setting` validates it (a Bool; both
  check nodes since 2026-10-19), called by the training function and, before any
  data is read, by `neural_bp_experiments.jl`.
  Recorded in the results CSV as `manual_backprop` (TOML value; whether it ran on
  the GPU is not recorded — that is USE_GPU at TRAINING time). With `--quiet false`
  training prints which gradient it uses, and always prints one summary line:
  gradients computed, total time, the first gradient (compilation included) and
  the mean of the rest. Test (`test_gradient.jl`, not yet run): two Adam steps on
  the two-check graph from one seed, with Enzyme, manual CPU and manual GPU, must
  agree to 1e-5; plus the validation and the enriched refusal.
  - **Measured 2026-10-15 (laptop, M-series, 50 layers): GPU and CPU about the same,
    ~10 min per training point.** At `batch_size = 20` the GPU gradient rebuilds and
    re-uploads every dense per-layer weight matrix and selector each step and is
    launch-bound. Caching the model-independent parts of the two states across steps
    is the first optimisation, larger batches the second. The cluster sweep trains on
    the GPU anyway (see the cluster notes), four points sharing a card.

## Measured 2026-10-15: the first GPU-trained cluster sweep (Narval, `hp_2026-10-15_02-38-18`)

CER arm = single-qubit priors with the TANH check node (no couplings); no-CER = uniform
priors. design A (last/last) and histpair (softmin + first, warmup 0) × 5 seeds × 4 test
sets × 1e6 samples, 90 layers, manual backprop on the GPU. The training set was the
Poisson(3) 10k subset (`training_samples = 10000`, `failure_weight_boundary = 3`,
inherited from `hyperparams_baseline.toml`).

- **Run health**: 20/20 trainings and 80/80 tests exited 0; 0 rollbacks; weight sd 0.173
  → 0.178–0.181 (moved, as before). Training took ~12 min per task (4 points sharing an
  A100, after a 5 min CUDA_Runtime_jll rebuild) against the 45 min estimate; a 1e6 test
  decodes in 96–107 s, ~2.4 min per test with startup. ~46 min from the first task to
  the last test.
- Pooled failures, 20M samples per row (per-seed CV of the total in brackets):

  | arm | total | coset | conv |
  |---|---|---|---|
  | CER, histpair | **19409** (5.8%) | 9743 | 9666 |
  | CER, design A | 22957 (6.8%) | 8836 | 14121 |
  | no-CER, histpair | 22931 (2.7%) | 9021 | 13910 |
  | no-CER, design A | 27945 (4.1%) | 7756 | 20189 |

- **Single-qubit priors alone are a real effect**: −17.8% (design A) and −15.4%
  (histpair), 5/5 seeds, paired t = −8.7 / −10.2 (df 4). All of it is convergence
  (−30%); coset rises +14% / +8% (higher in 5/5 seeds). The same per DISTINCT failing
  pattern (−18% / −15%), and on every test set (−13% to −24% per pattern).
- histpair beats design A again: −15.5% (CER), −17.9% (no-CER), 5/5 seeds; convergence
  −31%, coset +10% / +16%.
- Coset failures barely depend on the seed (CV 2–3.5%); convergence failures carry the
  seed variance, and in the CER arms much of it is **blocks**: 5 cells, all CER, where
  one frequent weight-3 pattern fails on every occurrence — (3,23,69) 424× on the
  training distribution, (11,27,57) 386×, (16,36,46) 272×, (2,24,54) 175×, one more
  97× — each failed by that one model only. 3% of CER failures; none in no-CER. Weight 3
  is the only weight where CER can be worse (training set, design A: 0.39% vs 0.23%;
  p = .0007: 0.37–0.50% vs 0.07–0.09%, nearly all convergence); at weights 4–12 CER is
  lower at every weight.
- On the SAME samples as the 2026-10-08 local run (first 200k of each set, seeds 1–2):
  no-CER design A 2290 (local 2331) and histpair 1830 (local 1898), so the GPU-trained
  decoders on the Poisson(3) set match the Enzyme-trained whole-file ones within 2–4%.
  CER priors (tanh) 1870 / 1535 against CER + couplings (enriched, local) 1481 / 1413:
  the couplings added −21% under design A (~3σ) and −8% under histpair (~1.5σ), almost
  all in convergence (1179 → 518, 786 → 417), and nearly all of the coset penalty
  (691 / 749 with priors only, 963 / 996 with couplings; no-CER 592 / 692).
- Under a perfect OSD only the coset column counts, and there no-CER design A is best.

## Measured 2026-10-19: enriched CER vs no-CER (Narval, `hp_2026-10-19_06-45-21`)

`sweeps/enriched_vs_nocer.toml`: CER = per-qubit priors + the enriched node (pairwise
table, α from 0.503, learned); no-CER = uniform priors, tanh. `softmin_first` (softmin +
first, warmup 0) and `ramp_last` (ramp + last, warmup 5, k = 3) × 5 seeds × 4 test sets ×
10⁶ (20M per row). Results in `data/72q_BB_cycles_1_trainable_alpha/` (the
`_spread_comparison` directory is the 2026-09-17 sweep; see the end).

- **No-CER `softmin_first` reproduced the 2026-10-15 `nocer_histpair` arm bit for bit**
  (every seed's total, coset and convergence count). GPU training + testing is
  deterministic, so arms from different sweeps pair exactly by seed.
- Verdict table (failures; t is seed-paired, df 4; CER lower in 5/5 seeds in every row):

  | | no-CER | CER | change |
  |---|---|---|---|
  | NBP alone, softmin_first | 22931 (9021 \| 13910) | 18276 (12241 \| 6035) | −20.3%, t −10.1 |
  | NBP alone, ramp_last | 24988 (7950 \| 17038) | 19642 (11993 \| 7649) | −21.4%, t −4.2 |
  | + BP+OSD fallback, softmin_first | 17531 | 15045 | −14.2%, t −8.9 |
  | + BP+OSD fallback, ramp_last | 17929 | 14927 | −16.7%, t −22.2 |

  Per DISTINCT failing pattern −20% / −27% (t −26 / −23). The NBP-alone gain is all
  convergence (−57% / −55%); coset rises +36% / +51% (0/5 seeds lower).
- **The fallback** (the OSD question, answered with a real decoder): ldpc 2.4.1
  `BpOsdDecoder(HZ, product_sum, max_iter 100, osd_cs, order 10)` run on every sample the
  NBP did not converge on (H = HZ, logical check LZ; each distinct pattern decoded once).
  The CER pipeline's fallback uses the test key's raw per-qubit CER rates, the no-CER one
  their mean. Coset failures stay failures (the NBP already returned a syndrome-valid
  answer). The fallback fails on 38–61% of the NBP's convergence failures (sample-weighted:
  CER arms 38–47% with CER priors, 46–57% uniform; no-CER arms 49–51% / 59–61%; by weight
  w3 0–17%, w4 15–34%, w5 and up 49–92%). With the SAME fallback for both arms CER is
  still −7% to −13% (5/5). Were the fallback to fail on one fraction f of both arms'
  convergence failures, the break-even would be f ≈ 0.41 (softmin_first) / 0.43
  (ramp_last); no-CER wins only under a post-processor that repairs most of them.
- Plain BP+OSD-CS(10), no NBP, per 10⁶ (uniform / CER priors): 405 / 279, 728 / 452,
  872 / 723, 3368 / 3364 on p.0005-sig.0005, p.0005-sig.001, p.0007-sig.001,
  p.0015-sig.0015 — 5373 / 4818 summed, against 4586 (no-CER) and 3655 (CER) for the
  NBP alone and 3506 / 3009 with the fallback (per seed, 4 sets). The NBP beats it.
- **Priors vs couplings** (softmin_first, paired by seed with the 2026-10-15 CER-tanh arm):
  priors alone 22931 → 19409 (−15.4%, t −10.2, 5/5; conv −30.5%, coset +8.0%); adding the
  couplings 19409 → 18276 (**−5.8%, t −1.5, 3/5 — not significant**; conv −37.6% t −4.1,
  coset +25.6% t +17.3, both 5/5). The couplings trade convergence for coset failures
  almost one for one (−3631 / +2498). Their value after the fallback is not computable:
  the 2026-10-15 per-sample files are gone. Re-running the CER-tanh arm reproduces it
  exactly (determinism, above) with per-sample files.
- **Training health**: 20/20 points completed 5/5 epochs, max nan_skip 0 everywhere — the
  enriched arms lost epochs in 8/20 points on 2026-10-05. Weight sd 0.177–0.181. Learned α
  0.493–0.526 from 0.503: still inert. A task of 2 enriched + 2 tanh points took ~25 min
  on one A100 (08:36 → 09:00 EDT, after a 6 min rebuild), so `seconds_per_train_point =
  1500` was right for a mixed wave; the 80 tests took ~25 min on 8 cards.
- **Blocks**: 9 frequent weight-3 patterns failed on every occurrence (65–983×, all
  convergence), all in CER arms: 4.4% (softmin_first) and 11.1% (ramp_last) of CER
  failures, and the cause of the CER ramp_last arm's 12% seed CV. The CER fallback rescues
  every weight-3 convergence failure, blocks included. On the training-distribution set
  CER fails less at every weight ≥ 4; at weight 3 it ties (softmin_first, 0.206% vs
  0.199%) or loses through a block (ramp_last, 0.86% vs 0.21%).
- Layer design: ramp_last is worse standalone (no-CER +9.0%, t +9.3, 5/5; CER +7.5%, n.s.)
  and within ~3% of softmin_first after the fallback.
- **The 2026-09-17 sweep in `data/72q_BB_cycles_1_spread_comparison`** (p = .0005,
  sig = .001, s_1–s_3 × 5 seeds, lr 0.01, warmup 10, init 0.1): CER tanh −24.1% vs no-CER
  (t −2.6, 11/15 cells; coset +0.2%, conv −33%) — consistent with the priors effect above.
  Its enriched arms (α 0.42 fixed/learned, +18%) hit the NaN rollback in 29/30 trainings,
  so they say nothing about the couplings.

## Submission pipeline: `submit.sh` → `batch_run.jl` (rewritten 2026-10-19/20)

The general-purpose driver for the `72q_BB_cycles_1_sweep` data (8 p × sigma ∈ {0, p,
2p}); `sweep_hyperparams.sh` remains the tool for hyperparameter sweeps.

- `bash submit.sh` writes `scripts/submission_settings_<ts>.toml` (defaults inline),
  opens it, runs `julia --project="./../" batch_run.jl --settings <file>`.
- **One submission, both jobs, chained (2026-10-20).** `train = true` + `test = true`
  (the defaults; `--train` / `--test` take true/false) writes the training job AND the
  test job, and `cluster/submit_<ts>.sh`, which `cd`s to expts/ and runs
  `TRAIN_ID=$(sbatch --parsable train_<ts>.sh)` then `sbatch
  --dependency=afterok:$TRAIN_ID --kill-on-invalid-dep=yes test_<ts>.sh` — the
  `sweep_hyperparams.sh` pattern. `test` alone tests models already on disk (commands
  whose model is missing are dropped with a warning; none left = refused); `train`
  alone only trains; Google_VM refuses both at once (its script halts the VM).
  - One timestamp per submission: `commands_<mode>_<ts>.txt`, `train_<ts>.sh` /
    `test_<ts>.sh`, and the arms' hyperparameters in
    `models/submissions/<ts>/<stem>_<arm>_<mode>.toml` — per submission, because a job
    reads its file when it STARTS (hours later for a chained test job) and the old
    fixed `<stem>_<arm>_<mode>.toml` let the next submission rewrite it under a queued
    job. `--hyperparams submissions/<ts>/…` works because neural_bp_experiments.jl
    reads `<prefix>/models/<name>`.
  - Everything is checked before anything is written: inputs (plus
    `bethe_table_<key>.txt` for the CER arm when the base is enriched + bethe), the
    walltime of BOTH jobs, and the VRAM a GPU test job sizes its batch to.
  - **Runtime model check in the test job.** `models_test_<ts>.txt` holds the weights
    file of each command, line for line (`expected_weights_name`, mirroring
    src/train.jl's name). After the stage-in each task reads its slice and runs only the
    commands whose model is in `$LOCAL_WORK_DIR/models`; the rest are reported ("model
    missing, test NOT run") and the task exits 1. Needed because a test with
    `retrain = false` and no weights file TRAINS a model of its own and reports it.
  - Both jobs share one depot preparation: lock/sentinel
    `<workdir>/.depot_locks/submission_<ts>_<gpu|cpu>_depot.{lock,ready}` (one folder
    for all codenames, since they share one depot; GPU and CPU jobs keep separate tags
    because only the GPU preparation rebuilds CUDA_Runtime_jll).
  - A training job warns when models of the same names exist (it will replace them).
- **Run tags (2026-10-20): `_cer` / `_nocer`**, plus `_<run_label>` when set (letters,
  digits, `-`, `_`; not "seed"), then the code's `_seed_<n>`:
  `…_trained_using_train_<key>_cer_seed_1.json`,
  `…_trained_using_train_<key>_no_cer_nocer_seed_1.json` (`_no_cer` is train.jl's own
  infix). Not `""` for the no-CER arm: `_no_cer_seed_1` would contain `_cer_seed_1`,
  so a CER-arm glob would match both. The no-CER arm is always written with
  `check_node = "tanh"` (the enriched node refuses `use_CER = false`).
- **Datasets are given by their numbers** (2026-10-20; the user asked for no symbolic
  sigma): `train_instance = [0.0015, 0.003, 1]` = (p, sigma, sample), and
  `test_pairs = [[0.0001, 0.0], [0.0001, 0.0001], …]` (N × 2) tested at every index
  in `test_samples = [1]` — the user's choice over an N × 3 list. "2p" or any
  non-number is refused (`setting_number`). Every pair × sample the data lacks is
  skipped with a warning (sigma = 0 comes with samples 1–3, every other sigma 1–5:
  samples 1–5 give 120 requested, 104 tested); a pair with a test file at NONE of the
  samples is refused as a mistyped number, with the same-p files listed. Empty
  `test_pairs` / `test_samples` = the training pair / sample. Command line:
  `--train_instance 0.0015,0.003,1 --test_pairs 0.0001,0 0.0001,0.0001 --test_samples 1 2 3`.
  Keys are written like the Python generator's `repr`: `p_0.0015_sig_0.003_s_1`,
  sigma 0 → `sig_0.0` (`python_float_text`); a typed 2 × 0.0007 is `0.0014`, as in the
  file names. Every input file is checked before anything is written. Old keys
  (`train_pair`, `train_sample`, `test_sample`, `train_p`, `test_pvals`, `pvals`, …) are
  refused with the new format in the message.
- Array layout (2026-10-20): tasks = cld(commands, commands per task), so no task is
  left empty (1040 tests at 18 per task are 58 tasks, not 60 with two idle cards).
- Arms: `cer_arms = ["cer", "nocer"]`, each from a copy of the base hyperparameters
  with `use_CER`, `retrain` and `run_tag` set (the base file is never edited). `seeds`
  are passed as `--seed`. `gpu_for_training` (needs `manual_backprop = true`;
  `trainings_per_gpu` share a card) and `gpu_for_testing` (one process per card,
  `GPU_MEMORY` = 85% of the card's VRAM).
- The job body is `submission/slurm_job_template.sh`, filled by `slurm.jl`: contiguous
  slices per array task, stage-in of only the files a slice reads, stage-out of only
  files newer than the stage-in, one depot preparation per tag (mkdir lock), CUDA gate,
  the model check (test jobs), `parallel` in the background with the TERM trap
  (`--signal=B:TERM@300`), exit 1 when any command fails or was not run. **The template
  was untracked in git on 2026-10-20** (`?? expts/submission/slurm_job_template.sh`):
  slurm.jl reads it for every SLURM job, so it must be committed for Narval to get it.
- **Walltime preflight** (2026-10-20, `measured_round_seconds` in batch_run.jl,
  `slurm_task_layout` + `walltime_estimate` in slurm.jl, run for every job before any
  is written): startup 600 s + rounds × (GPU test 150 s; GPU training wave n_epochs ×
  144 s tanh / 300 s enriched, scaled by updates per epoch and layers); refuses when an
  estimate reaches wall_time and names the max_nodes that would fit; warns at 70%;
  prints "no measured timing" for CPU jobs. It does not see `online_training = false`
  or `batch_size`, and skips a base with `n_gradient_updates_per_epoch = 0`.
- Big-run defaults (2026-10-20): train (0.0015, 0.003, 1), 24 test pairs at sample 1,
  cer + nocer, seeds 1–5, train = test = true, max_nodes 24, wall 1:00:00 per job →
  training 10 models in 3 tasks (est. 34 min), then testing 240 in 24 tasks of 10 (est.
  35 min). Samples 1–5 would be 1040 tests: refused at 24 tasks (needs ≥ 55 at 1 h;
  58 tasks of 18 = 91% of the wall).
- **Seeds vs samples (asked 2026-10-20), from the 2026-09-17 sweep in
  `72q_BB_cycles_1_spread_comparison`** (CER-tanh vs no-CER, samples 1–3 × seeds 1–5,
  each sample with its own models): the CER effect averaged over 5 seeds was −23.0%,
  −21.0%, −24.3% on samples 1, 2, 3 (between-sample variance ≈ 0), while single seed
  pairs on one sample ranged from +4.5% to −62.7% (cell sd of the log-ratio 38%,
  against 6.7% Poisson). So the replication that matters is seeds (the trained model);
  samples change the answer by nothing measurable. Its seed CVs (13–33%) were larger
  than the 2026-10-15/19 sweeps' (2.5–12%), older settings (lr 0.01, warmup 10).
- Sample index (asked 2026-10-20): the training sample (third entry of
  `train_instance`) and `test_samples` are independent; one index is enough. The index seeds the per-gate rate draw
  (misc/sweep_transfer.sh header): s_1's rate vector is a rescaling of one fixed
  pattern at every (p, sigma) (R² 0.977–0.994), s_2's is unrelated (R² 0.04); at sigma = 0
  the index changes only the shots. So same-index train/test = the same hot-gate
  pattern, rescaled; another index = another "device", which would measure
  device-to-device spread, not the CER effect. Test commands read the TEST key's CER
  file. The 2026-09-21 cross-index probe found transfer between indices free (−0.4%
  to +4.0%, every CI straddling 0).
- Verified 2026-10-20 WITHOUT Julia (unavailable in the sandbox): Julia files parse
  (tree-sitter); the settings TOML parses and every key is a flag; the 24 test keys and
  the training key exist in the Narval tree; a Python mirror of the Julia side filled the
  template and the job scripts ran under stub julia / parallel / module: 3/3 train and
  24/24 test tasks exit 0 with 10 models and 240 results staged out, the CPU path too,
  and a failing command makes its task exit 1 with the rest staged out. The Julia side
  itself first runs on Narval.
- Re-verified after the chaining (2026-10-20, same mirror + stub `sbatch`): the submit
  script makes two sbatch calls, the second `--dependency=afterok:<first id>
  --kill-on-invalid-dep=yes`; 3/3 train tasks (1 prepares the depot), then 24/24 test
  tasks find the sentinel and prepare nothing; 10 models, 240 results. One model removed
  between the jobs: its 24 tests are not run (no julia call), the 3 tasks holding them
  exit 1, 216 results land. Test-only with 2 of 10 models missing: 192 tests, all pass.
  A test job that cannot fit (max_nodes 4) refuses with 0 files written. An enriched
  base: the no-CER TOMLs say `check_node = "tanh"`. A separate desk-check of the Julia
  (signatures, typed locals, scoping) found no runtime error on the default path.
- Re-verified with numeric instances (2026-10-20), the fake dataset now built from the
  Narval tree's own file list (`refs/tree_sweep.sh`): defaults → 24 instances, 10
  models, 240 results; samples 1–5 → 120 requested, 104 tested (sigma 0 skipped at 4
  and 5), refused at 24 tasks, 58/58 tasks and 1040 results at max_nodes 60; sigma
  0.00014 for 0.0014 → refused, naming the pair. The 24 default keys and the training
  key exist in the tree.

## Gathering results: `postprocess.jl` (updated 2026-10-20 for the (p, sigma, sample) names)

- **Bringing the results home** (2026-10-20, skeleton 2026-10-21): on Narval, from
  expts/, `bash misc/compress.sh --results <codename>` packs everything EXCEPT the files
  in `training_data/` and `testing_data/` — those two folders go in EMPTY, so the archive
  has the folder's whole top-level skeleton (code/, correlated_weights/, models/, logs/,
  results/, cluster/ with contents; cluster/logs only with `--all`, which now combines
  with `--results`). It writes `data/<codename>_results.tar.gz` — a separate name, so a
  full archive is never replaced — and `tar -xzf <codename>_results.tar.gz -C data`
  recreates `data/<codename>/` on the laptop, leaving training/testing files already
  there in place. The empty folders come from `--exclude='<dir>/<folder>/*'` (GNU tar
  drops what is inside and keeps the folder); a check after packing lists the archive
  and exits 1 if either folder is missing or not empty, since another tar's matching
  might differ. All of compress.sh's empty-array expansions use
  `${exclude_args[@]+"…"}`: the plain form is an unbound-variable error under `set -u`
  in macOS's bash 3.2. Checked with pv, GNU tar, a non-GNU tar, and a tar made to drop
  the folders (caught).
- **Bringing the logs home, to see why a run crashed** (`--logs`, 2026-10-23):
  `bash misc/compress.sh --logs <codename>` writes `data/<codename>_logs.tar.gz` in the
  same skeleton (every top-level folder present) with only what the run wrote about
  itself: `cluster/` in full — job scripts, command lists, each array task's .out/.err,
  and `cluster/logs` (joblogs and every point's own stdout/stderr, where the crash's
  message is) with or without `--all`; `logs/` in full (training debug CSVs, ~60–90 MB
  uncompressed per run dir, mostly `*_individual_losses.csv`); and from `models/` only
  the generated TOMLs (`hyperparams_hp_*`, `hyperparams_xf_*`, `submissions/` — the
  split cleanup.py makes), each of which states its point's whole settings. Every other
  folder goes in EMPTY (data sets, results/, code/, correlated_weights/, plots/), and
  weights JSONs, base TOMLs, top-level files (.DS_Store, tree_log.sh) and top-level
  links stay out — so unpacking it at home never replaces the laptop's own base TOMLs.
  `--logs` with `--results` is refused (`--results --all` is everything but the data
  sets, cluster/logs included). After packing, the archive's entries are compared with
  the expected set (trailing slashes ignored, since tars differ there): exits 1 on
  anything missing or extra. Checked 2026-10-23 on a fake run dir and on name-for-name
  copies of trainable_alpha and the sweep dir (20 point TOMLs / submissions/ kept, 0
  JSONs, 0 base TOMLs): the default, `--all`, `--results` and `--results --all`
  listings are identical to the previous script's; the pv and non-GNU branches give
  the same archive; a tar that ignores `--exclude` and one that drops the emptied
  folders are both caught; a tar that lists folders without a trailing slash passes;
  unpacking over a laptop copy keeps its results, base TOML and data sets.
- `julia --project="./" postprocess.jl --config=postprocess_configs/postprocess_panel_sweep.toml --task=gather`
  (from expts/, on the laptop: postprocess.jl loads PlotsForBPDecoder/Plots) writes
  into `<codename>/results/`: `decoder_statistics_NeuralBP.csv` (one row per results
  CSV), `decoder_statistics_BP_OSD.csv` and `decoder_statistics_BP.csv` (one row per
  reference table). The default panel (`postprocess_panel.toml`, no `--config`) points
  at the sweep too since 2026-10-21, with the same three gathers. Its
  `standardbp_gather` / `bp_gather` had kept the July `files` lists
  (`72q_BB_p_<p>_cycles_1_…`), which name nothing in the sweep, so they failed with
  "no rows collected"; they now use the sweep panel's patterns. The other panels point
  at `data/update_23July2026`, which is not on the laptop.
- Gather sections take `pattern` (shell-style `*`/`?`, one or a list; `glob_regex`)
  and/or `files`, and REPLACE their output. They used to call `save_decoder_dataframe`,
  which APPENDS to an existing file, so every re-run duplicated all rows.
- **`neural_gather` selects by the submission's own keys (2026-10-21)**:
  `train_instance = [p, sigma, sample]`, `test_pairs = [[p, sigma], …]` × `test_samples`,
  `seeds`, `cer_arms` (`"cer"` / `"nocer"`) and `run_label` (→ run tags
  `_<arm>[_<label>]`; a label with no arms means both). Each narrows the candidates
  (every `simulation_results_*.csv` unless `pattern`/`files` say otherwise) by what the
  file NAME says — p and sigma compared as numbers (`same_dataset_number`, 1e-9
  relative), not text; a key left out or empty keeps every value. The parsers are the
  submission's: `submission/dataset_settings.jl` (new file; `python_float_text`,
  `dataset_key`, `DatasetPair`/`DatasetInstance`, `parse_dataset_instance`,
  `dataset_pairs`, `sample_indices`, `test_instance_list`), included by both
  `batch_commands.jl` and `postprocess.jl`, so "2p" is refused in both. With arms,
  seeds, pairs and samples all set, the gather warns about every requested (arm, seed,
  test dataset) with no file or with more than one (two training sets with
  `train_instance` open, say); nothing selected is an error. Checked through a Python
  mirror on the 240 real results names: the panel keeps 240 (0 missing, 0 repeated);
  `seeds = [1]` 48; the 8 sigma = 0 pairs 80; `test_samples = [1, 2]` 240 kept + 240
  missing; `cer_arms = ["cer"]` 120; `run_label = "big"` or a wrong `train_instance`
  → error; every key removed or empty → 240. The name's seed equals the CSV's `seed`
  column in all 240.
- **The reference gathers take `test_pairs` / `test_samples` too** (2026-10-21,
  `TestDatasetSelection`, `admits_test_dataset`, `requested_datasets_without_files`,
  shared with neural_gather): files whose name carries no dataset are then left out,
  a requested dataset with no table is warned about, none at all is an error. The data
  has 104 tables per decoder (samples 1–5, 1–3 at sigma = 0); both panels say
  `test_samples = [1]`, which gives exactly the 24 test datasets of the Neural BP run,
  so `test_key` joins the three CSVs one to one (checked: the 24 BP, 24 BP+OSD and 24
  neural keys are the same set; every table's own p and sample agree with its name).
  Leave `test_samples` out to gather all 104.
- **The arm and the datasets are only in the file names**: a results CSV has `seed`,
  `check_node`, the coset/convergence counts and the test file's path, but no
  `use_CER`, run tag or sigma column. `neural_gather` adds `results_file`, `arm`
  (`cer` / `nocer`: the run tag without its underscore), `use_CER`, `run_tag`,
  `test_key`, `test_p`, `test_sigma`, `test_sample`, `train_key`, `train_p`,
  `train_sigma`, `train_sample` from `NEURAL_RESULTS_NAME`; a name in an older format
  (no `sig_`) keeps its row without them, with a warning. The reference tables
  (`72q_BB_<key>_BP_failure_rates.txt`, `72q_BB_<key>BP+OSD_failure_rates_OSD_E_order_2.txt`)
  hold `p sample failures trials LER stderr` but not sigma, so the standard gathers add
  `decoder`, `test_key`, `test_p`, `test_sigma`, `test_sample` from the name, set the
  description to `testing_data/test_<key>.txt`, and warn if the table's p or sample
  disagrees with the name.
- Checked 2026-10-20 without Julia, on the 928 names of the run's `results/`
  (`data/tree_log.sh`) through Python copies of the patterns and regexes: 240 / 104 /
  104 files selected, all 240 names parsed (120 `cer`, 120 `nocer`, 5 seeds × 24 test
  datasets each, one training dataset), and the 208 real reference tables from the
  sweep zip agree with their names. postprocess.jl parses (tree-sitter).
- **The plot sections are not (p, sigma)-aware**: `_logical_error_rate_vs_p` groups rows
  by `test_p_<p>` alone and averages the rest, so on this run it would merge sigma = 0,
  p, 2p and both arms into one point per p. Off in the sweep panel until they learn the
  new columns. `misc/compare_decoders.py` (below) is the (p, sigma)-aware comparison.

## Measured 2026-10-21: the `72q_BB_cycles_1_sweep` run against BP and BP+OSD

The run: trained on `p_0.0015_sig_0.003_s_1`, tested on the 24 sets (8 p × sigma ∈
{0, p, 2p}) at sample 1, 10⁶ shots each, cer / nocer × seeds 1–5. **Both arms use the
TANH check node** (so CER = per-qubit priors only, no couplings), ramp + last at
warmup 0, 10 epochs, the Poisson(3) 10k training set; 0 rollbacks in all 10 trainings.

- **Tools.** `python3 misc/compare_decoders.py --workdir <codename> [--seed N]` reads the
  three gathered CSVs (+ `results/reference_decoders_rerun.csv` if present) and writes
  `results/decoder_comparison.csv` (failures and LER of every decoder per test set) and
  `plots/decoder_comparison.{pdf,png}` (LER vs p per sigma, seed bands, CER/no-CER
  ratio row). `python3 misc/reference_decoders.py --workdir <codename> --keys … |
  --all-keys --configs <bp|osd0|osd_e<k>|osd_cs<k>>:<cer|mean|number>:<max_iter>`
  re-runs ldpc BP / BP+OSD (each distinct syndrome once; loads ldpc's decoder modules
  without its `__init__`, which needs stim), refuses a `--testing-dir` whose samples
  differ from the ones the per-sample failure files list, and upserts
  `results/reference_decoders_rerun.csv`.
- **Seeds.** The user's rule: a FIXED seed (1) is the fair number; the best seed is
  chosen on the test data, so optimistic. Best = lowest geometric-mean LER over the 24
  sets (total failures and mean rank agree): **nocer 1, cer 3**. nocer seed 5 is 2.2×
  worse than seed 1 in geometric mean with no rollback — a genuinely worse model.
- **The shipped reference tables are a weak baseline.** They are ldpc with a uniform
  prior of **0.1** (6–100× the sets' mean single-qubit rate), 500 iterations,
  product-sum, parallel, and **OSD-E of order 2**, which is OSD-0 in practice (osd0 / E1
  / E2 / E3 give 3601 / 3600 / 3599 / 3598 on `p_0.0005_sig_0.0005_s_1`). Reproduced to
  the failure (`bp:0.1:500`, `osd_e2:0.1:500`) on the 5 sweep sets whose test files are
  on the laptop — `data/72q_BB_cycles_1_trainable_alpha/testing_data` holds the sweep's
  own samples for `p_0.0005_sig_{0.0,0.0005,0.001}_s_1` and `p_0.0015_sig_{0.0,0.0015}_s_1`
  (every failing sample's error weight in the sweep's per-sample files matches). Against
  them the NBP fails 0.10× BP+OSD (geometric mean; 0.013–0.38) and BP+OSD 0.40× BP on
  every set; the shipped decoders scale as p^0.93 (they fail on single-fault patterns),
  the NBPs as p^2.
- **With the sets' own rates the picture changes** (failures per 10⁶; NBP at seed 1,
  best seed in brackets where it differs):

  | set | shipped BP+OSD | BP, CER rates, 90 / 500 it | BP+OSD-CS10, CER rates, 90 / 500 it | BP+OSD-CS10, mean rate, 500 it | NBP nocer | NBP cer |
  |---|---|---|---|---|---|---|
  | p.0005 σ0 | 3155 | 2002 / 179 | 466 / 156 | 174 | 193 | 217 (216) |
  | p.0005 σ.0005 | 3599 | 2711 / 245 | 535 / 217 | 251 | 257 | 235 (229) |
  | p.0005 σ.001 | 4572 | 1894 / 388 | 534 / 343 | 428 | 433 | 359 (354) |
  | p.0015 σ0 | 8701 | 3295 / 2120 | 2151 / 1815 | 2129 | 2066 | 2242 (2098) |
  | p.0015 σ.0015 | 9771 | 6146 / 2784 | 2909 / 2430 | 2966 | 2782 | 2939 (2509) |

  BP needs ~500 iterations here (it is 1.5–11× worse at 90); at the NBP's 90-iteration
  budget the NBP beats BP by that much and BP+OSD-CS10 by up to ~55% at p = .0005 (tie
  at .0015). Given 500 iterations, plain BP with the CER rates is level with the NBP
  (±20%), and **BP+OSD-CS10 with the CER rates has fewer failures than every NBP model on
  all 5 sets** (3–19% below the best of the 10 models on each set; the two 3% cases are
  within noise). NBP no-CER ≈ BP+OSD-CS10 at
  the mean rate (−6% to +11%). The other 19 sets need the run on Narval (`--all-keys`,
  ~5 s per config per set). Not yet measured: the NBP with a BP+OSD fallback on its
  convergence failures (the 2026-10-19 pipeline), which is the comparison that matters
  if OSD is downstream.
- **CER vs no-CER (tanh, priors only)**, geometric-mean LER ratio cer/nocer per sigma
  group (sets where CER is lower): seed 1 — σ = 0 1.38 (1/8; the p = 1e-4 point is 57 vs
  9 failures), σ = p 0.93 (7/8), σ = 2p 0.84 (8/8); best seeds 1.03 / 0.92 / 0.82; median
  over seeds 1.25 / 0.84 / 0.73. The σ = 0 numbers are pulled up by the p = 1e-4 set,
  which is bimodal: cer seeds 1–5 fail 57 / 65 / 7 / 52 / 9 times, nocer 9 / 12 / 9 / 6 /
  70 — the "block" signature (one frequent pattern failing on every occurrence), in 3 of
  5 CER models and 1 of 5 no-CER. Without that set, σ = 0 is 1.11 / 1.07 / 1.01. The
  priors help in proportion to the gate-rate spread and not at all without it. Coset vs convergence over the 24 sets: nocer s1 11492 /
  16452, cer s1 12404 / 14452 — the usual trade.
- **The same rates help the classical decoders as much or more** (5 local sets, CER
  rates vs the uniform mean rate, 500 it): plain BP −10% / −25% / −15% / −25% / −25%
  and BP+OSD-CS10 −10% / −15% / −14% / −18% / −20% on p.0005 σ0, p.0015 σ0, p.0005 σp,
  p.0015 σp, p.0005 σ2p — INCLUDING σ = 0, where the NBP CER arm gains nothing (+12%,
  +9% at seed 1). At σ = 0 the 72 rates are nearly flat (max/min 1.3; 2.6 at σ = p, ~4
  at σ = 2p) and their pattern correlates 0.37 with the training set's (0.97 for the
  σ = p set), so the likely reason is that the CER model learned on the σ = 2p pattern
  and does not transfer to σ = 0's (inference, untested). The no-CER models did NOT
  quietly learn the training device's per-qubit pattern: their 90 × 72
  `weights_llrs` (layer-major), averaged over layers, correlate +0.23 / +0.14 / +0.03 /
  +0.14 / −0.04 with the training rates' log-odds (control, the s_2 pattern: −0.04 to
  −0.08), and their per-qubit spread (sd 0.016–0.020) is the initial noise averaged
  over 90 layers.

### Measured 2026-10-22: the enriched arm (`_cer_enr`) on the same run, and why it fails at low p

`cer_enr` = CER priors + the pairwise enriched node, α learned from 0.538 (5 seeds: 0.513,
0.580, 0.509, 0.514, 0.505), otherwise the 2026-10-20 CER arm; 10/10 epochs, 0 NaN skips.

- **Failures over the 24 sets, seed 1** (shipped BP / BP+OSD are prior 0.1, OSD-0-like):

  | | BP | BP+OSD | nocer | cer | cer_enr |
  |---|---|---|---|---|---|
  | σ = 0 | 93312 | 38578 | 5984 | 7009 | 8376 |
  | σ = p | 107982 | 43566 | 7903 | 7700 | 7564 |
  | σ = 2p | 134599 | 56580 | 14057 | 12147 | 12812 |
  | coset / conv (all) | | | 11492 / 16452 | 12404 / 14452 | 17762 / 10990 |

  Summed over 5 seeds and the 3 sigmas, per p: cer_enr / cer = 21.5, 4.2, 2.0, 1.15, then
  0.92, 0.89, 0.89, 0.90 at p = 1e-4 … 1.5e-3 (cer_enr / nocer 0.71–0.78 from 9e-4 up).
  So the couplings help ~10% on top of the priors near the training p and blow up below
  ~7e-4, in EVERY seed (p = 1e-4 σ = 0: 399–1257 vs cer 7–65, nocer 6–70).
- **By σ AND p** (5-seed sums; cer_enr / cer, seeds where cer_enr is lower):

  | | p ≤ 3e-4 | p = 5–7e-4 | p ≥ 9e-4 |
  |---|---|---|---|
  | σ = 0 | 7.8 (0/5) | 2.2 (0/5) | **0.71 (5/5)** |
  | σ = p | 9.0 (0/5) | 1.23 (4/5; seed 2 dominates) | **0.86 (5/5)** |
  | σ = 2p | 4.1 (0/5) | 1.05 (1/5) | 1.03 (0/5) |

  At σ = 2p — the training set's shape — the couplings add nothing over the priors
  (training set itself 21788 vs 21689); their gain is at σ = 0 and σ = p near the
  training p. Priors alone vs nocer at p ≥ 9e-4: 1.09 / 0.85 / 0.76 at σ = 0 / p / 2p.
  The σ-group totals are 80–90% p ≥ 9e-4 sets, so "cer_enr lowest at σ = p" (7564 vs
  7700, seed 1) is a −14% high-p gain netted against a 9× low-p loss, not a σ effect.
  Lowest NBP per set: σ = 0 nocer at p ≤ 7e-4, cer_enr above; σ = p cer at p ≤ 7e-4,
  cer_enr above; σ = 2p cer everywhere except p = 9e-4.
- **The excess is weight-3 convergence failures**: p = 1e-4 σ = 0 seed 1, 786 of 814;
  p = 5e-4 σ = 0, 756 of 964 — a few frequent cascade "blocks" (seed + two lattice
  neighbours, J ≈ 4.7 on both pairs, one check each, syndrome weight 5; (16,32,62) 258×,
  (15,35,45) 258×, (10,30,40) 229×) failed on (nearly) every occurrence, stuck at syndrome
  weight 3. Knife-edge: the GPU run fails (10,30,40) on 225 of 229 occurrences, and which
  blocks fail changes with the seed (p = 1e-4: 12 failing samples common to all 5 seeds,
  1787 in any). Seed 2 (largest α, 0.580) is the worst on most low-p sets.
- **Cause: one α for every test set, while the couplings grow as p falls.** J_ik in the
  test key's CER file is a marginal log-odds ratio, ≈ ln(q/p) for a cascade pair, so the
  median strong J is 2.8 on the training key and 4.6 at p = 1e-4; the priors are rescaled
  to median 0.1 at every p, so α* = ln 9 / ln((1−m)/m) falls from 0.538 (training) to 0.395
  (p = 5e-4 σ = 0) and 0.318 (p = 1e-4). α* × J (median strong pair) stays ≈ 1.4–1.5, but
  the learned α × J reaches 1.6× the training value at p = 1e-4.
- **Test-time rescale α_test = α · α*(test)/α*(train) fixes it** (`misc/nbp_mirror.py`,
  the numpy mirror of the test forward; enriched mirror ≠ GPU on knife-edge blocks, so
  compare mirror with mirror). p = 5e-4 σ = 0, weight-3 failures (9401 samples), seeds 1–5:
  learned α 470 / 1842 / 941 / 949 / 251 → rescaled (0.37–0.43) 6 / 10 / 2 / 5 / 3;
  training-set J with the learned α also fixes 4 of 5 seeds (3 / 4 / 4 / 248 / 4); α = 0 gives 5 (seed 1).
  Totals with the rescale: **158 / 155 / 152 / 152 / 153** (GPU, learned α: 964 / 1886 / 1361 /
  965 / 689; nocer 193–477, cer 206–630), so the enriched arm becomes the best decoder
  there and its seed spread vanishes. Controls: p = 5e-4 σ = p seed 1, 232 (GPU) → 212;
  first 250k of p = 1.5e-3 σ = 0 seed 1, 445 → 460 (GPU 446; cer 595, nocer 536), i.e.
  unchanged where α* barely moves. p ≤ 3e-4 is not on the laptop: untested, but its
  excess is the same weight-3 signature. Now a Julia option (next bullet).
- **`coupling_scale_transfer = "none" | "alpha_star"`** (hyperparameters TOML, added
  2026-10-22; default "none" = every earlier run). Test-time only, enriched node only
  (a tanh model is refused: it has no α). "alpha_star" sets the loaded model's logit to
  α_learned · α*(test)/α*(train) before testing (`transfer_coupling_scale`,
  `prior_softening_factor`, `training_key_cer_data_file` in `src/neuralbase.jl`; α* =
  ln((1−t)/t)/ln((1−m)/m), t = `single_qubit_rescale`, exactly 1 wherever the rescale
  leaves the rates alone; the training key's CER file is found from `--train
  train_<key>.txt`). The results, per-sample and layer-profile names get `_alphastar`
  after the run tag (`COUPLING_SCALE_TRANSFER_TAG`), the weights name does NOT, so the
  trained model is loaded and the learned-α results are neither overwritten nor
  reported as these (the gather reads the arm as `cer_enr_alphastar`). Every results
  CSV now also has `coupling_scale_transfer` and `learned_coupling_scale`
  (`coupling_scale` = the α used). Tests: `tests/test_coupling_scale_transfer.jl`
  (not yet run; parses with tree-sitter).
- **Local run: `bash misc/test_alpha_transfer_local.sh [--key <key>] [--seeds "1 2 3 4 5"]`**
  (from expts/; defaults p_0.0005_sig_0.0_s_1, the 5 `_cer_enr` models). Refuses before
  running anything if a model, CER file or test file is missing (a missing model would
  make the test TRAIN one); links `testing_data/test_<key>.txt` from
  `72q_BB_cycles_1_trainable_alpha/testing_data` when the run's own folder lacks it;
  checks every learned-α failure's error weight against the test file (the right file
  passes 4735/4735, p_0.0005_sig_0.0005 fails 4687/4735); writes the test TOML (base +
  `retrain = false`, `run_tag`, `coupling_scale_transfer = "alpha_star"`) under
  `models/submissions/local_alphastar_<ts>/`; tests each seed on Metal (`--cpu` for
  USE_GPU=0), skipping seeds whose `_alphastar` results exist; prints learned vs
  transferred side by side. Verified with a stub julia (5/5 runs, the TOML's keys, the
  link, the skip, and the refusals). Mirror expectation at p_0.0005_sig_0.0_s_1: α 0.377
  / 0.426 / 0.374 / 0.378 / 0.371, ~152–158 failures.
- **Measured 2026-10-22, the real decoder (Julia, Metal, ~455 s per 1e6 test)** on
  p_0.0005_sig_0.0_s_1, seeds 1–5 — α* **168 / 158 / 163 / 153 / 155** (coset 98 / 84 / 99
  / 98 / 97, convergence 70 / 74 / 64 / 55 / 58) against the learned α's 964 / 1886 / 1361 /
  965 / 689; the mirror was within 1–10%. Same set, same seeds: nocer 193 / 219 / 212 / 219
  / 477 (coset 55–63), cer (priors only) 217 / 206 / 216 / 630 / 441. So the NBP alone:
  α* below nocer in 5/5 seeds (797 vs 1320 summed, −40%), all of it convergence (321 vs
  1024); coset is HIGHER (476 vs 296). With a BP+OSD-CS10 fallback (ldpc, max_iter 100,
  CER rates; the mean rate for nocer) on each arm's convergence failures: nocer 650,
  α* 655, cer 664, learned-α cer_enr 735 (5 seeds; per seed α* 126–136, nocer 122–139) —
  level. Plain BP+OSD-CS10 with the CER rates (500 it): 156 on this set.
- **BP+OSD-CS10 with the NBP's own priors beats every NBP model** (measured 2026-10-22,
  ldpc, product-sum, parallel; the 5 local sets; priors = the CER rates RESCALED to median
  0.1, i.e. `single_qubit_rescale`'s input to the NBP; 90 iterations, the NBP's budget):
  116 / 186 / 305 / 1383 / 1992 on p.0005 σ0 / σp / σ2p, p.0015 σ0 / σp, against the best
  of all NBP models per set (any arm, any seed, α* included) 153 / 229 / 348 / 1704 / 2295
  — 12–24% fewer. At 500 iterations 109 / 182 / 290 / 1402 / 1947. The RAW rates were
  worse (466 / 535 / 534 / 2151 / 2909 at 90 it, the 2026-10-21 table), so the rescale
  helps OSD as it helps BP. Uniform 0.1 at 90 it: 113 / 172 / 315 / 1449 / 2096 — the
  per-qubit CER pattern is worth ~0–5% here. The shipped tables' weakness is uniform
  priors + 500 iterations + OSD-E2, together: uniform 0.1 at p.0005 σ0 gives 3155 (E2,
  500 it, = the shipped table), 1014 (CS10, 500 it), 113 (CS10, 90 it); with the
  rescaled CER rates the iteration count barely matters (116 vs 109). OSD-CS2 / E2 /
  CS0 with the rescaled rates at 90 it: 117 / 127 / 126. Not run on the other 19 sets
  (`reference_decoders.py` has no rescaled prior option yet).
- **The same-cost comparison: plain BP at the NBP's budget** (ldpc product-sum, parallel,
  max 90 iterations with early stop; measured 2026-10-22, the 5 local sets, NBP at seed 1):

  | set | BP, uniform 0.1 | BP, CER rescaled | NBP nocer | NBP cer | NBP cer_enr (α*) | BP+OSD-CS10 |
  |---|---|---|---|---|---|---|
  | p.0005 σ0 | 8778 | 194 | 193 | 217 | 964 (168) | 116 |
  | p.0005 σp | 10152 | 277 | 257 | 235 | 232 | 186 |
  | p.0005 σ2p | 13166 | 693 | 433 | 359 | 363 | 305 |
  | p.0015 σ0 | 23483 | 2019 | 2066 | 2242 | 1715 | 1383 |
  | p.0015 σp | 26990 | 3646 | 2782 | 2939 | 2317 | 1992 |

  BP with exactly uniform priors barely converges (87–97% of its failures are
  non-convergence); the trained weights fix that. BP's mean iterations per shot are
  0.3–1.1 (most syndromes are trivial), while the commit = last NBP always runs 90
  layers, and OSD runs on 0.01–0.23% of shots (125 / 155 / 484 / 1210 / 2337 per 1e6):
  so on AVERAGE work BP+OSD is cheaper than the NBP as trained; OSD's cost is the
  worst-case latency (an O(n³), sequential elimination) and its scaling with n.
- **NBP + the strong OSD fallback vs BP+OSD** (p.0005 σ0, seeds 1–5; fallback = ldpc
  BP+OSD-CS10 at 90 it, rescaled CER priors for the CER arms, uniform 0.1 for nocer, on
  each model's convergence failures): nocer 99 / 101 / 97 / 102 / 108, cer 106 / 115 /
  110 / 111 / 118, cer_enr α* 114 / 104 / 118 / 120 / 125, learned-α cer_enr 129 / 123 /
  110 / 134 / 126, against plain BP+OSD-CS10's 116. So the no-CER NBP in front of OSD
  is −13% (5/5 seeds) and the CER arms add nothing on top of it. (The earlier fallback,
  raw rates at 100 it, gave 122–139 for every arm.) One set only.
- **The Bethe table against that baseline (measured 2026-10-22, 5 local sets; tables from
  each key's own 1e6-sample TRAINING file, tested on its test file).**
  - *Inside untrained BP* (unit-weight enriched node, rescaled priors, 90 it,
    first-to-clear; numpy mirror): p.0005 σ0 tanh 195, pairwise α* 356, Bethe α* 14,731
    (none coset, all non-converged), Bethe α = 1 nothing converges. With BP+OSD-CS10
    as a fallback on the non-converged: 115 / 109 / 117 / 149 — no gain over plain
    BP+OSD (116). A within-check Bethe factor makes pair flips about as likely as single
    flips (which they are, locally), and loopy BP does not settle.
  - *As the judge of OSD's answer* (`misc/bethe_osd.py`: ldpc BP, OSD-CS10 written out
    in numpy, reproducing ldpc to ±2): choosing among the OSD candidates by
    log P_Bethe(x) = Σ_c T_c(x_c) + Σ_u log p_u(x_u) instead of the prior weight:
    110 / 173 / 282 / 1177 / 1811 vs 116 / 186 / 305 / 1383 / 1992 (−5 / −7 / −8 / −15 /
    −9%) on p.0005 σ0 / σp / σ2p, p.0015 σ0 / σp. Choosing by the raw pairwise-J model
    is worse than the prior weight (175 / 250 / 383 / 1812 / 3328). Letting the Bethe
    model also overrule BP where BP converged (`--always`, BP's answer as one more
    candidate) is far worse (949 at p.0005 σ0): it over-rates correlated alternatives on
    easy patterns.
  - *Headroom*: 59–70% of BP+OSD's failures are BP converging to a syndrome-valid wrong
    answer (69 / 122 / 209 / 809 / 1309), which OSD never sees. On ALL of BP+OSD's
    failures the true error is more probable than the decoder's answer under the Bethe
    model in 108/117, 166/182, 268/308 (independent priors: 20 / 21 / 44; pairwise J:
    105 / 162 / 260). The empirical maximum-likelihood class per syndrome (training data)
    fixes every OSD failure whose syndrome occurs in training (10 / 26 / 76 of them) —
    the rest (107 / 156 / 232) have syndromes never seen in 1e6 training samples.
- **Narval, all 24 sets (prepared 2026-10-22, not yet run):**
  `sweeps/hyperparams_cer_enriched_alphastar.toml` (= `hyperparams_cer_enriched.toml`
  + `coupling_scale_transfer = "alpha_star"`; body checked identical) and
  `sweeps/enriched_alphastar_submission.toml` (the enriched submission with `train =
  false`, that hyperparams file, `wall_time = "1:00:00"`; `run_label` stays "enr" so the
  `_cer_enr` models are the ones loaded). From expts/: `bash misc/precompile_depot.sh`
  (source changed), copy the hyperparams file into the run's models/, `julia
  --project="./../" batch_run.jl --settings sweeps/enriched_alphastar_submission.toml`,
  then the printed `sbatch …/test_<ts>.sh`. batch_run.jl keeps the base's other keys in
  the derived test TOML, the template stages all of `correlated_weights/` (so the
  training key's CER file is on the node), and results come back tagged
  `_cer_enr_alphastar`; gather with `cer_arms = [..., "cer_enr_alphastar"]`.
- `compare_decoders.py`'s ratio row now widens its (shared) y axis to the data.

### Measured 2026-10-22: commit first vs last, depth, epochs, and what priors do for BP

All from `misc/nbp_mirror.py` (numpy mirror; new `--rules last first`, `--depths`,
`--extra-layers`, `--unit-weights`, `--run-tag`, `--csv`), both readouts out of ONE
forward pass; rows in `results/mirror_commit_rules.csv` of both run directories.
Fidelity: tanh models within 0.1–0.4% of Julia under either rule (first:
22920 vs 22931 over the 20 no-CER softmin_first model-sets) except knife-edge weight-3
blocks; enriched models coset exact, convergence ±10%.

- **The production run's training**: all three arms of `72q_BB_cycles_1_sweep` trained
  with `loss_layer_selection = "ramp"`, **`warmup_layers = 0`** (the ramp was designed
  for 5), k = 3, 10 epochs × 500 updates × 20, lr 1e-3, init 0.1, the Poisson(3) 10k set,
  and test with `commit_layer_rule = "last"`. The commit rule is test-time only (train.jl
  never reads it), so "ramp + first" is the same models with another readout.
- **Ramp + first vs ramp + last** (`72q_BB_cycles_1_trainable_alpha`, the 2026-10-19
  ramp_last models: ramp, warmup 5; 5 seeds × 4 sets, 90 layers):
  no-CER 25014 → 22599 (**−9.7%**, every set −9 to −12%, 5/5 seeds, t −29; 2583 samples
  rescued, 168 hurt; coset +12%, convergence −20%); CER-enriched 18856 → 18615 (−1.3%;
  Julia's own per-sample files cap the possible gain at 1.75%: only 344 of its 19642
  failures ever cleared). **Ramp + first ≈ softmin + first** (no-CER −1.4%, t −1.3;
  enriched −0.8%, t −0.2): the 2026-10-19 "ramp_last worse by 7–9%" was the READOUT,
  not the loss.
- The same on the sweep models (5 local sets, 5 seeds): no-CER −12.4% (−21 to −24% at
  p = 5e-4, −10% at 1.5e-3, 5/5 everywhere), CER priors −2% to −7% (−15% on p.0015 σ0,
  block-driven), enriched α* −0.3%. The no-CER decoder clears and drifts; the CER ones
  rarely do. Mean committed layer under first: 1.3–1.6 (so an early-exit decoder would
  do ~1.5 layers per shot on average).
- **Same cost, same readout (seed 1, first-to-clear for both; ldpc BP stops at its first
  converged iteration)**: BP with rescaled CER priors 194 / 277 / 693 / 2019 / 3646 on
  p.0005 σ0 / σp / σ2p, p.0015 σ0 / σp; NBP no-CER 186 / 235 / 390 / 1964 / 2605; NBP CER
  priors 214 / 224 / 342 / 2159 / 2879; NBP enriched α* 158 / 211 / 382 / 1750 / 2393
  (−13% to −45% vs BP); BP+OSD-CS10 116 / 186 / 305 / 1383 / 1992.
- **What the CER priors do for plain BP is mostly symmetry breaking** (ldpc, 90 it,
  same 5 sets): uniform 0.1 fails 8778 / 10152 / 13166 / 23483 / 26990 (87–97%
  non-convergence); uniform × (1 + 10% random noise), 3 draws: 296–749 / 351–979 /
  655–1329 / 2725–3816 / 3744–4891; the CER rates randomly PERMUTED over the qubits:
  492–1379 / 281–763 / 503–926 / 4266–5728 / 3083–4110; the CER rates in place 194 / 277 /
  693 / 2019 / 3646. So any non-uniform prior buys 10–30×; the true pattern ~1.5–3× more
  at σ = 0 and σ = p, and no reliable gain at σ = 2p (one permutation beat it). The
  no-CER NBP gets the same lift from its weights, without the CER file.
- **Depth.** Failures still fall over the last trained layers (first rule, 75 → 90:
  no-CER −9%, enriched −5%). Running past 90 with layer 90's weights (`--extra-layers`,
  a probe, not a trained deeper model): to 120 layers no-CER −15%, enriched −13%; to 180
  −17% / −16%; all of it convergence (−51% / −63%) while coset rises (+35% / +9%: 45% of
  the samples newly cleared after layer 90 land in the wrong coset for no-CER, 21% for
  enriched, against 23% for those cleared in layers 76–90). With an OSD behind it
  (which fails on ~40–50% of convergence failures) that nets to about −2% / −4%.
- **Epochs** (training logs, batches whose LAST-layer loss shows a failure, 5 seeds ×
  500 per epoch): sweep run no-CER 36.2% → 18.6% over 10 epochs, still −5% per epoch at
  the end; CER priors 18.1% → 15.6% (flat from epoch 5); enriched 17.7% → 16.7%. The
  2026-10-19 5-epoch models: enriched flat at ~15.4%, no-CER 22.2% → 18.8%. Pattern
  coverage: of the failing test samples on the training distribution
  (p_0.0015_sig_0.0015_s_1, ramp_last models), 1% of the distinct failing patterns are in
  the model's own 10k training set and 16–19% anywhere in the 1e6 training file.
- **The real-decoder check: `sweeps/enriched_ramp_commit.toml`** (prepared 2026-10-22;
  the first submission, `hp_2026-10-22_16-25-01`, trained nothing — a cold depot, see
  the cluster notes — so not yet run). Enriched only (`enriched:0.503:learn`, no no-CER arm), trained on
  p_0.0015_sig_0.0015_s_1, the 2026-10-19 sweep's 4 test sets, 5 seeds; two optimizer
  arms that differ ONLY in `commit_layer_rule`: `rampw0_last` and `rampw0_first` (ramp,
  warmup 0, k = 3). Everything else is the user's `models/hyperparams_baseline.toml` of
  2026-10-22 in that directory (5 epochs, init 0.1, lr 1e-3, Poisson(3) 10k — the
  production run's settings but its 10 epochs). 10 trainings (5 tasks, 2 per card,
  1:30 wall, est. 44%), 40 tests (8 tasks, 1:00, est. 45%). Run from expts/:
  `bash sweep_hyperparams.sh --settings sweeps/enriched_ramp_commit.toml --no-edit`,
  then the two printed sbatch lines; afterwards `python3 misc/compare_sweep_arms.py
  --workdir 72q_BB_cycles_1_trainable_alpha --arms _cer_enr_rampw0_last
  _cer_enr_rampw0_first`.
  - The two arms train the same model at each seed (training never reads the commit
    rule; GPU training is deterministic), so the weights should be IDENTICAL —
    compare_sweep_arms.py checks that before comparing counts. The duplicate
    training is the price of using the generator unchanged.
  - **The generator ignores `loss_layer_selection` and `commit_layer_rule` in the base
    TOML** (it strips both; the settings file's top level, default softmin + first,
    or the arm decides). So those keys in hyperparams_baseline.toml do nothing in a
    sweep; every arm here names its layer design.
  - The tags avoid `ramp_last`: the 2026-10-19 `_cer_enr_ramp_last` models (warmup 5,
    init 0.3) have the same epochs and training key, and would be retrained and
    replaced.
  - Verified 2026-10-22 with stub julia / parallel / module and fake SLURM vars on a
    copy of the directory holding the 20 old models: generation gives 10 + 40 points,
    the ramp-warmup note, the two arms' TOMLs differ only in run_tag and
    commit_layer_rule; 5/5 train tasks and 8/8 test tasks exit 0, 10 models, 40
    results, every test loads its model (retrain = false); compare_sweep_arms.py
    reports identical weights. With one model removed every test task exits 1 naming
    it, with 29 *.json present; regenerating lists the 10 existing models.
- `misc/compare_sweep_arms.py --workdir <codename> --arms <run tag> <run tag> ...`
  (2026-10-22): per test set and over all of them, each arm's failures summed over
  seeds as total (coset | convergence), mean committed layer, and against the first
  arm the change, seeds lower and the seed-paired t of the log ratio; plus a seed-by-
  seed weights comparison. `sweep_hyperparams.sh --collect` cannot do this any more:
  its collector parses the pre-2026-10-09 `_hp<arm>` names. Checked on the 2026-10-19
  files: `_cer_enr_ramp_last` 19642 (11993 | 7649) vs `_cer_enr_softmin_first` 18276
  (12241 | 6035), `_nocer_*` 24988 vs 22931 (t −9.4), the numbers above.

### Next: the enriched check node on the same run (prepared and run 2026-10-21)

- **Status**: the enriched submission ran on 2026-10-21; its 120 `_cer_enr` results
  (written 16:57–17:17) and 5 `_cer_enr` models are in `data/72q_BB_cycles_1_sweep/`
  beside the untouched 2026-10-20 `_cer` / `_no_cer_nocer` files. Analysed 2026-10-22
  (next section); the Bethe submission has not run. (Before it, `git pull` on Narval failed with
  "unresolved deltas left after unpacking": `git fsck --full` showed missing trees and
  blobs in Narval's clone with the quota fine; the repair given was `git fetch
  --refetch origin`, or copying the packs of a fresh bare clone into `.git/objects/pack`.)

- **Why the 2026-10-20 CER arm used only half the CER file**: its
  `hyperparams_baseline.toml` has no `check_node` key, so the default `"tanh"` ran,
  and tanh reads only the per-qubit rates (as priors). The file's 540 pair couplings
  J_ik enter only through `check_node = "enriched"`.
- **Files (tracked, `expts/sweeps/`)**: `hyperparams_cer_enriched.toml` and
  `hyperparams_cer_bethe.toml` are that baseline byte for byte plus a check-node
  block (`check_node = "enriched"`, `enriched_table = "pairwise"` / `"bethe"`,
  `coupling_scale_init = 0.538` learned, `require_correlations = true`); at the same
  seed they start from the same weights, training draw and batch order as the
  2026-10-20 CER arm (α is set, not drawn). `enriched_submission.toml` /
  `bethe_submission.toml` are batch_run.jl settings: the 2026-10-20 datasets and
  seeds, `cer_arms = ["cer"]` (the no-CER arm would only repeat itself),
  `run_label = "enr"` / `"bethe"` (run tags `_cer_enr` / `_cer_bethe`, beside the old
  files), `trainings_per_gpu = 2` (enriched kernels are heavier: 3 training tasks),
  `wall_time = "2:00:00"` (batch_run.jl estimates 10 epochs × 300 s + 600 s = 60 min,
  which a 1 h wall refuses). Run on Narval from expts/: copy the hyperparams file into
  `../data/72q_BB_cycles_1_sweep/models/`, then `julia --project="./../" batch_run.jl
  --settings sweeps/enriched_submission.toml` and the printed `submit_<ts>.sh`. The
  Bethe one first needs `misc/build_bethe_tables.py` for the 24 s_1 keys (the
  training key is one of them; the loop is in its settings file), and should start
  after the enriched training job is running: each submission prepares the depot
  behind its own lock, so two at once could rebuild CUDA_Runtime_jll concurrently.
  Checked with the batch_run Python mirror + stub-executed template: 3/3 train tasks,
  5 models `…_cer_enr_seed_<n>.json`, 24/24 test tasks, 120 results, the derived
  TOMLs carry the enriched keys, the commands are the 2026-10-20 ones but for the
  hyperparams path.
- **Why not `sweep_hyperparams.sh`** for this run: its jobs stage the WHOLE codename
  directory (`tar -chf - -C $WORKDIR $CODENAME`), ~30 GB here (208 files of 144 MB),
  on every array task, and it writes softmin + first unless told otherwise; the
  submission pipeline stages only the files a slice reads and is what produced the
  results being compared.
- **α**: 0.538 is α* for the TRAINING set (median raw rate 0.01653). The learned α is
  used unchanged on every test set, whose own α* runs from 0.32 (p = 1e-4) to 0.54,
  so at low p the couplings are relatively too strong (up to 1.7×); a test-time
  rescale α_test = α · α*(test) / α*(train) is not implemented. Same argument for the
  Bethe table (its T is in raw units; the priors are rescaled), hence 0.538 there too
  rather than the raw-units Bayesian 1.
- **Gathering them**: `neural_gather`'s `cer_arms` now also takes an arm with its
  submission's label (`"cer_enr"`, `"cer_bethe"`: `^(cer|nocer)(_<label>)?$`, no
  "seed"), so `cer_arms = ["cer", "nocer", "cer_enr", "cer_bethe"]` gathers all four
  (commented in both panels). `compare_decoders.py` takes every gathered arm by
  default (nocer first, then cer, then the rest), prints each arm's ratio to the first
  per sigma group, and draws one ratio line per arm. Checked on the 240 real names
  plus 240 synthetic `_cer_enr` / `_cer_bethe` ones (mirror of the selection: 360 /
  480 kept, `cer_seed`, `CER`, `enr` refused) and a synthetic 4-arm figure.

## Run tags (changed 2026-10-09)

- **The run tag is `_<arm>_<optimizer tag>` and nothing else** — `_cer_init_0p1`,
  `_nocer_histpair` — or `_<arm>` alone for a lone, untagged `base` arm. Every
  `_hp<arm>`, `_cnenr<α>`, `_sch…`, `_opt…`, `_lsl…_clr…`, `_blsmooth`, `_fwb…`,
  `_ts…` tag mentioned above describes filenames written BEFORE this date; the
  settings a sweep holds fixed now live in the TOML and the results CSV, not the
  name. Consequently two points that would share a filename (two check-node
  arms in one sweep, say) are refused by the generator rather than merged.
  This included the generator's old cluster default, `check_node_arms =
  ["tanh", "enriched:0.42:learn"]` (both CER arms, both `_cer_<opt>`; refused,
  verified 2026-10-11). Since 2026-10-19 a non-tanh arm carries `_enr` / `_bethe`
  after the arm name in the run tag, so one sweep can hold tanh and one enriched
  node side by side (`["tanh", "bethe:1:learn"]`: 30 points, verified with stubs).
- Not part of the tag, and unchanged: the code's own `_no_cer` infix on a no-CER
  run (so `..._200000_no_cer_nocer_init_0p1_seed_1.json`), and the weights
  filename's `nlayers_<N>_epochs_<n>_trained_using_<training file>` fields.
- The sweep TOML keeps its `hyperparams_hp_` prefix (`hyperparams_hp_cer_init_0p1_<key>_seed1.toml`):
  cleanup.py and run_local find sweep TOMLs by it. Weights and results names carry
  no `_hp`, so every glob that used to match on `_hp` now matches on `_seed_`
  (run_local's model gate and results clearing, the generator's collect and
  stage-in steps).

## Running locally on Apple Silicon (`misc/run_local.sh`, added 2026-10-07)

- **The laptop runs Julia 1.13.1 on `Manifest.toml`** (since b207a50, 2026-10-15).
  It needs Enzyme ≥ 0.13.204: the 0.13.138 of the old 1.12-resolved manifest refused
  1.13 with `AssertionError: VERSION < v"1.13"` in `Enzyme.Compiler.fixup_1p12_sret!`.
  The cluster keeps its own `Manifest-v1.12.toml` (cluster notes, "One manifest per
  Julia minor version"). Runs on the new stack will not reproduce older runs bit for
  bit at the same seed — compare arms within one stack.

- **Metal does nothing for training under Enzyme.** Enzyme cannot differentiate
  through device-array allocation, so its training forward pass is always the CPU
  one. With `manual_backprop = true` USE_GPU=1 puts training's gradient on Metal
  too (both check nodes since 2026-10-19), but `run_local.sh` still trains with USE_GPU=0 — the
  hand-written gradient on the CPU, as fast per point on the laptop and with no
  sharing of one GPU between its concurrent trainings — and tests on Metal. The
  cluster sweep trains on the GPU.
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
  to its inputs. Rules are one table (`RULES`): logs/ cluster/ emptied; results/
  emptied **except `*.txt`** (since 2026-10-20: the dataset ships its reference
  decoders there — `72q_BB_<key>_BP_failure_rates.txt` and
  `72q_BB_<key>BP+OSD_failure_rates_OSD_E_order_2.txt`, one-line tables `p sample
  failures trials LER stderr` at 10⁶ trials, 208 in `72q_BB_cycles_1_sweep` — which the
  old rule would have deleted; nothing in the repo writes a .txt there except
  `reformat_standard_bp_failures.py`'s aggregate of the same tables, and only a
  `simulation_results_*.csv` makes a test skip); models/ loses `*.json`, sweep TOMLs
  (`hyperparams_hp_*`, `hyperparams_xf_*`) and everything under `submissions/`
  (batch_run.jl's per-submission arm files, matched by path), and keeps every other
  `*.toml` as the base config. Refuses `/`, `$HOME`, the repo, `data/` itself,
  anything outside `data/`, and any run whose models/ would be left with no `.toml`.
  Replaces `clean_up_data.sh`. Checked 2026-10-20 on a copy of the sweep zip's
  results/: 208 tables spared, the CSVs removed.

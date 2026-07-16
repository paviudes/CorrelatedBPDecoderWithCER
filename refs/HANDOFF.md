# Handoff — Neural BP Decoder for Quantum LDPC Codes

This document is a self-contained briefing so a fresh Claude session can pick up the project without re-deriving context.  Please read it end-to-end before touching any file.

---

## 0. Absolute must-do before anything else

**Do NOT walk, grep, or read anything under `CorrelatedBPDecoderWithCER/data/`.**  That directory holds gigabytes of experimental output — training/test datasets, model weight JSONs, per-sample result CSVs, correlation-strengths files.  Nothing there needs analysis; it's just where the pipeline stores outputs.  Reading it wastes token budget without adding context.  If a task genuinely needs one specific file from `data/`, ask me for the exact path and read just that one file with a byte range.

Everything relevant to the code lives in:

- `CorrelatedBPDecoderWithCER/src/` — the main Julia package
- `CorrelatedBPDecoderWithCER/expts/` — experiment drivers and cluster submission
- `CorrelatedBPDecoderWithCER/PlotsForBPDecoder/` — companion plotting package
- `CorrelatedBPDecoderWithCER/tests/` — tests (some are stale; see §7)
- `CorrelatedBPDecoderWithCER/Project.toml`, `LocalPreferences.toml` — deps + CUDA prefs

`CorrelatedBPDecoderWithCER/refs/` holds historical Python scripts from an earlier version of the data-generation pipeline — read only if I explicitly reference them.

---

## 1. Who I am

I'm Pavi (Pavithran Sridhar, pavithran.sridhar@gmail.com), a researcher at the Institute for Quantum Computing (IQC) in Prof. Jemerson's group.  Comfortable with:

- **BP math and quantum error correction.**  You can talk to me directly about check nodes, LLR passing, Tanner graphs, LDPC parity-check matrices, syndrome extraction — no hand-holding needed on the physics/theory.
- **Julia at a working level.**  I read and write Julia, use Revise, understand modules and includes.
- **Cluster workflows at an intermediate level.**  Comfortable with SLURM, sbatch, salloc, srun, GNU parallel, module load, and shell scripting; less comfortable with the deep systems details (cgroups, CUDA runtime initialization, MIG semantics).

I'm not a professional software engineer — I self-describe as a "software noob."  What I actually want from you: pick clean, principled designs; explain *why* choices matter; leave comments in code that document intent; when there's a legitimate architectural decision to make, walk me through the tradeoffs rather than picking silently.

I share this Claude subscription with my collaborator **Debankan**, who runs a lot of the actual cluster jobs (his username on Alliance Canada is `debankan`; his home is `/home/debankan`, scratch at `/scratch/debankan`).  When I paste his terminal output, it's from his sessions.

---

## 2. The project

I'm building a **Neural Network version of the Belief Propagation (BP) decoder for quantum LDPC codes**, specifically targeting the Bivariate Bicycle (BB) family — currently the 72-qubit BB code, referred to in file paths as `72q_BB_…`.

The neural model is a Nachmani-style unrolled BP: each of `n_hidden_layers` layers corresponds to one BP iteration, with learnable per-edge weights.  Training uses **Enzyme.jl for autodiff** on the CPU (we intentionally dropped Zygote and Flux — see §4).  Inference / testing runs on GPU via CUDA.jl.

The error model we're studying is a **correlated error model** (a.k.a. the "Ballistic Error Model" — a name we've now dropped from file/function names; see §5), parametrized by two probabilities:

- `p` — per-qubit error probability
- `q` — neighbour error probability

For each `(p, q)` pair we generate multiple "samples" (independent realisations of the correlation structure).  Each sample gets its own training set, test set, and correlation-strengths file.

**Comparison baseline:** BP-OSD (order-2), computed elsewhere and dumped as a single-file table.  See `collect_standard_decoder_statistics_correlated` in `src/postprocessing.jl`.

**Typical workflow for one experiment:**

1. On my Mac: generate training/test data files (correlation-strengths and error samples).
2. `rsync` (well, tar-pipe now) the codename directory to Alliance Canada `$SCRATCH`.
3. Generate `commands.txt` — one julia invocation per `(p, q, sample)` triple — via `expts/batch_run.jl`.
4. Submit as a SLURM job array.  Each array task runs GNU parallel over a chunk of `commands.txt`.
5. Each command invokes `expts/neural_bp_experiments.jl` with `--train …` and `--test …` files.  Loads weights, runs BP inference on GPU, writes per-sample result CSV.
6. Stage-out: tar-pipe `results/` back to `$SCRATCH`.
7. On my Mac: `collect_decoder_statistics_correlated` aggregates the per-sample CSVs, `PlotsForBPDecoder` renders the error-rate curves and violin plots.

---

## 3. Codebase layout

### Main package `CorrelatedBPDecoderWithCER/src/`

- `CorrelatedBPDecoderWithCER.jl` — top-level module.  Declares `USE_GPU`, `GPU_BACKEND` (`"cuda"` on Linux HPC, `"metal"` on Mac), imports CUDA or Metal conditionally, `include`s all submodules, and exports the public API.
- `utils.jl` — general helpers.  Owns the canonical **`fmt_probs(p, q) → "p_0.010_q_0.001"`** formatter (zero-padded, max-decimals form).  Also `random_values_around_one`, `safe_log_tanh_split[!]`, `safe_atanh_exp_signed[!]`, `sigmoid`, `binary_entropy`, `binary_entropy_of_sigmoid`, `xor_affine!`, `sparse_multiply!`.
- `tanner_graph.jl`, `hypergraph_product_code.jl`, `hamming.jl`, `quantum_code.jl` — code / graph structures.
- `error_model.jl` — `ErrorModel` subtypes: `IIDErrorModel`, `BallisticErrorModel`, `ExplicitErrorModel`, `RandomWalkErrorModel`, `RegenerativeErrorModel`.  Sampling and sweep helpers.
- `bp_algo.jl` — classic BP: `BPSettings`, `belief_propagation_decoder`, `run_bp`.
- `neuralbase.jl`, `nachmani.jl` — neural-BP model structs (`NeuralBPBase`, `NeuralBP`, `NachmaniNeuralBP`) plus soft-constraint plumbing and CER data parsing.
- `forward_pass_weights.jl`, `forward_gpu.jl`, `legacy.jl` — forward pass implementations (weight-aware, GPU, legacy debug).
- `train.jl` — `train_neuralbp!`, `train_neuralbp_enzyme!`, `train_Nachmani_neuralbp`.  Enzyme is the sole autodiff engine.
- `predict.jl` — `predict_neuralbp`, `neuralbp_test_predictions`, `predict_and_check_neuralbp`.
- `loss.jl` — loss functions.  **Loss wraps in sigmoid** (`sigmoid(μ) = 1 / (1 + exp(μ))`, so positive μ ⇒ low error probability) — important to know when reading gradient code.
- `log_model.jl`, `nbp_weights.jl` — model checkpoint save/load and BP-weight extraction.
- `postprocessing.jl` — result aggregation.  `DecoderStatistics` struct plus **`collect_decoder_statistics_correlated`** (neural BP) and **`collect_standard_decoder_statistics_correlated`** (BP-OSD baseline).  Uses `fmt_probs` for filename reconstruction.
- `command_line.jl` — ArgParse setup for the CLI plus `disable_retrain_in_hyperparams` (sed-based helper).
- `plot.jl` — **NOT** `include`d by the module.  Analysis-only helpers that depend on Plots.jl / StatsPlots.jl.  Used from a REPL on my Mac.  Also referenced by `PlotsForBPDecoder` (see below).

### Experiment / submission scripts `CorrelatedBPDecoderWithCER/expts/`

- `neural_bp_experiments.jl` — the actual worker script that per-sample commands invoke.  Parses CLI args, loads training + test data, runs `train_Nachmani_neuralbp` if needed (respects `retrain=false` in hyperparams), then `neuralbp_test_predictions`, writes result CSV.  Contains `extract_parameters_from_description` which now expects `test_p_…` filenames (was previously `test_ballistic_p_…`).
- `batch_run.jl` — top-level submission driver.  Loadable on Alliance Canada login nodes because it **deliberately does not `using CorrelatedBPDecoderWithCER`** — importing the main package on the login node triggers `lld SIGBUS` under the login node's resource throttling.  Duplicates two helpers inline: `disable_retrain_in_hyperparams` (from `src/command_line.jl`) and `fmt_probs` (in `submission/batch_commands.jl`).
- `submission/batch_commands.jl` — builds `commands.txt` and dispatches to a backend.  Contains its own local copy of `fmt_probs` (kept in sync with `src/utils.jl` — worth flagging in review comments if the canonical one changes).
- `submission/slurm.jl` — Alliance Canada SLURM script generator.  This file has had substantial hardening (§6); the bash it emits is now much longer than the Julia that emits it.
- `submission/local_runs.jl` — local runner (runtime GPU detection for Mac Metal / Linux CUDA / CPU-only).
- `submission/google_vm.jl` — Google Cloud VM runner + shutdown wrapper.  Rarely used.
- `scripts/rename_to_padded_pq.py` — one-shot on-disk rename script (§5, §7).
- `scripts/submission_settings_*.toml` — auto-generated submission configs from `submit.sh`.
- `submit.sh` — convenience wrapper.  Writes a defaults TOML, opens it in `$EDITOR`, runs `batch_run.jl --settings <toml>`.

### Companion package `CorrelatedBPDecoderWithCER/PlotsForBPDecoder/`

Sibling package that owns *all* Plots.jl / StatsPlots.jl usage.  Deliberately kept out of the main package's Project.toml so cluster jobs don't precompile ~1 GB of plotting artefacts.

- Uses Julia 1.11+ `[sources]` in its Project.toml to reference the sibling `CorrelatedBPDecoderWithCER` by relative path.  No `Pkg.develop` needed.
- `src/PlotsForBPDecoder.jl` — the module.  Does `using CorrelatedBPDecoderWithCER: fmt_probs` (canonical source of that formatter).
- `src/error_rate_curves.jl` — `plot_statistics_for_ballistic_error_model` (avg logical error rate vs. q, log-scale, one curve per p).
- `src/performance_spread.jl` — `plot_performance_spread` (violin plot of per-sample BP-OSD / Neural-BP LER ratios).
- `src/utils.jl` — deprecated stub; the file exists but no longer defines anything (waiting for me to `rm` it on the next branch cleanup).

### Tests `CorrelatedBPDecoderWithCER/tests/`

- `test_bp_algo.jl` — main tests.  **Still contain legacy `train_ballistic_p_…` / `test_ballistic_p_…` filenames.**  They'll fail against the new naming convention unless fixture files are also renamed.  Explicitly out-of-scope for the recent refactor; TODO for later.

---

## 4. Cluster setup — Alliance Canada Narval

The primary compute environment.  Notes I need you to internalise:

**Accounts:**
- `def-jemerson` — CPU jobs (training runs).
- `def-jemerson_gpu` — GPU jobs (test runs).  These are different accounts on Alliance; don't submit GPU jobs to `def-jemerson` (it'll queue forever) or CPU jobs to `def-jemerson_gpu` (wastes GPU allocation).

**Bundle ratios (Narval, per whole A100):**
- 12 CPU cores
- 124 GB RAM
- One A100-SXM4-40GB.  Can be sliced into MIG partitions of `a100_1g.5gb`, `a100_2g.10gb`, `a100_3g.20gb`.

For MIG requests, always scale cores/mem proportionally.  For a `a100_3g.20gb` (3/7 of a whole A100), reasonable request: 4 cores, ~40 GB.

**Julia + CUDA setup:**
- `module load julia/1.12.5` — the version we target.
- `module load cuda` — system CUDA (currently `cuda/12.2`).  **Required before any `julia --project=…` invocation that will load CUDA.jl.**
- `export JULIA_DEPOT_PATH=$SCRATCH/.julia` — depot lives on `/scratch`, not `/home` (avoids NFS ESHUTDOWN and preserves cache across jobs).
- `LocalPreferences.toml` at project root sets `[CUDA_Runtime_jll] local_toolkit = true` so CUDA.jl uses the system runtime rather than attempting to download an artefact (compute nodes have no internet).
- `USE_GPU=1 GPU_BACKEND=cuda` env vars enable the CUDA path inside `CorrelatedBPDecoderWithCER`.

**SCRATCH paths:**
- `/scratch/debankan/CorrelatedBPDecoderWithCER/` — the project checkout.
- `/scratch/debankan/CorrelatedBPDecoderWithCER/data/72q_BB_…/` — codename dirs containing `training_data/`, `testing_data/`, `correlated_weights/`, `assigned_probabilities/`, `models/`, `results/`, `cluster/`, `logs/`, `plots/`, `code/`.
- `$SLURM_TMPDIR` — node-local fast storage.  Data staged in via tar-pipe at job start; wiped when SLURM cleans up the cgroup.  Never rely on it surviving job cancellation.

---

## 5. Canonical formats now in force

**Filename tag for a `(p, q)` pair.**  Produced by `fmt_probs`:

```julia
fmt_probs(0.01, 0.001)   # → "p_0.010_q_0.001"    (max-decimals padding)
fmt_probs(0.1,  0.05)    # → "p_0.10_q_0.05"
```

Applies to training data (`train_p_…_s_N.txt`), test data (`test_p_…_s_N.txt`), correlation-strengths files (`correlated_weights_p_…_s_N.txt`), trained-weight JSONs (`neuralbp_weights_..._trained_using_train_p_…_s_N.json`), and per-sample result CSVs.  `fmt_probs` lives in `src/utils.jl` (canonical) and is duplicated inline in `expts/submission/batch_commands.jl` (needed by the login-node driver that can't `using` the main package).

**`ballistic_` prefix is gone.**  Legacy filenames like `train_ballistic_p_…`, `test_ballistic_p_…`, `simulation_results_test_ballistic_p_…`, `decoder_statistics_ballistic.csv` have been renamed:

- `train_ballistic_p_…` → `train_p_…`
- `test_ballistic_p_…` → `test_p_…`
- `simulation_results_test_ballistic_p_…_trained_using_train_ballistic_p_…` → `simulation_results_test_p_…_trained_using_train_p_…`
- `decoder_statistics_ballistic.csv` → `decoder_statistics_correlated.csv`
- `standard_decoder_statistics_ballistic.csv` → `standard_decoder_statistics_correlated.csv`

**Corresponding function renames:**

- `collect_decoder_statistics_for_ballistic_data` → `collect_decoder_statistics_correlated`
- `collect_standard_decoder_statistics_for_ballistic_data` → `collect_standard_decoder_statistics_correlated`

**On-disk migration script:** `expts/scripts/rename_to_padded_pq.py`.  Dry-run by default:

```bash
# Preview padding-only:
python3 expts/scripts/rename_to_padded_pq.py <codename_dir>

# Apply padding + strip _ballistic:
python3 expts/scripts/rename_to_padded_pq.py <codename_dir> --apply --strip-ballistic

# Also rename matching directory basenames (bottom-up):
python3 expts/scripts/rename_to_padded_pq.py <codename_dir> --apply --strip-ballistic --rename-dirs
```

The script only walks the tree it's pointed at — it will not touch `data/` unless I explicitly point it there.

---

## 6. Cluster failure modes and their fixes

Every one of these caused real wasted wall time before it got fixed.  Please DO NOT re-introduce any of them:

- **`lld SIGBUS` on login nodes** during Julia precompile.  Login nodes throttle memory / CPU aggressively.  Never run `Pkg.precompile()` on a login node — always `salloc` a compute node first.  This is the root reason `batch_run.jl` avoids `using CorrelatedBPDecoderWithCER`.
- **NFS ESHUTDOWN on `/home`** for the Julia depot.  Moved to `$SCRATCH/.julia` in the login-shell `.bashrc`.  Never store the depot on `/home`.
- **CUDA_Runtime_jll artefact download timeout on compute nodes** (no internet).  Fixed with `LocalPreferences.toml`:

  ```toml
  [CUDA_Runtime_jll]
  local_toolkit = true
  ```

  Pre-downloading the JLL artefact on a login node (which does have internet) was also done once.
- **MIG partition CUDA_VISIBLE_DEVICES override broke device visibility.**  For a MIG allocation SLURM exposes the partition via a UUID (`MIG-GPU-…`).  Overriding with an integer index like `0` hides the MIG partition from the CUDA runtime and julia silently falls back to CPU.  The current `slurm.jl` handles this correctly: for `n_gpus_per_node > 1` it captures SLURM's original list into `SLURM_CUDA_VISIBLE_DEVICES` and each parallel worker slices its own field with `cut -d, -f{%}` — works for both integer indices (whole GPUs) and MIG UUIDs.
- **OOM on MIG 1g.5gb (16 GB) with Julia + Flux + Zygote + Plots (~4.6 GB RES per worker).**  Fixed by removing Flux, Zygote, Plots, StatsPlots from the main package's deps.  Enzyme is the sole autodiff; Optimisers + Functors give us parameter iteration/updates without Flux.  `PlotsForBPDecoder` companion package owns Plots/StatsPlots.  RES per worker dropped to ~1.4 GB.
- **5-hour wall-time cancellation wiping `$SLURM_TMPDIR` before stage-out.**  Fixed with `#SBATCH --signal=B:TERM@300` plus a `trap term_handler TERM` in the SLURM script.  `term_handler` runs the `stage_out` bash function (tar-pipes `results/`, `models/`, `cluster/logs/` back to `$SCRATCH`) then `exit 0`s to short-circuit the wait on `parallel`.  A second `trap stage_out EXIT` fires on normal completion; both share an idempotency guard.  If your codename's `results/models/` trees grow large, bump the `@300` to `@600`.
- **Stale `CUDA_Runtime_jll` `.ji` from a prior killed job.**  `Pkg.precompile()` sees source hash unchanged and skips it; runtime `CUDA.functional()` returns false on an otherwise-healthy compute node.  Fixed in `slurm.jl` by force-recompiling `CUDA_Runtime_jll` via `Base.compilecache(pkg)` before `Pkg.precompile()`:

  ```julia
  pkg = Base.PkgId(Base.UUID("76a88914-d11a-5bdc-97e0-2f5a05c973a2"),
                   "CUDA_Runtime_jll")
  Base.compilecache(pkg)
  ```

  Costs 10–70 s at job start; guarantees a fresh JLL every time.
- **Silent-fallback failure when the CUDA sanity check errors.**  Old script's `@assert CUDA.functional()` fired but bash continued (no `set -e`), so `parallel` still ran 64 doomed workers.  Fixed by wrapping the sanity check in `if ! julia -e '…'; then <heredoc>; exit 1; fi`.  The heredoc dumps a diagnostic block into `.err` naming the three likely causes and the manual-recovery commands.

**Manual recovery if the shared depot gets into a bad state:**

```bash
salloc --account=def-jemerson_gpu \
       --gpus-per-node=a100_3g.20gb:1 \
       --time=00:30:00 --cpus-per-task=1 --mem-per-gpu=8G

# On the compute node:
module load julia/1.12.5 cuda
export JULIA_DEPOT_PATH=$SCRATCH/.julia
export USE_GPU=1 GPU_BACKEND=cuda
cd /scratch/debankan/CorrelatedBPDecoderWithCER

# Nuclear option — nuke all CUDA-related caches:
rm -rf $SCRATCH/.julia/compiled/v1.12/CUDA*
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. -e 'using CUDA; @assert CUDA.functional(); println(CUDA.name(CUDA.device()))'
```

---

## 7. Loose ends the last session flagged but didn't clean up

- `PlotsForBPDecoder/src/utils.jl` — was reduced to a stub comment.  Should be `rm`ed on the next branch cleanup (couldn't do it from that session due to a permissions issue on the file tools).
- `tests/test_bp_algo.jl:114, 173` — still hardcodes `train_ballistic_p_0.01_q_0.001_s_1.txt` / `test_ballistic_p_0.01_q_0.001_s_1.txt`.  Either rename the fixture files and update the tests, or delete the tests if they no longer represent something we care about.
- `refs/*.py`, `expts/sweep_epochs.sh`, `expts/run_multiple_epochs.sh`, `expts/error_analysis.jl`, `expts/profile_training.jl`, `expts/neural_vs_standard.jl` — legacy scripts with hardcoded `train_ballistic_p_…` / `test_ballistic_p_…` string literals.  Historical; touch only if I ask.
- `PlotsForBPDecoder/src/PlotsForBPDecoder.jl:17` docstring still says `decoder_statistics_ballistic.csv` in a usage example.  Cosmetic — fine to update in passing.

---

## 8. Where we left the last conversation

We had just:

1. Renamed the postprocessing collectors and dropped `_ballistic` from output CSV filenames.
2. Fixed a runtime regex in `extract_parameters_from_description` that was still matching `test_ballistic_p_…`.
3. Extended `expts/scripts/rename_to_padded_pq.py` with a `--strip-ballistic` flag.
4. Diagnosed a CUDA failure on a 2-GPU-per-node job caused by a stale `CUDA_Runtime_jll` cache.
5. Added three hardening measures to `expts/submission/slurm.jl`:
   - Force-recompile `CUDA_Runtime_jll` before `Pkg.precompile()`.
   - Strict CUDA sanity check with `exit 1` and a diagnostic heredoc.
   - MIG-safe per-slot `CUDA_VISIBLE_DEVICES` via `cut -d, -f{%}` from `SLURM_CUDA_VISIBLE_DEVICES`.

Next actions I have queued:

- Resubmit a `--gpus-per-node=a100_3g.20gb:2` test job with the new `slurm.jl`, watch for the `[cuda] force-recompiling CUDA_Runtime_jll on <hostname>...` and `[cuda] per-slot slicing from SLURM_CUDA_VISIBLE_DEVICES=…` lines in `.out`.
- Once the 2-GPU case is stable, sweep across more `(p, q)` pairs.
- Loose-end cleanup from §7 when I have time.

If I mention "the crash on job 64946651" or "the 5-hour timeout on 2026-07-08," those are the concrete incidents that motivated fixes in §6.

---

## 9. How to work with me

Things that help:

- **Ask before doing multi-step work.**  Use the AskUserQuestion tool when scope is ambiguous.  I'd rather answer one question up front than have you redo something after.
- **Show me the evidence, then the interpretation.**  For debugging cluster runs I'll paste `.out` / `.err` / `nvidia-smi` output.  Reason from what's actually visible, not from what you'd expect to be there.  When you're speculating, say so.
- **Write comments that explain WHY, not what.**  The code is usually clear enough about "what."  It's the "why did we choose this over the obvious alternative" that saves the next reader an hour.
- **Track multi-step work with TodoList.**  It renders as a widget in Cowork mode and helps me follow along.
- **Verify with a test where possible.**  Even a small bash-sandbox rehearsal (as we did for the rename-script regex and the `cut -d,` slicing) catches surprises cheaply.
- **When you make a mistake, own it and move on.**  Don't spiral into apology.  I'd rather have you say "you're right, my read was wrong, here's the corrected picture."

Things that don't help:

- Excessive summarising after each edit.  I can read the diffs.
- Guessing at cluster behaviour instead of asking me to run a diagnostic.
- Reading `data/` files "for context."  Really, don't.

---

## 10. Tools you have

Same Cowork mode setup I use in the other session:

- **File tools** (Read, Write, Edit) directly against the workspace folder I've mounted.
- **Shell** — a sandboxed Linux VM with the mounted folders visible under `/sessions/<name>/mnt/…`.  No `julia` in the sandbox by default; use it for python/bash/rg tasks.
- **TodoList** (TaskCreate / TaskUpdate / TaskList) — please use this for anything longer than one edit.
- **AskUserQuestion** — for clarifying questions.
- **Memory system** — you have a persistent memory dir.  Save memories about me and the project as you learn; don't save data-directory paths or ephemeral cluster job IDs.
- **Skills** — docx, xlsx, pptx, pdf all available if I ask for a document.  Otherwise stay in the code.

You do not have `julia` in the sandbox, so you can't run the code directly.  You can read/edit files and reason about them; when you need to validate behaviour, either ask me to run a command on the cluster, or (like we did for the rename script) simulate the mechanism in bash/python.

---

That's the state of things.  Ready when you are.

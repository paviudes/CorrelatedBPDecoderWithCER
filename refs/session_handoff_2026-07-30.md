# Session handoff — 2026-07-30

Context document for continuing work in a new chat. Covers code changes landed this
session, the current science picture, and open threads (most importantly the
correlation-loss redesign discussion at the end).

## Code changes landed (all in working tree)

1. **Standard-error fix.** `NeuralBPDecoderStatistics` scalar constructor
   (`src/postprocessing.jl:82`) now divides by `num_samples_per_error_rate`
   (was `n_layers` — inflated every neural std by ×100 at 10^6 samples).
   `compute_std_assuming_bernoulli` itself was always correct: `sqrt(mu(1-mu)/N)`
   is the SEM; no missing `sqrt(N)`/Bessel factor. One-off repair script for old
   CSVs: `expts/misc/fix_neuralbp_stderr.sh <results-folder> ...` (divides the
   std column by exactly 100; only touches
   `simulation_results_test_p_*_s_1_nlayers_100_epochs_20_trained_using_train_p_*_s_1.csv`
   with N=10^6 rows; writes `.bak`, idempotent). NOTE: the structurally identical
   bug in `src/legacy.jl:593` (`StandardBPDecoderStatistics` scalar ctor, uses
   `num_iterations_BP`) was deliberately left as behaviour-preserving — the
   standard gather path (legacy.jl:499) is correct.

2. **Three-way decoder comparison.** `expts/postprocess.jl`: gather handler is
   now decoder-generic (`run_standard_gather`) registered under `standardbp_gather`
   (BP+OSD files) and new `bp_gather` (plain `*_BP_failure_rates.txt` files);
   canonical output key `output_file` (legacy `output_bposd_file` still accepted).
   `PlotsForBPDecoder/src/plot_standard_vs_neural.jl`: optional `bp_dataframes`
   / `bp_label` — three line styles (solid = BP-OSD, dotted = BP, dashed = Neural),
   gain baseline = BP when present else BP-OSD, comparison CSV gains
   `average_logical_error_rate_BP`, `standard_error_BP`, `gain_baseline` columns.
   Gain-panel y-label reflects the baseline: `$\Delta_{\text{BP}}$` /
   `$\Delta_{\text{BP-OSD}}$`. Control panel (`expts/postprocess_configs/
   postprocess_panel.toml`): `[bp_gather]` section (19 plain-BP 72q files ->
   `decoder_statistics_BP.csv`), `bp_csv`/`bp_plot_label` keys in
   `[plot_standard_vs_neural]` (commented out until 18q BP data exists),
   `bp_gather` added to `gather`/`everything` task groups. All TOML panels now
   live in `expts/postprocess_configs/`.

3. **`use_CER` switch** (hyperparameters TOML key, default `true`).
   - `use_CER = false` => run pretends `correlated_weights/` doesn't exist:
     preset p=0.1 priors (`log(9)` LLRs), empty connectivity => `is_correlated=false`
     => correlation loss term dropped. Gate lives in `load_base_BP_model`
     (`src/log_model.jl`, `use_cer` kwarg).
   - Every output tagged `_no_cer` (results CSV in `expts/neural_bp_experiments.jl`,
     weights JSON in `src/train.jl`, preflight reconstruction in
     `expts/submission/batch_commands.jl` — all three verified identical).
   - Submission preflight (`expts/batch_run.jl`): with `use_CER=true`, errors
     before generating anything if `<data>/<codename>/correlated_weights/` is
     missing; with `false` the folder is not required.
   - Ready-made config: `data/update_23July2026/72q_BB_cycles_1/models/
     hyperparams_epochs_20_no_cer.toml` (copy of epochs_20 + `use_CER = false`).
   - Docs: README table row + directory-structure note; default documented in
     `src/command_line.jl` `parse_hyper_parameters`.

## Science picture (current)

- **Metric fix (earlier session, foundational):** `check_bp_solutions` commits to
  the FIRST syndrome-clearing layer (H only), then scores that layer against the
  logicals L (from `parity_check_matrix_dual = [H; L]`). No "any layer" oracle.
  With the fix, at 10^6 test samples: **18q** neural ~ tied with BP-OSD (gains
  0.98-1.25x, median ~1.09x); **72q** neural beats BP-OSD (median ~1.9x, up to
  ~6x at low p; vs plain BP the gains are larger still, e.g. ~13x at p=1e-4).
- **train@p vs train@max(p,0.002):** statistical wash. Only p<0.002 differs;
  pooled failures 23,443 vs 23,712 (z=1.24, n.s.). Per-p winners flip sign
  (`+++-+--+-+`), adjacent-p reversals with |z|~8-9 both ways => single-seed
  training variance dominates. 0.002 is NOT a sweet spot; train@p actually wins
  at the lowest p (z=3.6-7.8), contradicting the "train higher to avoid zeros"
  intuition (at 72q, even p=1e-4 gives ~1400-2900 non-trivial samples over
  20x10^4 draws). Conclusion: any strategy ranking needs K~5-10 training seeds
  per (strategy, p).
- **6-minute "12h" train job (90q, job 66607298):** total was ~6m15s of which
  ~4.5 min was CUDA precompile + 19s stage-in => commands ran seconds each.
  Almost certainly the skip path: weights exist + `retrain=false` =>
  `train_Nachmani_neuralbp` loads instead of training; train mode has no
  `--test` so it exits immediately. Diagnose via
  `find data/90q_BB_cycles_1/cluster/logs -name stdout|stderr` and
  `ls -la models/*.json` timestamps. Weights filename encodes only
  nlayers/epochs/training_source — changing other hyperparams silently reuses
  old weights; `retrain = true` forces fresh training.

## Correlation-loss redesign (open discussion — main thread to continue)

Current term (`src/loss.jl:104-170`):
`L_corr = exp(-beta*|s|) * sum_(i,k in C) lambda_ik (sigma_i - sigma_k)^2`,
where |s| is the SOFT sine-residue syndrome weight of the residual computed from
`sigma(mu)` (=> the gate is DIFFERENTIABLE), and beta anneals 0->1
(`correlation_syndrome_importance = "0.0,1.0,0.7,up"`). Observation: adding
correlations in this form DEGRADES performance. Analysis agreed on:

1. **Gate selects the wrong samples.** Weight=1 on already-solved samples (where
   syndrome gradient ~0, so the pull perturbs correct answers — worst for pairs
   with isolated flips, which are common since correlation != determinism);
   weight~0 on failing samples where prior guidance could help. At low p solved
   samples dominate => term mostly corrupts good solutions.
2. **Perverse gradient through the gate.** d/d|s| [exp(-beta|s|) L_corr] < 0:
   the network is rewarded for INCREASING |s| (breaking the syndrome) to escape
   the correlation penalty; annealing beta up makes it worst late in training.
3. **Bayesian view: don't gate on |s| at all.** Syndrome term = likelihood,
   correlation term = prior; prior applies unconditionally. If gating at all,
   the flipped, bounded, DETACHED form (w = 1 - exp(-beta|s|) or indicator
   1[|s|>0], no gradient through the gate) — never unbounded w ~ |s|.
4. **Functional form is the deeper bug.** (sigma_i - sigma_k)^2 = self-terms
   (extra sparsity push, double-counts sparsity term, punishes isolated flips —
   a WRONG prior) + co-activation. Principled replacement is the Ising log-prior
   co-activation term: `L_prior = - sum J_ik sigma_i sigma_k` with
   `J_ik = log[P11*P00/(P10*P01)]` from CER data (single-qubit rates already in
   initial LLRs). Indifferent to isolated flips, rewards joint activation.
5. **Training data already carries correlations implicitly** (samples drawn from
   the correlated model), so the explicit term is a regularizer, not the sole
   CER carrier. Proposed ablation (multi-seed, since single-seed deltas are
   noise-dominated): (a) alpha4=0, (b) current form, (c) Ising form with beta=0
   + alpha4 sweep {1e-3,1e-2,1e-1,1}, (d) use_CER=false corner.

**Standing offer (not yet confirmed):** implement the Ising-form term behind a
`correlation_form` hyperparameter (legacy default preserved).

## Other open threads

- Verify the 6-min job diagnosis from per-command parallel logs (on narval).
- `legacy.jl:593` std bug — align or keep as documented legacy behaviour.
- Optional: conditional CUDA force-recompile in `expts/submission/slurm.jl`;
  pin CUDA version in LocalPreferences.toml.
- 18q plain-BP failure files don't exist yet (BP dotted line currently 72q-only).

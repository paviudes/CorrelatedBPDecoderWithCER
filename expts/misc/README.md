# expts/misc

Miscellaneous / legacy helper scripts, moved out of `expts/` to keep the main
folder focused.

**Run these from the `expts/` directory, not from inside `misc/`.** They use
cwd-relative paths like `./../data`, so they resolve correctly only when the
working directory is `expts/`. For example:

```bash
cd expts
bash misc/reformat_standard_bp_failures.py --help    # (python3, see file)
python3 misc/reformat_standard_bp_failures.py <dir>
julia --project=expts misc/error_analysis.jl --analysis correlations
bash misc/sweep_epochs.sh
```

Contents:

- `error_analysis.jl` — error-weight and CER-correlation analysis (`--analysis`).
- `neural_vs_standard.jl` — neural-vs-standard comparison helpers.
- `explicit_errors.jl` — explicit error-model driver (also `include`d by
  `../quantum_BP_test.jl` as `misc/explicit_errors.jl`).
- `reformat_standard_bp_failures.py` — aggregate per-p BP-OSD failure files into
  the standard-decoder results format.
- `rename_to_padded_pq.py` — one-shot on-disk filename padding / `_ballistic`
  strip migration.
- `sweep_epochs.sh`, `run_multiple_epochs.sh` — epoch-sweep launchers.

[hp_sweep] 60 training point(s), 60 test point(s)
  test sets -> each model on its own training key's test set (test_keys = [])
  datasets  -> p_0.0007_sig_0.0_s_1 p_0.0011_sig_0.0_s_1 p_0.0015_sig_0.0_s_1
  seeds     -> 1 2 3 4 5
  check node-> tanh enriched:alpha_star:learn   (no-CER baseline: false)
  optimizer -> rampfirst:loss_layer_selection=ramp,commit_layer_rule=first
               ramplast:loss_layer_selection=ramp,commit_layer_rule=last
  layers    -> loss_layer_selection=softmin  commit_layer_rule=first  (the default; an optimizer arm may override either)
  cluster   -> trillium (clusters/trillium.toml)
  network   -> 90 layers (fixed), manual_backprop = true (hand-written gradient)
  train     -> the default account: array 0-14 (15 tasks x 1x h100, 24 cpu,
               188G host ram, both with the card), 4 trainings at a time per card, USE_GPU=1, 3:00:00
  test      -> the default account: array 0-7 (8 tasks x 1x h100 (80G vram), 24 cpu),
               188G host ram, both with the card, GPU_MEMORY=69632M, 1 at a time, USE_GPU=1, 1:00:00
  depot     -> JULIA_DEPOT_PATH=/scratch/debankan/.julia: (written into the jobs)
  commands  -> ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_06-36-17.txt
               ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_06-36-17.txt

submit — TRAIN, then TEST chained on it (both GPU, the default account, from trig-login01 and from this directory):

    TRAIN_ID=$(sbatch --parsable ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_06-36-17.sh)
    sbatch --dependency=afterok:$TRAIN_ID --kill-on-invalid-dep=yes ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_06-36-17.sh

  Or one at a time, checking the models trained before testing them:
    sbatch ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_06-36-17.sh
    julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_sweep_filtered/models/neuralbp_weights_nlayers_90_epochs_10_trained_using_train_p_0.0007_sig_0.0_s_1_cer_rampfirst_seed_1.json");
              println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
    # ~ initial_conditions_scale / sqrt(3) => never trained; larger => trained.
    sbatch ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_06-36-17.sh

  afterok on an array is ALL-OR-NOTHING: every one of the 15 train tasks must
  exit 0, so one failed point stops all 60 tests. --kill-on-invalid-dep=yes
  then cancels the test job outright instead of parking it in the queue as
  DependencyNeverSatisfied. A NaN-rolled-back epoch is NOT a failure (the
  point still exits 0 and writes weights), so rollbacks do not block testing.
[debankan@tri-login02 expts]$ 

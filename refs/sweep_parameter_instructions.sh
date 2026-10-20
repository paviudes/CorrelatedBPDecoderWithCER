[debankan@narval1 expts]$ bash sweep_hyperparams.sh --settings sweeps/enriched_vs_nocer.toml --no-edit
[hp_sweep] settings from: sweeps/enriched_vs_nocer.toml
[hp_sweep]   copied to:   /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/hp_sweep_settings_2026-10-19_06-45-21.toml

[hp_sweep] walltime preflight (129s/test, 1500s/train point, 900s startup)
  train     4 per task   est 0h40m of 3:00:00   (22% of budget)
  test     10 per task   est 0h36m of 4:00:00   (15% of budget)

[hp_sweep] 20 training point(s), 80 test point(s)
  test sets -> 4 key(s), every model scored on every one
  datasets  -> p_0.0015_sig_0.0015_s_1
  ref       ->    (CER tanh and no-CER only)
  seeds     -> 1  2  3  4  5
  check node-> enriched:0.503:learn   (no-CER baseline: true)
  optimizer -> softmin_first:loss_layer_selection=softmin,commit_layer_rule=first,warmup_layers=0,initial_conditions_scale=0.3
ramp_last:loss_layer_selection=ramp,commit_layer_rule=last,warmup_layers=5,loss_layer_ramp_sharpness=3.0,initial_conditions_scale=0.3
  layers    -> loss_layer_selection=softmin  commit_layer_rule=first
  cluster   -> narval
  network   -> 90 layers (fixed), manual_backprop = true (hand-written gradient)
  train     -> def-jemerson_gpu: array 0-4 (5 tasks x 1x a100, 12 cpu,
               --mem-per-gpu=32G host ram), 4 trainings at a time per card, USE_GPU=1, 3:00:00
  test      -> def-jemerson_gpu: array 0-7 (8 tasks x 1x a100 (40G vram), 12 cpu),
               --mem-per-gpu=32G host ram, GPU_MEMORY=34816M, 1 at a time, USE_GPU=1, 4:00:00
  commands  -> ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-19_06-45-21.txt
               ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-19_06-45-21.txt

submit — TRAIN, then TEST chained on it (both GPU, def-jemerson_gpu):

    TRAIN_ID=$(sbatch --parsable ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-19_06-45-21.sh)
    sbatch --dependency=afterok:$TRAIN_ID --kill-on-invalid-dep=yes ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-19_06-45-21.sh

  Or one at a time, checking the models trained before testing them:
    sbatch ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-19_06-45-21.sh
    julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_trainable_alpha/models/neuralbp_weights_nlayers_90_epochs_5_trained_using_train_p_0.0015_sig_0.0015_s_1_cer_enr_softmin_first_seed_1.json");
              println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
    # ~ initial_conditions_scale / sqrt(3) => never trained; larger => trained.
    sbatch ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-19_06-45-21.sh

  afterok on an array is ALL-OR-NOTHING: every one of the 5 train tasks must
  exit 0, so one failed point stops all 80 tests. --kill-on-invalid-dep=yes
  then cancels the test job outright instead of parking it in the queue as
  DependencyNeverSatisfied. A NaN-rolled-back epoch is NOT a failure (the
  point still exits 0 and writes weights), so rollbacks do not block testing.
[debankan@narval1 expts]$ 

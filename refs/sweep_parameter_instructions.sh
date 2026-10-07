[debankan@narval2 expts]$ bash sweep_hyperparams.sh --no-edit
[hp_sweep] wrote defaults to: /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/hp_sweep_settings_2026-10-05_04-15-32.toml

[hp_sweep] walltime preflight (129s/test, 6600s/train point, 900s startup)
  train     8 per task   est 2h05m of 4:00:00   (52% of budget)
  test     20 per task   est 0h58m of 4:00:00   (24% of budget)

[hp_sweep] 40 training point(s), 160 test point(s)
  test sets -> 4 key(s), every model scored on every one
  datasets  -> p_0.0015_sig_0.0015_s_1
  ref       ->    (CER tanh and no-CER only)
  seeds     -> 1  2  3  4  5
  check node-> tanh  enriched:0.42:learn  enriched:0.42:learn:step:12:3:learn   (no-CER baseline: true)
  optimizer -> designA:loss_layer_selection=last,commit_layer_rule=last,initial_conditions_scale=0.3
histpair:loss_layer_selection=softmin,commit_layer_rule=first,initial_conditions_scale=0.3
  layers    -> loss_layer_selection=last  commit_layer_rule=last
  cluster   -> narval
  train     -> def-jemerson: array 0-4 (5 x 54 cpu x 6G = 324G/node), 4:00:00
  test      -> def-jemerson_gpu: array 0-7 (8 tasks x 1x a100 (40G vram), 12 cpu),
               --mem-per-gpu=32G host ram, GPU_MEMORY=34816M, 1 at a time
  commands  -> ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-05_04-15-32.txt
               ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-05_04-15-32.txt

submit — TRAIN first (CPU, def-jemerson), then TEST (GPU, def-jemerson_gpu):

  # 1. training
  sbatch ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-05_04-15-32.sh

  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU:
  julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_trainable_alpha/models/neuralbp_weights_nlayers_90_epochs_5_trained_using_train_p_0.0015_sig_0.0015_s_1_hpcer_seed_1.json");
            println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
  # 0.058 => never trained (every batch NaN-skipped); larger => trained.

  # 3. testing
  sbatch ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-05_04-15-32.sh

  To chain them without the check instead:
    TRAIN=$(sbatch --parsable ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_train_2026-10-05_04-15-32.sh)
    sbatch --dependency=afterok:$TRAIN ./../data/72q_BB_cycles_1_trainable_alpha/cluster/hp_sweep_test_2026-10-05_04-15-32.sh
[debankan@narval2 expts]$ 

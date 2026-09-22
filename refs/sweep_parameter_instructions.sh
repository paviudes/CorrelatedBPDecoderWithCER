[xf_sweep] wrote defaults to: /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/xf_sweep_settings_2026-09-21_07-33-23.toml

[xf_sweep] 60 model(s), 320 test(s)
  train on  -> p_0.0015_sig_0.0015_s_1  p_0.0015_sig_0.0015_s_2  p_0.0015_sig_0.0015_s_3
  seeds     -> 1  2  3  4  5
  arms      -> tanh  enriched:0.42:fixed  enriched:0.42:learn   (no-CER baseline: true)
  matched   -> 280 test(s)
  probe     -> 40 test(s): p_0.0015_sig_0.0015_s_1 weights on p_0.0015_sig_0.0015_s_2  p_0.0015_sig_0.0015_s_3
  cluster   -> narval
  train     -> def-jemerson: array 0-4 (5 x 54 cpu x 6G = 324G/node), 4:00:00
               ~12 model(s) per task
  test      -> def-jemerson_gpu: array 0-15 (16 tasks x 1x a100 (40G vram), 12 cpu),
               --mem-per-gpu=32G host ram, GPU_MEMORY=34816M, 1 at a time
               ~20 test(s) per task
  commands  -> ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_train_2026-09-21_07-33-23.txt
               ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_test_2026-09-21_07-33-23.txt
  pairs     -> ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_pairs_2026-09-21_07-33-23.txt   (kind / train / test / arm / seed)

submit — TRAIN first (CPU, def-jemerson), then TEST (GPU, def-jemerson_gpu):

  # 1. training
  sbatch ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_train_2026-09-21_07-33-23.sh

  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU.
  #    (a) did the weights move at all?
  julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_soft_constraints/models/neuralbp_weights_nlayers_90_epochs_5_trained_using_train_p_0.0015_sig_0.0015_s_1_xfcer_seed_1.json");
            println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
  # 0.058 => never trained (every batch NaN-skipped); larger => trained.

  #    (b) did the enriched arm DIVERGE? base loss should fall across epochs,
  #        not climb. On the old dataset it went 0.011 -> 4.21 by epoch 5.
  awk -F, 'NR>1 && $3=="80"' ./../data/72q_BB_cycles_1_soft_constraints/logs/debugging_train_p_0.0015_sig_0.0015_s_1_xfcer_cnenr0p42F_seed_1_individual_losses.csv | head -1
  # If it climbs, lower learning_rate in hyperparams_epochs_5_corrs.toml and re-run this generator.

  # 3. testing
  sbatch ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_test_2026-09-21_07-33-23.sh

  To chain them without the check instead:
    TRAIN=$(sbatch --parsable ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_train_2026-09-21_07-33-23.sh)
    sbatch --dependency=afterok:$TRAIN ./../data/72q_BB_cycles_1_soft_constraints/cluster/xf_sweep_test_2026-09-21_07-33-23.sh
[debankan@narval2 expts]$ 

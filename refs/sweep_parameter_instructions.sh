[debankan@narval1 expts]$ bash sweep_hyperparams.sh --no-edit
[hp_sweep] wrote defaults to: /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/hp_sweep_settings_2026-09-16_23-44-01.toml

[hp_sweep] 60 point(s)
  datasets  -> p_0.0005_sig_0.001_s_1  p_0.0005_sig_0.001_s_2  p_0.0005_sig_0.001_s_3
  ref       ->    (CER tanh and no-CER only)
  seeds     -> 1  2  3  4  5
  check node-> tanh  enriched:0.42:fixed  enriched:0.42:learn   (no-CER baseline: true)
  cluster   -> narval
  train     -> def-jemerson: array 0-4 (5 x 54 cpu x 6G = 324G/node), 4:00:00
  test      -> def-jemerson_gpu: array 0-7 (8 tasks x 1x a100 (40G vram), 12 cpu),
               --mem-per-gpu=32G host ram, GPU_MEMORY=34816M, 1 at a time
  commands  -> ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-09-16_23-44-01.txt
               ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-09-16_23-44-01.txt

submit — TRAIN first (CPU, def-jemerson), then TEST (GPU, def-jemerson_gpu):

  # 1. training
  sbatch ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-09-16_23-44-01.sh

  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU:
  julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_spread_comparison/models/neuralbp_weights_nlayers_90_epochs_5_trained_using_train_p_0.0005_sig_0.001_s_1_hpcer_seed_1.json");
            println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
  # 0.058 => never trained (every batch NaN-skipped); larger => trained.

  # 3. testing
  sbatch ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-09-16_23-44-01.sh

  To chain them without the check instead:
    TRAIN=$(sbatch --parsable ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-09-16_23-44-01.sh)
    sbatch --dependency=afterok:$TRAIN ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-09-16_23-44-01.sh

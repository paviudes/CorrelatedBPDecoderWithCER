[debankan@narval1 expts]$ bash sweep_hyperparams.sh --no-edit
[hp_sweep] wrote defaults to: /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/hp_sweep_settings_2026-08-27_14-58-37.toml

[hp_sweep] 27 point(s)
  datasets  -> p_0.0005_sig_0.001_s_1  p_0.0005_sig_0.001_s_2  p_0.0005_sig_0.001_s_3
  ref       ->    (lambda = 0 and no-CER only)
  lambdas   -> 0.0  3.0   seeds -> 1
  gates     -> tau = 0.5  4.0  1e6 (swept);  certainty c = 2.2;  sparsity = 0.0
  train     -> def-jemerson: 27 cpu x 6G, 4:00:00
  test      -> def-jemerson_gpu: 1x a100_3g.20gb (20G vram), 6 cpu,
               --mem-per-gpu=16G host ram, GPU_MEMORY=17408M, 1 at a time
  commands  -> ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-08-27_14-58-37.txt
               ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-08-27_14-58-37.txt

submit — TRAIN first (CPU, def-jemerson), then TEST (GPU, def-jemerson_gpu):

  # 1. training
  sbatch ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-08-27_14-58-37.sh

  # 2. when it finishes, CHECK THE MODELS TRAINED before spending a GPU:
  julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_spread_comparison/models/neuralbp_weights_nlayers_90_epochs_5_trained_using_train_p_0.0005_sig_0.001_s_1_hpcer_sp0p0_lam3p0_seed_1.json");
            println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
  # 0.058 => never trained (every batch NaN-skipped); larger => trained.

  # 3. testing
  sbatch ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-08-27_14-58-37.sh

  To chain them without the check instead:
    TRAIN=$(sbatch --parsable ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_train_2026-08-27_14-58-37.sh)
    sbatch --dependency=afterok:$TRAIN ./../data/72q_BB_cycles_1_spread_comparison/cluster/hp_sweep_test_2026-08-27_14-58-37.sh
[debankan@narval1 expts]$ 

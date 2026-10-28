[debankan@trig-login01 expts]$ bash sweep_hyperparams.sh sweeps/sig0_lr_init_fwb.toml
[hp_sweep] enriched:alpha_star:learn starts each point from alpha* of its training key:
             p_0.0015_sig_0.0_s_1 -> 0.4875
[hp_sweep] note: loss_layer_selection=ramp with warmup_layers=0:
           the ramp weights the BP transient (layers 1-5) too. Measured knee is 5.
[hp_sweep] test sets: test_keys = [], so each model is scored on the test set of its own
           training key only:
             train_p_0.0015_sig_0.0_s_1.txt -> test_p_0.0015_sig_0.0_s_1.txt   (144 test(s))

[hp_sweep] walltime preflight (trillium: 900s startup, 150s per test;
           a training wave of 4 = 10 epochs x 300s (enriched) = 3000s)
  train     4 per task   est 1h05m of 2:00:00   (54% of budget)
  test      8 per task   est 0h35m of 1:00:00   (58% of budget)

[hp_sweep] 144 training point(s), 144 test point(s)
  test sets -> each model on its own training key's test set (test_keys = [])
  datasets  -> p_0.0015_sig_0.0_s_1
  seeds     -> 1 2 3
  check node-> tanh enriched:alpha_star:learn   (no-CER baseline: false)
  optimizer -> ic0p1_fwb3_lr0p001:initial_conditions_scale=0.1,failure_weight_boundary=3,learning_rate=0.001
               ic0p1_fwb1_lr0p001:initial_conditions_scale=0.1,failure_weight_boundary=1,learning_rate=0.001
               ic0p1_fwb2_lr0p001:initial_conditions_scale=0.1,failure_weight_boundary=2,learning_rate=0.001
               ic0p1_fwb4_lr0p001:initial_conditions_scale=0.1,failure_weight_boundary=4,learning_rate=0.001
               ic0p7_fwb1_lr0p001:initial_conditions_scale=0.7,failure_weight_boundary=1,learning_rate=0.001
               ic0p7_fwb2_lr0p001:initial_conditions_scale=0.7,failure_weight_boundary=2,learning_rate=0.001
               ic0p7_fwb3_lr0p001:initial_conditions_scale=0.7,failure_weight_boundary=3,learning_rate=0.001
               ic0p7_fwb4_lr0p001:initial_conditions_scale=0.7,failure_weight_boundary=4,learning_rate=0.001
               ic0p1_fwb1_lr0p01:initial_conditions_scale=0.1,failure_weight_boundary=1,learning_rate=0.01
               ic0p1_fwb2_lr0p01:initial_conditions_scale=0.1,failure_weight_boundary=2,learning_rate=0.01
               ic0p1_fwb3_lr0p01:initial_conditions_scale=0.1,failure_weight_boundary=3,learning_rate=0.01
               ic0p1_fwb4_lr0p01:initial_conditions_scale=0.1,failure_weight_boundary=4,learning_rate=0.01
               ic0p7_fwb1_lr0p01:initial_conditions_scale=0.7,failure_weight_boundary=1,learning_rate=0.01
               ic0p7_fwb2_lr0p01:initial_conditions_scale=0.7,failure_weight_boundary=2,learning_rate=0.01
               ic0p7_fwb3_lr0p01:initial_conditions_scale=0.7,failure_weight_boundary=3,learning_rate=0.01
               ic0p7_fwb4_lr0p01:initial_conditions_scale=0.7,failure_weight_boundary=4,learning_rate=0.01
               ic0p1_fwb1_lr0p1:initial_conditions_scale=0.1,failure_weight_boundary=1,learning_rate=0.1
               ic0p1_fwb2_lr0p1:initial_conditions_scale=0.1,failure_weight_boundary=2,learning_rate=0.1
               ic0p1_fwb3_lr0p1:initial_conditions_scale=0.1,failure_weight_boundary=3,learning_rate=0.1
               ic0p1_fwb4_lr0p1:initial_conditions_scale=0.1,failure_weight_boundary=4,learning_rate=0.1
               ic0p7_fwb1_lr0p1:initial_conditions_scale=0.7,failure_weight_boundary=1,learning_rate=0.1
               ic0p7_fwb2_lr0p1:initial_conditions_scale=0.7,failure_weight_boundary=2,learning_rate=0.1
               ic0p7_fwb3_lr0p1:initial_conditions_scale=0.7,failure_weight_boundary=3,learning_rate=0.1
               ic0p7_fwb4_lr0p1:initial_conditions_scale=0.7,failure_weight_boundary=4,learning_rate=0.1
  layers    -> loss_layer_selection=ramp  commit_layer_rule=first  (the default; an optimizer arm may override either)
  cluster   -> trillium (clusters/trillium.toml)
  network   -> 90 layers (fixed), manual_backprop = true (hand-written gradient)
  train     -> the default account: array 0-36 (37 tasks x 1x h100, 24 cpu,
               188G host ram, both with the card), 4 trainings at a time per card, USE_GPU=1, 2:00:00
  test      -> the default account: array 0-17 (18 tasks x 1x h100 (80G vram), 24 cpu),
               188G host ram, both with the card, GPU_MEMORY=69632M, 1 at a time, USE_GPU=1, 1:00:00
  depot     -> JULIA_DEPOT_PATH=/scratch/debankan/.julia: (written into the jobs)
  commands  -> ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_13-04-21.txt
               ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_13-04-21.txt

submit — TRAIN, then TEST chained on it (both GPU, the default account, from trig-login01 and from this directory):

    TRAIN_ID=$(sbatch --parsable ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_13-04-21.sh)
    sbatch --dependency=afterok:$TRAIN_ID ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_13-04-21.sh

  Or one at a time, checking the models trained before testing them:
    sbatch ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_train_2026-10-27_13-04-21.sh
    julia -e 'using JSON, Statistics; w=JSON.parsefile("./../data/72q_BB_cycles_1_sweep_filtered/models/neuralbp_weights_nlayers_90_epochs_10_trained_using_train_p_0.0015_sig_0.0_s_1_cer_ic0p1_fwb3_lr0p001_seed_1.json");
              println(std(vcat(w["weights_c2v_v2c"],w["weights_llrs"],w["weights_c2v_readout"])))'
    # ~ initial_conditions_scale / sqrt(3) => never trained; larger => trained.
    sbatch ./../data/72q_BB_cycles_1_sweep_filtered/cluster/hp_sweep_test_2026-10-27_13-04-21.sh

  afterok on an array is ALL-OR-NOTHING: every one of the 37 train tasks must
  exit 0, so one failed point stops all 144 tests. trillium's sbatch has no
  --kill-on-invalid-dep, so the test job then stays in the queue as
  DependencyNeverSatisfied until you scancel it. A NaN-rolled-back epoch is NOT
  a failure (the point still exits 0 and writes weights), so rollbacks do not
  block testing.
[debankan@trig-login01 expts]$ 

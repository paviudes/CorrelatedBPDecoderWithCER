[debankan@narval3 expts]$ bash submit.sh
[submit] wrote defaults to: /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/submission_settings_2026-10-20_01-11-12.toml
[submit] opening /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/submission_settings_2026-10-20_01-11-12.toml in nano...

[submit] Will run (from /scratch/debankan/CorrelatedBPDecoderWithCER/expts):
  julia --project=./../ batch_run.jl --settings /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/submission_settings_2026-10-20_01-11-12.toml 

[settings] loading /scratch/debankan/CorrelatedBPDecoderWithCER/expts/scripts/submission_settings_2026-10-20_01-11-12.toml

[72q_BB_cycles_1_sweep] submission 2026-10-20_01-12-59: training job, then the test job chained on it
  trained on : train_p_0.0015_sig_0.003_s_1.txt
  tested on  : 24 dataset(s) = 24 (p, sigma) pair(s) at sample(s) 1, 0 not in the data
  arms       : cer (run_tag _cer), nocer (run_tag _nocer)
  seeds      : 1, 2, 3, 4, 5
  models     : 10, e.g. neuralbp_weights_nlayers_90_epochs_10_trained_using_train_p_0.0015_sig_0.003_s_1_cer_seed_1.json
  settings   : ./../data/72q_BB_cycles_1_sweep/models/submissions/2026-10-20_01-12-59/
[72q_BB_cycles_1_sweep] wrote 10 train command(s) to ./../data/72q_BB_cycles_1_sweep/cluster/commands_train_2026-10-20_01-12-59.txt

=== Job Details ===
  Job              : train_2026-10-20_01-12-59  (train)
  Target codename  : 72q_BB_cycles_1_sweep
  Commands         : 10
  Devices          : GPU (a100), 1 card(s) per task, 4 process(es) per card
  Array tasks      : 3, 4 command(s) each, 4 at a time
  CPUs per task    : 12
  Memory           : 32G per GPU (host RAM)
  Wall time        : 1:00:00
  Estimate         : ~34m per task (1 round(s) of 24m + 10m startup), 57% of 1:00:00

=== Output Files ===
  Commands file    : ./../data/72q_BB_cycles_1_sweep/cluster/commands_train_2026-10-20_01-12-59.txt
  SLURM script     : ./../data/72q_BB_cycles_1_sweep/cluster/train_2026-10-20_01-12-59.sh
[72q_BB_cycles_1_sweep] wrote 240 test command(s) to ./../data/72q_BB_cycles_1_sweep/cluster/commands_test_2026-10-20_01-12-59.txt

=== Job Details ===
  Job              : test_2026-10-20_01-12-59  (test)
  Target codename  : 72q_BB_cycles_1_sweep
  Commands         : 240
  Devices          : GPU (a100), 1 card(s) per task, 1 process(es) per card
  Array tasks      : 24, 10 command(s) each, 1 at a time
  CPUs per task    : 12
  Memory           : 32G per GPU (host RAM)
  GPU_MEMORY       : 34816M (85% of the card, for the batch sizer)
  Wall time        : 1:00:00
  Estimate         : ~35m per task (10 round(s) of 2m + 10m startup), 58% of 1:00:00

=== Output Files ===
  Commands file    : ./../data/72q_BB_cycles_1_sweep/cluster/commands_test_2026-10-20_01-12-59.txt
  Models file      : ./../data/72q_BB_cycles_1_sweep/cluster/models_test_2026-10-20_01-12-59.txt  (checked by every task before it tests)
  SLURM script     : ./../data/72q_BB_cycles_1_sweep/cluster/test_2026-10-20_01-12-59.sh

=== Submission (from expts/, after `bash misc/precompile_depot.sh` on a login node) ===
  bash ./../data/72q_BB_cycles_1_sweep/cluster/submit_2026-10-20_01-12-59.sh
  It submits the training job, then the test job with --dependency=afterok on it: testing
  starts when EVERY training task has exited 0, and is cancelled if any has not.

[debankan@narval3 expts]$ 

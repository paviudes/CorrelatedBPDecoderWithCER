#!/usr/bin/env bash
# sweep_hyperparams.sh -- a hyperparameter sweep of the neural BP decoder, run as
# two chained SLURM GPU array jobs: training, then testing.
#
#   cd expts
#   bash sweep_hyperparams.sh sweeps/<sweep>.toml
#   bash sweep_hyperparams.sh sweeps/<sweep>.toml --local[=N]
#                       also print a one-point smoke test on N samples (5000)
#
# EVERYTHING COMES FROM TWO FILES, each key in exactly one of them:
#   sweeps/<sweep>.toml       WHAT to run: the run directory, the training keys,
#                             which test sets each model is scored on, the arms,
#                             the seeds -- and how many array tasks and how much
#                             walltime the two jobs get. sweeps/template.toml
#                             describes every key.
#   clusters/<cluster>.toml   WHERE it runs: account, modules, the card, cores and
#                             RAM of one array task, and the measured costs the
#                             walltime preflight uses. The same for every sweep;
#                             the sweep file names it:  cluster = "narval".
# A key in the wrong file, an unknown key, a key given twice and an array wrapped
# over two lines are refused, so nothing in either file is silently ignored. The
# sweep file is read as written; nothing is opened in an editor.
#
# THE CODE, in hp_sweep/:
#   read_settings.sh   the reader both files go through
#   sweep.sh           the points: one TOML per point, its training command and
#                      its test command(s); test_keys_of() decides which test
#                      set(s) each model is scored on
#   cluster.sh         the jobs: stage-in lists, walltime and depot preflights,
#                      the two SLURM scripts
#   local_smoke.sh     --local
#
# WRITES, only once every check has passed (<ts> = the time of the run):
#   <codename>/models/hyperparams_hp_*.toml           one per point
#   <codename>/cluster/hp_sweep_train_<ts>.txt        one julia command per training point
#   <codename>/cluster/hp_sweep_test_<ts>.txt         one julia command per test
#   <codename>/cluster/hp_sweep_models_<ts>.txt       the weights file each training
#                                                     point writes (the test job
#                                                     checks they are all there)
#   <codename>/cluster/hp_sweep_stage_{train,test}_<ts>.txt
#                                                     what each job copies to the node
#   <codename>/cluster/hp_sweep_{train,test}_<ts>.sh  the two job scripts
#   <codename>/cluster/hp_sweep_{settings,cluster}_<ts>.toml
#                                                     copies of the two input files
# Nothing is submitted: the two sbatch lines are printed at the end.
set -eu

SWEEP_FILE=""
LOCAL=0
SMOKE_N=5000
while [ $# -gt 0 ]; do
    case "$1" in
        --local)   LOCAL=1 ;;
        --local=*) LOCAL=1; SMOKE_N="${1#*=}" ;;
        --help|-h) awk 'NR==1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
        --settings|--settings=*|--no-edit)
            echo "$1 is gone (2026-10-27): the sweep file is the one argument, and it is always read" >&2
            echo "  as written -- there is no editor step:  bash sweep_hyperparams.sh sweeps/<sweep>.toml" >&2
            exit 2 ;;
        --collect)
            echo "--collect is gone (2026-10-27): it parsed the pre-2026-10-09 file names. Compare the" >&2
            echo "  arms of a finished sweep with" >&2
            echo "      python3 misc/compare_sweep_arms.py --workdir <codename> --arms <run tag> <run tag> ..." >&2
            exit 2 ;;
        -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
        *)
            if [ -n "$SWEEP_FILE" ]; then
                echo "one sweep file only (got $SWEEP_FILE and $1)." >&2
                exit 2
            fi
            SWEEP_FILE="$1" ;;
    esac
    shift
done
if [ -z "$SWEEP_FILE" ]; then
    echo "usage, from expts/:  bash sweep_hyperparams.sh sweeps/<sweep>.toml [--local[=N]]   (--help)" >&2
    exit 2
fi
if [ ! -f "$SWEEP_FILE" ]; then
    echo "no such sweep file: $SWEEP_FILE" >&2
    exit 1
fi
if ! echo "$SMOKE_N" | grep -qE '^[1-9][0-9]*$'; then
    echo "--local=$SMOKE_N: the sample count must be a positive integer." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
source "$SCRIPT_DIR/hp_sweep/read_settings.sh"
source "$SCRIPT_DIR/hp_sweep/sweep.sh"
source "$SCRIPT_DIR/hp_sweep/cluster.sh"
source "$SCRIPT_DIR/hp_sweep/local_smoke.sh"

# ------------------------------------------------- 1. the two settings files ---
read_sweep_file              # sweep.sh:   what to run, and the job sizes
read_cluster_profile         # cluster.sh: where it runs
check_run_directory          # sweep.sh:   the run directory and its base TOML
check_job_write_locations    # cluster.sh: run directory and depot where jobs can write
read_base_hyperparameters    # sweep.sh:   what every point inherits
check_dataset_files          # sweep.sh:   every input the points read is there

# ------------------------------------------------------------- 2. the points ---
TRAIN_CMDS="$CLUSTER_DIR/hp_sweep_train_${TS}.txt"
TEST_CMDS="$CLUSTER_DIR/hp_sweep_test_${TS}.txt"
EXPECTED_MODELS="$CLUSTER_DIR/hp_sweep_models_${TS}.txt"
TRAIN_STAGE_LIST="$CLUSTER_DIR/hp_sweep_stage_train_${TS}.txt"
TEST_STAGE_LIST="$CLUSTER_DIR/hp_sweep_stage_test_${TS}.txt"
SLURM_TRAIN="$CLUSTER_DIR/hp_sweep_train_${TS}.sh"
SLURM_TEST="$CLUSTER_DIR/hp_sweep_test_${TS}.sh"
SETTINGS_COPY="$CLUSTER_DIR/hp_sweep_settings_${TS}.toml"
CLUSTER_COPY="$CLUSTER_DIR/hp_sweep_cluster_${TS}.toml"
# The point TOMLs are written to a staging directory and moved into models/ only
# at the end: their names carry no timestamp, so writing them in place would
# replace an earlier sweep's TOMLs even when this generation is then refused.
POINT_TOML_DIR="$(mktemp -d)"
EMITTED_TOML_NAMES="$(mktemp)"
GENERATION_COMPLETE=0
discard_partial_generation() {
    rm -rf "$POINT_TOML_DIR" "$EMITTED_TOML_NAMES" "${GPU_PREAMBLE_FILE:-}"
    if [ "$GENERATION_COMPLETE" -eq 0 ]; then
        rm -f "$TRAIN_CMDS" "$TEST_CMDS" "$EXPECTED_MODELS" "$TRAIN_STAGE_LIST" "$TEST_STAGE_LIST" \
              "$SLURM_TRAIN" "$SLURM_TEST" "$SETTINGS_COPY" "$CLUSTER_COPY"
    fi
}
trap discard_partial_generation EXIT
: > "$TRAIN_CMDS"; : > "$TEST_CMDS"; : > "$EXPECTED_MODELS"

print_alpha_star_values      # sweep.sh: the alpha each alpha_star point starts from
emit_all_points              # sweep.sh: TOMLs, TRAIN_CMDS, TEST_CMDS, EXPECTED_MODELS
check_test_pairing           # sweep.sh: every test is on the set test_keys_of says
print_test_plan
warn_existing_models

# ---------------------------------------------------------- 3. preflights ---
walltime_preflight           # cluster.sh: refuses a job that cannot finish
depot_preflight              # cluster.sh: warns about a stale Julia depot
check_submit_host            # cluster.sh: warns when this is not the node to submit from

# --------------------------------------------------------------- 4. write ---
write_stage_lists            # cluster.sh
write_job_scripts            # cluster.sh
cp "$SWEEP_FILE" "$SETTINGS_COPY"
cp "$CLUSTER_FILE" "$CLUSTER_COPY"
mv "$POINT_TOML_DIR"/*.toml "$MODELS_DIR/"
GENERATION_COMPLETE=1

# -------------------------------------------------------------- 5. report ---
echo
echo "[hp_sweep] $N_POINTS training point(s), $N_TEST_POINTS test point(s)"
if [ -z "$(echo $TEST_KEYS)" ]; then
    echo "  test sets -> each model on its own training key's test set (test_keys = [])"
else
    echo "  test sets -> $(echo $TEST_KEYS | wc -w | tr -d ' ') key(s), every model scored on every one"
fi
echo "  datasets  -> $(echo $DATASETS)"
if [ -n "$(echo $REF_DATASETS)" ]; then
    echo "  ref       -> $(echo $REF_DATASETS)   (CER tanh and no-CER only)"
fi
echo "  seeds     -> $(echo $SEEDS)"
echo "  check node-> $(echo $CHECK_NODE_ARMS)   (no-CER baseline: $INCLUDE_NOCER)"
echo "  optimizer -> $(echo "$OPTIMIZER_ARMS" | head -1)"
echo "$OPTIMIZER_ARMS" | sed '1d; s/^/               /'
echo "  layers    -> loss_layer_selection=$LOSS_LAYER_SELECTION  commit_layer_rule=$COMMIT_LAYER_RULE  (the default; an optimizer arm may override either)"
echo "  cluster   -> $CLUSTER_NAME ($CLUSTER_PROFILE)"
echo "  network   -> $NLAYERS layers (fixed), manual_backprop = true (hand-written gradient)"
ACCOUNT_SHOWN="${ACCOUNT_GPU:-the default account}"
HOST_RAM_SHOWN="--mem-per-gpu=$MEM_PER_GPU host ram"
if [ "$REQUEST_CPUS_AND_MEMORY" = "false" ]; then
    HOST_RAM_SHOWN="$MEM_PER_GPU host ram, both with the card"
fi
echo "  train     -> $ACCOUNT_SHOWN: array 0-$((TRAIN_ARRAY-1)) ($TRAIN_ARRAY tasks x ${N_GPUS}x $GPU_TYPE, $TRAIN_CPUS cpu,"
echo "               $HOST_RAM_SHOWN), $TRAIN_JOBS trainings at a time per card, USE_GPU=1, $TRAIN_WALL"
echo "  test      -> $ACCOUNT_SHOWN: array 0-$((TEST_ARRAY-1)) ($TEST_ARRAY tasks x ${N_GPUS}x $GPU_TYPE (${VRAM_PER_GPU}G vram), $TEST_CPUS cpu),"
echo "               $HOST_RAM_SHOWN, GPU_MEMORY=${GPU_MEMORY_MB}M, $TEST_JOBS at a time, USE_GPU=1, $TEST_WALL"
if [ -n "$JOB_DEPOT_PATH" ]; then
    echo "  depot     -> JULIA_DEPOT_PATH=$JOB_DEPOT_PATH (written into the jobs)"
fi
echo "  commands  -> $TRAIN_CMDS"
echo "               $TEST_CMDS"
echo
# The first point's weights file: the CER arm of the first data set, seed,
# optimizer arm and check-node arm (emit_all_points writes it first).
FIRST_MODEL="$(head -1 "$EXPECTED_MODELS")"
SUBMIT_FROM=""
if [ -n "$SUBMIT_HOST" ]; then
    SUBMIT_FROM=", from $SUBMIT_HOST and from this directory"
fi
echo "submit — TRAIN, then TEST chained on it (both GPU, $ACCOUNT_SHOWN$SUBMIT_FROM):"
echo
echo "    TRAIN_ID=\$(sbatch --parsable $SLURM_TRAIN)"
echo "    sbatch --dependency=afterok:\$TRAIN_ID --kill-on-invalid-dep=yes $SLURM_TEST"
echo
echo "  Or one at a time, checking the models trained before testing them:"
echo "    sbatch $SLURM_TRAIN"
echo "    julia -e 'using JSON, Statistics; w=JSON.parsefile(\"$MODELS_DIR/$FIRST_MODEL\");"
echo "              println(std(vcat(w[\"weights_c2v_v2c\"],w[\"weights_llrs\"],w[\"weights_c2v_readout\"])))'"
echo "    # ~ initial_conditions_scale / sqrt(3) => never trained; larger => trained."
echo "    sbatch $SLURM_TEST"
echo
echo "  afterok on an array is ALL-OR-NOTHING: every one of the $TRAIN_ARRAY train tasks must"
echo "  exit 0, so one failed point stops all $N_TEST_POINTS tests. --kill-on-invalid-dep=yes"
echo "  then cancels the test job outright instead of parking it in the queue as"
echo "  DependencyNeverSatisfied. A NaN-rolled-back epoch is NOT a failure (the"
echo "  point still exits 0 and writes weights), so rollbacks do not block testing."

if [ "$LOCAL" -eq 1 ]; then
    write_local_smoke_test "$SMOKE_N"
fi

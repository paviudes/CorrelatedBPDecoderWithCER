#!/usr/bin/env bash
# Generate a commands.txt for a Neural BP epoch sweep — one self-contained
# Julia invocation per (n_epochs, rep) combination, each redirecting its
# stdout/stderr to a per-run log file.
#
# Usage:
#   ./sweep_epochs.sh                       # default: epochs 10 20 30 40 50, 1 rep each
#   ./sweep_epochs.sh 10 20 30              # custom epoch list, 1 rep each
#   N_REPS=3 ./sweep_epochs.sh              # 3 reps per epoch count
#
# After running, execute the sweep with:
#   parallel --bar -j 4 < <output_dir>/commands.txt
#
# Logs land in <output_dir>/epochs_<N>_rep_<R>.log. Extract num_failures with:
#   grep -h '"num_failures"' <output_dir>/epochs_*.log

set -u

# ---- paths and fixed experiment args ----
PROJECT_ROOT="/data/CorrelatedBPDecoderWithCER"
EXPTS_DIR="$PROJECT_ROOT/expts"
DATA_DIR="$PROJECT_ROOT/data"
PROJECT_DIR="$PROJECT_ROOT"

CODENAME="7q_Hamm_code_data_10000_train"
N_HIDDEN_LAYERS=100
BATCH_SIZE=8
CORR_FILE="correlated_weights_p_0.006_q_0.1_s_1.txt"
TRAIN_FILE="train_ballistic_p_0.006_q_0.1_s_1.txt"
TEST_FILE="test_ballistic_p_0.006_q_0.1_s_1.txt"

# ---- sweep config ----
if [ "$#" -gt 0 ]; then
    EPOCHS=("$@")
else
    EPOCHS=(5 10 20 30 40 50)
fi
N_REPS="${N_REPS:-1}"

# ---- output dir (timestamped) ----
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$DATA_DIR/sweep_results/$TS"
mkdir -p "$OUT_DIR"
COMMANDS_FILE="$OUT_DIR/commands.txt"
: > "$COMMANDS_FILE"

# ---- emit one command per (n_epochs, rep) ----
n_commands=0
for n_epochs in "${EPOCHS[@]}"; do
    for rep in $(seq 1 "$N_REPS"); do
        log_file="$OUT_DIR/epochs_${n_epochs}_rep_${rep}.log"
        printf "julia --project=%s neural_bp_experiments.jl --codename %s --n_hidden_layers %d --n_epochs %d --batch_size %d --correlation_strengths_file %s --train %s --test %s --retrain true > %s 2>&1\n" \
            "$PROJECT_DIR" "$CODENAME" \
            "$N_HIDDEN_LAYERS" "$n_epochs" "$BATCH_SIZE" \
            "$CORR_FILE" "$TRAIN_FILE" "$TEST_FILE" \
            "$log_file" >> "$COMMANDS_FILE"
        n_commands=$((n_commands + 1))
    done
done

echo "Generated $n_commands commands."
echo "  Output dir:    $OUT_DIR"
echo "  Commands file: $COMMANDS_FILE"
echo ""

# Number of CPUs = maximum of 32 and the number of commands, to avoid overloading the machine if there are too many commands.
N_CPUS=$(nproc)
if [ "$n_commands" -lt "$N_CPUS" ]; then
    N_CPUS="$n_commands"
fi

echo "Running with parallel:"
COMMAND_TO_RUN="parallel --bar --jobs $N_CPUS < $COMMANDS_FILE"
echo "  $COMMAND_TO_RUN"

# Take a shell prompt from the user before executing the parallel command. [y/n] and if the user enters 'y', execute the parallel command, else exit the script.
read -p "Do you want to execute the commands now? [y/n] " -n
if [[ "$REPLY" == "y" ]]; then
    eval "$COMMAND_TO_RUN"
else
    echo "Exiting without executing commands."
    exit 0
fi

# Shutdown the Google Cloud VM after the parallel job completes.
# Only do this on the Google Clould VM where `whoami` returns `pavithran_sridhar_gmail_com`, to avoid accidentally shutting down a local machine.
if [[ "$(whoami)" == "pavithran_sridhar_gmail_com" ]]; then
    echo "Parallel job completed. Shutting down the VM..."
    sudo shutdown -h now
fi

# Send an email notification when the `parallel` job completes using `mutt`
# parallel --bar --jobs 30 < $COMMANDS_FILE; echo "Neural BP epoch sweep complete" | mutt -s "Neural BP epoch sweep complete" pavithran.sridhar@gmail.com
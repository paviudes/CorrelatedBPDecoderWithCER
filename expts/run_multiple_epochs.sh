# Write a shell script to run training and testing for multiple epochs.
# We will run these epochs by creating multiple hyperparameter files where the line
# `n_epochs = 10` is changed to `n_epochs = y`.
# Each edited hyperparameter file will be saved in the same directory with the name `hyperparams_epochs_y.toml`.
# The directory of the hyperparameter files is `./../data/90q_BB_p_0.010_q_0.001_std_0.01_data/models/` and the one containing the default hyperparameters is `./../data/90q_BB_p_0.010_q_0.001_std_0.01_data/models/default_hyperparams.toml`.
# We will all the commands, one of each epoch, in parallel using GNU parallel.
# Step 1: Run only training in parallel for all epochs.
# parallel --jobs=<number of epochs> --bar 'julia --project="./../" neural_bp_experiments.jl --codename <codename> --n_hidden_layers 50 --hyperparams default_hyperparams_epochs_{}.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_<sample>.txt --train train_ballistic_p_0.01_q_0.001_s_<sample>.txt' ::: $(<list of epochs>)
# Step 2: After training, set `retrain = false` in all the hyperparameter files to avoid retraining when testing.
# Step 3: Run the testing for each epoch in serial (not in parallel) to avoid GPU memory issues by running the following command for each epoch:

# To avoid parsing issues from GNU Parallel, we will write all the commands to a temporary file in `./../data/<codename>/cluster/sweep_epochs_<timestamp>.txt` and then call
# parallel --jobs=<number of epochs> --bar < ./../data/<codename>/cluster/sweep_epochs_<timestamp>.txt

# To run this script, do the following.
# 1. If this is the first time, set the permission of the script to be executable by running `chmod +x run_multiple_epochs.sh` in the terminal from the `expts` directory.
# 2. For all subsequent runs, simply execute the script by running `./run_multiple_epochs.sh` in the terminal from the `expts` directory.

# List of epochs to run
epochs_list=(3 6)
n_cpus=2
n_hidden_layers=2
codename="90q_BB_p_0.010_q_0.001_std_0.01_data"
sample=1
use_gpu_for_testing=true

# Directory of the hyperparameter files
hyperparams_dir="./../data/${codename}/models"
old_hyperparams_file="${hyperparams_dir}/default_hyperparams.toml"

# Loop over each epoch and create a hyperparameter file with the corresponding number of epochs.
for n_epochs in "${epochs_list[@]}"; do
    # Create a new hyperparameter file by copying the default one and changing the number of epochs.
    new_hyperparams_file="${hyperparams_dir}/hyperparams_epochs_${n_epochs}.toml"
    cp "${old_hyperparams_file}" "${new_hyperparams_file}"
    sed -i '' "s/n_epochs = [0-9]\+/n_epochs = ${n_epochs}/" "${new_hyperparams_file}"
    # Edit the line `retrain = false` to `retrain = true` in the new hyperparameter file.
    sed -i '' "s/retrain = false/retrain = true/" "${new_hyperparams_file}"
done

# Run the training and testing for each epoch in parallel using GNU parallel.
# create a timestamp special characters replaced by underscores to avoid parsing issues from GNU Parallel
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p "./../data/${codename}/cluster"
commands_file="./../data/${codename}/cluster/sweep_epochs_${timestamp}.txt"
touch "${commands_file}"
for n_epochs in "${epochs_list[@]}"; do
    command="julia --project=\"./../\" neural_bp_experiments.jl --codename ${codename} --n_hidden_layers ${n_hidden_layers} --quiet true --hyperparams hyperparams_epochs_${n_epochs}.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_${sample}.txt --train train_ballistic_p_0.01_q_0.001_s_${sample}.txt"
    echo "${command}" >> "${commands_file}"
done

if [ "$use_gpu_for_testing" = true ]; then
    echo "GPU usage for testing is enabled. Testing will be run in serial to avoid GPU memory issues."
    # Run only training in parallel for all epochs.
    parallel --jobs ${n_cpus} --bar < "${commands_file}"

    # After training, set `retrain = false` in all the hyperparameter files to avoid retraining when testing.
    for n_epochs in "${epochs_list[@]}"; do
        new_hyperparams_file="${hyperparams_dir}/hyperparams_epochs_${n_epochs}.toml"
        sed -i '' "s/retrain = true/retrain = false/" "${new_hyperparams_file}"
    done

    # Turn on GPU usage by setting the environment variable `USE_GPU` to `1`.
    export USE_GPU="1"
    # Run the testing for each epoch in serial (not in parallel) to avoid GPU memory issues.
    for n_epochs in "${epochs_list[@]}"; do
        julia --project="./../" neural_bp_experiments.jl --codename ${codename} --n_hidden_layers ${n_hidden_layers} --hyperparams hyperparams_epochs_${n_epochs}.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_${sample}.txt --train train_ballistic_p_0.01_q_0.001_s_${sample}.txt --test test_ballistic_p_0.01_q_0.001_s_${sample}.txt
    done
else
    echo "GPU usage for testing is disabled. Training and testing will be run at once in parallel."
    export USE_GPU="0"
    # Append the test command to each training command in <commands_file> by adding `--test test_ballistic_p_0.01_q_0.001_s_${sample}.txt` at the end of each command.
    sed -i '' "s/\$/ --test test_ballistic_p_0.01_q_0.001_s_${sample}.txt/" "${commands_file}"
    parallel --jobs ${n_cpus} --bar < "${commands_file}"
fi
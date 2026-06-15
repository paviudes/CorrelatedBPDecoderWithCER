###################################################
#       READ THIS BEFORE RUNNING THE SCRIPT
###################################################
# To run this script, do the following.
# 1. Set all the parameters under the section "PARAMETERS TO SET" below according to your needs.
# 2. If this is the first time, set the permission of the script to be executable by running `chmod +x run_multiple_epochs.sh` in the terminal from the `expts` directory.
# 3. For all subsequent runs, simply execute the script by running `./run_multiple_epochs.sh` in the terminal from the `expts` directory.
###################################################

###################################################
#       DESCRIPTION OF THE SCRIPT
####################################################
# Write a shell script to run training and testing for multiple epochs.
# We will run these epochs by creating multiple hyperparameter files where the line
# `n_epochs = 10` is changed to `n_epochs = y`.
# Each edited hyperparameter file will be saved in the same directory with the name `hyperparams_epochs_y.toml`.
# The directory of the hyperparameter files is `./../data/${codename}/models/` and the one containing the default hyperparameters is `./../data/${codename}/models/default_hyperparams.toml`.
# We will all the commands, one of each epoch, in parallel using GNU parallel.
# Step 1: Run only training in parallel for all epochs.
# parallel --jobs=<number of epochs> --bar 'julia --project="./../" neural_bp_experiments.jl --codename <codename> --n_hidden_layers 50 --hyperparams default_hyperparams_epochs_{}.toml --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_<sample>.txt --train train_ballistic_p_0.01_q_0.001_s_<sample>.txt' ::: $(<list of epochs>)
# Step 2: After training, set `retrain = false` in all the hyperparameter files to avoid retraining when testing.
# Step 3: Run the testing for each epoch in serial (not in parallel) to avoid GPU memory issues by running the following command for each epoch:

# To avoid parsing issues from GNU Parallel, we will write all the commands to a temporary file in `./../data/<codename>/cluster/sweep_epochs_<timestamp>.txt` and then call
# parallel --jobs=<number of epochs> --bar < ./../data/<codename>/cluster/sweep_epochs_<timestamp>.txt
# where <timestamp> is a timestamp with special characters replaced by underscores to avoid parsing issues from GNU Parallel.
###################################################

###################################################
#       PARAMETERS TO SET
###################################################
epochs_list=(3 6 9 12) # List of epochs to run.
n_cpus=4 # Number of CPUs to use for parallel processing.
n_hidden_layers=50 # Number of hidden layers in the Neural BP model.
codename="90q_BB_p_0.010_q_0.001_std_0.01_data" # Folder name in `./../data/` for the dataset.
sample=1 # Sample index for the training and testing data files.
use_gpu_for_testing=true # If true, GPU usage for testing is enabled.
###################################################

# Directory of the hyperparameter files
hyperparams_dir="./../data/${codename}/models"
old_hyperparams_file="${hyperparams_dir}/default_hyperparams.toml"

# Loop over each epoch and create a hyperparameter file with the corresponding number of epochs.
for n_epochs in "${epochs_list[@]}"; do
    # Create a new hyperparameter file by copying the default one and changing the number of epochs.
    new_hyperparams_file="${hyperparams_dir}/hyperparams_epochs_${n_epochs}.toml"
    cp "${old_hyperparams_file}" "${new_hyperparams_file}"
    sed -i '' -E "s/n_epochs = [0-9]+/n_epochs = ${n_epochs}/" "${new_hyperparams_file}"
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

# Collect the results from all epochs and save them in a single file.
# Each epoch will have its own results file in `./../data/${codename}/results/simulation_results_test_ballistic_p_0.01_q_0.001_s_<sample>_nlayers_<n_hidden_layers>_epochs_<epochs>_trained_using_train_ballistic_p_0.01_q_0.001_s_<sample>.csv`. 
# algo,average_logical_error_rate,error_model_name,error_model_parameters_description,n_iterations_BP,num_failures,num_samples_per_error_rate,rounds_per_BP,runtime,std_logical_error_rate,weight_soft_constraint
# NN,0.01484,ExplicitErrorModel,./../data/90q_BB_p_0.010_q_0.001_std_0.01_data/testing_data/test_ballistic_p_0.01_q_0.001_s_1.txt,50,1484,100000,3,31.207653999328613,0.017099575667249758,0.0
# The aggregate results file will be saved in `./../data/${codename}/results/sweep_epochs_${timestamp}.csv`.
# The file should have 3 columns:
# the first column is the number of epochs
# The second column is the number of failures (available in results as `num_failures`)
# The third column is the total number of samples. (available in results as `num_samples_per_error_rate`)
# The fourth column is the estimated logical error rate (LER) which is the number of failures divided by the total number of samples. (available in results as `average_logical_error_rate`)
results_dir="./../data/${codename}/results"
mkdir -p "${results_dir}"
aggregate_results_file="${results_dir}/sweep_epochs_${timestamp}.csv"
echo "n_epochs,n_failures,n_samples,LER" > "${aggregate_results_file}"
for n_epochs in "${epochs_list[@]}"; do
    results_file="${results_dir}/simulation_results_test_ballistic_p_0.01_q_0.001_s_${sample}_nlayers_${n_hidden_layers}_epochs_${n_epochs}_trained_using_train_ballistic_p_0.01_q_0.001_s_${sample}.csv"
    if [ -f "${results_file}" ]; then
        # Extract the number of failures, total number of samples, and average logical error rate from the results file.
        num_failures=$(tail -n +2 "${results_file}" | cut -d',' -f6)
        num_samples=$(tail -n +2 "${results_file}" | cut -d',' -f7)
        average_logical_error_rate=$(tail -n +2 "${results_file}" | cut -d',' -f2)
        # Append the results to the aggregate results file.
        echo "${n_epochs},${num_failures},${num_samples},${average_logical_error_rate}" >> "${aggregate_results_file}"
    else
        echo "Results file ${results_file} not found. Skipping."
    fi
done
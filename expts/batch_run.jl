using Dates
using Printf
using LinearAlgebra

function generate_parallel_commands(
    pvals::AbstractVector{<:Real},
    qvals::AbstractVector{<:Real},
    n_samples::Int;
    codename::String="aps",
    # Hyperparameters for the Neural BP model
    n_hidden_layers::Int=100,
    hyperparams_file::String="default_hyperparams.toml",
    julia_project::String="./../",
    commands_file::String="commands.txt",
    output_file::String="simulation_results.log",
    ncpus::Int=10,
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    cluster_backend::String="Google_VM" # "SLURM" or "Google_VM"
)
    """
    Generate shell commands for parallel execution of neural BP experiments.
    """
    open(commands_file, "w") do io
        for p in pvals, q in qvals, s in 1:n_samples

            p_str = @sprintf("%.3g", p)
            q_str = @sprintf("%.3g", q)
            samples_str = @sprintf("%d", s)

            cer_file = "correlated_weights_p_$(p_str)_q_$(q_str)_s_$(samples_str).txt"
            train_file = "train_ballistic_p_$(p_str)_q_$(q_str)_s_$(samples_str).txt"
            test_file = "test_ballistic_p_$(p_str)_q_$(q_str)_s_$(samples_str).txt"

            cmd = """julia --project="$(julia_project)" neural_bp_experiments.jl \
                --codename $(codename) \
                --n_hidden_layers $(n_hidden_layers) \
                --hyperparams $(hyperparams_file) \
                --correlation_strengths_file $(cer_file) \
                --train $(train_file)"""

            cmd = replace(cmd, "\n" => " ")

            println(io, cmd)
        end
    end

    println("$(length(pvals) * length(qvals) * n_samples) commands written to: $commands_file\n")

    # Calculate the number of simulations and determine how many CPUs to use for parallel execution.
    n_simulations = length(pvals) * length(qvals) * n_samples
    n_cpus_to_use = min(ncpus, n_simulations)

    # Write a shell script to run the commands in `commands_file` in parallel, save results to `output_file`, and halt the Google Cloud VM when done.
    if lowercase(cluster_backend) == "google_vm"
        run_on_Google_VM(commands_file, output_file, n_cpus_to_use)
    elseif lowercase(cluster_backend) == "slurm"
        run_on_SLURM(commands_file, n_simulations; n_cpus=n_cpus_to_use, max_nodes=max_nodes, wall_time=wall_time)
    else
        # Meant for local execution.
        println("Run simulations with:")
        println("parallel --bar --keep-order --jobs $(n_cpus_to_use) --results $(output_file) :::: $(commands_file)\n")
    end
end

function run_on_SLURM(commands_file::String, n_commands::Int; n_cpus::Int=10, max_nodes::Int=10, wall_time::String="4:00:00")
    """
    Run the commands in `commands_file` in parallel on a SLURM cluster.
    The SLURM job script will be named `run_<timestamp>.slurm` and will be saved in the same directory as `commands_file`.
    If there are more commands than CPUs, we want to use multiple nodes in a job array to run the commands in parallel.
    However, we don't want to use more than max_nodes nodes, so we will calculate the number of nodes to use based on the number of commands and the number of CPUs per node.
    """
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    commands_dir = dirname(commands_file)
    
    slurm_script_file = joinpath(commands_dir, "run_$(timestamp).sh")
    output_file = joinpath(commands_dir, "nbp_$(timestamp).out")
    error_file = joinpath(commands_dir, "nbp_$(timestamp).err")

    # --- Compute node usage ---
    n_nodes_needed = ceil(Int, n_commands / n_cpus)
    n_nodes = min(n_nodes_needed, max_nodes)
    commands_per_node = ceil(Int, n_commands / n_nodes)

    println("Total commands: $n_commands")
    println("CPUs per node: $n_cpus")
    println("Using nodes: $n_nodes")
    println("Commands per node: $commands_per_node")

    slurm_script_lines = [
        "#!/bin/bash",
        "#SBATCH --account=default",
        "#SBATCH --job-name=nbp_$(timestamp)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --array=0-$(n_nodes-1)",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
        "#SBATCH --time=$(wall_time)",
        "#SBATCH --partition=cpu",
        "",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=pavithran.sridhar@gmail.com",
        "",
        "echo \"Running SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID}\"",
        "",
        "# Determine line range for this task",
        "START=\$((SLURM_ARRAY_TASK_ID * $(commands_per_node) + 1))",
        "END=\$((START + $(commands_per_node) - 1))",
        "",
        "# Extract commands for this node",
        "sed -n \"\${START},\${END}p\" $(commands_file) > $(commands_dir)/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
        "",
        "# Load necessary modules and set up environment",
        "module load julia", # The default version on Trillium is 1.12.
        "cp -r ~/.julia \$SLURM_TMPDIR/",
        "export JULIA_DEPOT_PATH=\"\$SLURM_TMPDIR/.julia\"",
        "",
        "# Disable GPU usage since the cluster nodes we have access to do not have GPUs.",
        "export USE_GPU=0",
        "",
        "# Run commands in parallel",
        "parallel --bar --keep-order --jobs $(n_cpus) --results $(commands_dir)/logs/\${SLURM_ARRAY_TASK_ID} ::: $(commands_dir)/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt"
    ]

    # Write the SLURM job script
    open(slurm_script_file, "w") do io
        println(io, join(slurm_script_lines, "\n"))
    end
    println("SLURM job script written to: $slurm_script_file\n")
    println("Run with:")
    println("sbatch $slurm_script_file\n")
end

function run_on_Google_VM(commands_file::String, output_file::String, n_cpus::Int=10)
    """
    Run the commands in `commands_file` in parallel on a Google Cloud VM, save results to `output_file`, and halt the VM when done.
    """
    # Write a shell script to run the commands in `commands_file` in parallel, save results to `output_file`, and halt the Google Cloud VM when done.
    # The shell script should be named `run_<timestamp>.sh` and should be saved in the same directory as `commands_file`.
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    shell_script_file = joinpath(dirname(commands_file), "run_$(timestamp).sh")
    job_cmd = """parallel --bar --keep-order --jobs $(n_cpus) --results $(output_file) --arg-file $(commands_file)"""
    open(shell_script_file, "w") do io
        wrapped_cmd = shut_down_vm(job_cmd)
        println(io, wrapped_cmd)
    end
    println("Run simulations with:")
    println("bash $(shell_script_file)\n")
end
    

function shut_down_vm(job_cmd::String)
    r"""
    Shutdown the Google Cloud VM after all simulations are done.
    We want to generate the shell script lines that stop the VM after all simulations are done.
    We will add these lines to wrap the shell script generated by `generate_parallel_commands`.
    """
    wrapped_cmd = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Run the commands in parallel and save results to a file",
        job_cmd,
        "",
        "# Stop the instance",
        "sudo shutdown -h now"
    ]

    return join(wrapped_cmd, "\n")
end

function main()
    """
    This function generates commands for running simulations over a range of error parameters.
    
    Data set for plots in APS: aps_7q_Hamm_code_data
    p: 0.001:0.001:0.005
    q: 0.3:0.04:0.66
    """
    dirname = "90q_BB_p_0.010_q_0.001_std_0.01_data"
    generate_parallel_commands(
        [0.01], # set of p values
        [0.001], # set of q values
        6; # number of samples per (p, q) pair. For optimal usage of the machine, please set this to be a multiple of the number of CPUs available.
        codename = dirname,
        # Hyperparameters for the Neural BP model
        n_hidden_layers = 100,
        hyperparams_file = "default_hyperparams.toml",
        # File paths and project settings for running the commands
        julia_project = "./../",
        commands_file = "./../data/$(dirname)/cluster/commands.txt",
        output_file = "./../data/$(dirname)/logs/simulation_results.log",
        # Cluster settings.
        ncpus = 6,
        max_nodes = 1,
        wall_time = "3:00:00",
        cluster_backend = "SLURM" # "SLURM" or "Google_VM" or "local"
    )
end
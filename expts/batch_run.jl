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
    n_epochs::Int=10,
    batch_size::Int=2,
    retrain::Bool=false,
    # Hyperparameters for training the Neural BP model
    learning_rate::Float32=1f-1,
    max_grad_norm::Float32=2.0f0,
    julia_project::String="./../",
    commands_file::String="commands.txt",
    output_file::String="simulation_results.log",
    ncpus::Int=10
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
                --n_epochs $(n_epochs) \
                --batch_size $(batch_size) \
                --correlation_strengths_file $(cer_file) \
                --train $(train_file) \
                --test $(test_file) \
                --retrain $(retrain) \
                --learning_rate $(learning_rate) \
                --max_grad_norm $(max_grad_norm)"""

            cmd = replace(cmd, "\n" => " ")

            println(io, cmd)
        end
    end

    println("$(length(pvals) * length(qvals) * n_samples) commands written to: $commands_file\n")

    # Calculate the number of simulations and determine how many CPUs to use for parallel execution.
    n_simulations = length(pvals) * length(qvals) * n_samples
    n_cpus_to_use = min(ncpus, n_simulations)

    # Write a shell script to run the commands in `commands_file` in parallel, save results to `output_file`, and halt the Google Cloud VM when done.
    run_on_Google_VM(commands_file, n_cpus_to_use)
end

function run_on_SLURM(commands_file::String, n_cpus::Int=10)
    """
    Run the commands in `commands_file` in parallel on a SLURM cluster.
    The SLURM job script will be named `run_<timestamp>.slurm` and will be saved in the same directory as `commands_file`.
    The format of the script will be as follows:
    #!/bin/bash
    #SBATCH --job-name=nbp_<timestamp>
    #SBATCH --output=<directory_of_commands_file>/nbp_<timestamp>.out
    #SBATCH --error=<directory_of_commands_file>/nbp_<timestamp>.err
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=<n_cpus>
    #SBATCH --time=4:00:00
    #SBATCH --partition=cpu
    #SBATCH --mem=16G
    # Email notifications
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=pavithran.sridhar@gmail.com
    # Load necessary modules (if any)
    # module load julia parallel
    # Run the commands in parallel
    parallel --bar --keep-order --jobs $(n_cpus) --results $(output_file) --arg-file $(commands_file)
    """
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    commands_dir = dirname(commands_file)
    slurm_script_file = joinpath(commands_dir, "run_$(timestamp).sh")
    output_file = joinpath(commands_dir, "nbp_$(timestamp).out")
    error_file = joinpath(commands_dir, "nbp_$(timestamp).err")

    slurm_script_lines = [
        "#!/bin/bash",
        "#SBATCH --account=default",
        "#SBATCH --job-name=nbp_$(timestamp)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
        "#SBATCH --time=4:00:00",
        "#SBATCH --partition=cpu",
        "#SBATCH --mem=16G",
        "# Email notifications",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=pavithran.sridhar@gmail.com",
        "# Load necessary modules (if any)",
        "# module load julia parallel",
        "# Run the commands in parallel",
        "parallel --bar --keep-order --jobs $(n_cpus) --results $(output_file) --arg-file $(commands_file)"
    ]
    open(slurm_script_file, "w") do io
        println(io, join(slurm_script_lines, "\n"))
    end
    println("SLURM job script written to: $slurm_script_file\n")
    println("Run the SLURM job with:")
    println("sbatch $(slurm_script_file)\n")
end

function run_on_Google_VM(commands_file::String, n_cpus::Int=10)
    """
    Run the commands in `commands_file` in parallel on a Google Cloud VM, save results to `simulation_results.log`, and halt the VM when done.
    """
    # Write a shell script to run the commands in `commands_file` in parallel, save results to `output_file`, and halt the Google Cloud VM when done.
    # The shell script should be named `run_<timestamp>.sh` and should be saved in the same directory as `commands_file`.
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    shell_script_file = joinpath(dirname(commands_file), "run_$(timestamp).sh")
    job_cmd = """parallel --bar --keep-order --jobs $(n_cpus_to_use) --results $(output_file) --arg-file $(commands_file)"""
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
    Data set for plots in APS: aps_7q_Hamm_code_data
    p: 0.001:0.001:0.005
    q: 0.3:0.04:0.66
    """
    dirname = "72q_BB_p_0.006_q_0.1_std_0.1"
    generate_parallel_commands(
        [0.006],
        [0.1],
        56;
        codename = dirname,
        n_hidden_layers = 100,
        n_epochs = 20,
        batch_size = 8,
        retrain = false,
        learning_rate = 1f-1,
        max_grad_norm = 2.0f0,
        julia_project = "./../",
        commands_file = "./../data/$(dirname)/commands.txt",
        output_file = "./../data/$(dirname)/simulation_results.log",
        ncpus = 56
    )
end
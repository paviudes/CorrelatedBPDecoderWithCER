using Dates
using Printf
using ArgParse
using LinearAlgebra

function generate_parallel_commands(
    pvals::AbstractVector{<:Real},
    qvals::AbstractVector{<:Real},
    n_samples::Int,
    codename::String="aps";
    # Hyperparameters for the Neural BP model
    n_hidden_layers::Int=100,
    hyperparams_file::String="default_hyperparams.toml",
    # File paths and project settings for running the commands
    julia_project::String="./../",
    commands_file::String="commands.txt",
    output_file::String="simulation_results.log",
    working_dir::String=joinpath(@__DIR__, ".."),
    # Cluster settings.
    ncpus::Int=10,
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    cluster_backend::String="Google_VM", # "SLURM" or "Google_VM"
    skip_testing::Bool=false,
)
    """
    Generate shell commands for parallel execution of neural BP experiments.
    """
    train_files = [
        "train_p_$(p)_q_$(q)_s_$(samp).txt"
        for p in pvals for q in qvals for samp in 1:n_samples
    ]
    test_files = [
        "test_p_$(p)_q_$(q)_s_$(samp).txt"
        for p in pvals for q in qvals for samp in 1:n_samples
    ]
    cer_files = [
        "correlated_weights_p_$(p)_q_$(q)_s_$(samp).txt"
        for p in pvals for q in qvals for samp in 1:n_samples
    ]
    hyperparams_files = [hyperparams_file for _ in 1:length(cer_files)]
    generate_parallel_commands(
        cer_files,
        train_files,
        test_files,
        codename=codename;
        # Hyperparameters for the Neural BP model
        n_hidden_layers=n_hidden_layers,
        hyperparams_files=hyperparams_files,
        julia_project=julia_project,
        # File paths and project settings for running the commands
        commands_file=commands_file,
        output_file=output_file,
        working_dir=working_dir,
        # Cluster settings.
        ncpus=ncpus,
        max_nodes=max_nodes,
        wall_time=wall_time,
        email_address=email_address,
        cluster_backend=cluster_backend,
        # Flags for ommiting testing or training
        skip_testing=skip_testing,
    )
end

function generate_parallel_commands(
    cer_files::AbstractVector{<:String},
    train_files::AbstractVector{<:String},
    test_files::AbstractVector{<:String},
    codename::String="aps";
    # Hyperparameters for the Neural BP model
    n_hidden_layers::Int=100,
    hyperparams_files::AbstractVector{<:String}=["default_hyperparams.toml"],
    julia_project::String="./../",
    commands_file::String="commands.txt",
    output_file::String="simulation_results.log",
    working_dir::String=joinpath(@__DIR__, ".."),
    # Cluster settings.
    ncpus::Int=10,
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    cluster_backend::String="Google_VM", # "SLURM" or "Google_VM"
    # Flags for ommiting testing or training
    skip_testing::Bool=false
)
    """
    Generate shell commands for parallel execution of neural BP experiments.
    """
    commands_dir = joinpath(working_dir, "cluster")
    if !isdir(commands_dir)
        mkpath(commands_dir)
    end

    commands_file_path = joinpath(commands_dir, commands_file)
    open(commands_file_path, "w") do io
        for (cer_file, train_file, test_file, hyperparams_file) in zip(cer_files, train_files, test_files, hyperparams_files)

            cmd = """julia --project="$(julia_project)" neural_bp_experiments.jl \
                --workdir $(working_dir) \
                --codename $(codename) \
                --n_hidden_layers $(n_hidden_layers) \
                --hyperparams $(hyperparams_file) \
                --correlation_strengths_file $(cer_file) \
                --quiet true \
                --train $(train_file)"""
            
            if !skip_testing
                cmd *= """ \
                --test $(test_file)"""
            end

            cmd = replace(cmd, "\n" => " ")

            println(io, cmd)
        end
    end

    n_params = length(cer_files)

    # Calculate the number of simulations and determine how many CPUs to use for parallel execution.
    n_simulations = n_params
    n_cpus_to_use = min(ncpus, n_simulations)

    # Write a shell script to run the commands in `commands_file` in parallel, save results to `output_file`, and halt the Google Cloud VM when done.
    if lowercase(cluster_backend) == "google_vm"
        run_on_Google_VM(commands_file_path, output_file, n_cpus_to_use)
    elseif lowercase(cluster_backend) == "slurm"
        run_on_SLURM(
            commands_file_path,
            n_simulations;
            n_cpus        = n_cpus_to_use,
            max_nodes     = max_nodes,
            wall_time     = wall_time,
            email_address = email_address,
            working_dir   = "$(working_dir)",
            codename      = "$(codename)"
        )
    else
        # Meant for local execution.
        println("Run simulations with:")
        println(
            "parallel --bar --keep-order" *
            " --jobs $(n_cpus_to_use)" *
            " --results $(output_file)" *
            " :::: $(commands_file_path)\n"
        )
    end
end

using Dates

function run_on_SLURM(
    commands_file::String,
    n_commands::Int,
    codename::String;
    n_cpus::Int=10,
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    working_dir::String=joinpath(@__DIR__, ".."),
)
    """
    Run the commands in `commands_file` in parallel on a SLURM cluster.
    Utilizes targeted directory mirroring to $SLURM_TMPDIR to isolate 
    the specific `codename` folder and bypass network I/O bottlenecks.
    """
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Target the specific codename directory for all cluster outputs
    target_dir = joinpath(working_dir, codename)
    commands_dir = joinpath(target_dir, "cluster")

    # Create the logs directory if it doesn't exist.
    logs_dir = joinpath(commands_dir, "logs")
    if !isdir(logs_dir)
        mkpath(logs_dir)
    end
    
    slurm_script_file = joinpath(commands_dir, "run_$(timestamp).sh")
    output_file = joinpath(commands_dir, "nbp_$(timestamp).out")
    error_file = joinpath(commands_dir, "nbp_$(timestamp).err")

    # --- Compute node usage ---
    n_nodes_needed = ceil(Int, n_commands / n_cpus)
    n_nodes = min(n_nodes_needed, max_nodes)
    commands_per_node = ceil(Int, n_commands / n_nodes)

    jobname = "nbp_$(timestamp)"
    slurm_script_lines = [
        "#!/bin/bash",
        "#SBATCH --account=def-jemerson",
        "#SBATCH --job-name=$(jobname)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --array=0-$(n_nodes-1)",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
        "#SBATCH --time=$(wall_time)",
        "",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=$(email_address)",
        "",
        "echo \"Running SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID}\"",
        "",
        "# Determine line range for this task",
        "START=\$((SLURM_ARRAY_TASK_ID * $(commands_per_node) + 1))",
        "END=\$((START + $(commands_per_node) - 1))",
        "",
        "# Extract commands for this node and write them to the network target directory",
        "sed -n \"\${START},\${END}p\" $(commands_file) > $(commands_dir)/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
        "",
        "# Load necessary modules and set up environment",
        "module load julia/1.12.5",
        "cp -r ~/.julia \$SLURM_TMPDIR/",
        "export JULIA_DEPOT_PATH=\"\$SLURM_TMPDIR/.julia\"",
        "",
        "# Disable GPU usage",
        "export USE_GPU=\"0\"",
        "",
        "######################################################################",
        "# Thread Safety & Precompilation",
        "######################################################################",
        "# Force 1 thread per process to avoid CPU thrashing with GNU parallel",
        "export JULIA_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export BLAS_NUM_THREADS=1",
        "export JULIA_NUM_PRECOMPILE_TASKS=1",
        "",
        "# Precompile serially on the isolated local disk",
        "julia --project=$(target_dir) -e 'using Pkg; Pkg.precompile()'",
        "",
        "######################################################################",
        "# 1. STAGE IN: Mirror Targeted Project Folder to Fast Local Storage",
        "######################################################################",
        "LOCAL_WORK_DIR=\"\$SLURM_TMPDIR/$(codename)\"",
        "mkdir -p \$LOCAL_WORK_DIR",
        "",
        "echo \"Mirroring $(codename) directory to \$SLURM_TMPDIR...\"",
        "# rsync -a preserves structure. We strictly copy the codename folder contents.",
        "rsync -a $(target_dir)/ \$LOCAL_WORK_DIR/",
        "",
        "######################################################################",
        "# 2. COMPUTE",
        "######################################################################",
        "LOCAL_LOGS=\"\$LOCAL_WORK_DIR/cluster/logs/node_\${SLURM_ARRAY_TASK_ID}\"",
        "mkdir -p \$LOCAL_LOGS",
        "",
        "# Move INTO the local NVMe drive's expts folder",
        "# This ensures your script's relative paths flawlessly resolve to the local drive",
        "cd \$LOCAL_WORK_DIR/expts",
        "",
        "echo \"Running parallel computations locally...\"",
        "# Because we rsynced the codename folder, the commands_chunk exists on the local drive",
        "parallel --jobs $(n_cpus) --results \$LOCAL_LOGS < ../cluster/commands_chunk_\${SLURM_ARRAY_TASK_ID}.txt",
        "",
        "######################################################################",
        "# 3. STAGE OUT: Save Results via Rsync",
        "######################################################################",
        "echo \"Computation finished. Staging results back to network storage...\"",
        "# rsync -au strictly transfers newly generated or modified files.",
        "rsync -au \$LOCAL_WORK_DIR/ $(target_dir)/",
        "",
        "echo \"Job completed and data safely transferred.\""
    ]

    # Write the SLURM job script
    open(slurm_script_file, "w") do io
        println(io, join(slurm_script_lines, "\n"))
    end
    
    println("\n=== Job Details ===")
    println("  Job name         : $jobname")
    println("  Target codename  : $codename")
    println("  Total commands   : $n_commands")
    println("  CPUs per node    : $n_cpus")
    println("  Nodes in use     : $n_nodes")
    println("  Commands per node: $commands_per_node")
    println("  Wall time        : $wall_time")
    println("\n=== Output Files ===")
    println("  Commands file    : $commands_file")
    println("  SLURM script     : $slurm_script_file")
    println("\n=== Submission ===")
    println("  sbatch $slurm_script_file\n")
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

function main_test()
    """
    Generate commands for running simulations for different testing ansatzes.
    We will use different training files, and hyperparameters files combinations.
    """
    codename = "90q_BB_p_0.010_q_0.001_std_variable_data_v2"
    individual_training_files = [
        "train_ballistic_p_0.01_q_0.001_s_1.txt",
        "train_ballistic_p_0.01_q_0.001_s_2.txt",
        "train_ballistic_p_0.01_q_0.001_s_3.txt",
        "train_ballistic_p_0.01_q_0.001_s_4.txt",
        "train_ballistic_p_0.01_q_0.001_s_5.txt",
        "train_ballistic_p_0.01_q_0.001_s_6.txt"
    ]
    # The CER files are formatted as "correlated_weights_fname" for each `fname` in `individual_training_files`.
    individual_testing_files = [
        "test_ballistic_p_0.01_q_0.001_s_1.txt"
    ]
    individual_hyperparams_files = [
        "hyperparams_epochs_10.toml",
        "hyperparams_epochs_15.toml",
        "hyperparams_epochs_20.toml",
    ]

    n_param_combinations = length(individual_training_files) * length(individual_testing_files) * length(individual_hyperparams_files)

    # Generate each combination of training file, testing file, and hyperparameters file.
    combinations = vec(collect(Iterators.product(
        individual_training_files,
        individual_testing_files,
        individual_hyperparams_files,
    )))
    train_files       = [combo[1]                         for combo in combinations]
    cer_files         = ["correlated_weights_$(combo[1])" for combo in combinations]
    test_files        = [combo[2]                         for combo in combinations]
    hyperparams_files = [combo[3]                         for combo in combinations]

    # Generate commands for running the simulations in parallel.
    generate_parallel_commands(
        cer_files,
        train_files,
        test_files;
        codename=codename,
        # Hyperparameters for the Neural BP model
        n_hidden_layers=200,
        hyperparams_files=hyperparams_files,
        julia_project="./../",
        commands_file="./../data/$(codename)/cluster/commands.txt",
        output_file="./../data/$(codename)/logs/simulation_results.log",
        # Cluster settings.
        ncpus=6,
        max_nodes=1,
        wall_time="1:00:00",
        email_address="pavithran.sridhar@gmail.com",
        cluster_backend="SLURM",
        skip_testing=true
    )
end

function main(;
    codenames::AbstractVector{<:String}=["72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data"],
    p_vals::AbstractVector{<:Real}=[0.01],
    qvals::AbstractVector{<:Real}=[0.001],
    n_samples::Int=64,
    hyperparams_file::String="hyperparams_epochs_10.toml",
    n_hidden_layers::Int=200,
    n_cpus::Int=64,
    wall_time::String="1:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    max_nodes::Int=1,
    data_dir::String="./../data"
)
    """
    This function generates commands for running simulations over a range of error parameters.
    
    Note:
    Data set for plots in APS:
    dirname: aps_7q_Hamm_code_data
    p: 0.001:0.001:0.005
    q: 0.3:0.04:0.66
    """
    for codename in codenames
        generate_parallel_commands(
            p_vals, # set of p values
            qvals, # set of q values
            n_samples, # number of samples per (p, q) pair. For optimal usage of the machine, please set this to be a multiple of the number of CPUs available.
            codename = codename;
            # Hyperparameters for the Neural BP model
            n_hidden_layers = n_hidden_layers,
            hyperparams_file = hyperparams_file,
            # File paths and project settings for running the commands
            julia_project = "./../",
            commands_file = "commands.txt",
            output_file = "simulation_results.log",
            working_dir = "$(data_dir)",
            # Cluster settings.
            ncpus = n_cpus,
            max_nodes = max_nodes,
            wall_time = wall_time,
            email_address = email_address,
            cluster_backend = "SLURM" # "SLURM" or "Google_VM" or "local"
            # Flags for ommiting testing or training
            skip_testing = true
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Use ArgParse to parse command line arguments for the main function.
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--working_dir"
            help = "Working directory for the simulations."
            arg_type = String
            default = "./../data"
        "--dirnames"
            help = "List of directory names for different simulation settings."
            nargs = '+'
            default = ["72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data"]
        "--p_vals"
            help = "List of p values for the simulations."
            nargs = '+'
            default = [0.01]
        "--qvals"
            help = "List of q values for the simulations."
            nargs = '+'
            default = [0.001]
        "--n_samples"
            help = "Number of samples per (p, q) pair."
            arg_type = Int
            default = 64
        "--hyperparams_file"
            help = "Path to the hyperparameters file."
            arg_type = String
            default = "hyperparams_epochs_10.toml"
        "--n_hidden_layers"
            help = "Number of hidden layers in the neural BP model."
            arg_type = Int
            default = 200
        "--n_cpus"
            help = "Number of CPUs to use for parallel execution."
            arg_type = Int
            default = 64
        "--wall_time"
            help = "Wall time for the SLURM job."
            arg_type = String
            default = "1:00:00"
        "--email"
            help = "Email address for SLURM job notifications."
            arg_type = String
            default = "pavithran.sridhar@gmail.com"
        "--max_nodes"
            help = "Maximum number of nodes to use for the SLURM job."
            arg_type = Int
            default = 1
    end

    # Parse the command line arguments and call the main function with the parsed arguments.
    parsed_args = parse_args(settings)
    dirnames = String.(parsed_args["dirnames"])
    p_vals = [parse(Float64, p) for p in parsed_args["p_vals"]]
    qvals = [parse(Float64, q) for q in parsed_args["qvals"]]

    # Call the main function with the parsed arguments.
    main(;
        data_dir = parsed_args["working_dir"],
        codenames = dirnames,
        p_vals = p_vals,
        qvals = qvals,
        n_samples = parsed_args["n_samples"],
        hyperparams_file = parsed_args["hyperparams_file"],
        n_hidden_layers = parsed_args["n_hidden_layers"],
        n_cpus = parsed_args["n_cpus"],
        wall_time = parsed_args["wall_time"],
        email_address = parsed_args["email"],
        max_nodes = parsed_args["max_nodes"]
    )

    # Example usage (from Shell in the `expts` directory):
    # julia --project="./../" batch_run.jl --dirnames 72q_BB_p_0.010_std_0.01_q_0.000_std_0.00_data --p_vals 0.01 --qvals 0.001 --n_samples 64 --hyperparams_file hyperparams_epochs_10.toml --n_hidden_layers 200 --n_cpus 64 --wall_time 1:00:00 --max_nodes 1
end
# ============================================================================
# batch_commands.jl — build commands.txt and dispatch to a backend
# ============================================================================
#
# Exposes two overloads of generate_parallel_commands:
#
#   (1) grid form:      (pvals, qvals, n_samples, codename; ...)
#         Constructs train/test/CER filenames from the p×q×samples grid, then
#         delegates to the explicit form.
#
#   (2) explicit form:  (cer_files, train_files, test_files, codename; ...)
#         Writes commands.txt with one julia neural_bp_experiments.jl invocation
#         per (cer, train, test, hyperparams) tuple. Then dispatches to one of
#         three backends selected by cluster_backend:
#           "slurm"     → run_on_SLURM      (slurm.jl)
#           "google_vm" → run_on_Google_VM  (google_vm.jl)
#           "local"     → run_locally       (local_runs.jl)
#
# `include`d by batch_run.jl, which also `include`s the three backend files.
# ============================================================================

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
    mem_per_cpu::String="4G",
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    cluster_backend::String="Google_VM", # "SLURM", "local", or "Google_VM"
    skip_testing::Bool=false,
    account::String="def-jemerson",
    mode::Symbol=:train,
    n_gpus_per_node::Int=1,
    gpu_type::String="",
    cuda_module::String="cuda",
    mem_per_gpu::String="",   # SLURM `--mem-per-gpu` (test mode only). Empty ⇒ use `--mem-per-cpu`.
)
    """
    Grid form: build cer/train/test filenames from `pvals × qvals × 1:n_samples`
    then delegate to the explicit-list form.
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
        codename;
        n_hidden_layers   = n_hidden_layers,
        hyperparams_files = hyperparams_files,
        julia_project     = julia_project,
        commands_file     = commands_file,
        output_file       = output_file,
        working_dir       = working_dir,
        ncpus             = ncpus,
        mem_per_cpu       = mem_per_cpu,
        max_nodes         = max_nodes,
        wall_time         = wall_time,
        email_address     = email_address,
        cluster_backend   = cluster_backend,
        skip_testing      = skip_testing,
        account           = account,
        mode              = mode,
        n_gpus_per_node   = n_gpus_per_node,
        gpu_type          = gpu_type,
        cuda_module       = cuda_module,
        mem_per_gpu       = mem_per_gpu,
    )
end

function generate_parallel_commands(
    cer_files::AbstractVector{<:String},
    train_files::AbstractVector{<:String},
    test_files::AbstractVector{<:String},
    codename::String;
    n_hidden_layers::Int=100,
    hyperparams_files::AbstractVector{<:String}=["default_hyperparams.toml"],
    julia_project::String="./../",
    commands_file::String="commands.txt",
    output_file::String="simulation_results.log",
    working_dir::String=joinpath(@__DIR__, ".."),
    ncpus::Int=10,
    mem_per_cpu::String="4G",
    max_nodes::Int=10,
    wall_time::String="4:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    cluster_backend::String="Google_VM",
    skip_testing::Bool=false,
    account::String="def-jemerson",
    mode::Symbol=:train,
    n_gpus_per_node::Int=1,
    gpu_type::String="",
    cuda_module::String="cuda",
    mem_per_gpu::String="",   # SLURM `--mem-per-gpu` (test mode only). Empty ⇒ use `--mem-per-cpu`.
)
    """
    Explicit-list form: write commands.txt (one julia neural_bp_experiments.jl
    invocation per tuple), then dispatch to the selected backend.
    """
    commands_dir = joinpath(working_dir, codename, "cluster")
    isdir(commands_dir) || mkpath(commands_dir)

    commands_file_path = joinpath(commands_dir, commands_file)
    open(commands_file_path, "w") do io
        for (cer_file, train_file, test_file, hyperparams_file) in
                zip(cer_files, train_files, test_files, hyperparams_files)

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

    n_simulations = length(cer_files)
    n_cpus_to_use = min(ncpus, n_simulations)

    backend = lowercase(cluster_backend)

    if backend == "google_vm"
        run_on_Google_VM(commands_file_path, output_file, n_cpus_to_use)
    elseif backend == "slurm"
        run_on_SLURM(
            commands_file_path,
            n_simulations,
            codename;
            n_cpus          = n_cpus_to_use,
            mem_per_cpu     = mem_per_cpu,
            max_nodes       = max_nodes,
            wall_time       = wall_time,
            email_address   = email_address,
            working_dir     = "$(working_dir)",
            account         = account,
            mode            = mode,
            n_gpus_per_node = n_gpus_per_node,
            gpu_type        = gpu_type,
            cuda_module     = cuda_module,
            mem_per_gpu     = mem_per_gpu,
        )
    elseif backend == "local"
        # Generate a runnable train_local_<ts>.sh or test_local_<ts>.sh (per mode)
        # with runtime GPU detection (Metal on Apple Silicon, CUDA on Linux if
        # nvidia-smi works, else USE_GPU=0). See local_runs.jl for details.
        run_locally(
            commands_file_path,
            n_simulations,
            codename;
            n_cpus          = n_cpus_to_use,
            working_dir     = "$(working_dir)",
            mode            = mode,
            n_gpus_per_node = n_gpus_per_node,
            gpu_type        = gpu_type,
        )
    else
        error("Unknown cluster_backend=\"$(cluster_backend)\". " *
              "Must be one of: \"SLURM\", \"local\", \"Google_VM\".")
    end
end

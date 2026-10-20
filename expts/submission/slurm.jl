# ============================================================================
# slurm.jl — SLURM job-script generator for Alliance Canada clusters
# ============================================================================
#
# Exposes:  run_on_SLURM(commands_file, n_commands, codename; ...) -> script path
#           SlurmTaskLayout, slurm_task_layout, walltime_estimate
#
# Called from batch_commands.jl when cluster_backend == "SLURM". Fills in
# slurm_job_template.sh (next to this file) and writes the result as
# <working_dir>/<codename>/cluster/{train,test}_<timestamp>.sh; batch_run.jl
# prints how to submit it (one sbatch, or the chained submit script). The job
# BODY lives in the template, not here, so that it can be read as a shell script
# and executed with stub binaries to test it; this file only computes the values
# that go into it. `include`d by batch_run.jl.
#
# Either job can run on GPUs or on CPUs (`use_gpu`):
#   training on a GPU  — `jobs_per_gpu` trainings share each card. A training at
#                        the batch of 20 keeps an A100 ~20% busy (it is host-bound),
#                        so ~4 per card is the useful packing. Needs
#                        manual_backprop = true (checked by batch_run.jl).
#   testing on a GPU   — one process per card: the prediction batch is sized to
#                        fill the card's VRAM (GPU_MEMORY below), so two would not fit.
#   either on CPUs     — `n_cpus` single-threaded processes per array task.
# ============================================================================

using Dates
using Printf

"""
    card_vram_gb(gpu_type) -> Int

VRAM of one card, or one MIG slice, of `gpu_type`, in GB: what the prediction
batch sizer may fill. 0 for a type this table does not list; set `vram_per_gpu`
in the settings then.
"""
function card_vram_gb(gpu_type::String)::Int
    vram_by_type::Dict{String, Int} = Dict(
        "a100" => 40, "a100_3g.20gb" => 20, "a100_2g.10gb" => 10, "a100_1g.5gb" => 5,
        "h100" => 80, "h100_3g.40gb" => 40, "h100_2g.20gb" => 20, "h100_1g.10gb" => 10,
        "l40s" => 48, "v100" => 32,
    )
    vram_gb::Int = get(vram_by_type, gpu_type, 0)
    return vram_gb
end

"""
    walltime_seconds(wall_time) -> Int

A SLURM time limit in seconds. SLURM reads "MM", "MM:SS", "HH:MM:SS", "D-HH",
"D-HH:MM" and "D-HH:MM:SS".
"""
function walltime_seconds(wall_time::String)::Int
    day_parts::Vector{SubString{String}} = split(strip(wall_time), '-')
    clock_fields::Vector{Int} = Int[]
    parsed_fields::Vector{Union{Int, Nothing}} = [tryparse(Int, field) for field in split(day_parts[end], ':')]
    if length(day_parts) > 2 || any(field -> field === nothing, parsed_fields) || length(parsed_fields) > 3
        throw(ArgumentError("wall_time = $(repr(wall_time)) is not a SLURM time limit; write it like \"1:00:00\"."))
    end
    clock_fields = Int[field for field in parsed_fields]
    total_seconds::Int = 0
    if length(day_parts) == 2
        days_value::Union{Int, Nothing} = tryparse(Int, day_parts[1])
        if days_value === nothing
            throw(ArgumentError("wall_time = $(repr(wall_time)) is not a SLURM time limit; write it like \"1-00:00:00\"."))
        end
        # D-HH[:MM[:SS]]
        padded_fields::Vector{Int} = vcat(clock_fields, zeros(Int, 3 - length(clock_fields)))
        total_seconds = ((days_value * 24 + padded_fields[1]) * 60 + padded_fields[2]) * 60 + padded_fields[3]
    elseif length(clock_fields) == 3
        total_seconds = (clock_fields[1] * 60 + clock_fields[2]) * 60 + clock_fields[3]
    elseif length(clock_fields) == 2
        total_seconds = clock_fields[1] * 60 + clock_fields[2]
    else
        total_seconds = clock_fields[1] * 60
    end
    if total_seconds <= 0
        throw(ArgumentError("wall_time = $(repr(wall_time)) is not a positive time limit."))
    end
    return total_seconds
end

"""
    duration_text(seconds) -> String

"1h05m", "34m" or "45s".
"""
function duration_text(seconds::Int)::String
    text::String = "$(seconds)s"
    if seconds >= 3600
        text = @sprintf("%dh%02dm", seconds ÷ 3600, (seconds % 3600) ÷ 60)
    elseif seconds >= 60
        text = "$(seconds ÷ 60)m"
    end
    return text
end

"How an array job spreads its commands over array tasks."
struct SlurmTaskLayout
    jobs_per_task::Int        # processes at a time in one array task
    n_tasks::Int              # array tasks
    commands_per_task::Int    # each task takes this many consecutive lines of the commands file
end

"""
    slurm_task_layout(n_commands, use_gpu, n_cpus, n_gpus_per_node, jobs_per_gpu, max_nodes) -> SlurmTaskLayout

A task runs `n_gpus_per_node × jobs_per_gpu` processes at a time on GPUs and
`n_cpus` on CPUs; the commands are split into at most `max_nodes` contiguous
slices.
"""
function slurm_task_layout(n_commands::Int, use_gpu::Bool, n_cpus::Int, n_gpus_per_node::Int,
                           jobs_per_gpu::Int, max_nodes::Int)::SlurmTaskLayout
    if n_commands < 1
        throw(ArgumentError("slurm_task_layout: there are no commands to run."))
    end
    jobs_per_task::Int = n_cpus
    if use_gpu
        jobs_per_task = n_gpus_per_node * jobs_per_gpu
    end
    n_tasks::Int = min(cld(n_commands, jobs_per_task), max_nodes)
    commands_per_task::Int = cld(n_commands, n_tasks)
    layout::SlurmTaskLayout = SlurmTaskLayout(jobs_per_task, n_tasks, commands_per_task)
    return layout
end

"""
    walltime_estimate(phase, n_commands, layout, max_nodes, wall_time, seconds_per_round, startup_seconds) -> Tuple{String, Int}

WALLTIME PREFLIGHT. A task that hits the wall loses every command it had not
finished (the TERM trap saves the finished ones), so a job whose estimate does
not fit is refused: this throws, saying which of max_nodes / wall_time to
raise. A task runs its commands `layout.jobs_per_task` at a time, one round
taking `seconds_per_round` (measured), after `startup_seconds`. Returns the
estimate as text and as a percentage of the wall; 0 per cent and a note when
there is no measurement (`seconds_per_round = 0`, a job on CPUs).
"""
function walltime_estimate(phase::String, n_commands::Int, layout::SlurmTaskLayout, max_nodes::Int,
                           wall_time::String, seconds_per_round::Int, startup_seconds::Int)::Tuple{String, Int}
    rounds_per_task::Int = cld(layout.commands_per_task, layout.jobs_per_task)
    wall_seconds::Int = walltime_seconds(wall_time)
    estimate_text::String = "no measured timing for a job on CPUs; check wall_time yourself"
    percent_of_wall::Int = 0
    if seconds_per_round > 0
        estimated_seconds::Int = startup_seconds + rounds_per_task * seconds_per_round
        percent_of_wall = round(Int, 100 * estimated_seconds / wall_seconds)
        estimate_text = "~$(duration_text(estimated_seconds)) per task ($(rounds_per_task) round(s) of " *
                        "$(duration_text(seconds_per_round)) + $(duration_text(startup_seconds)) startup), " *
                        "$(percent_of_wall)% of $(wall_time)"
        if estimated_seconds >= wall_seconds
            # The most rounds whose estimate stays strictly under the wall.
            rounds_that_fit::Int = (wall_seconds - startup_seconds - 1) ÷ seconds_per_round
            advice::String = "Raise wall_time."
            if rounds_that_fit >= 1
                tasks_needed::Int = cld(n_commands, rounds_that_fit * layout.jobs_per_task)
                advice = "Raise max_nodes to at least $(tasks_needed) (now $(max_nodes)), or raise wall_time."
            end
            throw(ArgumentError("walltime preflight: $(n_commands) $(phase) command(s) over $(layout.n_tasks) " *
                                "array task(s) need $(estimate_text) — the job cannot finish. $(advice) " *
                                "Nothing was written."))
        end
    end
    estimate::Tuple{String, Int} = (estimate_text, percent_of_wall)
    return estimate
end

"""
    fill_job_template(template_text, values) -> String

Replace every `{{NAME}}` in `template_text` with `values["NAME"]`, literally, and
refuse to return a script that still holds a placeholder.
"""
function fill_job_template(template_text::String, values::Dict{String, String})::String
    filled_text::String = template_text
    for (placeholder_name, placeholder_value) in values
        filled_text = replace(filled_text, "{{" * placeholder_name * "}}" => placeholder_value)
    end
    leftover_placeholder::Union{RegexMatch, Nothing} = match(r"\{\{[A-Z_]+\}\}", filled_text)
    if leftover_placeholder !== nothing
        throw(ErrorException("slurm.jl: the job template's $(leftover_placeholder.match) was not filled in."))
    end
    return filled_text
end

function run_on_SLURM(
    commands_file::String,
    n_commands::Int,
    codename::String;
    n_cpus::Int=12,
    mem_per_cpu::String="4G",
    max_nodes::Int=8,
    wall_time::String="1:00:00",
    email_address::String="pavithran.sridhar@gmail.com",
    working_dir::String=joinpath(@__DIR__, ".."),
    account::String="def-jemerson_gpu",
    mode::Symbol=:train,         # :train or :test — which of the two jobs this is
    use_gpu::Bool=true,          # request GPUs and run every process on one (CUDA)
    jobs_per_gpu::Int=1,         # processes sharing one card (GPU jobs only)
    n_gpus_per_node::Int=1,      # cards per array task (GPU jobs only)
    gpu_type::String="",         # Alliance model specifier: "a100", "a100_3g.20gb", "h100", ...
    cuda_module::String="cuda",
    mem_per_gpu::String="",      # SLURM --mem-per-gpu (host RAM). Empty ⇒ --mem-per-cpu.
    vram_per_gpu::Int=0,         # GB per card for the test batch sizer; 0 ⇒ from gpu_type
    julia_module::String="julia/1.12.5",
    seconds_per_round::Int=0,    # measured time of one round (a wave of processes); 0 ⇒ no preflight
    startup_seconds::Int=0,      # per-job depot preparation and stage-in
    timestamp::String="",        # names the job (<phase>_<timestamp>); empty ⇒ now
    models_file::String="",      # test jobs: the model each command loads, line for line; empty ⇒ no check
    depot_tag::String="",        # jobs sharing a tag prepare the depot once between them; empty ⇒ this job alone
)::String
    """
    Write the SLURM array job that runs the commands in `commands_file`. Each array
    task takes a contiguous slice of the file and runs it through GNU parallel,
    `n_gpus_per_node × jobs_per_gpu` processes at a time on GPUs, or `n_cpus` at a
    time on CPUs. Returns the script's path; the caller prints how to submit it.
    """
    if !(mode in (:train, :test))
        throw(ArgumentError("mode must be :train or :test, got :$(mode)"))
    end
    if n_commands < 1
        throw(ArgumentError("run_on_SLURM: there are no commands to run."))
    end
    if jobs_per_gpu < 1
        throw(ArgumentError("jobs_per_gpu = $(jobs_per_gpu) must be at least 1."))
    end
    is_test_mode::Bool = mode == :test
    phase::String = "train"
    if is_test_mode
        phase = "test"
    end
    job_timestamp::String = timestamp
    if isempty(job_timestamp)
        job_timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    end

    target_dir::String = joinpath(working_dir, codename)
    commands_dir::String = joinpath(target_dir, "cluster")
    mkpath(joinpath(commands_dir, "logs"))
    job_name::String = "$(phase)_$(job_timestamp)"
    slurm_script_file::String = joinpath(commands_dir, "$(job_name).sh")
    # One .out/.err per array task: without %a every task writes the same file.
    output_file::String = joinpath(commands_dir, "$(job_name)_task%a.out")
    error_file::String = joinpath(commands_dir, "$(job_name)_task%a.err")
    # Every job shares ONE Julia depot ($SCRATCH/.julia), so a tag's lock lives in
    # one folder for all codenames: the jobs of a submission over several
    # codenames still prepare the depot once, never two at a time.
    depot_name::String = job_name
    depot_lock_dir::String = commands_dir
    if !isempty(depot_tag)
        depot_name = depot_tag
        depot_lock_dir = joinpath(working_dir, ".depot_locks")
        mkpath(depot_lock_dir)
    end

    # Processes at a time in one array task, how the commands spread over tasks,
    # and whether the tasks can finish (batch_run.jl has already run the same
    # check for every job of the submission before writing any of them).
    layout::SlurmTaskLayout = slurm_task_layout(n_commands, use_gpu, n_cpus, n_gpus_per_node, jobs_per_gpu,
                                                max_nodes)
    jobs_per_task::Int = layout.jobs_per_task
    n_tasks::Int = layout.n_tasks
    commands_per_task::Int = layout.commands_per_task
    estimate::Tuple{String, Int} = walltime_estimate(phase, n_commands, layout, max_nodes, wall_time,
                                                     seconds_per_round, startup_seconds)
    estimate_text::String = estimate[1]
    if estimate[2] >= 70
        @warn "walltime preflight: $(phase) job $(estimate_text) is tight; consider a longer wall_time or more " *
              "max_nodes."
    end
    if !use_gpu && endswith(account, "_gpu")
        @warn "run_on_SLURM: this job requests no GPU but account = \"$(account)\" is a GPU account; " *
              "CPU jobs normally go to the account without the _gpu suffix."
    end

    # A test process that holds a whole card may use the cores that come with it;
    # everything else is one thread per process.
    threads_per_job::Int = 1
    if is_test_mode
        threads_per_job = max(1, n_cpus ÷ jobs_per_task)
    end

    # The test job's batch sizer must be given the card's VRAM: SLURM_MEM_PER_GPU,
    # its fallback, is HOST memory, and a batch sized to it can overrun the card.
    gpu_memory_mb::String = ""
    if use_gpu && is_test_mode
        vram_gb::Int = vram_per_gpu
        if vram_gb <= 0
            vram_gb = card_vram_gb(gpu_type)
        end
        if vram_gb <= 0
            throw(ArgumentError("no VRAM known for gpu_type = \"$(gpu_type)\": set vram_per_gpu " *
                                "(GB per card) in the settings."))
        end
        gpu_memory_mb = string(vram_gb * 1024 * 85 ÷ (100 * jobs_per_gpu))
    end

    sbatch_lines::Vector{String} = [
        "#SBATCH --account=$(account)",
        "#SBATCH --job-name=$(job_name)",
        "#SBATCH --output=$(output_file)",
        "#SBATCH --error=$(error_file)",
        "#SBATCH --array=0-$(n_tasks - 1)",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=$(n_cpus)",
    ]
    # --mem-per-gpu and --mem-per-cpu are mutually exclusive in SLURM.
    if use_gpu && !isempty(mem_per_gpu)
        push!(sbatch_lines, "#SBATCH --mem-per-gpu=$(mem_per_gpu)")
    else
        push!(sbatch_lines, "#SBATCH --mem-per-cpu=$(mem_per_cpu)")
    end
    if use_gpu
        # The model specifier is how Alliance clusters pick the card; without it
        # the job "may be rejected or sent to an arbitrary GPU".
        gpu_request::String = string(n_gpus_per_node)
        if isempty(gpu_type)
            @warn "run_on_SLURM: no gpu_type given; the cluster may reject the job or pick any GPU."
        else
            gpu_request = "$(gpu_type):$(n_gpus_per_node)"
        end
        push!(sbatch_lines, "#SBATCH --gpus-per-node=$(gpu_request)")
    elseif !isempty(mem_per_gpu)
        @warn "run_on_SLURM: mem_per_gpu = $(repr(mem_per_gpu)) is ignored: this job requests no GPU, " *
              "so it uses --mem-per-cpu=$(mem_per_cpu)."
    end
    append!(sbatch_lines, [
        "#SBATCH --time=$(wall_time)",
        # SIGTERM to the batch shell 300 s before the wall, so the trap in the
        # template stages out every finished command before the hard kill.
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=$(email_address)",
    ])

    template_values::Dict{String, String} = Dict(
        "SBATCH_LINES" => join(sbatch_lines, "\n"),
        "JOB_NAME" => job_name,
        "PHASE" => phase,
        "USE_GPU" => string(Int(use_gpu)),
        "CODENAME" => codename,
        "DATA_ROOT" => working_dir,
        "COMMANDS_FILE" => commands_file,
        "COMMANDS_PER_TASK" => string(commands_per_task),
        "JOBS_PER_TASK" => string(jobs_per_task),
        "THREADS_PER_JOB" => string(threads_per_job),
        "GPU_MEMORY_MB" => gpu_memory_mb,
        "JULIA_MODULE" => julia_module,
        "CUDA_MODULE" => cuda_module,
        "MODELS_FILE" => models_file,
        # One depot preparation per tag: one task holds the lock, the rest wait —
        # including the tasks of a test job chained on a training job of the same tag.
        "DEPOT_LOCK" => joinpath(depot_lock_dir, "$(depot_name)_depot.lock"),
        "DEPOT_READY" => joinpath(depot_lock_dir, "$(depot_name)_depot.ready"),
    )
    template_text::String = read(joinpath(@__DIR__, "slurm_job_template.sh"), String)
    script_text::String = fill_job_template(template_text, template_values)
    write(slurm_script_file, script_text)

    device_label::String = "CPU, $(n_cpus) process(es) per task"
    if use_gpu
        device_label = "GPU ($(gpu_type)), $(n_gpus_per_node) card(s) per task, $(jobs_per_gpu) process(es) per card"
    end
    println("\n=== Job Details ===")
    println("  Job              : $(job_name)  ($(phase))")
    println("  Target codename  : $(codename)")
    println("  Commands         : $(n_commands)")
    println("  Devices          : $(device_label)")
    println("  Array tasks      : $(n_tasks), $(commands_per_task) command(s) each, $(jobs_per_task) at a time")
    println("  CPUs per task    : $(n_cpus)")
    if use_gpu && !isempty(mem_per_gpu)
        println("  Memory           : $(mem_per_gpu) per GPU (host RAM)")
    else
        println("  Memory           : $(mem_per_cpu) per CPU")
    end
    if !isempty(gpu_memory_mb)
        println("  GPU_MEMORY       : $(gpu_memory_mb)M (85% of the card, for the batch sizer)")
    end
    println("  Wall time        : $(wall_time)")
    println("  Estimate         : $(estimate_text)")
    println("\n=== Output Files ===")
    println("  Commands file    : $(commands_file)")
    if !isempty(models_file)
        println("  Models file      : $(models_file)  (checked by every task before it tests)")
    end
    println("  SLURM script     : $(slurm_script_file)")

    return slurm_script_file
end

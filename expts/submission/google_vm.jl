# ============================================================================
# google_vm.jl — Google Cloud VM runner
# ============================================================================
#
# Exposes:  run_on_Google_VM(commands_file, output_file, n_cpus)
#           shut_down_vm(job_cmd)
#
# Emits a run_<ts>.sh that runs the commands via GNU parallel and then
# shuts the VM down (`sudo shutdown -h now`). Called from batch_commands.jl
# when cluster_backend == "google_vm".
# ============================================================================

using Dates

function run_on_Google_VM(commands_file::String, output_file::String, n_cpus::Int=10)
    """
    Run the commands in `commands_file` in parallel on a Google Cloud VM, save
    results to `output_file`, and halt the VM when done. Writes a run_<ts>.sh
    next to the commands file and prints the `bash <script>` invocation.
    """
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    shell_script_file = joinpath(dirname(commands_file), "run_$(timestamp).sh")
    job_cmd = """parallel --bar --keep-order --jobs $(n_cpus) --results $(output_file) --arg-file $(commands_file)"""
    open(shell_script_file, "w") do io
        wrapped_cmd = shut_down_vm(job_cmd)
        println(io, wrapped_cmd)
    end
    println("Run simulations with:")
    println("bash $(shell_script_file)\n")
    return shell_script_file
end

function shut_down_vm(job_cmd::String)
    """
    Wrap `job_cmd` with a bash `set -e` preamble and a `sudo shutdown -h now`
    tail, so the VM halts as soon as the parallel job finishes (success or
    fail — `set -e` makes the shutdown reachable in both cases because the
    script exits on the parallel line if it errored, without shutdown).
    """
    wrapped_cmd = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Run the commands in parallel and save results to a file",
        job_cmd,
        "",
        "# Stop the instance",
        "sudo shutdown -h now",
    ]
    return join(wrapped_cmd, "\n")
end

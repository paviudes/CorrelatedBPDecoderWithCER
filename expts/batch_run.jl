using Printf
using LinearAlgebra

function generate_parallel_commands(
    pvals::AbstractVector{<:Real},
    qvals::AbstractVector{<:Real},
    n_samples::Int;
    codename::String="aps",
    n_hidden_layers::Int=100,
    n_epochs::Int=10,
    batch_size::Int=2,
    retrain::Bool=false,
    julia_project::String="./../",
    commands_file::String="commands.txt",
    results_file::String="simulation_results.json",
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
                --retrain $(retrain)"""

            cmd = replace(cmd, "\n" => " ")

            println(io, cmd)
        end
    end

    println("$(length(pvals) * length(qvals) * n_samples) commands written to: $commands_file\n")

    println("Run simulations with:")
    println("parallel --keep-order --jobs $ncpus --results $(results_file) --arg-file $commands_file")
end

function main()
    generate_parallel_commands(
        0.001:0.001:0.005,
        0.3:0.04:0.66,
        10;
        codename = "aps_7q_Hamm_code_data",
        n_hidden_layers = 5,
        n_epochs = 5,
        batch_size = 2,
        retrain = false,
        julia_project = "./../",
        commands_file = "./../data/aps_7q_Hamm_code_data/commands.txt",
        results_file = "./../data/aps_7q_Hamm_code_data/simulation_results.json",
        ncpus = 56
    )
end
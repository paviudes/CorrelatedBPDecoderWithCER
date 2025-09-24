using LinearAlgebra
using DataFrames
using CSV

struct DecoderStatistics
    """
    Statistics for the Belief Propagation decoder.
    """
    error_model_name::String
    error_model_parameters_description::String
    num_samples_per_error_rate::Int
    num_iterations_BP::Int
    num_failures::Int
    failures::Vector{Bool}
    average_logical_error_rate::Float64
    std_logical_error_rate::Float64
    runtime::Float64

    function DecoderStatistics(error_model_name::String, error_model_parameters_description::String, num_samples_per_error_rate::Int=0; num_failures::Int=0, failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), num_iterations_BP::Int=0, runtime::Float64=0.0)
        if num_samples_per_error_rate < 0
            error("Number of samples per error rate must be non-negative.")
        end

        if (num_failures == 0) && (length(failures) > 0)
            num_failures = count(failures)
        end

        if (num_samples_per_error_rate == 0) || (num_failures == 0)
            # warning("Number of failures is zero. Standard deviation will be set to zero.")
            average_logical_error_rate = 0.0
            std_logical_error_rate = 0.0
        else
            average_logical_error_rate = num_failures / num_samples_per_error_rate 
            std_logical_error_rate = compute_std_assuming_bernoulli(average_logical_error_rate, num_iterations_BP)
        end
        new(error_model_name, error_model_parameters_description, num_samples_per_error_rate, num_iterations_BP, num_failures, failures, average_logical_error_rate, std_logical_error_rate, runtime)
    end

    function DecoderStatistics(error_model_name::String, error_model_parameters_description::String, num_samples_per_error_rate::Int, num_iterations_BP::Int; num_failures::Int=0, failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), average_logical_error_rate::Float64=0.0, std_logical_error_rate::Float64=0.0, runtime::Float64=0.0)
        new(error_model_name, error_model_parameters_description, num_samples_per_error_rate, num_iterations_BP, num_failures, failures, average_logical_error_rate, std_logical_error_rate, runtime)
    end
end

function compute_std_assuming_bernoulli(μ::Float64, n::Int)::Float64
    """
    Compute the standard deviation of a Bernoulli random variable given its mean (μ) and number of trials (n).
    The standard deviation is given by sqrt(μ * (1 - μ) / n).
    """
    if n == 0
        return 0.0
    end
    if (μ < 0.0 || μ > 1.0)
        error("Mean (μ) must be in the range [0, 1].")
    end
    σ = sqrt(μ * (1 - μ) / n)
    return σ
end

function print_decoder_statistics(stats::DecoderStatistics; io::IO=stdout)
    """
    Print the decoder statistics in a readable format.
    """
    println(io, "Decoder Statistics:")
    println(io, "-------------------")
    println(io, "Error Model: ", stats.error_model_name)
    println(io, "Error Model Parameters: ", stats.error_model_parameters_description)
    println(io, "Number of Samples per Error Rate: ", stats.num_samples_per_error_rate)
    println(io, "Number of BP Iterations: ", stats.num_iterations_BP)
    println(io, "Number of Failures: ", stats.num_failures)
    println(io, "Average Logical Error Rate: ", stats.average_logical_error_rate)
    println(io, "Standard Deviation of Logical Error Rate: ", stats.std_logical_error_rate)
    println(io, "Runtime: ", stats.runtime)
end

function get_output_filename(error_model_name::String, error_model_parameters_description::String; prefix::String="./../data")::String
    """
    Generate an output filename based on the error model name and parameters.
    Eg. decoder_stats_Ballistic_Error_Model_per_qubit_error_prob_0_09__neighbour_error_prob_0_03.json
    """
    sanitized_name = replace(error_model_name, r"\s+" => "_")
    sanitized_params = replace(error_model_parameters_description, r"[^\w]" => "_")
    output_filename = "$(prefix)/decoder_stats_$(sanitized_name)_$(sanitized_params).json"
    return output_filename
end

function get_output_filename(stats::DecoderStatistics, prefix::String="./../data")::String
    """
    Generate an output filename based on the error model name and parameters.
    Eg. decoder_stats_Ballistic_Error_Model_per_qubit_error_prob_0_09__neighbour_error_prob_0_03.json
    """
    output_filename = get_output_filename(stats.error_model_name, stats.error_model_parameters_description; prefix=prefix)
    return output_filename
end

function save_decoder_statistics(stats::DecoderStatistics; outputfile::String=get_output_filename(stats))
    """
    Write the decoder statistics to a JSON file.
    """
    stats_dict = Dict(
        "error_mode_name" => stats.error_model_name,
        "error_model_parameters" => stats.error_model_parameters_description,
        "num_samples_per_error_rate" => stats.num_samples_per_error_rate,
        "num_iterations_BP" => stats.num_iterations_BP,
        "num_failures" => stats.num_failures,
        "average_logical_error_rate" => stats.average_logical_error_rate,
        "std_logical_error_rate" => stats.std_logical_error_rate,
        "runtime" => stats.runtime
    )

    open(outputfile, "w") do io
        JSON.print(io, stats_dict)
    end
end

function load_decoder_statistics(inputfile::String)::DecoderStatistics
    """
    Load decoder statistics from a JSON file.
    """
    stats_dict = open(inputfile, "r") do io
        JSON.parse(io)
    end

    error_model_name = stats_dict["error_mode_name"]
    error_model_parameters = stats_dict["error_model_parameters"]

    num_samples_per_error_rate = stats_dict["num_samples_per_error_rate"]
    num_iterations_BP = stats_dict["num_iterations_BP"]
    num_failures = stats_dict["num_failures"]
    average_logical_error_rate = stats_dict["average_logical_error_rate"]
    std_logical_error_rate = stats_dict["std_logical_error_rate"]
    runtime = stats_dict["runtime"]

    # Create a DecoderStatistics instance
    stats = DecoderStatistics(
        error_model_name,
        error_model_parameters,
        num_samples_per_error_rate,
        num_iterations_BP;
        num_failures=num_failures,
        average_logical_error_rate=average_logical_error_rate,
        std_logical_error_rate=std_logical_error_rate,
        runtime=runtime
    )

    return stats
end

function collect_decoder_statistics(error_model_name::String, parameter_ranges::Dict{String, <:AbstractVector}; prefix::String="./../data")::DataFrame
    """
    Collect decoder statistics from simulations with different error model parameters.
    The `parameter_ranges` dictionary describes the ranges of parameters that have been swept over.
    It is of the format: Dict("param1" => [val1, val2, ...], "param2" => [val1, val2, ...], ...)
    This function assumes that the simulations have already been run and their statistics saved in JSON files.
    We want to read the data from these files and into a Dataframe which has N + 2 columns:
    where N is the number of parameters, i.e., the keys of `parameter_ranges`.
    The last two columns are the average logical error rate and its standard deviation.
    """
    param_names = collect(keys(parameter_ranges))
    # Create a dataframe with each column corresponding to a parameter name (key) in `parameter_ranges`
    stats_dataframe = DataFrame(
        [Symbol(param) => Float64[] for param in param_names]...,
        Symbol("summary") => DecoderStatistics[]
    )
    # Iterate over all combinations of parameter values
    parameter_combinations = Iterators.product(values(parameter_ranges)...)
    for param_values in parameter_combinations
        ballistic_error_model = BallisticErrorModel(param_values...)
        output_filename = get_output_filename(ballistic_error_model.name, ballistic_error_model.parameters_description; prefix=prefix)
        if isfile(output_filename)
            stats = load_decoder_statistics(output_filename)
            new_row = (; [Symbol(k) => v for (k, v) in zip(param_names, param_values)]..., Symbol("summary") => stats)
            push!(stats_dataframe, new_row)
        else
            @warn "File $(output_filename) does not exist. Skipping this parameter set."
        end
    end
    save_dataframe(error_model_name, stats_dataframe; prefix=prefix)
    return stats_dataframe
end

function save_dataframe(error_model_name::String, df::DataFrame; prefix::String="./../data")::String
    """
    Save the dataframe to a CSV file.
    """
    sanitized_error_model_name = replace(error_model_name, r"\s+" => "_")
    output_filename = "$(prefix)/$(sanitized_error_model_name)_dataframe.csv"
    CSV.write(output_filename, df)
    return output_filename
end
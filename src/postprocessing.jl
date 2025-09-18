using LinearAlgebra

struct DecoderStatistics
    """
    Statistics for the Belief Propagation decoder.
    """
    error_model::ErrorModel
    num_samples_per_error_rate::Int
    num_iterations_BP::Int
    num_failures::Int
    failures::Vector{Bool}
    average_logical_error_rate::Float64
    std_logical_error_rate::Float64
    runtime::Float64

    function DecoderStatistics(error_model::ErrorModel, num_samples_per_error_rate::Int=0; num_failures::Int=0, failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), num_iterations_BP::Int=0, runtime::Float64=0.0)
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
        new(error_model, num_samples_per_error_rate, num_iterations_BP, num_failures, failures, average_logical_error_rate, std_logical_error_rate, runtime)
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
    println(io, "Error Model: ", stats.error_model.name)
    println(io, "Error Model Parameters: ", stats.error_model.parameters_description)
    println(io, "Number of Samples per Error Rate: ", stats.num_samples_per_error_rate)
    println(io, "Number of BP Iterations: ", stats.num_iterations_BP)
    println(io, "Number of Failures: ", stats.num_failures)
    println(io, "Average Logical Error Rate: ", stats.average_logical_error_rate)
    println(io, "Standard Deviation of Logical Error Rate: ", stats.std_logical_error_rate)
    println(io, "Runtime: ", stats.runtime)
end

function get_output_filename(stats::DecoderStatistics, prefix::String="./../data/decoder_stats")::String
    """
    Generate an output filename based on the error model name and parameters.
    """
    sanitized_name = replace(stats.error_model.name, r"\s+" => "_")
    sanitized_params = replace(stats.error_model.parameters_description, r"[^\w]" => "_")
    output_filename = "$(prefix)_$(sanitized_name)_$(sanitized_params).json"
    return output_filename
end

function save_decoder_statistics(stats::DecoderStatistics; outputfile::String=get_output_filename(stats))
    """
    Write the decoder statistics to a JSON file.
    """
    stats_dict = Dict(
        "error_mode_name" => stats.error_model.name,
        "error_model_parameters" => stats.error_model.parameters_description,
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
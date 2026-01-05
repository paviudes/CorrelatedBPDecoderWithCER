using LinearAlgebra
using DataFrames
using DataStructures
using CSV

struct DecoderStatistics
    """
    Statistics for the Belief Propagation decoder.
    """
    algo::String
    error_model_name::String
    error_model_parameters_description::String
    num_samples_per_error_rate::Int
    n_iterations_BP::Int # denotes n_layers in Neural Network BP
    rounds_per_BP::Int # denotes n_epochs in Neural Network BP
    weight_soft_constraint::Float64 # denotes correlation_strength for Neural Network BP
    num_failures::Int
    average_logical_error_rate::Float64
    std_logical_error_rate::Float64
    runtime::Float64

    function DecoderStatistics(algo::String, error_model_name::String, error_model_parameters_description::String, num_samples_per_error_rate::Int, num_iterations_BP::Int, num_rounds_per_iteration_BP::Int, weight_soft_constraint::Float64; num_failures::Int=0, failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), runtime::Float64=0.0)
        if !(algo in ("SumProduct", "MinSum", "NN"))
            throw(ArgumentError("Algorithm must be either 'SumProduct', 'MinSum' or 'NN'."))
        end
        if num_samples_per_error_rate < 0
            throw(ArgumentError("Number of samples per error rate must be non-negative."))
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
        new(algo, error_model_name, error_model_parameters_description, num_samples_per_error_rate, num_iterations_BP, num_rounds_per_iteration_BP, weight_soft_constraint, num_failures, average_logical_error_rate, std_logical_error_rate, runtime)
    end
end

function record_decoder_statistics(stats::DecoderStatistics)
    """
    Print the statistics in a JSON format so that they can be easily printed into a file using GNU `parallel`.
    """
    stats_dict = Dict(
        name => getfield(stats, name) for name in fieldnames(DecoderStatistics)
    )
    println(JSON.json(stats_dict)) # Ensure a newline after the JSON object
end

"""
Check that all provided symbols are valid fields of `DecoderStatistics`.
Throws an ArgumentError if an invalid key is found.
"""
function check_valid_fields_DecoderStatistics(keys::Vector{Symbol})::Vector{Symbol}
    allowed = fieldnames(DecoderStatistics)
    valid = intersect(keys, allowed)
    invalid = setdiff(keys, allowed)
    if !isempty(invalid)
        @warn ("Invalid keys found: $(collect(invalid))")
    end
    return valid
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

function collect_decoder_statistics(simulation_output_file::String)::DataFrame
    """
    Collect decoder statistics from simulations with settings.
    The `output_data_file` is is a text file where each line corresponds to a dictionary specifying all the parameters of a simulation run, which are also parameters that define `DecoderStatistics`.
    Each line should be a valid dictionary of the form: {`parameter_name1` => `value1`, `parameter_name2` => `value2`, ...}.
    """
    # Create an empty DataFrame to store the statistics
    parameter_names = fieldnames(DecoderStatistics)
    # Create an empty DataFrame with the field names in `parameter_names` and the corresponding types in `parameter_types`
    stats_dataframe = DataFrame(
        [name => fieldtype(DecoderStatistics, name)[] for name in parameter_names]...
    )
    # println("Empty DataFrame:\n", stats_dataframe)
    fp = open(simulation_output_file, "r")
    for line in eachline(fp)
        # Interpret the line as a dictionary, where the line is given as a JSON string
        line_dict = JSON.parse(line)
        # println("Parsed line dictionary: ", line_dict)
        # Create a new row in the DataFrame with the values from the dictionary, where the keys correspond to the parameter names
        new_dataframe_row = DataFrame(
            [Symbol(name) => [line_dict[name]] for name in keys(line_dict)]...
        )
        # println("New DataFrame row:\n", new_dataframe_row)
        append!(stats_dataframe, new_dataframe_row)
    end
    close(fp)
    return stats_dataframe
end

function save_decoder_dataframe(decoder_stats::DataFrame, output_filename::String="./../data/debankan/explicit_error_model_focused_data.csv")::String
    """
    Save the dataframe to a CSV file.
    """
    # If a file with the same name already exists, then load the data and append the new data to it
    if isfile(output_filename)
        existing_data = CSV.read(output_filename, DataFrame)
        # Append the new data dataframe to the existing dataframe
        append!(decoder_stats, existing_data)
    end
    CSV.write(output_filename, decoder_stats)
    return output_filename
end

function check_approximate(col::AbstractVector, val; atol::Float64=1e-8, rtol::Float64=1e-5)::BitVector
    """
    Check if the values in the column are approximately equal to the given value `val` using `isapprox` for Real values,
    or == for String values. Returns a BitVector indicating which elements are approximately equal or equal.
    """
    if eltype(col) <: Real && isa(val, Real)
        # println("Check if elements the column\n", col, "\n are approximately equal to ", val, ". Result: ", isapprox.(col, val; atol=atol, rtol=rtol))
        return isapprox.(col, val; atol=atol, rtol=rtol)
    elseif eltype(col) <: AbstractString && isa(val, AbstractString)
        # println("Check if elements the column\n", col, "\n are equal to ", val, ". Result: ", col .== val)
        return col .== val
    else
        # Fallback to == for other types
        return col .== val
    end
end

function extract_collected_data(stats_dataframe::DataFrame, select_parameters::Dict{Symbol, AbstractVector{<:Any}}, display_parameters::Vector{Symbol})::DataFrame
    """
    Extract data from a dataframe that has a specific set of columns and rows corresponding to the values for the columns.
    # Define a readable dataframe which has the following columns:
    # - All the columns in `display_parameters`.
    # - All the rows that whose values in the columns corresponding to the keys in `select_parameters` match the values in `select_parameters`.
    """
    # Check if the selected parameters are valid
    valid_parameter_names = check_valid_fields_DecoderStatistics(collect(keys(select_parameters)))
    # println("Valid parameter names: ", valid_parameter_names)
    valid_parameter_values = Iterators.product(collect([select_parameters[param] for param in valid_parameter_names])...)
    # println("Valid parameter values: ", collect(valid_parameter_values))

    # Create a focused dataframe with columns as the parameter names.
    focused_dataframe = DataFrame(
        #[name => fieldtype(DecoderStatistics, name)[] for name in valid_parameter_names]...,
        [name => fieldtype(DecoderStatistics, name)[] for name in display_parameters]...
    )
    # println("Focused DataFrame columns: ", names(focused_dataframe))
    # Filter the stats_dataframe to only include rows that match the selected parameter values
    for values in valid_parameter_values
        # Search for rows in the `stats_dataframe` where the columns corresponding to the `valid_parameter_names` match the values in `values`
        filter_condition = reduce((acc, (param, val)) -> acc .& check_approximate(stats_dataframe[!, param], val), zip(valid_parameter_names, values), init=trues(nrow(stats_dataframe)))
        # For the selected rows, print the columns in `valid_parameter_names` and `display_parameters`
        matching_rows = stats_dataframe[filter_condition, :]
        # println("Parameter names: ", valid_parameter_names)
        # println("Matching rows for values $(values):\n", matching_rows)
        # If there are matching rows, add them to the focused dataframe
        if nrow(matching_rows) > 0
            # append!(focused_dataframe, matching_rows[:, vcat(valid_parameter_names, display_parameters)])
            append!(focused_dataframe, matching_rows[:, display_parameters])
        end
    end

    # Add the selected parameters to the readable dataframe
    # println("=============================")
    # println("Summary\n", focused_dataframe)
    return focused_dataframe
end
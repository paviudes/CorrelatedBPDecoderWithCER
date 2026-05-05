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

function record_decoder_statistics(stats::DecoderStatistics, output_filename::String="./../data/decoder_statistics.csv")::String
    """
    Print the statistics in a JSON format so that they can be easily printed into a file using GNU `parallel`.
    """
    stats_dict = Dict(
        name => getfield(stats, name) for name in fieldnames(DecoderStatistics)
    )
    println(JSON.json(stats_dict)) # Ensure a newline after the JSON object
    
    # Save the statistics to a CSV file
    stats_dataframe = DataFrame(stats_dict)
    CSV.write(output_filename, stats_dataframe)
    return output_filename
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

function collect_decoder_statistics_for_ballistic_data(per_qubit_error_probs::AbstractVector{<:Real}, neighbour_error_probs::AbstractVector{<:Real}, num_samples_per_error_rate::Int, n_layers::Int, n_epochs::Int; prefix::String="./../data")::DataFrame
    """
    Collect decoder statistics for the ballistic error model data.
    We have one file summarizing the result for each simulation run, whose name is
        simulation_results_test_ballistic_p_<p>_q_<q>_s_<s>_nlayers_<n_layers>_epochs_<n_epochs>_trained_using_<training_file>.csv
    where
        - <p> is the per-qubit error probability
        - <q> is the neighbour error probability
        - <s> is the number of samples per error rate
        - <n_layers> is the number of layers in the Neural BP model (denoted as n_iterations_BP in `DecoderStatistics`)
        - <n_epochs> is the number of epochs for training the Neural BP model (denoted as rounds_per_iteration_BP in `DecoderStatistics`)
        - <training_file> is the name of the file used for training the Neural BP model (without path and extension): train_ballistic_p_0.001_q_0.3_s_1
    In each of these files we have a DataFrame of the type `DecoderStatistics` with the statistics for that simulation run.
    We will collect all these statistics into a single DataFrame and return it.
    """
    # precompute the number of entries: this is the number of combinations of per_qubit_error_probs, neighbour_error_probs and num_samples_per_error_rate for which we have data files.
    num_files = 0
    missing_files = String[]
    for p in per_qubit_error_probs, q in neighbour_error_probs, s in 1:num_samples_per_error_rate
        training_file = "train_ballistic_p_$(p)_q_$(q)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_ballistic_p_$(p)_q_$(q)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
        if isfile(results_file)
            num_files += 1
        else
            push!(missing_files, results_file)
        end
    end
    if (size(missing_files, 1) > 0)
        @warn ("$(size(missing_files, 1)) files are missing:\n$(missing_files)")
    end

    all_stats = DataFrame(
        algo = Vector{String}(undef, num_files),
        error_model_name = Vector{String}(undef, num_files),
        error_model_parameters_description = Vector{String}(undef, num_files),
        num_samples_per_error_rate = Vector{Int}(undef, num_files),
        n_iterations_BP = Vector{Int}(undef, num_files),
        rounds_per_BP = Vector{Int}(undef, num_files),
        weight_soft_constraint = Vector{Float64}(undef, num_files),
        num_failures = Vector{Int}(undef, num_files),
        average_logical_error_rate = Vector{Float64}(undef, num_files),
        std_logical_error_rate = Vector{Float64}(undef, num_files),
        runtime = Vector{Float64}(undef, num_files)
    )
    file_index = 1
    for p in per_qubit_error_probs, q in neighbour_error_probs, s in 1:num_samples_per_error_rate
        training_file = "train_ballistic_p_$(p)_q_$(q)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_ballistic_p_$(p)_q_$(q)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
        if isfile(results_file)
            stats_dataframe = CSV.read(results_file, DataFrame)
            # Fill the dataframe fields
            all_stats[file_index, :algo] = stats_dataframe[1, :algo]
            all_stats[file_index, :error_model_name] = stats_dataframe[1, :error_model_name]
            all_stats[file_index, :error_model_parameters_description] = stats_dataframe[1, :error_model_parameters_description]
            all_stats[file_index, :num_samples_per_error_rate] = stats_dataframe[1, :num_samples_per_error_rate]
            all_stats[file_index, :n_iterations_BP] = stats_dataframe[1, :n_iterations_BP]
            all_stats[file_index, :rounds_per_BP] = stats_dataframe[1, :rounds_per_BP]
            all_stats[file_index, :weight_soft_constraint] = stats_dataframe[1, :weight_soft_constraint]
            all_stats[file_index, :num_failures] = stats_dataframe[1, :num_failures]
            all_stats[file_index, :average_logical_error_rate] = stats_dataframe[1, :average_logical_error_rate]
            all_stats[file_index, :std_logical_error_rate] = compute_std_assuming_bernoulli(all_stats[file_index, :average_logical_error_rate], all_stats[file_index, :num_samples_per_error_rate])
            all_stats[file_index, :runtime] = stats_dataframe[1, :runtime]
            file_index += 1
        end
    end
    return all_stats
end

function collect_standard_decoder_statistics_for_ballistic_data(prefix::String="./../data", ntrials::Int=100000; standard_BP_output_file::String="standard_bp_failure_rates.txt")::DataFrame
    """
    Collect decoder statistics for the ballistic error model data for the standard BP decoder (i.e., not the Neural BP decoder).
    We have one file summarizing the result for each simulation run, whose name is results/standard_BP_failure_rates.txt.
    This file contains one line for each combination of per_qubit_error_probs, neighbour_error_probs and num_samples_per_error_rate, where
    each line is formatted as follows.
    <per_qubit_error_prob> <neighbour_error_prob> <sample> <total number of failures>

    We will read this file and collect the statistics into a DataFrame and return it.
    The DataFrame will have the following columns:
    algo::String (set to "SumProduct")
    error_model_name::String (set to "ExplicitErrorModel")
    error_model_parameters_description::String (set to "./../data/$(dirname)/testing_data/test_ballistic_p_<per_qubit_error_prob>_q_<neighbour_error_prob>_s_<sample>.txt")
    num_samples_per_error_rate::Int (set to ntrials)
    n_iterations_BP::Int # denotes n_layers in Neural Network BP (set to 0, since we don't have this data for the standard BP decoder)
    rounds_per_BP::Int # denotes n_epochs in Neural Network BP (set to 0, since we don't have this data for the standard BP decoder)
    weight_soft_constraint::Float64 # denotes correlation_strength for Neural Network BP (set to 0.0, since we don't have this data for the standard BP decoder)
    num_failures::Int (set to the total number of failures read from the file)
    average_logical_error_rate::Float64 (set to num_failures / ntrials)
    std_logical_error_rate::Float64 (computed using compute_std_assuming_bernoulli with μ = num_failures / ntrials and n = ntrials)
    runtime::Float64 (set to 0.0, since we don't have runtime data for the standard BP decoder)
    """
    results_file = "$(prefix)/results/$(standard_BP_output_file)"
    if !isfile(results_file)
        @warn ("File $(results_file) is missing.")
        return DataFrame()
    end

    # Estimate the number of lines in the file to preallocate the DataFrame
    num_lines = 0
    open(results_file, "r") do fp
        for _ in eachline(fp)
            num_lines += 1
        end
    end
    stats_dataframe = DataFrame(
        algo = Vector{String}(undef, num_lines),
        error_model_name = Vector{String}(undef, num_lines),
        error_model_parameters_description = Vector{String}(undef, num_lines),
        num_samples_per_error_rate = Vector{Int}(undef, num_lines),
        n_iterations_BP = Vector{Int}(undef, num_lines),
        rounds_per_BP = Vector{Int}(undef, num_lines),
        weight_soft_constraint = Vector{Float64}(undef, num_lines),
        num_failures = Vector{Int}(undef, num_lines),
        average_logical_error_rate = Vector{Float64}(undef, num_lines),
        std_logical_error_rate = Vector{Float64}(undef, num_lines),
        runtime = Vector{Float64}(undef, num_lines)
    )
    line_index = 1
    open(results_file, "r") do fp
        for line in eachline(fp)
            split_line = split(line)
            if length(split_line) != 4
                @warn ("Invalid line format: $line")
                continue
            end
            per_qubit_error_prob = split_line[1]
            neighbour_error_prob = split_line[2]
            sample = split_line[3]
            num_failures = parse(Int, split_line[4])
            # Fill the DataFrame row with the corresponding values
            stats_dataframe[line_index, :algo] = "SumProduct"
            stats_dataframe[line_index, :error_model_name] = "ExplicitErrorModel"
            stats_dataframe[line_index, :error_model_parameters_description] = "$(prefix)/testing_data/test_ballistic_p_$(per_qubit_error_prob)_q_$(neighbour_error_prob)_s_$(sample).txt"
            stats_dataframe[line_index, :num_samples_per_error_rate] = ntrials
            stats_dataframe[line_index, :n_iterations_BP] = 0
            stats_dataframe[line_index, :rounds_per_BP] = 0
            stats_dataframe[line_index, :weight_soft_constraint] = 0.0
            stats_dataframe[line_index, :num_failures] = num_failures
            stats_dataframe[line_index, :average_logical_error_rate] = num_failures / ntrials
            stats_dataframe[line_index, :std_logical_error_rate] = compute_std_assuming_bernoulli(num_failures / ntrials, ntrials)
            stats_dataframe[line_index, :runtime] = 0.0
            line_index += 1
        end
    end
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
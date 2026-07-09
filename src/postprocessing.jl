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

function record_decoder_statistics(stats::DecoderStatistics, output_filename::String="./../data/decoder_statistics.csv")::DataFrame
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
    return stats_dataframe
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
    The simulation output file is expected to be a CSV file containing a DataFrame saved using `record_decoder_statistics`.
    """
    if !isfile(simulation_output_file)
        @warn ("File $(simulation_output_file) is missing.")
        return DataFrame()
    end
    stats_dataframe = CSV.read(simulation_output_file, DataFrame)
    return stats_dataframe
end

function collect_decoder_statistics_correlated(per_qubit_error_probs::AbstractVector{<:Real}, neighbour_error_probs::AbstractVector{<:Real}, num_samples_per_error_rate::Int, n_layers::Int, n_epochs::Int; prefix::String="./../data")::DataFrame
    """
    Collect Neural BP decoder statistics for the correlated (Ballistic) error
    model. There's one file per simulation run, produced by
    `neural_bp_experiments.jl` and named by the same convention it builds
    (see line ~272 of that file):

        simulation_results_test_<pq_tag>_s_<s>_nlayers_<n_layers>_epochs_<n_epochs>_trained_using_train_<pq_tag>_s_<s>.csv

    where `<pq_tag>` is `fmt_probs(p, q)` — the canonical
    max-decimals-padded formatter — so e.g. `p_0.010_q_0.001`. The
    `ballistic_` infix that used to sit here (and in older test/train
    filenames) was dropped when the file-naming convention was unified;
    an on-disk rename script is available at
    `expts/scripts/rename_to_padded_pq.py` if you need to bring legacy
    result CSVs into the new form.

    Arguments:
      - `per_qubit_error_probs`, `neighbour_error_probs`, `num_samples_per_error_rate`
        — the grid we iterate over to reconstruct the expected filenames.
      - `n_layers`, `n_epochs` — pulled from the hyperparams TOML at
        submission time; must match what was passed to
        `neural_bp_experiments.jl`.
      - `prefix` — root path of the codename directory (contains `results/`).
    """
    # precompute the number of entries: this is the number of combinations of per_qubit_error_probs, neighbour_error_probs and num_samples_per_error_rate for which we have data files.
    num_files = 0
    missing_files = String[]
    for p in per_qubit_error_probs, q in neighbour_error_probs, s in 1:num_samples_per_error_rate
        pq_tag = fmt_probs(Float64(p), Float64(q))
        training_file = "train_$(pq_tag)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_$(pq_tag)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
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
        pq_tag = fmt_probs(Float64(p), Float64(q))
        training_file = "train_$(pq_tag)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_$(pq_tag)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
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

function collect_standard_decoder_statistics_correlated(prefix::String="./../data", ntrials::Int=100000; standard_BP_output_file::String="standard_bp_failure_rates.txt")::DataFrame
    """
    Collect standard BP-OSD decoder statistics for the correlated (Ballistic)
    error model. Unlike the Neural BP twin above, standard-decoder results
    are ALREADY aggregated into a single file — `results/<standard_BP_output_file>`
    — with one row per `(per_qubit_prob, neighbour_prob, sample)` triple:

        <per_qubit_error_prob> <neighbour_error_prob> <sample> <total number of failures>

    We parse it into a DataFrame with the same shape as
    `collect_decoder_statistics_correlated`'s output so downstream code
    (plots, comparisons) can consume both interchangeably.

    The `error_model_parameters_description` field is synthesised to look
    like a testing-data filename — kept in sync with the canonical
    `fmt_probs` naming used everywhere else in the pipeline — because the
    plotting code filters DataFrame rows by `occursin(fmt_probs(p, q), ...)`
    against this field.

    Returned columns:
      algo                              — "SumProduct"
      error_model_name                  — "ExplicitErrorModel"
      error_model_parameters_description — "\$(prefix)/testing_data/test_<pq_tag>_s_<sample>.txt"
      num_samples_per_error_rate        — ntrials
      n_iterations_BP, rounds_per_BP, weight_soft_constraint, runtime — 0
      num_failures                      — parsed from the file
      average_logical_error_rate        — num_failures / ntrials
      std_logical_error_rate            — Bernoulli std at μ = LER, n = ntrials
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
            # Build the same `p_<X>_q_<Y>` tag that the neural-BP path and the
            # plotting code use, so DataFrames from both decoders can be
            # filtered/joined with the same `occursin(fmt_probs(p, q), ...)`
            # predicate. Parse the raw strings to Float64 first so
            # `fmt_probs`'s max-decimals-padding runs on numbers, not text.
            pq_tag = fmt_probs(parse(Float64, per_qubit_error_prob),
                               parse(Float64, neighbour_error_prob))
            # Fill the DataFrame row with the corresponding values
            stats_dataframe[line_index, :algo] = "SumProduct"
            stats_dataframe[line_index, :error_model_name] = "ExplicitErrorModel"
            stats_dataframe[line_index, :error_model_parameters_description] = "$(prefix)/testing_data/test_$(pq_tag)_s_$(sample).txt"
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
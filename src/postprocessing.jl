using LinearAlgebra
using DataFrames
using DataStructures
using CSV

struct DecoderStatistics
    """
    Statistics for the Belief Propagation decoder.

    Every field is a Vector, so a single `DecoderStatistics` can hold ONE record
    (all fields length 1) or MANY (e.g. after concatenating the results of a
    parameter sweep). The two inner constructors are:

      1. Scalar/per-simulation form (unchanged call signature): pass one
         simulation's scalar values; the constructor computes the logical error
         rate and its Bernoulli std, then stores every field as a 1-element
         Vector. This keeps existing call sites (neural_bp_experiments.jl etc.)
         working untouched.
      2. Raw form: pass every field as an already-built Vector (all the same
         length). Used by `vcat`/concat and the `DataFrame` builder below.
    """
    algo::Vector{String} # for standard BP, this is either "SumProduct" or "MinSum"; for Neural BP, this is "NN"
    error_model_name::Vector{String} # e.g. "ExplicitErrorModel" or "IsingModel" or "CircuitLevelModel"
    error_model_parameters_description::Vector{String} # e.g. "p=0.0011,q=0.0007" or "p=0.0011" or a filename for an explicit error model
    num_samples_per_error_rate::Vector{Int} # for standard BP, this is the number of trials (ntrials) for a given error rate; for Neural BP, this is the number of test samples.
    n_iterations_BP::Vector{Int} # for standard BP, this is the number of iterations of BP; for Neural BP, this is the number of layers in the trained neural network
    rounds_per_BP::Vector{Int} # for standard BP, this is the number of rounds of BP; for Neural BP, this denotes n_epochs in the trained neural network
    weight_soft_constraint::Vector{Float64} # for standard BP, this is the weight of the soft constraint in the BP decoder; for Neural BP, this parameter is not used and is set to 0.0
    num_failures::Vector{Int} # for standard BP, this is the number of logical failures observed in the trials; for Neural BP, this is the number of logical failures observed in the test samples
    average_logical_error_rate::Vector{Float64} # average logical error rate = num_failures / num_samples_per_error_rate
    std_logical_error_rate::Vector{Float64} # standard deviation of the logical error rate, computed assuming a Bernoulli distribution
    runtime::Vector{Float64} # runtime of the decoder in seconds (for standard BP, this is the total runtime for all trials; for Neural BP, this is the total runtime for all test samples)

    # --- Raw (array) constructor: every field already a Vector. ---------------
    function DecoderStatistics(algo::Vector{String}, error_model_name::Vector{String},
            error_model_parameters_description::Vector{String},
            num_samples_per_error_rate::Vector{Int}, n_iterations_BP::Vector{Int},
            rounds_per_BP::Vector{Int}, weight_soft_constraint::Vector{Float64},
            num_failures::Vector{Int}, average_logical_error_rate::Vector{Float64},
            std_logical_error_rate::Vector{Float64}, runtime::Vector{Float64})
        n = length(algo)
        lengths_match = all(==(n), (length(error_model_name), length(error_model_parameters_description),
                    length(num_samples_per_error_rate), length(n_iterations_BP),
                    length(rounds_per_BP), length(weight_soft_constraint),
                    length(num_failures), length(average_logical_error_rate),
                    length(std_logical_error_rate), length(runtime)))
        if !lengths_match
            throw(ArgumentError("All DecoderStatistics field vectors must have the same length."))
        end
        new(algo, error_model_name, error_model_parameters_description,
            num_samples_per_error_rate, n_iterations_BP, rounds_per_BP,
            weight_soft_constraint, num_failures, average_logical_error_rate,
            std_logical_error_rate, runtime)
    end

    # --- Scalar/per-simulation constructor (unchanged public signature). ------
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
        # Store every field as a 1-element Vector (delegates to the raw form).
        new([algo], [error_model_name], [error_model_parameters_description],
            [num_samples_per_error_rate], [num_iterations_BP], [num_rounds_per_iteration_BP],
            [weight_soft_constraint], [num_failures], [average_logical_error_rate],
            [std_logical_error_rate], [runtime])
    end
end

function Base.vcat(stats::DecoderStatistics...)::DecoderStatistics
    """
    Concatenate several `DecoderStatistics` into one by stacking each field vector.
    Lets you accumulate a sweep's worth of single-record structs into one
    multi-record struct.

    Arguments:
    - `stats...`: One or more `DecoderStatistics` instances to concatenate.
    Returns:
    - A new `DecoderStatistics` instance containing the concatenated data.
    """
    if isempty(stats)
        throw(ArgumentError("vcat needs at least one DecoderStatistics."))
    end
    combined = DecoderStatistics(
        (reduce(vcat, getfield(s, f) for s in stats) for f in fieldnames(DecoderStatistics))...
    )
    return combined
end

"""
    DecoderStatistics(df::DataFrame) -> DecoderStatistics

Build a multi-record `DecoderStatistics` from a DataFrame whose columns are the
struct's field names (e.g. the output of `collect_decoder_statistics`).
"""
function DecoderStatistics(df::DataFrame)::DecoderStatistics
    stats = DecoderStatistics(
        (Vector(df[!, f]) for f in fieldnames(DecoderStatistics))...
    )
    return stats
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

function collect_decoder_statistics(simulation_out_files::Vector{String})::DataFrame
    """
    Load each per-simulation result CSV in `simulation_out_files` (each written
    by `record_decoder_statistics`) and stack them into one DataFrame. Missing
    files are warned about and skipped; the surviving frames are concatenated
    row-wise (`vcat`), giving one combined table over all the input files.
    """
    frames = DataFrame[]
    missing_files = String[]
    for f in simulation_out_files
        if isfile(f)
            push!(frames, CSV.read(f, DataFrame))
        else
            push!(missing_files, f)
        end
    end
    if !isempty(missing_files)
        @warn ("$(length(missing_files)) file(s) missing and skipped:\n$(missing_files)")
    end
    if isempty(frames)
        empty_df = DataFrame()
        return empty_df
    end
    combined = reduce(vcat, frames)
    return combined
end

# Convenience overload for a single file path.
function collect_decoder_statistics(simulation_out_file::String)::DataFrame
    combined = collect_decoder_statistics([simulation_out_file])
    return combined
end

function collect_standard_decoder_statistics(error_type::Symbol; prefix::String="./../data", standard_BP_output_file::String="standard_bp_failure_rates.txt")::DataFrame
    """
    Collect standard BP-OSD decoder statistics from a single aggregated results
    file, for either the two-parameter Ising model or the single-parameter
    circuit-level model. Selected by `error_type`:

      :Ising   — file rows are `<p> <q> <sample> <failures> <total_trials>
                 <average> <sigma>` (7 columns). `num_samples_per_error_rate =
                 total_trials` (column 5), and the average logical error rate
                 (column 6) and its std (column 7) are READ DIRECTLY from the
                 file. The description tag uses the two-parameter `fmt_probs(p, q)`.

      :Circuit — file rows are `<p> <sample> <failures> <total_trials> <average>
                 <sigma>` (6 columns) — the same layout without the `q` column.
                 `num_samples_per_error_rate = total_trials` (column 4), average
                 (column 5) and std (column 6) are READ DIRECTLY. The description
                 tag uses the single-parameter `fmt_prob(p)`.

    Both branches read the trial count, average, and std straight from the file,
    so there is no `ntrials` argument. Both return the same DataFrame schema as
    `collect_decoder_statistics` (and `DecoderStatistics`'s fields) so downstream
    code can consume them interchangeably. Rows with the wrong column count are
    warned about and skipped. `error_type` must be `:Ising` or `:Circuit`.
    """
    if !(error_type in (:Ising, :Circuit))
        throw(ArgumentError("error_type must be :Ising or :Circuit, got $(repr(error_type))."))
    end

    results_file = "$(prefix)/results/$(standard_BP_output_file)"
    if !isfile(results_file)
        @warn ("File $(results_file) is missing.")
        empty_df = DataFrame()
        return empty_df
    end

    expected_cols::Int = 6
    if error_type == :Ising
        expected_cols = 7
    end

    descriptions = String[]
    num_samples  = Int[]
    failures_col = Int[]
    averages     = Float64[]
    stds         = Float64[]

    open(results_file, "r") do fp
        for line in eachline(fp)
            fields = split(line)
            if isempty(fields)
                continue
            end
            if startswith(fields[1], "#")
                continue  # skip comment / header lines
            end
            if length(fields) != expected_cols
                @warn ("Skipping line (expected $(expected_cols) columns for $(error_type)): $line")
                continue
            end
            if error_type == :Ising
                # p q sample failures total_trials average sigma
                p = parse(Float64, fields[1])
                q = parse(Float64, fields[2])
                sample = fields[3]
                failures = parse(Int, fields[4])
                n = parse(Int, fields[5])
                avg = parse(Float64, fields[6])
                sigma = parse(Float64, fields[7])
                tag = fmt_probs(p, q)
            else # :Circuit — p sample failures total_trials average sigma
                p = parse(Float64, fields[1])
                sample = fields[2]
                failures = parse(Int, fields[3])
                n = parse(Int, fields[4])
                avg = parse(Float64, fields[5])
                sigma = parse(Float64, fields[6])
                tag = fmt_prob(p)
            end
            push!(descriptions, "$(prefix)/testing_data/test_$(tag)_s_$(sample).txt")
            push!(num_samples, n)
            push!(failures_col, failures)
            push!(averages, avg)
            push!(stds, sigma)
        end
    end

    n_rows = length(descriptions)
    stats_df = DataFrame(
        algo = fill("SumProduct", n_rows),
        error_model_name = fill("ExplicitErrorModel", n_rows),
        error_model_parameters_description = descriptions,
        num_samples_per_error_rate = num_samples,
        n_iterations_BP = zeros(Int, n_rows),
        rounds_per_BP = zeros(Int, n_rows),
        weight_soft_constraint = zeros(Float64, n_rows),
        num_failures = failures_col,
        average_logical_error_rate = averages,
        std_logical_error_rate = stds,
        runtime = zeros(Float64, n_rows),
    )
    return stats_df
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

# NOTE: `check_approximate` and `extract_collected_data` moved to legacy.jl
# (still exported). They are only used by the legacy expts drivers
# (ballistic_errors.jl, misc/explicit_errors.jl).

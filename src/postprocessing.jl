using LinearAlgebra
using DataFrames
using DataStructures
using CSV

struct NeuralBPDecoderStatistics
    """
    Statistics for the Neural BP decoder.

    Every field is a Vector, so a single `NeuralBPDecoderStatistics` can hold ONE
    record (all fields length 1) or MANY (e.g. after concatenating the results of
    a parameter sweep). The two inner constructors are:

      1. Scalar/per-simulation form: pass one simulation's scalar values; the
         constructor computes the logical error rate and its Bernoulli std, then
         stores every field as a 1-element Vector.
      2. Raw form: pass every field as an already-built Vector (all the same
         length). Used by `vcat`/concat and the `DataFrame` builder below.

    The field names are Neural-BP-specific. The standard-BP counterpart
    (`StandardBPDecoderStatistics`, in legacy.jl) stored the layer/epoch counts
    under the standard-BP names `n_iterations_BP`/`rounds_per_BP` and carried a
    `weight_soft_constraint`; here those are named `n_layers`/`n_epochs` and there
    is no soft-constraint weight (a standard-BP-only knob).
    """
    algo::Vector{String} # always "NN" for the Neural BP decoder
    error_model_name::Vector{String} # e.g. "ExplicitErrorModel"
    error_model_parameters_description::Vector{String} # e.g. a filename for an explicit error model
    num_samples_per_error_rate::Vector{Int} # number of test samples for a given error rate
    n_layers::Vector{Int} # number of layers in the trained neural network
    n_epochs::Vector{Int} # number of epochs the network was trained for
    num_failures::Vector{Int} # number of logical failures observed in the test samples
    average_logical_error_rate::Vector{Float64} # average logical error rate = num_failures / num_samples_per_error_rate
    std_logical_error_rate::Vector{Float64} # standard deviation of the logical error rate, computed assuming a Bernoulli distribution
    runtime::Vector{Float64} # total runtime of the decoder in seconds over all test samples

    # --- Raw (array) constructor: every field already a Vector. ---------------
    function NeuralBPDecoderStatistics(algo::Vector{String}, error_model_name::Vector{String},
            error_model_parameters_description::Vector{String},
            num_samples_per_error_rate::Vector{Int}, n_layers::Vector{Int},
            n_epochs::Vector{Int}, num_failures::Vector{Int},
            average_logical_error_rate::Vector{Float64},
            std_logical_error_rate::Vector{Float64}, runtime::Vector{Float64})
        n = length(algo)
        lengths_match = all(==(n), (length(error_model_name), length(error_model_parameters_description),
                    length(num_samples_per_error_rate), length(n_layers),
                    length(n_epochs), length(num_failures),
                    length(average_logical_error_rate),
                    length(std_logical_error_rate), length(runtime)))
        if !lengths_match
            throw(ArgumentError("All NeuralBPDecoderStatistics field vectors must have the same length."))
        end
        new(algo, error_model_name, error_model_parameters_description,
            num_samples_per_error_rate, n_layers, n_epochs,
            num_failures, average_logical_error_rate,
            std_logical_error_rate, runtime)
    end

    # --- Scalar/per-simulation constructor. -----------------------------------
    function NeuralBPDecoderStatistics(algo::String, error_model_name::String,
            error_model_parameters_description::String, num_samples_per_error_rate::Int,
            n_layers::Int, n_epochs::Int; num_failures::Int=0,
            failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), runtime::Float64=0.0)
        if algo != "NN"
            throw(ArgumentError("Algorithm for the Neural BP decoder must be 'NN'."))
        end
        if num_samples_per_error_rate < 0
            throw(ArgumentError("Number of samples per error rate must be non-negative."))
        end
        if (num_failures == 0) && (length(failures) > 0)
            num_failures = count(failures)
        end
        if (num_samples_per_error_rate == 0) || (num_failures == 0)
            average_logical_error_rate = 0.0
            std_logical_error_rate = 0.0
        else
            average_logical_error_rate = num_failures / num_samples_per_error_rate
            # KNOWN BUG (fix pending): the Bernoulli std should divide by
            # num_samples_per_error_rate, NOT n_layers. Kept as-is so this
            # rename stays behaviour-preserving; the one-line fix is to pass
            # num_samples_per_error_rate as the second argument here.
            std_logical_error_rate = compute_std_assuming_bernoulli(average_logical_error_rate, n_layers)
        end
        # Store every field as a 1-element Vector (delegates to the raw form).
        new([algo], [error_model_name], [error_model_parameters_description],
            [num_samples_per_error_rate], [n_layers], [n_epochs],
            [num_failures], [average_logical_error_rate],
            [std_logical_error_rate], [runtime])
    end
end

function Base.vcat(stats::NeuralBPDecoderStatistics...)::NeuralBPDecoderStatistics
    """
    Concatenate several `NeuralBPDecoderStatistics` into one by stacking each field
    vector. Lets you accumulate a sweep's worth of single-record structs into one
    multi-record struct.

    Arguments:
    - `stats...`: One or more `NeuralBPDecoderStatistics` instances to concatenate.
    Returns:
    - A new `NeuralBPDecoderStatistics` instance containing the concatenated data.
    """
    if isempty(stats)
        throw(ArgumentError("vcat needs at least one NeuralBPDecoderStatistics."))
    end
    combined = NeuralBPDecoderStatistics(
        (reduce(vcat, getfield(s, f) for s in stats) for f in fieldnames(NeuralBPDecoderStatistics))...
    )
    return combined
end

"""
    NeuralBPDecoderStatistics(df::DataFrame) -> NeuralBPDecoderStatistics

Build a multi-record `NeuralBPDecoderStatistics` from a DataFrame whose columns
are the struct's field names (e.g. the output of `collect_decoder_statistics` run
over CSVs produced by the current Neural BP pipeline).
"""
function NeuralBPDecoderStatistics(df::DataFrame)::NeuralBPDecoderStatistics
    stats = NeuralBPDecoderStatistics(
        (Vector(df[!, f]) for f in fieldnames(NeuralBPDecoderStatistics))...
    )
    return stats
end

function record_decoder_statistics(stats, output_filename::String="./../data/decoder_statistics.csv")::DataFrame
    """
    Print the statistics in a JSON format so that they can be easily printed into a file using GNU `parallel`.

    Works for any decoder-statistics struct (e.g. `NeuralBPDecoderStatistics` or
    the legacy `StandardBPDecoderStatistics`): the CSV columns are taken from the
    struct's own field names, so each decoder writes its own schema.
    """
    stats_dict = Dict(
        name => getfield(stats, name) for name in fieldnames(typeof(stats))
    )
    println(JSON.json(stats_dict)) # Ensure a newline after the JSON object

    # Save the statistics to a CSV file
    stats_dataframe = DataFrame(stats_dict)
    CSV.write(output_filename, stats_dataframe)
    return stats_dataframe
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
    so there is no `ntrials` argument. Both return a DataFrame whose schema
    matches the legacy `StandardBPDecoderStatistics` fields so downstream code can
    consume standard results interchangeably. Rows with the wrong column count are
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

# NOTE: `StandardBPDecoderStatistics` (the former `DecoderStatistics`),
# `check_valid_fields_StandardBPDecoderStatistics`, `check_approximate`, and
# `extract_collected_data` now live in legacy.jl — standard BP is not the current
# focus, and those are only used by the legacy expts drivers (ballistic_errors.jl,
# misc/explicit_errors.jl, quantum_BP_test.jl).

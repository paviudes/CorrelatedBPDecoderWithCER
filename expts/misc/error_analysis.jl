using CSV
using Plots
using ArgParse
using Statistics                    # mean / cor for the CER calibration stats
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER    # parse_cer_data (for the CER-analysis wrapper)

# NOTE ON ENVIRONMENT: this script mixes Plots (analysis-only) with
# CorrelatedBPDecoderWithCER. Run it under the experiments environment
# `expts/Project.toml`, which has both plus the other analysis deps:
#
#     julia --project=expts expts/misc/error_analysis.jl --analysis correlations
#
# Do NOT run it with `--project=./../` (the main package deliberately excludes
# Plots, so plotting won't resolve), and never on a cluster job.

function count_error_weights(errors_filenames::Vector{String}, subdir::String; prefix::String="")
    """
    Compute the frequency of error weights in each file.
    Each file is expected to have one error pattern per column, with each error pattern being a binary vector (0s and 1s).
    The function returns a Matrix where the i-th column corresponds to the counts of error patterns of each weight for the i-th file in `errors_filenames`.
    The output file is saved as a DataFrame where the first column contains each file name and the subsequent columns contain the counts of error patterns of each weight.
    """
    data_directory = "$(prefix)/$(subdir)"
    weight_distribution_filename = "$(data_directory)/weight_counts.csv"
    
    if isfile(weight_distribution_filename)
        return weight_distribution_filename
    end
    
    error_weight_counts = [Int[] for _ in 1:length(errors_filenames)]
    
    for (i, errors_filename) in enumerate(errors_filenames)
        # Load the error patterns from the file
        error_pattern = convert.(Bool, readdlm("$(data_directory)/$(errors_filename)", Int))

        # Count the number of errors of each weight
        (n_bits, n_samples) = size(error_pattern)
        # Sum along the columns to get the weight of each error pattern
        error_weights = sum(error_pattern, dims=1)
        
        # Count the number of error patterns for each weight    max_weight = maximum(error_weights)
        error_weight_counts[i] = zeros(Int, n_bits + 1)  # Initialize counts for weights from 0 to n_bits
        for s in 1:n_samples
            weight = error_weights[s]
            error_weight_counts[i][weight + 1] += 1  # Increment count for this weight
        end

        # println("File: $(errors_filename), Total samples: $(n_samples), Error weight distribution: ", error_weight_counts[i])
    end

    # Create a DataFrame to store the counts of error weights for each file
    n_bits = length(error_weight_counts[1]) - 1  # Get the number of bits from the length of the first counts array
    weight_counts_df = DataFrame(
        :filename => errors_filenames,
        [Symbol("weight_$w") => zeros(Int, length(errors_filenames)) for w in 0:n_bits]...
    )
    for i in 1:length(errors_filenames)
        for w in 0:n_bits
            weight_counts_df[i, Symbol("weight_$w")] = error_weight_counts[i][w + 1]
        end
    end
    # println("Weight counts DataFrame:")
    # println(weight_counts_df)
    
    # Save the counts in a CSV file
    CSV.write(weight_distribution_filename, weight_counts_df)

    return weight_distribution_filename
end

function plot_error_weight_distribution(weight_distribution_filename::String)
    """
    Plot the distribution of error weights from a file containing the counts of error weights.
    The CSV file is formatted as follows.
    - The first column contains the file names corresponding to different error patterns.
    - The subsequent columns contain the counts of error patterns of each weight, with column names like `weight_0`, `weight_1`, ..., `weight_n`.
    
    We want to create a bar plot, where the x-axis corresponds to the error weights (0, 1, ..., n) and the y-axis corresponds to the counts of error patterns of each weight.
    For each file (i.e., for each row in the CSV), we will use a different color.

    The plot will be saved as a PDF file in the `plots` directory, with the same name as the input file but with a `.pdf` extension.
    """
    # Load the error weight counts from the file
    weight_counts_df = CSV.read("$(weight_distribution_filename)", DataFrame)

    # We don't want to plot zeros, so we will identify the maximum weight that has a non-zero count across all files, and only plot up to that weight.
    weight_columns = names(weight_counts_df)[2:end]  # Get the names of the weight columns
    max_weight_to_plot = 0
    for col in weight_columns
        weight_num = parse(Int, replace(string(col), "weight_" => ""))
        max_count = maximum(weight_counts_df[!, col])
        if max_count > 0 && weight_num > max_weight_to_plot
            max_weight_to_plot = weight_num
        end
    end
    println("Maximum weight to plot (with non-zero count): ", max_weight_to_plot)
    
    # Extract the file names and the counts of error weights
    error_weights = 0:max_weight_to_plot

    # Create an empty plot with the appropriate labels and title
    plt = plot(
        title = "Distribution of Error Weights",
        xlabel = "Error Weight",
        ylabel = "Count of Error Patterns",
        xticks = (error_weights, string.(error_weights)),
        legend = :topright,
        size = (800, 600)
    )
    
    # Add a bar plot for each file
    colors = [:blue, :orange, :green, :red, :purple, :brown, :pink, :gray, :cyan, :magenta]
    
    for i in 1:nrow(weight_counts_df)
        # println("Plotting file: ", weight_counts_df[i, :filename])
        # println("Error weights: ", error_weights)
        counts = [weight_counts_df[i, Symbol("weight_$w")] for w in error_weights]
        # println("Counts: ", counts)
        bar!(
            plt,
            error_weights,
            counts,
            label = weight_counts_df[i, :filename],
            color = colors[mod1(i, length(colors))]
        )
    end

    # Save the plot as a PDF file
    plot_filename = replace(weight_distribution_filename, ".csv" => ".pdf")
    savefig(plt, "$(plot_filename)")
    return nothing
end

function analyze_correlations_with_cer_data(
    connectivity_matrix::Matrix{Int},
    cer_two_qubit_marginals::Vector{Float64},
    error_patterns::BitMatrix,
    output_plot_file::String;
    codename::String = "",
)
    """
    Compare the CER two-qubit marginals against the empirical two-qubit
    statistics of an actual set of error patterns.

    Arguments
    - `connectivity_matrix` : one edge per row, `[u v]` (1-based qubit indices).
    - `cer_two_qubit_marginals` : the CER marginal for edge `e`, aligned with the
      rows of `connectivity_matrix`. Interpreted as the joint two-qubit marginal
      P(e_u = e_v = 1) — i.e. directly comparable to the empirical co-occurrence.
    - `error_patterns` : BitMatrix, one error pattern per COLUMN (rows = qubits).
    - `output_plot_file` : output PDF filename (a bare filename; `.pdf` appended
      if missing).
    - `codename` (keyword) : if given, the figure is written to
      `<codename>/plots/<output_plot_file>`; otherwise to `plots/<output_plot_file>`.
      (Added as a keyword because the 4 positional args don't carry the codename
      that the requested save path `<codename>/plots/...` needs.)

    For each edge e = (u, v) we compute the empirical joint marginal
        emp[e] = mean_samples( error[u] .& error[v] )
    and compare it to `cer_two_qubit_marginals[e]`.

    We make three plots, combined into a single figure (calibration scatter on
    top, the other two side-by-side below):

    1. Calibration scatter (top). A log-log scatter of the CER marginal (x) vs
       the empirical marginal (y), with the y = x reference line overlaid. Points
       on the line are well-calibrated; points above it are edges the CER
       UNDER-estimates, points below are edges it OVER-estimates. The panel title
       reports the log-space Pearson r between the two and the number of edges
       whose empirical co-occurrence is exactly 0 (CER claims a correlation the
       data never realises jointly). This is the panel that actually answers "are
       the CER marginals consistent with what the errors do?".

    2. Sorted CER-vs-empirical overlay (bottom-left). Both series plotted on a log
       y-axis against edges sorted by CER marginal (edges with empirical 0 are
       dropped from the log axis). Reading left-to-right walks from the weakest to
       the strongest CER couplings, so a gap between the two curves that grows or
       shifts across the range exposes whether mis-calibration is systematic over
       the ~2-orders-of-magnitude spread of strengths.

    3. Product metric (bottom-right). A scatter of the
       per-edge `emp[e] * cer[e]` (empirical co-occurrence times CER marginal) vs
       edge index. Kept for reference, but note it mixes the two quantities
       multiplicatively, so a high/low point does NOT tell you whether the CER
       value AGREES with the data — that is what plot 1 is for.

    (If your CER value is actually a connected correlation / covariance rather
    than a joint marginal, compare it instead to emp[e] - mean(e_u)*mean(e_v);
    say the word and I'll switch the empirical quantity.)

    Returns a NamedTuple with the computed arrays and calibration stats.
    """
    n_edges = size(connectivity_matrix, 1)
    if n_edges != length(cer_two_qubit_marginals)
        error(
            "connectivity_matrix has $(n_edges) rows but cer_two_qubit_marginals " *
            "has $(length(cer_two_qubit_marginals)) entries.")
    end
    (_, n_samples) = size(error_patterns)

    # Empirical joint two-qubit marginal per edge, P(e_u = e_v = 1) averaged over the samples.
    empirical = zeros(Float64, n_edges)
    @inbounds for e in 1:n_edges
        u = connectivity_matrix[e, 1]
        v = connectivity_matrix[e, 2]
        # Count the average number of occurrences of errors on both qubits `u` and `v`.
        empirical[e] = count(error_patterns[u, :] .& error_patterns[v, :]) / n_samples
    end

    # Data for plot 3: average number of occurrences of two-qubit errors times the CER marginal.
    product_metric = empirical .* cer_two_qubit_marginals

    # Compute the Pearson correlation in log space, ignoring edges with empirical = 0 (log undefined).
    positive_edges = findall(e -> empirical[e] > 0 && cer_two_qubit_marginals[e] > 0, 1:n_edges)
    log_pearson::Float64 = 0.0
    if length(positive_edges) >= 2
        log_pearson = cor(log10.(cer_two_qubit_marginals[positive_edges]), log10.(empirical[positive_edges]))
    else
        log_pearson = NaN
    end

    # --- Panel 1: Log-Log scatter of the CER marginals vs empirical co-occurrence. ----------------------
    n_empirical_zero = count(e -> cer_two_qubit_marginals[e] > 0 && empirical[e] == 0, 1:n_edges)
    panel_scatter = scatter(
        cer_two_qubit_marginals[positive_edges], empirical[positive_edges];
        xscale = :log10, yscale = :log10,
        xlabel = "CER two-qubit marginal", ylabel = "empirical  P(e_u = e_v = 1)",
        # title = "Calibration: log-space r = $(round(log_pearson, digits = 3)), $(n_empirical_zero) edges with empirical = 0",
        label = "r = $(round(log_pearson, digits = 3))", markersize = 3, markerstrokewidth = 0, legend = :topleft,
    )
    # Add the Y = X reference line to the scatter plot.
    if !isempty(positive_edges)
        low = min(minimum(cer_two_qubit_marginals[positive_edges]), minimum(empirical[positive_edges]))
        high = max(maximum(cer_two_qubit_marginals[positive_edges]), maximum(empirical[positive_edges]))
        plot!(panel_scatter, [low, high], [low, high]; label = "y = x", linestyle = :dash, color = :black)
    end

    # --- Panel 2: CER marginals and empirical co-occurrence sorted by CER marginal. ----------------------
    order = sortperm(cer_two_qubit_marginals)
    panel_sorted = plot(
        1:n_edges, cer_two_qubit_marginals[order];
        yscale = :log10, xlabel = "edge (sorted by CER marginal)",
        ylabel = "CER / empirical", label = "CER marginal", lw = 2,
        # title = "CER vs empirical co-occurrence",
    )
    # Skip edges with empirical = 0 for the log plot, replacing them with NaN so that they don't appear.
    empirical_sorted = [empirical[order[i]] > 0 ? empirical[order[i]] : NaN for i in 1:n_edges]
    scatter!(panel_sorted, 1:n_edges, empirical_sorted; label = "empirical", markersize = 2, markerstrokewidth = 0)

    # --- Panel 3: mean(e_u * e_v) * CER marginal. -------
    panel_product = scatter(
        1:n_edges, product_metric;
        xlabel = "edge index", ylabel = "\$\\{\\langle e_u ~ e_v \\rangle ~ : ~ (u,v) \\in E \\}\$ * CER marginal",
        # title = "Empirical co-occurrence * CER",
        label = "", markersize = 2, markerstrokewidth = 0,
    )
    
    # -- Combine the three panels into a single figure with a 2x2 layout. ----------------------
    plt = plot(panel_scatter, panel_sorted, panel_product; layout = @layout([a{0.5h}; b c]), size = (1000, 1000))

    # --- Save to <codename>/plots/<output_plot_file>. -------------------------
    plots_dir = isempty(codename) ? "plots" : joinpath(codename, "plots")
    if !isdir(plots_dir)
        mkpath(plots_dir)
    end
    if !endswith(lowercase(output_plot_file), ".pdf")
        output_plot_file *= ".pdf"
    end
    save_path = joinpath(plots_dir, output_plot_file)
    savefig(plt, save_path)

    # Print summary statistics.
    println("==================================================")
    println("Summary of correlation analysis for $(output_plot_file):")
    println("edges = $(n_edges)")
    println("Positive empirical and CER marginals = $(length(positive_edges))")
    println("Empirical-zero = $(n_empirical_zero)")
    println("log-space Pearson r = $(round(log_pearson, digits = 3))")
    println("Plot saved to $(save_path).")
    println("==================================================")

    return nothing
end

function analyze_correlations_with_cer_data(
    codename::String,
    correlation_strengths_file::String,
    error_patterns_file::String,
)
    """
    Convenience overload: load the CER data and error patterns from disk, then
    delegate to the matrix/vector form above.

    - `parse_cer_data("<codename>/<correlation_strengths_file>")` supplies the
      connectivity matrix and CER marginals (its third return, the single-qubit
      rates, is unused here).
    - error patterns are read from `<codename>/<error_patterns_file>` as a
      BitMatrix (one pattern per column).
    - the output PDF is named `correlations_in_<errors-basename>.pdf` and written
      under `<codename>/plots/`.
    """
    (connectivity_matrix, cer_marginals_f32, _) =
        parse_cer_data(joinpath(codename, correlation_strengths_file))
    cer_two_qubit_marginals = Float64.(cer_marginals_f32)

    error_patterns = BitMatrix(convert.(Bool, readdlm(joinpath(codename, error_patterns_file), Int)))

    errors_basename = splitext(basename(error_patterns_file))[1]
    output_plot_file = "correlations_in_$(errors_basename).pdf"

    return analyze_correlations_with_cer_data(
        connectivity_matrix, cer_two_qubit_marginals, error_patterns, output_plot_file;
        codename = codename,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ------------------------------------------------------------------------
    # Analysis dispatcher.
    #
    # Pick an analysis with `--analysis <name>`. Each analysis is a self-
    # contained if-block below with its parameters hard-coded
    # The intended workflow is to edit the parameters in the relevant block and re-run,
    # rather than exposing every knob as a CLI flag. Running with no arguments
    # (or `--help` / `-h`) prints the list of analyses and their descriptions.
    # ------------------------------------------------------------------------

    # Working directory holding the codename subfolders.
    work_dir::String = "./../data"

    # Registry of available analyses: (name, description). The descriptions are
    # printed by --help / no-args below and are mirrored as a comment at the top
    # of each analysis's if-block.
    analyses = [
        ("weights", "Compute and plot the weight distribution of the errors"),
        ("correlations", "Compare the CER two-qubit marginals against the empirical statistics for a set of error patterns"),
    ]

    # ArgParse's built-in help (add_help defaults to true) is the single source
    # of usage text. The per-analysis descriptions from the registry above are
    # folded into the --analysis option's help string, so ArgParse's terse
    # `--help` / `-h` output — and the no-args case below, which reuses it —
    # displays them.
    settings = ArgParseSettings(
        description = "Run one error-analysis routine; each analysis's parameters " *
                      "are hard-coded in its if-block below.",
    )
    @add_arg_table settings begin
        "--analysis"
        help = "Which analysis to run. \n Options:\n" *
               join(["$(name): $(description)" for (name, description) in analyses], "\n") * "."
        arg_type = String
        default = ""
    end

    # No arguments → show the same terse ArgParse help as --help, then exit.
    if isempty(ARGS)
        parse_args(["--help"], settings)
    end

    parsed_args = parse_args(settings)
    analysis = lowercase(strip(parsed_args["analysis"]))

    if isempty(analysis)
        # Nothing chosen → show the ArgParse help (with the descriptions) and exit.
        parse_args(["--help"], settings)

    elseif analysis == "weights"
        # Error-weight distribution. Counts, for one or more error files, how
        # many sampled patterns have each Hamming weight, and saves a bar plot
        # of the weight histograms.
        codename         = "18q_BB_p_0.0005_cycles_1"
        subdir           = "testing_data"
        errors_filenames = ["test_errors_p_0.0005.txt"]

        prefix = joinpath(work_dir, codename)
        weight_distribution_filename = count_error_weights(errors_filenames, subdir; prefix = prefix)
        plot_error_weight_distribution(weight_distribution_filename)

    elseif analysis == "correlations"
        # CER correlation calibration. Compares the CER two-qubit marginals
        # against the empirical co-occurrence statistics of an error file, and
        # saves the calibration scatter + companion panels.
        codename                   = "18q_BB_p_0.0005_cycles_1"
        correlation_strengths_file = "correlated_weights/correlated_weights_p_0.0005_s_1.txt"
        error_patterns_file        = "testing_data/test_p_0.0005_s_1.txt"

        analyze_correlations_with_cer_data(
            joinpath(work_dir, codename),
            correlation_strengths_file,
            error_patterns_file
        )

    else
        println("Unknown analysis: $(repr(parsed_args["analysis"])). Run with --help to see the options.")
        exit(1)
    end
end
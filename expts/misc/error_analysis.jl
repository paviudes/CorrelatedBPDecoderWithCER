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

function empirical_ising_coupling(
    error_patterns::BitMatrix, u::Int, v::Int; haldane::Float64 = 0.5,
)::Tuple{Float64, Float64, NTuple{4, Int}, Bool}
    """
    Empirical log-odds Ising coupling for the qubit pair (u, v):

        J_emp = log[ (n00 * n11) / (n01 * n10) ]

    from the 2x2 contingency table of co-occurrences across all samples. This is
    the SAME quantity the CER file stores, so the two are directly comparable.

    Zero cells are the norm at low p (n11 is often 0), and they send J to ±Inf.
    We apply the standard Haldane–Anscombe correction — add `haldane` (default
    0.5) to every cell — whenever ANY cell is zero, which keeps J finite and
    is the usual estimator for sparse contingency tables. The last return value
    flags whether the correction was applied, so those edges can be marked in the
    plots rather than silently trusted.

    We also return Woolf's standard error for a log odds ratio,

        SE(J) = sqrt(1/n00 + 1/n01 + 1/n10 + 1/n11)  ~  1/sqrt(n11)

    which is what makes this comparison interpretable: at low p, n11 is tiny, so
    a weak edge can easily show |J_emp - J_file| ~ 0.5 purely from Poisson noise
    on a handful of co-occurrences. Without the SE you cannot tell "the CER file
    is wrong" from "this edge is unmeasurable with this many samples". Verified
    against synthetic data with known J: all deviations fall within 3 SE.

    Returns `(J_emp, SE, (n00, n01, n10, n11), was_corrected)`.
    """
    eu = error_patterns[u, :]
    ev = error_patterns[v, :]
    n11 = count(eu .& ev)
    n10 = count(eu .& .!ev)
    n01 = count(.!eu .& ev)
    n00 = length(eu) - n11 - n10 - n01

    was_corrected = (n00 == 0) || (n01 == 0) || (n10 == 0) || (n11 == 0)
    a, b, c, d = Float64(n00), Float64(n01), Float64(n10), Float64(n11)
    if was_corrected
        a += haldane; b += haldane; c += haldane; d += haldane
    end
    J_emp = log((a * d) / (b * c))
    standard_error = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return (J_emp, standard_error, (n00, n01, n10, n11), was_corrected)
end

function analyze_correlations_with_cer_data(
    connectivity_matrix::Matrix{Int},
    cer_couplings::Vector{Float64},
    error_patterns::BitMatrix,
    output_plot_file::String;
    codename::String = "",
)
    """
    Validate the CER file's log-odds couplings J against the empirical couplings
    measured directly from a set of error patterns.

    `cer_couplings[e]` is the file's value for edge `e` (row `e` of
    `connectivity_matrix`): the SIGNED Ising coupling
    J = log[P00*P11 / (P01*P10)]. For each edge we measure the same quantity from
    the data (see `empirical_ising_coupling`) and compare.

    This is the check that answers "does the CER file describe the errors we
    actually generated?". If the file claims J ~ 4 (a ~16% conditional co-flip at
    p ~ 0.004) where the data says J ~ 0, the decoder is being handed a strong,
    confidently WRONG prior — which is far more damaging than a weak one.

    Three panels:
      1. Calibration scatter (top): file J (x) vs empirical J (y) on LINEAR axes
         (J is already a log quantity — do NOT log it again), with the y = x
         reference. Points on the line are well calibrated; the panel reports the
         Pearson r, the best-fit slope, and the mean signed bias. Edges needing
         the Haldane correction are drawn hollow, since their J is an estimate
         from a table with an empty cell.
      2. Sorted overlay (bottom-left): both series against edges sorted by file J,
         so systematic mis-calibration across the strength range is visible.
      3. Residual (bottom-right): (empirical - file) J per edge, with a zero line.

    Returns a NamedTuple with the arrays and calibration statistics.
    """
    n_edges = size(connectivity_matrix, 1)
    if n_edges != length(cer_couplings)
        error(
            "connectivity_matrix has $(n_edges) rows but cer_couplings " *
            "has $(length(cer_couplings)) entries.")
    end
    (_, n_samples) = size(error_patterns)

    # Empirical log-odds coupling per edge, plus the contingency table and whether
    # the Haldane correction had to be applied (an empty cell in that table).
    empirical_J  = zeros(Float64, n_edges)
    standard_err = zeros(Float64, n_edges)
    corrected    = falses(n_edges)
    n11_counts   = zeros(Int, n_edges)
    @inbounds for e in 1:n_edges
        u = connectivity_matrix[e, 1]
        v = connectivity_matrix[e, 2]
        (J_emp, se, counts, was_corrected) = empirical_ising_coupling(error_patterns, u, v)
        empirical_J[e]  = J_emp
        standard_err[e] = se
        corrected[e]    = was_corrected
        n11_counts[e]   = counts[4]
    end

    residual = empirical_J .- cer_couplings
    # How many standard errors the file sits from the measurement. |z| > 3 on an
    # edge with a small SE is a real disagreement; a large residual on an edge
    # with a large SE is just an unmeasurable edge.
    z_scores = residual ./ standard_err

    # Calibration statistics. `clean` = edges whose table had no empty cell, i.e.
    # where the empirical J is measured rather than Haldane-imputed.
    clean = findall(.!corrected)
    pearson_all::Float64 = length(cer_couplings) >= 2 ? cor(cer_couplings, empirical_J) : NaN
    pearson_clean::Float64 = length(clean) >= 2 ? cor(cer_couplings[clean], empirical_J[clean]) : NaN
    # Least-squares slope of empirical vs file (1.0 == perfectly calibrated).
    slope::Float64 = NaN
    if length(clean) >= 2
        x = cer_couplings[clean]; y = empirical_J[clean]
        xbar = mean(x); ybar = mean(y)
        denom = sum((x .- xbar) .^ 2)
        slope = denom > 0 ? sum((x .- xbar) .* (y .- ybar)) / denom : NaN
    end
    bias = mean(residual)

    # --- Panel 1: calibration scatter, file J vs empirical J (LINEAR axes). -----
    panel_scatter = scatter(
        cer_couplings[clean], empirical_J[clean];
        xlabel = "CER file coupling  \$J = \\log[P_{00}P_{11}/(P_{01}P_{10})]\$",
        ylabel = "empirical \$J\$ from error patterns",
        label = "measured (r = $(round(pearson_clean, digits = 3)))",
        markersize = 3, markerstrokewidth = 0, legend = :topleft,
    )
    if any(corrected)
        scatter!(panel_scatter, cer_couplings[corrected], empirical_J[corrected];
            label = "Haldane-corrected (empty cell)", markersize = 3,
            markerstrokewidth = 1, markercolor = :white, markerstrokecolor = :red)
    end
    low  = min(minimum(cer_couplings), minimum(empirical_J))
    high = max(maximum(cer_couplings), maximum(empirical_J))
    plot!(panel_scatter, [low, high], [low, high]; label = "y = x (calibrated)",
          linestyle = :dash, color = :black)

    # --- Panel 2: both series sorted by the file's J. --------------------------
    order = sortperm(cer_couplings)
    panel_sorted = plot(
        1:n_edges, cer_couplings[order];
        xlabel = "edge (sorted by CER \$J\$)", ylabel = "\$J\$",
        label = "CER file", lw = 2,
    )
    scatter!(panel_sorted, 1:n_edges, empirical_J[order];
        label = "empirical", markersize = 2, markerstrokewidth = 0)
    hline!(panel_sorted, [0.0]; color = :gray, linestyle = :dot, label = "")

    # --- Panel 3: residual in units of the measurement error. ------------------
    # Plotting z rather than the raw residual is the point: it separates "the CER
    # file is wrong here" from "this edge has too few co-occurrences to measure".
    panel_residual = scatter(
        1:n_edges, z_scores[order];
        xlabel = "edge (sorted by CER \$J\$)",
        ylabel = "(empirical \$J\$ - CER \$J\$) / SE",
        label = "", markersize = 2, markerstrokewidth = 0,
    )
    hline!(panel_residual, [0.0]; color = :black, linestyle = :dash, label = "")
    hline!(panel_residual, [-3.0, 3.0]; color = :red, linestyle = :dot, label = "±3 SE")

    # -- Combine the three panels into a single figure with a 2x2 layout. ----------------------
    plt = plot(panel_scatter, panel_sorted, panel_residual; layout = @layout([a{0.5h}; b c]), size = (1000, 1000))

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
    println("CER coupling validation for $(output_plot_file):")
    println("  samples                     = $(n_samples)")
    println("  edges                       = $(n_edges)")
    println("  edges with an empty cell    = $(count(corrected))  (Haldane-corrected)")
    println("  edges with n11 == 0         = $(count(==(0), n11_counts))")
    println("  file J:      mean = $(round(mean(cer_couplings), digits = 3))   " *
            "mean|J| = $(round(mean(abs.(cer_couplings)), digits = 3))   " *
            "max|J| = $(round(maximum(abs.(cer_couplings)), digits = 3))")
    println("  empirical J: mean = $(round(mean(empirical_J), digits = 3))   " *
            "mean|J| = $(round(mean(abs.(empirical_J)), digits = 3))   " *
            "max|J| = $(round(maximum(abs.(empirical_J)), digits = 3))")
    println("  Pearson r (all edges)       = $(round(pearson_all, digits = 3))")
    println("  Pearson r (measured only)   = $(round(pearson_clean, digits = 3))")
    println("  best-fit slope (measured)   = $(round(slope, digits = 3))   [1.0 = calibrated]")
    println("  mean bias (empirical - file)= $(round(bias, digits = 3))")
    println("  median SE(J)                = $(round(median(standard_err), digits = 3))   " *
            "[edges with SE > ~0.5 are effectively unmeasurable at this sample count]")
    n_significant = count(z -> isfinite(z) && abs(z) > 3, z_scores)
    println("  edges disagreeing by >3 SE  = $(n_significant) / $(n_edges)")
    println("  VERDICT: " * (
        isnan(pearson_clean)  ? "not enough measured edges to judge." :
        pearson_clean > 0.8 && abs(slope - 1) < 0.25 ? "CER file AGREES with the data." :
        pearson_clean > 0.5 ? "partial agreement — check the slope/bias above." :
        "CER file does NOT track the data (a strong but WRONG prior)."))
    println("  Plot saved to $(save_path).")
    println("==================================================")

    return (
        connectivity = connectivity_matrix,
        cer_J = cer_couplings,
        empirical_J = empirical_J,
        standard_error = standard_err,
        z_scores = z_scores,
        residual = residual,
        corrected = corrected,
        pearson_all = pearson_all,
        pearson_clean = pearson_clean,
        slope = slope,
        bias = bias,
    )
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
      connectivity matrix and the SIGNED log-odds couplings J (its third return,
      the single-qubit rates, is unused here).
    - error patterns are read from `<codename>/<error_patterns_file>` as a
      BitMatrix (one pattern per column).
    - the output PDF is named `cer_coupling_validation_<errors-basename>.pdf` and
      written under `<codename>/plots/`.
    """
    (connectivity_matrix, cer_couplings_f32, _) =
        parse_cer_data(joinpath(codename, correlation_strengths_file))
    cer_couplings = Float64.(cer_couplings_f32)

    error_patterns = BitMatrix(convert.(Bool, readdlm(joinpath(codename, error_patterns_file), Int)))

    errors_basename = splitext(basename(error_patterns_file))[1]
    output_plot_file = "cer_coupling_validation_$(errors_basename).pdf"

    return analyze_correlations_with_cer_data(
        connectivity_matrix, cer_couplings, error_patterns, output_plot_file;
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
        codename         = "update_23July2026/72q_BB_cycles_1"
        subdir           = "testing_data"
        errors_filenames = ["test_p_0.0005_s_1.txt"]

        prefix = joinpath(work_dir, codename)
        weight_distribution_filename = count_error_weights(errors_filenames, subdir; prefix = prefix)
        plot_error_weight_distribution(weight_distribution_filename)

    elseif analysis == "correlations"
        # CER coupling validation. Compares the SIGNED log-odds couplings
        # J = log[P00*P11/(P01*P10)] stored in the CER file against the same
        # quantity measured empirically from an error file, and saves the
        # calibration scatter + companion panels. Use the TRAINING errors if you
        # want to know what the model was actually taught.
        codename                   = "72q_BB_cycles_1"
        correlation_strengths_file = "correlated_weights/correlated_weights_p_0.0005_s_1.txt"
        error_patterns_file        = "training_data/train_p_0.0005_s_1.txt"

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
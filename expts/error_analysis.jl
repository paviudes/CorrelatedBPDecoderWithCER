using CSV
using Plots
using ArgParse
using DataFrames
using DelimitedFiles

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

if abspath(PROGRAM_FILE) == @__FILE__
    """
    Main function to count error weights and plot the distribution for a given errors file.
    We will use argument parsing to specify the prefix for the data directory and the list of error files to analyze.
    """
    settings = ArgParseSettings()
    @add_arg_table settings begin
        "--codename"
        help = "Prefix for the data directory"
        arg_type = String
        default = "90q_BB_p_0.008_q_0.2_std_0.2_data"

        "--subdir"
        help = "Whether to analyze test or train error files"
        arg_type = String
        default = "testing_data"

        "--errors"
        help = "List of error files to analyze"
        nargs = '+'
        default = ["test_ballistic_p_0.008_q_0.2_s_$(sample).txt" for sample in 1:2]
    end
    parsed_args = parse_args(settings)
    prefix = "./../data/$(parsed_args["codename"])"
    subdir = parsed_args["subdir"]
    errors_filename = convert.(String, parsed_args["errors"])
    weight_distribution_filename = count_error_weights(errors_filename, subdir; prefix=prefix)
    plot_error_weight_distribution(weight_distribution_filename)
end
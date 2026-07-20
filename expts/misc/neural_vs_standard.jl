using CSV
using DataFrames
using DelimitedFiles
using CorrelatedBPDecoderWithCER   # for save_decoder_dataframe (used by identify_best_performing_samples)

function extract_failed_test_samples(test_errors_file::String, logical_errors_file::String, selected_errors_file::String; prefix::String=".")
    """
    Extract the errors patterns on which the Neural BP decoder failed whereas the standard BP decoder succeeded, and save these error patterns to a file.
    The logical_errors_file is a CSV file with the following columns:
    sample_index, weight, Bp_logical_fail
    where
    - sample_index is the index of the error in `test_errors_file` on which the Neural BP decoder failed.
    - weight is the weight of the error pattern (i.e. the number of bit flips in the error pattern).
    - Bp_logical_fail is a string: "Yes" if the standard BP decoder also failed on this error pattern, and "No" if the standard BP decoder succeeded on this error pattern.
    
    We need to collect all those error patterns whose index is "No" in the Bp_logical_fail column, and save these error patterns to `selected_errors_file`.    
    """

    # Read the logical errors CSV file to get the indices of the failed error patterns and their weights.
    logical_errors_df = CSV.read("$(prefix)/results/$(logical_errors_file)", DataFrame)
    failure_indices = logical_errors_df[logical_errors_df.Bp_logical_fail .== "No", :sample_index]

    # Read the test errors from the test_errors_file and select the error patterns corresponding to the failure indices.
    test_errors = readdlm("$(prefix)/testing_data/$(test_errors_file)", Int)
    selected_error_patterns = test_errors[:, failure_indices]

    # Save the selected error patterns to a new text file.
    writedlm("$(prefix)/testing_data/$(selected_errors_file)", selected_error_patterns, ' ')

    println("Selected error patterns saved to $(prefix)/testing_data/$(selected_errors_file)")

    return nothing
end

function extract_parameters_from_description(description::String)::Tuple{Float64, Float64, Int}
    """
    Extract the error parameters p, q, and sample index from the
    `error_model_parameters_description` string.  The description looks
    like `<prefix>/testing_data/test_<pq_tag>_s_<sample>.txt`, where
    `<pq_tag>` is `fmt_probs(p, q)` — canonically `p_0.010_q_0.001`.

    The regex tolerates any decimal count in either p or q so that
    old-style filenames like `test_p_0.01_q_0.001_s_1.txt` (fewer
    decimals) still parse; but new filenames written by the pipeline
    will always be zero-padded to the max of the two.
    """
    filename = basename(description)
    m = match(r"test_p_(\d+\.\d+)_q_(\d+\.\d+)_s_(\d+)\.txt", filename)
    if m !== nothing
        p = parse(Float64, m.captures[1])
        q = parse(Float64, m.captures[2])
        sample = parse(Int, m.captures[3])
        return (p, q, sample)
    else
        @warn ("Filename does not match expected format: $(filename). Returning default values (-1.0, -1.0, -1).")
        return (-1.0, -1.0, -1)
    end
end

function identify_best_performing_samples(neuralbp_results::DataFrame, standardbp_results::DataFrame; performance_threshold::Float64=10.0, prefix::String="./../data")::DataFrame
    """
    Identify the best performing samples where the ratio of the logical error rate of the standard decoder to the logical error rate of the neural BP decoder is higher than a given threshold.
    """
    # Create a new dataframe that contains the error parameters, sample index, and the neural BP and standard decoder logical error rates.
    # The `error_model_parameters_description` column contains a string description of the error parameters in the format
    # <prefix>/testing_data/test_<pq_tag>_s_<sample>.txt   where <pq_tag> is fmt_probs(p, q), e.g. `p_0.010_q_0.001`.
    # So we just want to extract the basename of the file: test_<pq_tag>_s_<sample>.txt and then extract the values of p, q, and sample from this string.
    compare_results = DataFrame(
        p = zeros(Float64, nrow(neuralbp_results)),
        q = zeros(Float64, nrow(neuralbp_results)),
        sample = zeros(Int, nrow(neuralbp_results)),
        logical_error_rate_neural = zeros(Float64, nrow(neuralbp_results)),
        logical_error_rate_standard = zeros(Float64, nrow(neuralbp_results)),
        performance_ratio = zeros(Float64, nrow(neuralbp_results)),
    )

    # Iterate over each row of the standard and neural BP results dataframes and extract the error parameters.
    for i in 1:nrow(neuralbp_results)
        description = neuralbp_results[i, :error_model_parameters_description]
        (p, q, sample) = extract_parameters_from_description(description)
        compare_results[i, :p] = p
        compare_results[i, :q] = q
        compare_results[i, :sample] = sample

        # Assign logical error rates.
        compare_results[i, :logical_error_rate_neural] = neuralbp_results[i, :average_logical_error_rate]

        # Search for the corresponding entry in the standard BP results dataframe with the same p, q, and sample index in `error_model_parameters_description`
        # There should be only one such entry.
        standard_entry = filter(row -> extract_parameters_from_description(row[:error_model_parameters_description]) == (p, q, sample), standardbp_results)
        if nrow(standard_entry) >= 1
            compare_results[i, :logical_error_rate_standard] = standard_entry[1, :average_logical_error_rate]
            # Set the performance ratio to be the logical error rate of the standard decoder divided by the logical error rate of the neural BP decoder.
            compare_results[i, :performance_ratio] = compare_results[i, :logical_error_rate_standard] / compare_results[i, :logical_error_rate_neural]
        else
            @warn ("No matching entry found in standard BP results for description: $(description).")
            # Set the performance ratio to -1
            compare_results[i, :performance_ratio] = -1.0
        end
    end

    # Remove all entries where the performance ratio is -1, as these correspond to entries where we could not find a matching entry in the standard BP results dataframe.
    compare_results = filter(row -> row[:performance_ratio] != -1.0, compare_results)

    # Save the performance comparison results to a CSV file for later analysis.
    performance_comparison_csv_file = "$(prefix)/results/performance_comparison_neuralbp_vs_standardbp.csv"
    save_decoder_dataframe(compare_results, performance_comparison_csv_file)
    println("Performance comparison saved to file: $performance_comparison_csv_file")

    # Filter the samples where the performance ratio is higher than the threshold
    best_samples = filter(row -> row[:performance_ratio] >= performance_threshold, compare_results)

    return best_samples
end

function main()
    """
    Select the error patterns on which the original Neural BP decoder failed whereas the standard BP decoder succeeded, and save these error patterns to a file.
    """

    prefix = "./../data/90q_BB_p_0.008_q_0.2_std_0.2_data" # Change this to the appropriate path if needed
    test_errors_file = "test_ballistic_p_0.008_q_0.2_s_2.txt"
    logical_errors_file = "failures_test_ballistic_p_0.008_q_0.2_s_2_updated.csv"
    selected_errors_file = "selected_p_0.008_q_0.2_s_2.txt"

    if !isfile("$(prefix)/testing_data/$(selected_errors_file)")
        extract_failed_test_samples(test_errors_file, logical_errors_file, selected_errors_file; prefix=prefix)
    else
        println("Selected error patterns file already exists at $(prefix)/testing_data/$(selected_errors_file). Skipping extraction.")
    end
end
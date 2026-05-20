using CSV
using DataFrames
using DelimitedFiles

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
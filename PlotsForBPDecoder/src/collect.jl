# ============================================================================
# collect.jl — result aggregation / analysis entry points
# ============================================================================
# Moved here from expts/neural_bp_experiments.jl. These are analysis-only and
# belong in PlotsForBPDecoder because `collect_results` calls the plotting
# helpers (which depend on Plots/StatsPlots, kept out of the main package).
# ============================================================================

"""
    postprocess_neuralbp_results(result_csv_files::Vector{String}, output_csv_file::String) -> String

Combine several per-simulation result CSVs (each written by
`record_decoder_statistics`) into one database and write it out.

`result_csv_files` is the list of CSV paths to merge; they are stacked row-wise
by the multi-file `collect_decoder_statistics`. The combined table is written to
`output_csv_file` (taken as an argument), whose path is returned.
"""
function postprocess_neuralbp_results(result_csv_files::Vector{String}, output_csv_file::String)::String
    decoder_stats = collect_decoder_statistics(result_csv_files)
    filename = save_decoder_dataframe(decoder_stats, output_csv_file)
    return filename
end

function collect_results()
    """
    Collect results from the Neural BP experiments and render the comparison
    plots. ANALYSIS-ONLY entry point — it calls plotting helpers
    (`plot_statistics_for_ballistic_error_model`, `plot_performance_spread`),
    so run it with the PlotsForBPDecoder project active:

        julia --project=PlotsForBPDecoder
        julia> using PlotsForBPDecoder
        julia> collect_results()

    Never call this from a cluster job. The parameters below are hard-coded for a
    specific experiment — edit them to point at your codename/grid.
    """
    per_qubit_error_probs = [0.01]
    neighbour_error_probs = [0.001]
    n_samples = 56
    codename = "72q_BB_p_0.010_q_0.001_std_0.01_data_no_cer"

    prefix = "./../data/$(codename)"
    n_hidden_layers = 100
    n_epochs = 10

    # Create the plots directory if it doesn't exist
    plots_dir = "$(prefix)/plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
    end

    # Collect results for the Neural BP decoder. If the results file already exists, load it instead of re-computing.
    output_csv_file_neural = "$(prefix)/results/decoder_statistics_correlated.csv"
    if (isfile(output_csv_file_neural))
        neuralbp_results = CSV.read(output_csv_file_neural, DataFrame)
    else
        neuralbp_results = collect_decoder_statistics_correlated(
            per_qubit_error_probs,
            neighbour_error_probs,
            n_samples,
            n_hidden_layers,
            n_epochs;
            prefix=prefix
        )
        save_decoder_dataframe(neuralbp_results, output_csv_file_neural)
        println("Decoder statistics saved to file: $output_csv_file_neural")
    end

    # Collect results for the standard decoder. If the results file already exists, load it instead of re-computing.
    output_csv_file_standard = "$(prefix)/results/standard_decoder_statistics_correlated.csv"
    if (isfile(output_csv_file_standard))
        standardbp_results = CSV.read(output_csv_file_standard, DataFrame)
    else
        standardbp_results = collect_standard_decoder_statistics_correlated(
            prefix;
            standard_BP_output_file="72q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt"
        )
        save_decoder_dataframe(standardbp_results, output_csv_file_standard)
        println("Standard decoder statistics saved to file: $output_csv_file_standard")
    end

    #=
    # Plot results for the neural BP decoder
    plot_statistics_for_ballistic_error_model(
        neuralbp_results,
        per_qubit_error_probs,
        neighbour_error_probs;
        prefix="$(prefix)/plots",
        data_to_compare=standardbp_results
    )
    =#

    # Violin plots to show the spread of the logical error rates across different samples for a given set of error parameters.
    violin_error_parameters = [
        (0.01, 0.001)
    ]
    plot_performance_spread(
        neuralbp_results,
        standardbp_results,
        violin_error_parameters;
        prefix="$(prefix)/plots"
    )

    # NOTE: the best-performing-samples analysis used `identify_best_performing_samples`,
    # which now lives in the standalone `expts/misc/neural_vs_standard.jl` (not part
    # of this package). Re-enable by including that file and uncommenting:
    #
    # best_samples = identify_best_performing_samples(neuralbp_results, standardbp_results; performance_threshold=10.0, prefix=prefix)
    # println("Best performing samples (where standard decoder performs >10X worse than neural BP):")
    # println(best_samples)

    return nothing
end

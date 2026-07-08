"""
    plot_performance_spread(neuralbp_stats_dataframe, standardbp_stats_dataframe,
        error_parameters; prefix="./../plots")

Draw a violin plot of the per-sample **performance gain**:

    gain = LER(standard BP-OSD)  /  LER(Neural BP)

grouped by `(per_qubit_error_prob, neighbour_error_prob)` pairs. This shows
how much decoder-quality variance the sample-to-sample randomness introduces
at a fixed set of noise parameters.

Arguments:
- `neuralbp_stats_dataframe::DataFrame`, `standardbp_stats_dataframe::DataFrame`
  — both contain columns `error_model_parameters_description` and
  `average_logical_error_rate`; they must have matching sample counts per
  `(p, q)` pair (this function pairs them index-wise).
- `error_parameters::Vector{Tuple{Float64,Float64}}` — the `(p, q)` combinations
  to plot as separate violins.
- `prefix::String` — output directory. The CSV of computed gains lands at
  `\$(prefix)/performance_spread_data.csv` (kept for record-keeping), and the
  violin PDF at `\$(prefix)/performance_spread_violin_plot.pdf`.
"""
function plot_performance_spread(
    neuralbp_stats_dataframe::DataFrame,
    standardbp_stats_dataframe::DataFrame,
    error_parameters::Vector{Tuple{Float64, Float64}};
    prefix::String="./../plots",
)
    # One label per (p, q) — used both for DataFrame filtering and x-axis ticks.
    error_parameters_labels = [
        fmt_probs(per_qubit_prob, neighbour_prob)
        for (per_qubit_prob, neighbour_prob) in error_parameters
    ]

    # Assemble the long-format DataFrame that StatsPlots.@df consumes.
    data_for_violin = DataFrame(
        error_parameter_index  = Int[],
        error_parameter_labels = String[],
        neuralbp_logerr        = Float64[],
        standardbp_logerr      = Float64[],
        performance_gain       = Float64[],
    )

    for (label_index, label) in enumerate(error_parameters_labels)
        neuralbp_logerrs = filter(
            row -> occursin(label, row.error_model_parameters_description),
            neuralbp_stats_dataframe,
        )
        standardbp_logerrs = filter(
            row -> occursin(label, row.error_model_parameters_description),
            standardbp_stats_dataframe,
        )
        for i in 1:nrow(neuralbp_logerrs)
            neuralbp_logerr    = neuralbp_logerrs.average_logical_error_rate[i]
            standardbp_logerr  = standardbp_logerrs.average_logical_error_rate[i]
            performance_gain   = standardbp_logerr / neuralbp_logerr
            push!(data_for_violin,
                  (label_index, label, neuralbp_logerr, standardbp_logerr, performance_gain))
        end
    end

    # Persist the underlying data alongside the plot for later inspection.
    output_csv_file = "$(prefix)/performance_spread_data.csv"
    CSV.write(output_csv_file, data_for_violin)
    println("Data for performance spread violin plot saved to file: $output_csv_file")

    # Pretty tick labels: "p=..., q=..."
    xtick_error_parameter_labels = [
        "\$p=$(p)\$, \$ q=$(q)\$"
        for (p, q) in error_parameters
    ]

    plt = @df data_for_violin violin(
        :error_parameter_index,
        :performance_gain;
        xlabel        = "Ballistic Error Model Parameters (p, q)",
        ylabel        = "Performance Gain",
        legend        = false,
        xrotation     = 30,
        xticks        = (1:length(error_parameters_labels), xtick_error_parameter_labels),
        labelfontsize = 14,
        tickfontsize  = 14,
        size          = (800, 600),
        bottom_margin = 10mm,
    )

    savefig(plt, "$(prefix)/performance_spread_violin_plot.pdf")
end

"""
    plot_statistics_for_ballistic_error_model(stats_dataframe, per_qubit_error_probs,
        neighbour_error_probs; prefix="./../plots", data_to_compare=nothing)

Plot the average logical error rate as a function of the Ballistic Error Model
parameters. The x-axis is `neighbour_error_prob`, the y-axis is logical error
rate (log-scale), and each `per_qubit_error_prob` value gets its own curve.

For each (p, q) pair the DataFrame is expected to contain multiple simulation
rows (one per sample); we average the `average_logical_error_rate` field across
those to get the plotted point, and use the mean `std_logical_error_rate` as
the error bar.

Arguments:
- `stats_dataframe::DataFrame` — rows with columns
  `error_model_parameters_description`, `average_logical_error_rate`,
  `std_logical_error_rate`.
- `per_qubit_error_probs`, `neighbour_error_probs` — grids to iterate over.
- `prefix::String` — output directory. The plot lands at
  `\$(prefix)/ballistic_error_model_plot.pdf`.
- `data_to_compare::Union{DataFrame,Nothing}` — optional second DataFrame
  (typically standard BP-OSD results) drawn as dashed lines for comparison.
"""
function plot_statistics_for_ballistic_error_model(
    stats_dataframe::DataFrame,
    per_qubit_error_probs::AbstractVector{Float64},
    neighbour_error_probs::AbstractVector{Float64};
    prefix::String="./../plots",
    data_to_compare::Union{DataFrame, Nothing}=nothing,
)
    plt = plot(;
        xlabel="\nNeighbour Error Probability \$(q)\$",
        ylabel="Average Logical Error Rate\n",
        legend=:bottomright,
        legendfontsize=12,
        yscale=:log10,
    )

    # Add empty-plot labels to distinguish solid (Neural BP) from dashed (BP-OSD) in the legend.
    if !isnothing(data_to_compare)
        plot!(plt, [NaN], [NaN]; label="Neural BP", color=:black, marker=:none, markersize=3, linewidth=1)
        plot!(plt, [NaN], [NaN]; label="BP-OSD",    color=:black, marker=:none, markersize=3, linewidth=1, linestyle=:dash)
    end

    colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :yellow]
    for (i, per_qubit_prob) in enumerate(per_qubit_error_probs)

        # Per-p accumulators.
        average_logical_error_rates            = Float64[]
        std_logical_error_rates                = Float64[]
        average_logical_error_rates_to_compare = Float64[]
        std_logical_error_rates_to_compare     = Float64[]

        for neighbour_prob in neighbour_error_probs
            # Filter to rows whose `error_model_parameters_description` matches this (p, q).
            data_per_pair = filter(
                row -> occursin(fmt_probs(per_qubit_prob, neighbour_prob),
                                row.error_model_parameters_description),
                stats_dataframe,
            )

            if !isnothing(data_to_compare)
                data_per_pair_to_compare = filter(
                    row -> occursin(fmt_probs(per_qubit_prob, neighbour_prob),
                                    row.error_model_parameters_description),
                    data_to_compare,
                )
            end

            # Average across samples.
            push!(average_logical_error_rates, mean(data_per_pair.average_logical_error_rate))
            push!(std_logical_error_rates,     mean(data_per_pair.std_logical_error_rate))

            if !isnothing(data_to_compare)
                push!(average_logical_error_rates_to_compare,
                      mean(data_per_pair_to_compare.average_logical_error_rate))
                push!(std_logical_error_rates_to_compare,
                      mean(data_per_pair_to_compare.std_logical_error_rate))
            end
        end

        # Solid curve for this p-value.
        plot!(plt,
            neighbour_error_probs,
            average_logical_error_rates;
            yerr       = std_logical_error_rates,
            label      = "p = $(per_qubit_prob)",
            color      = colors[i],
            marker     = :circle,
            markersize = 3,
            linewidth  = 1,
        )

        # Dashed comparison curve (unlabelled — legend takes care of it above).
        if !isnothing(data_to_compare)
            plot!(plt,
                neighbour_error_probs,
                average_logical_error_rates_to_compare;
                yerr       = std_logical_error_rates_to_compare,
                color      = colors[i],
                label      = "",
                marker     = :square,
                markersize = 3,
                linewidth  = 1,
                linestyle  = :dash,
            )
        end
    end

    # Legend outside the plot area so it doesn't overlap the curves.
    plot!(plt;
        legend        = :outertopright,
        labelfontsize = 24,
        tickfontsize  = 24,
        size          = (880, 600),
        bottom_margin = 5mm,
        left_margin   = 5mm,
        legendfontsize = 18,
        labelpadding  = 2cm,
    )

    savefig(plt, "$(prefix)/ballistic_error_model_plot.pdf")
end

# ============================================================================
# plot.jl — analysis-only plotting helpers
# ============================================================================
#
# This file is intentionally NOT `include`d by the CorrelatedBPDecoderWithCER
# module. Plots.jl + StatsPlots.jl are heavyweight (~1 GB of precompiled
# artifacts, several GB of RAM per Julia process) and only needed when you're
# reading result CSVs and generating figures on your Mac — the cluster path
# (`neural_bp_experiments.jl` invoked with `--train ... --test ...`) never
# touches these functions.
#
# Usage from a REPL on your Mac (Plots + StatsPlots must be installed in a
# reachable env — either the active project or the global `@v1.12` stack):
#
#     julia> using CorrelatedBPDecoderWithCER
#     julia> include(joinpath(pkgdir(CorrelatedBPDecoderWithCER), "src", "plot.jl"))
#     julia> plot_statistics_for_ballistic_error_model(df, ps, qs)
#
# Install Plots + StatsPlots globally once:
#
#     julia> using Pkg
#     julia> Pkg.activate()                # activates @v1.12
#     julia> Pkg.add(["Plots", "StatsPlots"])
#
# After that any project you activate can `using Plots` via the LOAD_PATH
# fallback into @v1.12.
# ============================================================================

using DataFrames
using Statistics
using Plots
using Plots.PlotMeasures
using StatsPlots

function plot_statistics_for_ballistic_error_model(
    stats_dataframe::DataFrame,
    per_qubit_error_probs::AbstractVector{Float64},
    neighbour_error_probs::AbstractVector{Float64}; 
    prefix::String="./../plots", 
    data_to_compare::Union{DataFrame, Nothing}=nothing
)
    """
    Plot the average logical error rate as a function of the error model parameters for the Ballistic Error Model.
    We will plot the `neighbour_error_prob` on the x-axis and the logical error rate on the y-axis, with different curves for different `per_qubit_error_prob` values.
    Note that for each unique `per_qubit_error_prob` and `neighbour_error_prob` combination, we have multiple simulation runs, each corresponding to a different `sample`.
    We will average over the `sample` dimension to get the average logical error rate.
    Arguments:
    - `stats_dataframe::DataFrame`: A DataFrame containing the statistics for the ballistic error model, with columns `per_qubit_error_prob`, `neighbour_error_prob`, and `average_logical_error_rate` and `std_logical_error_rate`.
    - `prefix::String`: The prefix for the output plot file (default: "./../plots"). The plot will be saved as `$(prefix)/ballistic_error_model_plot.pdf`.
    - `data_to_compare::Union{DataFrame, Nothing}`: An optional DataFrame containing data to compare with the ballistic error model (default: `nothing`).
    """
    plt = plot(;
                xlabel="\nNeighbour Error Probability \$(q)\$",
                ylabel="Average Logical Error Rate\n",
                legend=:bottomright,
                legendfontsize=12,
                yscale=:log10
    )
    
    # Make an empty plot to add label for the solid line (Neural BP decoder) and dashed line (standard decoder) in the legend.
    if !isnothing(data_to_compare)
        plot!(plt, [NaN], [NaN], label="Neural BP", color=:black, marker=:none, markersize=3, linewidth=1)
        plot!(plt, [NaN], [NaN], label="BP-OSD", color=:black, marker=:none, markersize=3, linewidth=1, linestyle=:dash)
    end

    colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :yellow]
    for (i, per_qubit_prob) in enumerate(per_qubit_error_probs)
        
        # Collect the neighbour error probabilities and corresponding average logical error rates, std logical error rates for this `per_qubit_error_prob` value.
        average_logical_error_rates = Float64[]
        std_logical_error_rates = Float64[]
        average_logical_error_rates_to_compare = Float64[]
        std_logical_error_rates_to_compare = Float64[]

        for neighbour_prob in neighbour_error_probs
            # Select the rows corresponding to a specific `per_qubit_error_prob` and `neighbour_error_prob` pair.
            # These rows have their `error_model_parameters_description` field of the form ./../data/aps_7q_Hamm_code_data/testing_data/test_ballistic_p_<per_qubit_prob>_q_<neighbour_prob>_*.txt
            # We have to filter the dataframe based on the `error_model_parameters_description` column to select the relevant rows for this pair of parameters.
            data_per_pair = filter(
                row -> occursin(fmt_probs(per_qubit_prob, neighbour_prob), row.error_model_parameters_description), 
                stats_dataframe
            )
            
            if !isnothing(data_to_compare)
                data_per_pair_to_compare = filter(
                    row -> occursin(fmt_probs(per_qubit_prob, neighbour_prob), row.error_model_parameters_description), 
                    data_to_compare
                )
            end
            
            # Now we have multiple rows corresponding to different `sample` values. We will average over these to get the average logical error rate for this pair of parameters.
            average_logical_error_rate = mean(data_per_pair.average_logical_error_rate)
            std_logical_error_rate = mean(data_per_pair.std_logical_error_rate)
            
            # Add this to the datasets for plotting.
            push!(average_logical_error_rates, average_logical_error_rate)
            push!(std_logical_error_rates, std_logical_error_rate)

            # Extract the average logical error rate and std logical error rate for the data to compare, if provided.
            if !isnothing(data_to_compare)
                average_logical_error_rate_to_compare = mean(data_per_pair_to_compare.average_logical_error_rate)
                std_logical_error_rate_to_compare = mean(data_per_pair_to_compare.std_logical_error_rate)
                push!(average_logical_error_rates_to_compare, average_logical_error_rate_to_compare)
                push!(std_logical_error_rates_to_compare, std_logical_error_rate_to_compare)
            end
        end
        
        #=
        println("X values (neighbour error probabilities): ", neighbour_error_probs
                , "\nY values (average logical error rates): ", average_logical_error_rates
                , "\nError bars (std logical error rates): ", std_logical_error_rates)
        =#
        
        # Plot the curve for this `per_qubit_error_prob` value, with error bars.
        plot!(plt, 
              neighbour_error_probs, 
              average_logical_error_rates, 
              yerr=std_logical_error_rates, 
              label="p = $(per_qubit_prob)", 
              color=colors[i],
              marker=:circle,
              markersize=3,
              linewidth=1
        )

        # If we have data to compare, plot that as well.
        if !isnothing(data_to_compare)
            #=
            println("X values (neighbour error probabilities): ", neighbour_error_probs
                , "\nY values (average logical error rates): ", average_logical_error_rates_to_compare
                , "\nError bars (std logical error rates): ", std_logical_error_rates_to_compare)
            =#
            plot!(plt, 
                neighbour_error_probs, 
                average_logical_error_rates_to_compare, 
                yerr=std_logical_error_rates_to_compare, 
                color=colors[i],
                label="", # No label for the standard decoder curves, since we already have a label for the dashed line in the legend.
                marker=:square,
                markersize=3,
                linewidth=1,
                linestyle=:dash
            )
        end
    end

    # Add the legend outside the plot area to avoid overlapping with the curves.
    plot!(plt, 
        legend=:outertopright, 
        labelfontsize=24,
        tickfontsize=24,
        size=(880, 600), 
        bottom_margin=5mm, 
        left_margin=5mm,
        legendfontsize=18,
        labelpadding=2cm
    )

    # Save the plot to a file.
    savefig(plt, "$(prefix)/ballistic_error_model_plot.pdf") 
end

function plot_performance_spread(
    neuralbp_stats_dataframe::DataFrame, 
    standardbp_stats_dataframe::DataFrame,
    error_parameters::Vector{Tuple{Float64, Float64}}; 
    prefix::String="./../plots"
)
    """
    Plot the spread of the performance gain:
    
    gain = ratio of the logical error rate of the standard BP decoder to the neural BP decoder,

    across different samples for specific pairs of error model parameters (`per_qubit_error_prob` and `neighbour_error_prob`).
    This will give us an idea of how much the performance varies across different samples for the same error model parameters.
    We will do violin plots of the performance gains for different samples, grouped by the error model parameters.
    """

    # Isolate the data needed for the violin plots.
    error_parameters_labels = [
        fmt_probs(per_qubit_prob, neighbour_prob)
        for (per_qubit_prob, neighbour_prob) in error_parameters
    ]
    
    # DataFrame for the violin plot, with columns `error_parameter_labels` and `performance_gain`.
    data_for_violin = DataFrame(
        error_parameter_index = Int[],
        error_parameter_labels = String[],
        neuralbp_logerr = Float64[],
        standardbp_logerr = Float64[],
        performance_gain = Float64[]
    )

    # Filter out the rows in the data frame that correspond to the specific (`per_qubit_error_prob`, `neighbour_error_prob`) pairs we are interested in.
    for (label_index, label) in enumerate(error_parameters_labels)
        neuralbp_logerrs = filter(
            row -> occursin(label, row.error_model_parameters_description), 
            neuralbp_stats_dataframe
        )
        standardbp_logerrs = filter(
            row -> occursin(label, row.error_model_parameters_description), 
            standardbp_stats_dataframe
        )
        for i in 1:nrow(neuralbp_logerrs)
            neuralbp_logerr = neuralbp_logerrs.average_logical_error_rate[i]
            standardbp_logerr = standardbp_logerrs.average_logical_error_rate[i]
            performance_gain = standardbp_logerr / neuralbp_logerr
            push!(data_for_violin, (label_index, label, neuralbp_logerr, standardbp_logerr, performance_gain))
        end
    end

    # Save the data for the violin plot in a CSV file for record-keeping and potential future use.
    output_csv_file = "$(prefix)/performance_spread_data.csv"
    CSV.write(output_csv_file, data_for_violin)
    println("Data for performance spread violin plot saved to file: $output_csv_file")

    # Define the tick labels
    xtick_error_parameter_labels = [
        "\$p=$(p)\$, \$ q=$(q)\$"
        for (p, q) in error_parameters
    ]

    # Do the violin plot.
    plt = @df data_for_violin violin(
        :error_parameter_index,
        :performance_gain,
        xlabel = "Ballistic Error Model Parameters (p, q)",
        ylabel = "Performance Gain",
        legend = false,
        #yscale = :log10,
        xrotation = 30,
        xticks = (1:length(error_parameters_labels), xtick_error_parameter_labels),
        labelfontsize = 14,
        tickfontsize = 14,
        size = (800, 600),
        bottom_margin = 10mm
    )

    # Save the plot to a file.
    savefig(plt, "$(prefix)/performance_spread_violin_plot.pdf")
end

function fmt_probs(prob1::Float64, prob2::Float64)::String
    """
    Format the error model parameters `per_qubit_error_prob` and `neighbour_error_prob` to a string of the form `p_<per_qubit_error_prob>_q_<neighbour_error_prob>`, where the number of decimal places is determined by the maximum number of decimal places in either `prob1` or `prob2`.
    """
    ndig = max(length(split(string(prob1), ".")[end]),
               length(split(string(prob2), ".")[end]))
    fmt = Printf.Format("%.$(ndig)f")
    prob_string = "p_$(Printf.format(fmt, prob1))_q_$(Printf.format(fmt, prob2))"
    return prob_string
end

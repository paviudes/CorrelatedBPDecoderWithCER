using Plots
using DelimitedFiles

function plot_statistics_for_ballistic_error_model(stats_dataframe::DataFrame; prefix::String="./../plots")
    """
    Plot the average logical error rate as a function of the error model parameters for the Ballistic Error Model.
    We the `neighbour_error_prob` on the x-axis and the logical error rate on the y-axis, with different curves for different `per_qubit_error_prob` values.
    """
    plt = plot(;
                title="Logical Error Rate vs Neighbour Error Probability",
                xlabel="Neighbour Error Probability",
                ylabel="Average Logical Error Rate",
                legend=:bottomright,
                legendfontsize=12
            )
    unique_per_qubit_probs = unique(stats_dataframe.per_qubit_error_prob)
    
    # Iterate over each unique `per_qubit_error_prob` value to plot separate curves
    for site_error_prob in unique_per_qubit_probs
        neighbour_error_probs = stats_dataframe[stats_dataframe.per_qubit_error_prob .== site_error_prob, :neighbour_error_prob]
        stats_for_error_prob = stats_dataframe[stats_dataframe.per_qubit_error_prob .== site_error_prob, :summary]
        average_logical_error_rates = [stat.average_logical_error_rate for stat in stats_for_error_prob]
        std_logical_error_rates = [stat.std_logical_error_rate for stat in stats_for_error_prob]
        
        if length(stats_for_error_prob) > 0
            # Plot the logical error rates with error bars corresponding to the standard deviation
            plot!(
                plt,
                neighbour_error_probs,
                average_logical_error_rates;
                label="\$ p = $(site_error_prob)\$",
                marker=:o,
                yscale=:log10,
                linewidth=2,
                # yerror=std_logical_error_rates
            )
        end
    end

    # Remove any trailing "/" from `prefix`
    prefix = replace(prefix, r"/$" => "")
    
    # Close the plot and save it to a file
    savefig(plt, "$(prefix)/ballistic_error_model_plot.pdf")
end

function print_collected_data(error_model_name::String, parameter_ranges::Dict{String, <:AbstractVector}; prefix::String="./../data")
    stats_dataframe = collect_decoder_statistics(error_model_name, parameter_ranges; prefix=prefix)
    for each_stat in stats_dataframe[!, :summary]
        print_decoder_statistics(each_stat; io=stdout)
    end
end
using Plots
using DelimitedFiles

function plot_statistics_for_ballistic_error_model(stats_dataframe::DataFrame, per_qubit_error_probs::AbstractVector{Float64}, neighbour_error_probs::AbstractVector{Float64}; prefix::String="./../plots", data_to_compare::Union{DataFrame, Nothing}=nothing)
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
                xlabel="Neighbour Error Probability \$(q)\$",
                ylabel="Average Logical Error Rate",
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
            data_per_pair = filter(row -> occursin("p_$(per_qubit_prob)_q_$(neighbour_prob)", row.error_model_parameters_description), stats_dataframe)
            
            if !isnothing(data_to_compare)
                data_per_pair_to_compare =  filter(row -> occursin("p_$(per_qubit_prob)_q_$(neighbour_prob)", row.error_model_parameters_description), data_to_compare)
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

        println("X values (neighbour error probabilities): ", neighbour_error_probs
                , "\nY values (average logical error rates): ", average_logical_error_rates
                , "\nError bars (std logical error rates): ", std_logical_error_rates)
        
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
            println("X values (neighbour error probabilities): ", neighbour_error_probs
                , "\nY values (average logical error rates): ", average_logical_error_rates_to_compare
                , "\nError bars (std logical error rates): ", std_logical_error_rates_to_compare)
            
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
    plot!(plt, legend=:outertopright)

    # Save the plot to a file.
    savefig(plt, "$(prefix)/ballistic_error_model_plot.pdf") 
end
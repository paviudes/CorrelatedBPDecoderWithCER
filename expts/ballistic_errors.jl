function generate_runs_for_ballistic_errormodel(commands_file::String="./../expts/run_commands_ballistic_errormodel.txt")
	"""
	Generate commands for running simulations over a range of error parameters for the Ballistic error model.
	"""
	# Define values for all the parameters.
	# Define values for all the parameters.
	parameter_ranges = OrderedDict(
		"ballistic_per_qubit_error_prob" => 0.001:0.002:0.01,
		"ballistic_neighbour_error_prob" => 0.0:0.1:1.0,
		"num_error_samples" => [10000],
		"algo" => ["MinSum"],
		"n_iterations_BP" => [5],
		"rounds_per_BP" => [50],
		"llr_convergence_threshold" => [1e-6],
		"llr_confidence_threshold" => [4.0],
		"weight_soft_constraint" => [0.75],
		"debug" => [false],
		"verbose" => [false]
	)
	
	generate_runs(parameter_ranges, commands_file)
end

function show_data_ballistic_error_model()
    # Define values for the parameters to collect statistics for.
    outdir = "./../data/debankan"
    error_model_name = "Ballistic Error Model"
	# parameter_ranges = Dict(
	# 	"error_files" => ["10000_th0.0020_p0.10", "10000_th0.0040_p0.30", "10000_th0.0060_p0.50", "10000_th0.0100_p0.20", "10000_th0.0020_p0.20", "10000_th0.0040_p0.40", "10000_th0.0080_p0.10", "10000_th0.0100_p0.30", "10000_th0.0020_p0.30", "10000_th0.0040_p0.50", "10000_th0.0080_p0.20", "10000_th0.0100_p0.40", "10000_th0.0020_p0.40", "10000_th0.0060_p0.10", "10000_th0.0080_p0.30", "10000_th0.0100_p0.50", "10000_th0.0020_p0.50", "10000_th0.0060_p0.20", "10000_th0.0080_p0.40", "10000_th0.0040_p0.10", "10000_th0.0060_p0.30", "10000_th0.0080_p0.50", "10000_th0.0040_p0.20", "10000_th0.0060_p0.40", "10000_th0.0100_p0.10"]
	# )
	stats_dataframe = collect_decoder_statistics("$(outdir)/ballistic_error_model_MinSum_results.txt")
	println("Full DataFrame:\n", stats_dataframe)
	ballistic_per_qubit_error_prob = 0.001:0.002:0.01
	ballistic_neighbour_error_prob = 0.0:0.1:1.0
	ballistic_error_model_descriptions = ["per_qubit_error_prob=$(per_qubit_error_prob),neighbour_error_prob=$(neighbour_error_prob)" for per_qubit_error_prob in ballistic_per_qubit_error_prob for neighbour_error_prob in ballistic_neighbour_error_prob]
    selected_parameters = Dict(
		Symbol("error_model_name") => [error_model_name],
		Symbol("error_model_parameters_description") => ballistic_error_model_descriptions,
		Symbol("algo") => ["MinSum"],
		Symbol("n_iterations_BP") => [5],
		Symbol("rounds_per_BP") => 50:50:50,
		Symbol("weight_soft_constraint") => [0.75]
	)
	# Add additional parameters to display
	display_parameters = [Symbol("error_model_parameters_description"), Symbol("num_failures"), Symbol("average_logical_error_rate")]
	focused_dataframe = extract_collected_data(stats_dataframe, selected_parameters, display_parameters)
	println("Focused DataFrame:\n", focused_dataframe)
	save_decoder_dataframe(focused_dataframe, "$(outdir)/ballistic_error_model_focused_data.csv")
end

function plot_data_ballistic_error_model()
    # Define values for the parameters to collect statistics for.
    error_model_name = "Ballistic Error Model"
    parameter_ranges = Dict(
        "per_qubit_error_prob" => 0.001:0.002:0.01,
        "neighbour_error_prob" => 0.01:0.02:0.1
    )
	print_collected_data(error_model_name, parameter_ranges; prefix="./../data")
    # Load the collected statistics into a DataFrame
    stats_dataframe = collect_decoder_statistics(error_model_name, parameter_ranges; prefix="./../data")
	# Plot the statistics
    plot_statistics_for_ballistic_error_model(stats_dataframe; prefix="./../plots")
end
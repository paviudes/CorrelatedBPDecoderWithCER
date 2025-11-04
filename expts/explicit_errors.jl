function generate_runs_for_explicit_errormodel(commands_file::String="./../expts/run_commands_explicit_errormodel.txt")
	"""
	Generate commands for running simulations over a range of error parameters for a explicit error model.
	"""
	# Define values for all the parameters.
	# error_files = ["10000_th0.0020_p0.10", "10000_th0.0040_p0.30", "10000_th0.0060_p0.50", "10000_th0.0100_p0.20", "10000_th0.0020_p0.20", "10000_th0.0040_p0.40", "10000_th0.0080_p0.10", "10000_th0.0100_p0.30", "10000_th0.0020_p0.30", "10000_th0.0040_p0.50", "10000_th0.0080_p0.20", "10000_th0.0100_p0.40", "10000_th0.0020_p0.40", "10000_th0.0060_p0.10", "10000_th0.0080_p0.30", "10000_th0.0100_p0.50", "10000_th0.0020_p0.50", "10000_th0.0060_p0.20", "10000_th0.0080_p0.40", "10000_th0.0040_p0.10", "10000_th0.0060_p0.30", "10000_th0.0080_p0.50", "10000_th0.0040_p0.20", "10000_th0.0060_p0.40", "10000_th0.0100_p0.10"]
	error_files = ["10000_th0.0500_p0.50_high_error"]
	parameter_ranges = OrderedDict(
		"errors_filename" => error_files,
		"algo" => ["MinSum"],
		"weight_soft_constraint" => 0.4:0.05:0.9,
		"n_iterations_BP" => [5],
		"rounds_per_BP" => 100:100:100,
		"llr_convergence_threshold" => [1e-6],
		"llr_confidence_threshold" => [4.0],
		"debug" => [false],
		"verbose" => [false]
	)

	generate_runs(parameter_ranges, commands_file)
end

function show_data_explicit_error_model()
    # Define values for the parameters to collect statistics for.
    outdir = "./../data/debankan"
    error_model_name = "Explicit Error Set"
	# parameter_ranges = Dict(
	# 	"error_files" => ["10000_th0.0020_p0.10", "10000_th0.0040_p0.30", "10000_th0.0060_p0.50", "10000_th0.0100_p0.20", "10000_th0.0020_p0.20", "10000_th0.0040_p0.40", "10000_th0.0080_p0.10", "10000_th0.0100_p0.30", "10000_th0.0020_p0.30", "10000_th0.0040_p0.50", "10000_th0.0080_p0.20", "10000_th0.0100_p0.40", "10000_th0.0020_p0.40", "10000_th0.0060_p0.10", "10000_th0.0080_p0.30", "10000_th0.0100_p0.50", "10000_th0.0020_p0.50", "10000_th0.0060_p0.20", "10000_th0.0080_p0.40", "10000_th0.0040_p0.10", "10000_th0.0060_p0.30", "10000_th0.0080_p0.50", "10000_th0.0040_p0.20", "10000_th0.0060_p0.40", "10000_th0.0100_p0.10"]
	# )
	stats_dataframe = collect_decoder_statistics("$(outdir)/explicit_error_model_MinSum_results.txt")
	println("Full DataFrame:\n", stats_dataframe)
	error_file_names = ["10000_th0.0020_p0.10", "10000_th0.0040_p0.30", "10000_th0.0060_p0.50", "10000_th0.0100_p0.20", "10000_th0.0020_p0.20", "10000_th0.0040_p0.40", "10000_th0.0080_p0.10", "10000_th0.0100_p0.30", "10000_th0.0020_p0.30", "10000_th0.0040_p0.50", "10000_th0.0080_p0.20", "10000_th0.0100_p0.40", "10000_th0.0020_p0.40", "10000_th0.0060_p0.10", "10000_th0.0080_p0.30", "10000_th0.0100_p0.50", "10000_th0.0020_p0.50", "10000_th0.0060_p0.20", "10000_th0.0080_p0.40", "10000_th0.0040_p0.10", "10000_th0.0060_p0.30", "10000_th0.0080_p0.50", "10000_th0.0040_p0.20", "10000_th0.0060_p0.40", "10000_th0.0100_p0.10"]
    selected_parameters = Dict(
		Symbol("error_model_name") => [error_model_name],
		Symbol("error_model_parameters_description") => "errorfile_" .* error_file_names,
		Symbol("algo") => ["MinSum"],
		Symbol("n_iterations_BP") => [5],
		Symbol("rounds_per_BP") => 500:500:500,
		Symbol("weight_soft_constraint") => [0.75]
	)
	# Add additional parameters to display
	display_parameters = [Symbol("error_model_parameters_description"), Symbol("num_failures"), Symbol("average_logical_error_rate")]
	focused_dataframe = extract_collected_data(stats_dataframe, selected_parameters, display_parameters)
	println("Focused DataFrame:\n", focused_dataframe)
	save_decoder_dataframe(focused_dataframe, "$(outdir)/explicit_error_model_focused_data.csv")
end
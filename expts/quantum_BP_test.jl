using ArgParse
using LinearAlgebra
using DelimitedFiles
using CorrelatedBPDecoderWithCER

function main(read_from_file::Bool, explicit_error_file::String, ballistic_per_qubit_error_prob::Float64, ballistic_neighbour_error_prob::Float64, n_iterations_of_BP::Int, rounds_per_BP::Int, num_error_samples::Int; llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=4.0, weight_soft_constraint::Float64=0.8, debug::Bool=false, verbose::Bool=false)
	start_time = time()
	# Construct the parity-check matrices for the Hypergraph Product Code of two (7,4) Hamming codes
	# parity_check_Hamming = generate_Hamming_Parity_Check_Matrix(3)
	# (HX, HZ) = get_hypergraph_product_code_H(parity_check_Hamming, parity_check_Hamming)

	# Load the matrices from files
	# X Parity-Check matrix
	HX = readdlm("./../data/hgp_HX_matrix.txt", Int)
	correlations_X = readdlm("./../data/X_correlations.txt", Int)  # Extra rows to accomodate for correlations.
	# Z Parity-Check matrix
	HZ = readdlm("./../data/hgp_HZ_matrix.txt", Int)
	correlations_Z = readdlm("./../data/Z_correlations.txt", Int)  # Extra rows to accomodate for correlations.
	# X Logical operators
	LX = readdlm("./../data/hgp_LX_matrix.txt", Int)
	# Z Logical operators
	LZ = readdlm("./../data/hgp_LZ_matrix.txt", Int)
	
	# Maintain a log of the experiment
	io_log = Base.stdout
	if (debug)
		io_log = open("./../logs/hgp_bp_log.txt", "w")
	end

	# Specify the Hypergraph Product Code
	hgp_hamming = QuantumCode(HX, HZ, LX, LZ; correlations_X=correlations_X, correlations_Z=correlations_Z, name="HGP of (7,4) Hamming Codes")
	if (debug || verbose)	
		print_quantum_code_info(hgp_hamming; io=io_log)
		println(io_log, "----------------------------------------")
	end
	
	if read_from_file
		errormodel = ExplicitErrorModel(explicit_error_file)
		explicit_error_set = sample_errors(errormodel, hgp_hamming.n)
		num_error_samples = size(explicit_error_set, 1)
	else
		# Specify the error model
		errormodel = BallisticErrorModel(ballistic_per_qubit_error_prob, ballistic_neighbour_error_prob; correlations=correlations_X, name="Ballistic Error Model")
		if (debug || verbose)
			print_error_model_info(errormodel; io=io_log)
		end
	end
	
	if (debug || verbose)
		println(io_log, "----------------------------------------")
	end
	
	# Decode using Belief Propagation
	if (debug || verbose)
		println(io_log, "Decoding using Belief Propagation (BP)")
	end
	prior_probabilities = [0.9 for _ in 1:hgp_hamming.n]  # A Prior to get the BP started. It needn't be related to the physical error rate.
	if (debug || verbose)
		println(io_log, "----------------------------------------")
	end
	
	# Montecarlo simulation to estimate the performance of the decoder for the given error model.
	is_decoder_failures = Vector{Bool}(undef, num_error_samples)
	for trial in 1:num_error_samples
		if read_from_file
			error = explicit_error_set[trial, :]
		else
			# Sample error from the IID model.
			error = sample_error(errormodel, hgp_hamming.n)
		end
		
		if (debug || verbose)
			println(io_log, "================= Trial $(trial) ================")
			println(io_log, "Error: ", error)
		end
		
		# Decode the error using BP.
		bpset = quantum_belief_propagation_decoder(
			hgp_hamming, # The quantum code
			error, # The sampled error
			prior_probabilities, # Initial probabilities for each qubit being in error
			rounds_per_BP, # Number of rounds of BP to run
			n_iterations_of_BP; # Number of iterations of BP to run. Each interation consists of several rounds.
			llr_convergence_threshold=llr_convergence_threshold, # Threshold for convergence based on LLR changes
			llr_confidence_threshold=llr_confidence_threshold, # Threshold for confidence in LLR values to consider decoding successful
			weight_soft_constraint=weight_soft_constraint, # Weight for the soft constraints in the Tanner graph.
			verbose=verbose, # Whether to print detailed logs and print statements
			io=(debug ? io_log : stdout)
		)
		is_decoder_failures[trial] = bpset.is_decoder_failure
	end
	
	# Summary of the experiment
	stats = DecoderStatistics(errormodel.name, errormodel.parameters_description, num_error_samples; failures=is_decoder_failures, num_iterations_BP=n_iterations_of_BP, runtime=time() - start_time)
	save_decoder_statistics(stats)

	print_decoder_statistics(stats; io=stdout)
	
	if (debug)
		close(io_log)
	end
	
	return is_decoder_failures
end

function parse_command_line_args()::Dict{String, Any}
	"""
	Parse command-line arguments and return them as a dictionary using `ArgParse`.
	# Input modes
		- **File mode**: Provide `errors_filename` as the first argument, followed by 
		`n_iterations_of_BP` and `rounds_per_BP`.
		- **Parameter mode**: Provide three parameters 
		(`ballistic_per_qubit_error_prob`, `ballistic_neighbour_error_prob`, 
		`num_error_samples`) followed by `n_iterations_of_BP` and `rounds_per_BP`.
	
	# Compulsory line arguments
		- `errors_filename::String` (exclusive with error parameters):  
		Path to file containing precomputed error samples.

		- `ballistic_per_qubit_error_prob::Float64` (positional, parameter mode):  
		Probability of ballistic error per qubit.

		- `ballistic_neighbour_error_prob::Float64` (positional, parameter mode):  
		Probability of ballistic error between neighbouring qubits.

		- `num_error_samples::Int` (positional, parameter mode):  
		Number of error samples to generate.

		- `n_iterations_of_BP::Int` (positional, required):  
		Number of iterations of belief propagation to run.

		- `rounds_per_BP::Int` (positional, required):  
		Number of rounds per BP iteration.

	# Optional keyword arguments
		- `--llr_convergence_threshold::Float64` (default = `1e-6`):  
		Convergence threshold for log-likelihood ratios.

		- `--llr_confidence_threshold::Float64` (default = `4.0`):  
		Confidence threshold for log-likelihood ratios.

		- `--weight_soft_constraint::Float64` (default = `0.8`):  
		Weight applied to soft constraints in BP.

		- `--debug::Bool` (default = `false`):  
		Enable debug mode with extra diagnostics.

		- `--verbose::Bool` (default = `false`):  
		Enable verbose logging of BP progress.

	# Examples
	The script should be run from the folder `expts` as follows:
	```sh
	# Run using error file
	julia --project="./../" quantum_BP_test.jl error_file 50 5
	# where `./../data/error_file.txt` is a file containing the n-qubit error strings.

	# Run using generated errors
	julia --project="./../" quantum_BP_test.jl 0.01 0.02 1000 50 5 --llr_convergence_threshold 1e-6
	"""
	settings = ArgParseSettings()
	
	@add_arg_table! settings begin
		# Input modes
		"--errors_filename"
			help = "Path to file containing precomputed error samples."
			arg_type = String
			default = ""

		"--ballistic_per_qubit_error_prob"
			help = "Probability of an error on each qubit."
			arg_type = Float64
			default = -1.0

		"--ballistic_neighbour_error_prob"
			help = "Probability of an error on neighbouring qubits given that one qubit has an error."
			arg_type = Float64
			default = -1.0

		"--num_error_samples"
			help = "Number of error samples to generate."
			arg_type = Int
			default = -1

		"--n_iterations_of_BP"
			help = "Number of iterations of belief propagation to run."
			arg_type = Int
			default = -1

		"--rounds_per_BP"
			help = "Number of rounds per BP iteration."
			arg_type = Int
			default = -1

		"--llr_convergence_threshold"
			help = "Convergence threshold for log-likelihood ratios."
			arg_type = Float64
			default = 1e-6

		"--llr_confidence_threshold"
			help = "Confidence threshold for log-likelihood ratios."
			arg_type = Float64
			default = 4.0

		"--weight_soft_constraint"
			help = "Weight applied to soft constraints in BP."
			arg_type = Float64
			default = 0.8

		"--debug"
			help = "Enable debug mode."
			arg_type = Bool
			default = false

		"--verbose"
			help = "Enable verbose logging."
			arg_type = Bool
			default = false
	end
	args_dict = parse_args(settings)

	# Ensure that either errors_filename is provided or error parameters are provided, but not both.
	if (args_dict["errors_filename"] != "" && (args_dict["ballistic_per_qubit_error_prob"] != -1.0 || args_dict["ballistic_neighbour_error_prob"] != -1.0 || args_dict["num_error_samples"] != -1))
		throw(ArgumentError("Provide either 'errors_filename' or all three error parameters ('ballistic_per_qubit_error_prob', 'ballistic_neighbour_error_prob', 'num_error_samples'), but not both."))
	else
		if (args_dict["errors_filename"] == "")
			args_dict["read_from_file"] = false
		else
			args_dict["read_from_file"] = true
			# Check if the file exists
			if !isfile("./../data/$(args_dict["errors_filename"]).txt")
				throw(FileNotFoundError("The specified error file './../data/$(args_dict["errors_filename"]).txt' does not exist."))
			end
		end
	end

	return args_dict
end

function print_arguments(args_dict::Dict{String, Any}; io::IO=stdout)
	"""
	Print the parsed command-line arguments in a readable format.
	"""
	println(io, "** Parsed Command-Line Arguments **")
	for (key, value) in args_dict
		println(io, "$(key): $(value)")
	end
	println(io, "----------------------------------------")
end

function generate_runs_for_ballistic(commands_file::String="./../expts/run_commands.txt")
	"""
	Generate commands for running simulations over a range of error parameters for the given error model.
	These commands are of the form:
	julia --project="./../" quantum_BP_test.jl <per_qubit_error_prob> <neighbour_error_prob> <num_error_samples> <n_iterations_of_BP> <rounds_per_BP> --llr_convergence_threshold <llr_convergence_threshold> --llr_confidence_threshold <llr_confidence_threshold> --weight_soft_constraint <weight_soft_constraint> --debug <debug> --verbose <verbose>
	"""
	# Define values for all the parameters.
	ballistic_per_qubit_error_probs = 0.01:0.02:0.1
	ballistic_neighbour_error_probs = 0.01:0.02:0.1
	n_iterations_of_BP = 5
	rounds_per_BP = 50
	num_error_samples = 100
	llr_convergence_threshold = 1e-6
	llr_confidence_threshold = 4.0
	weight_soft_constraint = 0.8
	debug = false
	verbose = false

	run_commands = String[]
	for per_qubit_error_prob in ballistic_per_qubit_error_probs
		for neighbour_error_prob in ballistic_neighbour_error_probs
			command = "julia --project=\"./../\" quantum_BP_test.jl --ballistic_per_qubit_error_prob $(per_qubit_error_prob) --ballistic_neighbour_error_prob $(neighbour_error_prob) --num_error_samples $(num_error_samples) --n_iterations_of_BP $(n_iterations_of_BP) --rounds_per_BP $(rounds_per_BP) --llr_convergence_threshold $(llr_convergence_threshold) --llr_confidence_threshold $(llr_confidence_threshold) --weight_soft_constraint $(weight_soft_constraint) --debug $(debug) --verbose $(verbose)"
			push!(run_commands, command)
		end
	end

	# Write the commands to the specified file
	open(commands_file, "w") do io
		for cmd in run_commands
			println(io, cmd)
		end
	end
end

function plot_data()
    # Define values for the parameters to collect statistics for.
    error_model_name = "Ballistic_Error_Model"
    parameter_ranges = Dict(
        "per_qubit_error_prob" => 0.01:0.02:0.1,
        "neighbour_error_prob" => 0.01:0.02:0.1
    )
	print_collected_data(error_model_name, parameter_ranges; prefix="./../data")
    # Load the collected statistics into a DataFrame
    stats_dataframe = collect_decoder_statistics(error_model_name, parameter_ranges; prefix="./../data")
	# println("Collected DataFrame:\n", stats_dataframe)
    # Plot the statistics
    plot_statistics_for_ballistic_error_model(stats_dataframe; prefix="./../plots")
end

# Run the main function if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
	# Create the './../data' and './../logs' directories if they don't exist
	if !isdir("./../data")
		mkdir("./../data")
	end
	if !isdir("./../logs")
		mkdir("./../logs")
	end
	if !isdir("./../plots")
		mkdir("./../plots")
	end

	# Parse command-line arguments
	args_dict = parse_command_line_args()
	print_arguments(args_dict; io=stdout)

	# Extract arguments
	errors_filename = args_dict["errors_filename"]
	ballistic_per_qubit_error_prob = args_dict["ballistic_per_qubit_error_prob"]
	ballistic_neighbour_error_prob = args_dict["ballistic_neighbour_error_prob"]
	num_error_samples = args_dict["num_error_samples"]
	
	n_iterations_of_BP = args_dict["n_iterations_of_BP"]
	rounds_per_BP = args_dict["rounds_per_BP"]
	llr_convergence_threshold = args_dict["llr_convergence_threshold"]
	llr_confidence_threshold = args_dict["llr_confidence_threshold"]
	weight_soft_constraint = args_dict["weight_soft_constraint"]
	debug = args_dict["debug"]
	verbose = args_dict["verbose"]
	
	# Call the main function with parsed arguments
	main(
		args_dict["read_from_file"],
		errors_filename,
		ballistic_per_qubit_error_prob,
		ballistic_neighbour_error_prob,
		n_iterations_of_BP,
		rounds_per_BP,
		num_error_samples;
		llr_convergence_threshold=llr_convergence_threshold,
		llr_confidence_threshold=llr_confidence_threshold,
		weight_soft_constraint=weight_soft_constraint,
		debug=debug,
		verbose=verbose
	)
end
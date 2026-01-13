using LinearAlgebra
using DelimitedFiles
using DataStructures
using CorrelatedBPDecoderWithCER
include("ballistic_errors.jl")
include("explicit_errors.jl")

function main(read_from_file::Bool, explicit_error_file::String, ballistic_per_qubit_error_prob::Float64, ballistic_neighbour_error_prob::Float64, num_error_samples::Int, algo::String, n_iterations_BP::Int, rounds_per_BP::Int; llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=4.0, weight_soft_constraint::Float64=0.8, debug::Bool=false, verbose::Bool=false, outdir::String="./../data")
	start_time = time()
	# Construct the parity-check matrices for the Hypergraph Product Code of two (7,4) Hamming codes
	# parity_check_Hamming = generate_Hamming_Parity_Check_Matrix(3)
	# (HX, HZ) = get_hypergraph_product_code_H(parity_check_Hamming, parity_check_Hamming)

	# Load the parity check matrices from files
	# X Parity-Check matrix
	HX = readdlm("./../data/hamming/HX.txt", Int)
	correlations_X = readdlm("./../data/hamming/connectivity_matrix.txt", Int)  # Extra rows to accomodate for correlations.
	# Z Parity-Check matrix
	HZ = readdlm("./../data/hamming/HZ.txt", Int)
	correlations_Z = readdlm("./../data/hamming/connectivity_matrix.txt", Int)  # Extra rows to accomodate for correlations.
	# X Logical operators
	LX = readdlm("./../data/hamming/LX.txt", Int)
	# Z Logical operators
	LZ = readdlm("./../data/hamming/LZ.txt", Int)

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
		errormodel = ExplicitErrorModel("./../data/hamming/$(explicit_error_file).txt")
		explicit_error_set = sample_errors(errormodel, hgp_hamming.n)
		num_error_samples = size(explicit_error_set, 2)
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
			error = explicit_error_set[:, trial]
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
			hgp_hamming, # The quantum code.
			error, # The sampled error.
			prior_probabilities, # Initial probabilities for each qubit being in error.
			rounds_per_BP, # Number of rounds of BP to run.
			n_iterations_BP; # Number of iterations of BP to run. Each iteration consists of several rounds.
			algo, # The BP algorithm to use (:SumProduct or :MinSum).
			llr_convergence_threshold=llr_convergence_threshold, # Threshold for convergence based on LLR changes.
			llr_confidence_threshold=llr_confidence_threshold, # Threshold for confidence in LLR values to consider decoding successful.
			weight_soft_constraint=weight_soft_constraint, # Weight for the soft constraints in the Tanner graph.
			verbose=verbose, # Whether to print detailed logs and print statements.
			io=(debug ? io_log : stdout)
		)
		is_decoder_failures[trial] = bpset.is_decoder_failure
	end
	
	# Summary of the experiment
	stats = DecoderStatistics(
		algo, 
		errormodel.name, 
		errormodel.parameters_description, 
		num_error_samples, 
		n_iterations_BP, 
		rounds_per_BP, 
		weight_soft_constraint;
		failures=is_decoder_failures, 
		runtime=time() - start_time
	)
	record_decoder_statistics(stats)
	if (debug)
		close(io_log)
	end
	
	# return is_decoder_failures
end

# Run the main function if this script is executed directly
# Run the parallel command with `parallel --line-buffer < run_commands_explicit_errormodel_varrying_alpha.txt > ./../data/debankan/explicit_error_model_results.txt 2>&1`
if abspath(PROGRAM_FILE) == @__FILE__
	# Create the './../data' and './../logs' directories if they don't exist
	prefix="./../data/hamming"
	if !isdir(prefix)
		mkdir(prefix)
	end
	if !isdir("./../logs")
		mkdir("./../logs")
	end
	if !isdir("./../plots")
		mkdir("./../plots")
	end

	# Parse command-line arguments
	args_dict = parse_command_line_args_BP(;prefix=prefix)
	# print_arguments(args_dict; io=stdout)

	# Extract arguments
	errors_filename = args_dict["errors_filename"]
	ballistic_per_qubit_error_prob = args_dict["ballistic_per_qubit_error_prob"]
	ballistic_neighbour_error_prob = args_dict["ballistic_neighbour_error_prob"]
	num_error_samples = args_dict["num_error_samples"]
	
	algo = args_dict["algo"]
	if !(algo in ("SumProduct", "MinSum"))
		throw(ArgumentError("Invalid value for 'algo'. Must be either 'SumProduct' or 'MinSum'."))
	end
	n_iterations_BP = args_dict["n_iterations_BP"]
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
		num_error_samples,
		algo,
		n_iterations_BP,
		rounds_per_BP;
		llr_convergence_threshold=llr_convergence_threshold,
		llr_confidence_threshold=llr_confidence_threshold,
		weight_soft_constraint=weight_soft_constraint,
		debug=debug,
		verbose=verbose,
		outdir=prefix
	)
end
using ArgParse

function parse_command_line_args_BP(;prefix="./../data")::Dict{String, Any}
	"""
	Parse command-line arguments and return them as a dictionary using `ArgParse`.
	# Input modes
		- **File mode**: Provide `errors_filename` as the first argument, followed by 
		`n_iterations_BP` and `rounds_per_BP`.
		- **Parameter mode**: Provide three parameters 
		(`ballistic_per_qubit_error_prob`, `ballistic_neighbour_error_prob`, 
		`num_error_samples`) followed by `n_iterations_BP` and `rounds_per_BP`.
	
	# Compulsory line arguments
		- `errors_filename::String` (exclusive with error parameters):  
		Path to file containing precomputed error samples.

		- `ballistic_per_qubit_error_prob::Float64` (positional, parameter mode):  
		Probability of ballistic error per qubit.

		- `ballistic_neighbour_error_prob::Float64` (positional, parameter mode):  
		Probability of ballistic error between neighbouring qubits.

		- `num_error_samples::Int` (positional, parameter mode):  
		Number of error samples to generate.

		- `n_iterations_BP::Int` (positional, required):  
		Number of iterations of belief propagation to run.

		- `rounds_per_BP::Int` (positional, required):  
		Number of rounds per BP iteration.

	# Optional keyword arguments
		- `-- algo::String` (default = `SumProduct`):
		The BP algorithm to use. Options are `SumProduct` or `MinSum`.

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
		
		"--algo"
			help = "The BP algorithm to use. Options are 'SumProduct' or 'MinSum'."
			arg_type = String
			default = "SumProduct"

		"--n_iterations_BP"
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
			if !isfile("$(prefix)/$(args_dict["errors_filename"]).txt")
				throw(ArgumentError("The specified error file '$(prefix)/$(args_dict["errors_filename"]).txt' does not exist."))
			end
		end
	end

	return args_dict
end

function parse_command_line_args_NN(;prefix::String="./../data")::Dict{String, Any}
	"""
	Parse command-line arguments for Neural BP experiments and return them as a dictionary using `ArgParse`.
	# Compulsory line arguments
		- `-- codename::String`:  
		Name of a directory containing the parity-check matrices and the logical operators.
		The directory should contain the following files:
			- `HX.txt`: Parity-check matrix for X errors.
			- `LX.txt`: Logical operators for X errors.
			- `connectivity_matrix.txt`: Connectivity matrix for correlated errors.
			- `train_error_patterns_Z.txt`: Training error patterns for Z errors.
	
	# Keywords for training the Neural BP model
		- `-- n_hidden_layers::Int` (default = 5):
		Number of hidden layers in the Neural BP model.

		- `--n_epochs::Int` (default = `5`):  
		Number of training epochs.

		- `--batch_size::Int` (default = `100`):  
		Batch size for training.

		- `--n_samples::Int` (default = `-1`):  
		Number of samples to use for training. Use all available samples if set to -1.

		- `--retrain::Bool` (default = `false`):  
		Retrain the model even if trained weights are available.

	# Examples
	The script should be run from the folder `expts` as follows:
	```sh
	julia --project="./../" neural_bp_experiments.jl --codename hamming --n_hidden_layers 5 --n_epochs 5 --batch_size 100 --retrain false
	```
	Where `codename` is a folder inside `./../data/` containing the required files.
	"""
	settings = ArgParseSettings()
	
	@add_arg_table! settings begin
		"--codename"
			help = "Name of a directory containing the parity-check matrices and the logical operators."
			arg_type = String
			default = ""
		"--n_hidden_layers"
			help = "Number of hidden layers in the Neural BP model."
			arg_type = Int
			default = 5
		"--n_epochs"
			help = "Number of training epochs."
			arg_type = Int
			default = 5
		"--batch_size"
			help = "Batch size for training."
			arg_type = Int
			default = 100
		"--n_samples"
			help = "Number of samples to use for training. Use all available samples if set to -1."
			arg_type = Int
			default = -1
		"--correlation_strength"
			help = "Strength of correlation in the error model."
			arg_type = Float64
			default = 0.0
		"--retrain"
			help = "Retrain the model even if trained weights are available."
			arg_type = Bool
			default = false
		"--train"
			help = "Name of the file used for training the Neural BP model."
			arg_type = String
			default = ""
		"--test"
			help = "Name of the file used for testing the trained Neural BP model."
			arg_type = String
			default = ""
	end
	args_dict = parse_args(settings)

	# Ensure that the `./../data/codename` has all the required files.
	if isdir("$(prefix)/$(args_dict["codename"])")
        # Check for required files
		required_files = ["HX.txt", "LX.txt", "connectivity_matrix.txt", args_dict["train"], args_dict["test"]]
		for file in required_files
			if !isfile("$(prefix)/$(args_dict["codename"])/$(file)")
				throw(error("The required file '$(file)' is missing in the directory '$(prefix)/$(args_dict["codename"])'. Please add the file and try again."))
			end
		end
		# All required files are present.
    else
		throw(error("The specified data directory does not exist: $(prefix). Please create the directory and add the required data files."))
	end

	args_dict = parse_args(settings)
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

function generate_runs(parameter_ranges::OrderedDict{String, AbstractVector{<:Any}}, commands_file::String="./../expts/run_commands.txt")
	"""
	Generate commands for running simulations over a range of error parameters for the given error model.
	These commands are of the form:
	julia --project="./../" quantum_BP_test.jl <per_qubit_error_prob> <neighbour_error_prob> <num_error_samples> <n_iterations_BP> <rounds_per_BP> --llr_convergence_threshold <llr_convergence_threshold> --llr_confidence_threshold <llr_confidence_threshold> --weight_soft_constraint <weight_soft_constraint> --debug <debug> --verbose <verbose>
	"""
	# Take a catresian product of all the values in the `parameter_ranges` dictionary.
	parameter_names = collect(keys(parameter_ranges))
	parameter_values = Iterators.product(collect(values(parameter_ranges))...)
	
	run_commands = String[]
	for param_vals in parameter_values
		param_dict = Dict{String, Any}()
		for (i, param_name) in enumerate(parameter_names)
			param_dict[param_name] = param_vals[i]
		end
		command = "julia --project=\"./../\" quantum_BP_test.jl"
		for (param_name, param_value) in param_dict
			command *= " --$(param_name) $(param_value)"
		end
		push!(run_commands, command)
	end
	
	# Write the commands to the specified file
	open(commands_file, "w") do io
		for cmd in run_commands
			println(io, cmd)
		end
	end
end
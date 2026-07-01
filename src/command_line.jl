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

function parse_command_line_args_NN()::Dict{String, Any}
	"""
	Parse command-line arguments for Neural BP experiments and return them as a dictionary using `ArgParse`.
	# Compulsory line arguments
		- `-- workdir::String`:  
		Working directory where the code and data are located. Default is `./../data`.
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
		"--workdir"
			help = "Working directory where the code and data are located."
			arg_type = String
			default = "./../data"
		"--codename"
			help = "Name of a directory containing the parity-check matrices and the logical operators."
			arg_type = String
			default = ""
		"--n_hidden_layers"
			help = "Number of hidden layers in the Neural BP model."
			arg_type = Int
			default = 5
		"--n_samples"
			help = "Number of samples to use for training. Use all available samples if set to -1."
			arg_type = Int
			default = -1
		"--correlation_strengths_file"
			help = "File containing the correlation strengths for the additional loss term for correlations. The file should contain a vector of correlation strengths corresponding to the rows of the connectivity matrix."
			arg_type = String
			default = "unspecified.txt"
		"--hyperparams"
			help = "JSON file containing the hyperparameters for training the Neural BP model. If not provided, default hyperparameters will be used."
			arg_type = String
			default = "hyperparams.json"
		"--train"
			help = "Name of the file used for training the Neural BP model."
			arg_type = String
			default = ""
		"--test"
			help = "Name of the file used for testing the trained Neural BP model."
			arg_type = String
			default = ""
		"--isdebug"
			help = "Enable debug mode: log per-batch loss components and weight statistics."
			arg_type = Bool
			default = false
		"--quiet"
			help = "Suppress progress bars."
			arg_type = Bool
			default = false
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

import TOML

function parse_hyper_parameters(hyperparams_file::String=""; prefix::String="./../data")::Dict{String, Any}
    """
    Parse hyperparameters from a TOML file or use default values.

    # Arguments
    - `hyperparams_file::String`: Name of the TOML file containing hyperparameters.
    - `prefix::String`: Directory prefix where the hyperparameters file is located.

    # Returns
    - `Dict{String, Any}`: Dictionary of hyperparameters.
    """
    default_hyperparams::Dict{String, Any} = Dict(
        "retrain" => false, # Whether to retrain the model even if trained weights are available.
        "learning_rate" => 1f-1, # Learning rate for training the Neural BP model using the ADAM optimizer
        "max_grad_norm" => 2f0, # Gradient clipping threshold
        "weight_decay" => 1f-4, # L2 regularization strength for ADAM optimizer
        "nanskip" => 5, # Number of consecutive NaN occurrences in the loss before skipping the batch during training
        "adam_eps" => 1f-4, # Epsilon parameter for the ADAM optimizer to improve numerical stability
        "batch_size" => 100, # Batch size for training
        "n_epochs" => 5, # Number of training epochs
		"warmup_layers" => 10, # First number of layers to leave unconstrained in the loss function.
		"online_training" => false, # If true, we will generate training samples on the fly instead of reading from a file. However, right now we don't have an implementation for this, so we will simply read a random subset of `batch_size` samples from the training dataset to simulate the online training scenario. Important: when this is set to true, explicitly make sure that the batch size divides the number of samples in the training dataset.
		"n_gradient_updates_per_epoch" => 0, # If `online_training` is true, this specifies the number of gradient updates to perform per epoch. If set to 0, it defaults to the number of batches in the training dataset.
        # Annealing schedule for the loss hyperparameters
        "loss_layer_temperature" => "0.1,5.0,0.9,down", # Smooth minimum approximation temperature
        "correlation_importance" => "0.1,1.0,0.1,down", # Correlation penalty importance
        "llr_certainty_importance" => "0.001,0.01,0.1,down", # LLR convergence term importance
        "sparsity_importance" => "0.0,0.01,0.5,up", # Sparsity encouragement term importance
		# Initial conditions: all weights are initialized to Gaussian random values around 1, with a standard deviation of σ = 0.3.
		"initial_conditions_scale" => 0.3f0
    )

    hyperparams_file_path = "$(prefix)/models/$(hyperparams_file)"
    if hyperparams_file != "" && isfile(hyperparams_file_path)
        
        # Use Julia's built-in TOML parser
        file_hyperparams = TOML.parsefile(hyperparams_file_path)
        
        # Merge default hyperparameters with those from the file, giving priority to the file values
        updated_hyperparams = merge(default_hyperparams, file_hyperparams)

        # --- Convert specific keys to Float32 ---
        float32_keys = ["learning_rate", "max_grad_norm", "weight_decay", "adam_eps", "initial_conditions_scale"]
        for key in float32_keys
            if haskey(updated_hyperparams, key)
                updated_hyperparams[key] = Float32(updated_hyperparams[key])
            end
        end
		
		# Parse annealing schedules from strings into structured dictionaries
        for key in ["loss_layer_temperature", "correlation_importance", "llr_certainty_importance", "sparsity_importance"]
            # Added `isa String` check for safety, in case the TOML file is ever updated to use inline tables
            if haskey(updated_hyperparams, key) && isa(updated_hyperparams[key], String)
                schedule_parts = split(updated_hyperparams[key], ",")
                updated_hyperparams[key] = Dict(
                    "min" => parse(Float32, schedule_parts[1]),
                    "max" => parse(Float32, schedule_parts[2]),
                    "decay" => parse(Float32, schedule_parts[3]),
					"direction" => lowercase(schedule_parts[4])
                )

				# If the direction is neither "up" nor "down", throw an error
				if !(updated_hyperparams[key]["direction"] in ["up", "down"])
					throw(ArgumentError("Unknown direction direction for annealing schedule of key \"$(key)\": $(updated_hyperparams[key]["direction"]). Must be 'up' or 'down'."))
				end
            end
        end

        return updated_hyperparams
    else
        println("Hyperparameters file not provided or does not exist. Using default values.")

        # Convert default annealing schedules from strings to structured dictionaries
        for key in ["loss_layer_temperature", "correlation_importance", "llr_certainty_importance", "sparsity_importance"]
            schedule_parts = split(default_hyperparams[key], ",")
            default_hyperparams[key] = Dict(
                "min" => parse(Float32, schedule_parts[1]),
                "max" => parse(Float32, schedule_parts[2]),
                "decay" => parse(Float32, schedule_parts[3]),
				"direction" => lowercase(schedule_parts[4])
            )
        end

        return default_hyperparams
    end
end

function disable_retrain_in_hyperparams(hyperparams_file::String)
	"""
	Stream-edit the TOML file at `hyperparams_file` to flip `retrain = true` to
	`retrain = false`, in place. Uses `sed` so comments, blank lines, key ordering,
	and TOML formatting are preserved exactly — only the literal `true` token
	following `retrain =` is replaced.

	The regex anchors on whole lines (`^...\$`) and requires `true` to be followed
	either by end-of-line, whitespace, or a `#` comment, so it won't touch
	values like `truecolor` or quoted strings `"true"`. Lines that already have
	`retrain = false` (or no `retrain` key) are passed through unchanged.

	`sed -E` (extended regexes) is supported on both BSD sed (macOS) and GNU sed
	(Linux clusters). We pipe through a fresh buffer instead of using `sed -i`
	because BSD and GNU disagree on the in-place flag's syntax.
	"""
	if !isfile(hyperparams_file)
		error("Hyperparams file not found: $(hyperparams_file)")
	end

	sed_expr = raw"s|^([[:space:]]*retrain[[:space:]]*=[[:space:]]*)true([[:space:]]*(#.*)?)$|\1false\2|"
	new_contents = read(`sed -E $(sed_expr) $(hyperparams_file)`, String)
	write(hyperparams_file, new_contents)

	return hyperparams_file
end
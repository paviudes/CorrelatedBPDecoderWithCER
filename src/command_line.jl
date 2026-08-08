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
		"--cer_data", "--correlation_strengths_file"
			# `--correlation_strengths_file` is kept as a DEPRECATED ALIAS: the file
			# holds single-qubit marginals as well as pairwise couplings, so the old
			# name undersold it — but ~180 occurrences across generated command files
			# and shell scripts still use it, and silently breaking them is worse than
			# a second name. ArgParse keys the result on the FIRST long name, so this
			# lands in args_dict["cer_data"].
			help = "CER data file: single-qubit error rates and two-qubit couplings J_ij, read from <codename>/correlated_weights/. (Formerly --correlation_strengths_file, still accepted.)"
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
		"--diagnose"
			help = "Split decode failures into COSET failures (syndrome cleared, wrong logical coset) and CONVERGENCE failures (no layer ever cleared the syndrome). Adds four columns to the results CSV, writes one row per FAILED sample to `per_sample_failures_*.csv`, and writes the committed-layer histogram of the successes to `layer_profile_*.csv`. Costs no extra forward pass."
			arg_type = Bool
			default = false
		"--seed"
			help = "RNG seed for a reproducible training run. Overrides `seed` in the hyperparameters TOML. When neither is given the global RNG is left untouched, which is the historical (non-deterministic) behaviour."
			arg_type = Int
			# NO `default` on purpose: ArgParse then yields `nothing` when the flag
			# is absent, which is the only way to distinguish "unset" from any
			# particular integer the user might legitimately want as a seed.
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

    # The `seed` key
    `seed` (integer) makes a training run reproducible: it seeds the global RNG
    before the initial weights are drawn and before any batch is sampled, and it
    is appended to the weights / results / debug filenames as `_seed_<n>` so runs
    that differ only in seed cannot overwrite one another.

    It is deliberately ABSENT from `default_hyperparams`. When it is unset the
    global RNG is not touched at all, so existing runs keep their historical,
    non-deterministic behaviour rather than silently becoming `seed = 0`. Use
    `hyperparameter_seed`, `seed_tag_for` and `apply_training_seed!` to read it.
    """
    default_hyperparams::Dict{String, Any} = Dict(
        "retrain" => false, # Whether to retrain the model even if trained weights are available.
        "single_qubit_rescale" => 0.0f0, # Put the CER SINGLE-QUBIT rates at this scale: rescale them so their MEDIAN lands here, UNLESS their maximum already reaches it, in which case leave them exactly as parsed. 0 (the default) disables it entirely. One value, two roles — read it as "priors at this scale, unless they already get there". It can therefore only ever SOFTEN, never sharpen. This is an INFERENCE TEMPERATURE, not a calibration fix: the raw rates were measured correct to 0.5%, and deliberately softening them to median 0.1 decoded 12.5x better because BP cannot iterate out of a prior at LLR 5.46. Couplings J_ij are never touched. Prefer this to `prior_llr_clip`, which clamps and therefore flattens all qubits to one constant when every raw LLR exceeds the clip.
        "prior_llr_clip" => 0.0f0, # Cap on |initial LLR| (0 = disabled). SUPERSEDED for CER data by `single_qubit_rescale`: every raw CER LLR here is 5.33..5.58, so any clip below that collapses all 72 qubits to a single constant and destroys the per-qubit information. Separates the CER prior's INFORMATION from its MAGNITUDE: CER rates give LLR ~ 5.4 (tanh' ~ 0.018) vs the no-CER fallback's 2.2 (tanh' ~ 0.36), a ~20x weaker gradient through the message nonlinearity. Clip to ~2.5 to equalise conditioning between the arms.
        "use_CER" => true, # Whether to use correlated-error-rate (CER) priors. If false, the correlated_weights/ folder is ignored: single-qubit priors default to p=0.1 and the correlation loss term is dropped. Outputs are tagged `_no_cer` so CER and no-CER runs don't overwrite each other.
        "learning_rate" => 1f-1, # Learning rate for training the Neural BP model using the ADAM optimizer
        "max_grad_norm" => 2f0, # Gradient clipping threshold
        "weight_decay" => 1f-4, # L2 regularization strength for ADAM optimizer
        "nanskip" => 5, # Number of consecutive NaN occurrences in the loss before skipping the batch during training
        "adam_eps" => 1f-4, # Epsilon parameter for the ADAM optimizer to improve numerical stability
        "batch_size" => 100, # Batch size for training
        "gpu_memory" => "", # GPU memory available for TESTING, e.g. "16G" or "20480M". Used to size the prediction batch without editing predict.jl (which would force a recompile). Empty = fall back to ENV["GPU_MEMORY"], then SLURM_MEM_PER_GPU (exported automatically by --mem-per-gpu), then the built-in 16384. Has no effect on training.
        "prediction_batch_size" => 0, # Explicit prediction batch size. Overrides `gpu_memory` when > 0; 0 = derive it.
        "n_epochs" => 5, # Number of training epochs
		"warmup_layers" => 10, # First number of layers to leave unconstrained in the loss function.
		"online_training" => false, # If true, we will generate training samples on the fly instead of reading from a file. However, right now we don't have an implementation for this, so we will simply read a random subset of `batch_size` samples from the training dataset to simulate the online training scenario. Important: when this is set to true, explicitly make sure that the batch size divides the number of samples in the training dataset.
		"n_gradient_updates_per_epoch" => 0, # If `online_training` is true, this specifies the number of gradient updates to perform per epoch. If set to 0, it defaults to the number of batches in the training dataset.
        # Annealing schedule for the loss hyperparameters
        "loss_layer_temperature" => "0.1,5.0,0.9,down", # Smooth minimum approximation temperature
        "correlation_weight" => "1.0,1.0,0.1,down", # Overall weight α₄ on the correlation term (constant 1.0 by default; raise to strengthen). NOTE: the correlation term has no internal counterweight, so tune this together with `sparsity_importance` (α₃).
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
        float32_keys = ["learning_rate", "max_grad_norm", "weight_decay", "adam_eps", "initial_conditions_scale",
                        "prior_llr_clip", "single_qubit_rescale"]
        for key in float32_keys
            if haskey(updated_hyperparams, key)
                updated_hyperparams[key] = Float32(updated_hyperparams[key])
            end
        end

        # --- Convert specific keys to Int -------------------------------------
        # `seed` must stay an integer: `Random.seed!` takes an Integer, and a
        # Float32 seed would also produce a filename tag like `_seed_7.0`.
        # `Int(7.5)` throws rather than silently truncating, which is what we want.
        integer_keys = ["seed"]
        for key in integer_keys
            if haskey(updated_hyperparams, key) && updated_hyperparams[key] !== nothing
                updated_hyperparams[key] = Int(updated_hyperparams[key])
            end
        end
		
		# Parse annealing schedules from strings into structured dictionaries
        for key in ["loss_layer_temperature", "correlation_weight", "llr_certainty_importance", "sparsity_importance"]
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
        for key in ["loss_layer_temperature", "correlation_weight", "llr_certainty_importance", "sparsity_importance"]
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

# ============================================================================
#                        Reproducible runs: the `seed` key
# ============================================================================
# Training draws from the global RNG in four places — `random_values_around_one`
# for the three weight vectors (expts/neural_bp_experiments.jl, and the fallback
# inside `train_Nachmani_neuralbp`), the `randn` fallbacks in the
# `NachmaniNeuralBP`/`NeuralBP` constructors, `rand(1:n_samples, batch_size)` in
# the online-training path, and `randperm` in the fixed-dataset path.
#
# Left unseeded, two runs of a provably identical configuration were measured at
# 309 and 620 logical failures out of 10^6 — a factor of two from RNG alone,
# which is larger than any effect this project is trying to resolve. Seeding
# makes a run a function of (config, seed) so comparisons can be pooled over a
# fixed set of seeds.
#
# These three helpers exist so the seed is read, applied, and turned into a
# filename tag in exactly ONE place each. In particular `seed_tag_for` is used by
# both the weights filename and the results filename; if they disagreed, a
# `retrain = false` run would load weights that do not correspond to its results.
# ============================================================================

"""
    hyperparameter_seed(hyperparameters) -> Union{Int, Nothing}

The `seed` hyperparameter as an `Int`, or `nothing` when it is unset.

`nothing` means "do not touch the global RNG" — NOT "seed with 0". Keeping those
distinct is what lets existing, unseeded runs behave exactly as they did before.
"""
function hyperparameter_seed(hyperparameters::Dict)::Union{Int, Nothing}
    if !haskey(hyperparameters, "seed")
        return nothing
    end
    raw_seed_value = hyperparameters["seed"]
    if raw_seed_value === nothing
        return nothing
    end
    seed::Int = Int(raw_seed_value)
    return seed
end

"""
    seed_tag_for(hyperparameters) -> String

`"_seed_<n>"` when a seed is set, `""` otherwise.

The empty string matters as much as the tag: with no seed every filename this
package writes is byte-identical to what it wrote before `seed` existed, so old
results stay addressable and the sweep scripts keep matching.
"""
function seed_tag_for(hyperparameters::Dict)::String
    seed::Union{Int, Nothing} = hyperparameter_seed(hyperparameters)
    seed_tag::String = ""
    if seed !== nothing
        seed_tag = "_seed_$(seed)"
    end
    return seed_tag
end

"""
    apply_training_seed!(hyperparameters) -> Union{Int, Nothing}

Seed the global RNG from the `seed` hyperparameter and return the seed applied,
or `nothing` if none was set (in which case the RNG is left alone).

Call this BEFORE anything random happens. It is called in two places, and both
are necessary:

  - `expts/neural_bp_experiments.jl`, before it builds `initial_conditions` —
    those weight draws happen in the script, so seeding only inside the library
    would leave the initial weights non-reproducible;
  - the top of `train_Nachmani_neuralbp`, so a caller using the package directly
    (without the script) still gets a reproducible run.

Calling it twice is harmless: re-seeding restarts the same stream, so the run
remains a pure function of (config, seed). It does mean the batch-sampling draws
begin at the same stream position as the weight draws did, which is of no
practical consequence — they consume different distributions for different
purposes.

The global RNG is the right target here rather than a threaded-through RNG
object, because nothing in the training path is multi-threaded (there is no
`@threads`/`ThreadsX` in it). See the note in the README about BLAS threads.
"""
function apply_training_seed!(hyperparameters::Dict)::Union{Int, Nothing}
    seed::Union{Int, Nothing} = hyperparameter_seed(hyperparameters)
    if seed !== nothing
        Random.seed!(seed)
    end
    return seed
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
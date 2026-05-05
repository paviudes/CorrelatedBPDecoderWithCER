function generate_ballistic_training_data(
    prefix::String,
    ballistic_per_qubit_error_probs::AbstractVector{Float64},
    ballistic_neighbour_error_probs::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/ballistic_training_data.txt"
)::String
    """
    Generate training data for the Ballistic error model.
    The training data consists of error patterns generated according to the Ballistic error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, p_neighbour) for p_qubit in ballistic_per_qubit_error_probs for p_neighbour in ballistic_neighbour_error_probs]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate
    
    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)
    # Iterate over all combinations of error parameters
    for (i, (ballistic_per_qubit_error_prob, ballistic_neighbour_error_prob)) in enumerate(error_rates)
        errormodel = BallisticErrorModel(ballistic_per_qubit_error_prob, ballistic_neighbour_error_prob; correlations=connectivity_matrix, name="Ballistic Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1
    
    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')
    return output_errors_file
end

function generate_randomwalk_training_data(
    prefix::String,
    randomwalk_per_qubit_error_probs::AbstractVector{Float64},
    randomwalk_lengths::AbstractVector{Int},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/randomwalk_training_data.txt"
)::String
    """
    Generate training data for the Random Walk error model.
    The training data consists of error patterns generated according to the Random Walk error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, length) for p_qubit in randomwalk_per_qubit_error_probs for length in randomwalk_lengths]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate
    
    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)
    # Iterate over all combinations of error parameters
    for (i, (randomwalk_per_qubit_error_prob, randomwalk_length)) in enumerate(error_rates)
        errormodel = RandomWalkErrorModel(randomwalk_per_qubit_error_prob, randomwalk_length, nqubits; correlations=connectivity_matrix, name="Random Walk Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1
    
    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')
    return output_errors_file
end

function generate_randomwalk_testing_data(
    prefix::String,
    randomwalk_per_qubit_error_probs::AbstractVector{Float64},
    randomwalk_lengths::AbstractVector{Int},
    samples_per_error_rate::Int;
    output_errors_dir::String="./../data/hamming"
)::Vector{String}
    """
    Generate training data for the Random Walk error model.
    The training data consists of error patterns generated according to the Random Walk error model for a small range of error parameters.
    The generated error patterns are saved to a file for later use in training the Neural BP decoder.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)
    
    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [(p_qubit, length) for p_qubit in randomwalk_per_qubit_error_probs for length in randomwalk_lengths]
    
    # Define the error model and generate error patterns
    output_error_files = String[]
    # Iterate over all combinations of error parameters
    for (randomwalk_per_qubit_error_prob, randomwalk_length) in error_rates
        errormodel = RandomWalkErrorModel(randomwalk_per_qubit_error_prob, randomwalk_length, nqubits; correlations=connectivity_matrix, name="Random Walk Error Model")
        error_patterns = sample_errors(errormodel, nqubits, samples_per_error_rate)
        # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
        # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
        error_patterns[error_patterns .== 1] .= 0
        error_patterns[error_patterns .== 2] .= 1
        error_patterns[error_patterns .== 3] .= 1
        # Save the generated error patterns to a file
        output_errors_file = "$(output_errors_dir)/test_error_patterns_Z_p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length).txt"
        writedlm(output_errors_file, error_patterns', ' ')
        push!(output_error_files, output_errors_file)
        # Print the command to run the test with this error patterns file
        println("julia --project=./../ neural_bp_experiments.jl " *
                "--codename hamming " *
                "--n_hidden_layers 50 " *
                "--n_epochs 5 " *
                "--batch_size 32 " *
                "--retrain false " *
                "--train randomwalk_training_data.txt " *
                "--test test_error_patterns_Z_p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length).txt " *
                "--correlation_strength 0.5")
        println("echo \"Testing done for p_$(randomwalk_per_qubit_error_prob)_nb_$(randomwalk_length)\" >&2")
    end
    return output_error_files
end

function generate_regenerative_training_data(
    prefix::String,
    regenerative_block_sizes::AbstractVector{Int},
    regenerative_block_probabilities::AbstractVector{Float64},
    regenerative_error_probs_within_block::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_file::String="./../data/hamming/regenerative_training_data.txt"
)::String
    """
    Generate training data for the Regenerative error model.
    The training data consists of error patterns generated according to the Regenerative error model for a small range of error parameters.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    # connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)

    # Determine the number of qubits and the number of error patterns to generate
    nqubits = size(parity_check_matrix, 2)
    error_rates = [
        (block_size, block_prob, error_prob_within_block) 
        for block_size in regenerative_block_sizes 
        for block_prob in regenerative_block_probabilities 
        for error_prob_within_block in regenerative_error_probs_within_block
    ]
    n_error_rates = length(error_rates)
    nsamples = n_error_rates * samples_per_error_rate

    # Define the error model and generate error patterns
    error_patterns = zeros(Int, nsamples, nqubits)

    # Iterate over all combinations of error parameters
    for (i, (block_size, block_prob, error_prob_within_block)) in enumerate(error_rates)
        errormodel = RegenerativeErrorModel(block_size, block_prob, error_prob_within_block, nqubits; name="Regenerative Error Model")
        start_index = (i - 1) * samples_per_error_rate + 1
        end_index = i * samples_per_error_rate
        error_patterns[start_index:end_index, 1:nqubits] = sample_errors(errormodel, nqubits, samples_per_error_rate)
    end

    # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
    # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
    error_patterns[error_patterns .== 1] .= 0
    error_patterns[error_patterns .== 2] .= 1
    error_patterns[error_patterns .== 3] .= 1

    # Save the generated error patterns to a file
    writedlm(output_errors_file, error_patterns', ' ')

    return output_errors_file
end

function generate_regenerative_testing_data(
    prefix::String,
    regenerative_block_sizes::AbstractVector{Int},
    regenerative_block_probabilities::AbstractVector{Float64},
    regenerative_error_probs_within_block::AbstractVector{Float64},
    samples_per_error_rate::Int;
    output_errors_dir::String="./../data/hamming"
)::Vector{String}
    """
    Generate testing data for the Regenerative error model.
    The testing data consists of error patterns generated according to the Regenerative error model for a small range of error parameters.
    """
    # Load the parity-check matrix and the connectivity matrix for the code
    parity_check_matrix = readdlm("$(prefix)/HX.txt", Int)
    # connectivity_matrix = readdlm("$(prefix)/connectivity_matrix.txt", Int)

    # Determine the number of qubits
    nqubits = size(parity_check_matrix, 2)
    error_rates = [
        (block_size, block_prob, error_prob_within_block) 
        for block_size in regenerative_block_sizes 
        for block_prob in regenerative_block_probabilities 
        for error_prob_within_block in regenerative_error_probs_within_block
    ]

    output_error_files = String[]

    # Iterate over all combinations of error parameters
    for (block_size, block_prob, error_prob_within_block) in error_rates
        errormodel = RegenerativeErrorModel(block_size, block_prob, error_prob_within_block, nqubits; name="Regenerative Error Model")
        error_patterns = sample_errors(errormodel, nqubits, samples_per_error_rate)

        # Turn Y errors (2) into Z (1) and turn X errors (1) into I (0) for training the Z decoder.
        # Apply the following transformations to ensure we have only I and Z errors: 1 -> 0, 2 -> 1, 3 -> 1.
        error_patterns[error_patterns .== 1] .= 0
        error_patterns[error_patterns .== 2] .= 1
        error_patterns[error_patterns .== 3] .= 1

        # Save the generated error patterns to a file
        output_errors_file = "$(output_errors_dir)/test_error_patterns_Z_bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block).txt"
        writedlm(output_errors_file, error_patterns', ' ')
        push!(output_error_files, output_errors_file)

        # Print the command to run the test with this error patterns file
        println("julia --project=./../ neural_bp_experiments.jl " *
                "--codename hamming " *
                "--n_hidden_layers 50 " *
                "--n_epochs 5 " *
                "--batch_size 32 " *
                "--retrain false " *
                "--train regenerative_training_data.txt " *
                "--test test_error_patterns_Z_bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block).txt " *
                "--correlation_strength 0.5")
        println("echo \"Testing done for bs_$(block_size)_bp_$(block_prob)_epb_$(error_prob_within_block)\" >&2")
    end

    return output_error_files
end
function predict_recovery(bpnn::NeuralBP, syndrome::BitVector)::BitMatrix
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    
    Note that the LLRs are provided in the form
    [
        μ_1
        μ_2
        .
        .
        .
        μ_K
    ]
    where each μ_k is of shape (n bits, n_samples) and K is the number of layers.
    Each μ_k contains the LLRs after k layers of forward propagation.
    The predicted recoveries are computed by taking the sign of the LLRs at each layer.
    
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndrome::BitVector`: A vector representing a syndrome corresponding to an error pattern.
    
    Returns:
    - `predicted_recoveries::BitMatrix`: A matrix where each column represents the predicted recovery (error pattern) corresponding to the syndrome.
    """
    (output_LLRs, _, _) = bpnn(bpnn.initial_llrs, syndrome)
    # If LLR is negative, predict an error (1), else predict no error (0).
    predicted_recoveries = convert.(Bool, (output_LLRs' .< 0))
    return predicted_recoveries
end

function check_bp_solutions(parity_check_matrix_dual::Matrix{Int}, error::BitVector, recoveries::BitMatrix)::Bool
    """
    Check if the error + recovery is a stabilizer by verifying if it commutes with the elements of the dual code.
     In other words, we want to check if H^⟂ * (e + r) = 0, where H^⟂ is the parity-check matrix of the dual code, e is the error pattern, and r is the recovery pattern.
     If this condition is satisfied, then the total pattern (error + recovery) is a stabilizer and thus the recovery is correct.
    
    Arguments:
    - `parity_check_matrix_dual::Matrix{Int}`: The parity-check matrix of the dual code.
    - `error::BitVector`: A vector representing the error pattern.
    - `recoveries::BitMatrix`: A matrix where each column represents a choice for the recovery obtained from the LLRs at different layers of the NeuralBP model.

    Returns:
    - `is_correct::Bool`: A boolean indicating whether any of the recoveries correct the corresponding error.
    """
    n_layers = size(recoveries, 2)
    for l in 1:n_layers
        recovery = recoveries[:, l]
        total_pattern = xor.(error, recovery)
        syndrome = mod.(parity_check_matrix_dual * total_pattern, 2)
        if all(syndrome .== 0)
            return true
        end
    end
    return false
end

function predict_and_validate(bpnn::NeuralBP, parity_check_matrix_dual::Matrix{Int}, syndromes::BitMatrix, expected_recoveries::BitMatrix)::BitVector
    """
    Predict the recovery for the given syndrome and check if it is correct.
    
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `parity_check_matrix_dual::Matrix{Int}`: The parity-check matrix of the dual code.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.

    Returns:
    - `failures::BitVector`: A vector indicating whether each predicted recovery fails to correct the corresponding error.
    """
    n_samples = size(syndromes, 2)
    failures = falses(n_samples)
    n_failures::Int = 0
    # Use a progress bar to track the prediction and validation process.
    predprogress = Progress(n_samples, desc="Predicting and validating")
    @Threads.threads for i in 1:n_samples
        syndrome = syndromes[:, i]
        expected_recovery = expected_recoveries[:, i]
        predicted_recoveries = predict_recovery(bpnn, syndrome)
        is_correct = check_bp_solutions(parity_check_matrix_dual, expected_recovery, predicted_recoveries)
        failures[i] = !is_correct
        n_failures += failures[i]
        next!(predprogress, showvalues = [(:Sample, i), (:Failures, n_failures)])
    end
    return failures
end
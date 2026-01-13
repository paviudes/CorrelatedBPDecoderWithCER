function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix)::BitMatrix
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    
    Returns:
    - `predicted_recoveries::BitMatrix`: A matrix where each column represents the predicted recovery (error pattern) corresponding to the syndrome.
    """
    batch_size = size(syndromes, 2)
    initial_llrs_batch = repeat(bpnn.base.initial_llrs, 1, batch_size)
    predicted_recoveries_LLRs = bpnn(initial_llrs_batch, syndromes)
    # print_neuralbp_summary(bpnn; final_llrs=predicted_recoveries_LLRs)
    predicted_recoveries = convert.(Bool, (predicted_recoveries_LLRs .< 0))
    return predicted_recoveries
end


function check_bp_solutions(parity_check_matrix_dual::Matrix{Int}, errors::BitMatrix, recoveries::BitMatrix)::BitVector
    """
    Check if the provided recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `recoveries::BitMatrix`: A matrix where each row represents the recovery pattern corresponding to the error.

    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.
    """
    n_samples = size(errors, 2)
    is_correct = BitVector(undef, n_samples)

    for i in 1:n_samples
        total_pattern = xor.(errors[1:end, i], recoveries[1:end, i])
        syndrome = mod.(parity_check_matrix_dual * total_pattern, 2)
        is_correct[i] = all(syndrome .== 0)

        # For debugging purposes: if a weight-0 error is not corrected, print details.
        if (sum(errors[1:end, i]) == 0) && (!is_correct[i])
            println("Debug Info: Weight-0 error not corrected for sample $i.")
            println("Error pattern: ", errors[1:end, i])
            println("Recovery pattern: ", recoveries[1:end, i])
            println("Total pattern (error + recovery): ", total_pattern)
            println("Syndrome: ", syndrome)
        end
    end
    return is_correct
end
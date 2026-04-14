function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix)::Array{Bool, 3}
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
    predicted_recoveries_LLRs = forward_pass_with_weights(
        bpnn.weights_c2v_v2c,
        bpnn.weights_llrs,
        bpnn.weights_c2v_readout,
        bpnn.base,
        initial_llrs_batch,
        syndromes
    )
    # print_neuralbp_summary(bpnn; final_llrs=predicted_recoveries_LLRs)
    predicted_recoveries = convert.(Bool, (predicted_recoveries_LLRs .< 0))
    return predicted_recoveries
end

function check_bp_solutions(parity_check_matrix_dual::Matrix{Int}, errors::BitMatrix, proposed_recoveries::Array{Bool, 3})::BitVector
    """
    Check if the provided recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `proposed_recoveries::Array{3, Bool}`: A 3-dimensional array where each slice along the first dimension represents the recovery pattern corresponding to the error, derived from the LLRs of a particular layer.

    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.

    We check if the recovery at any of the intermediate layers correctly fixes the error, since the final layer's LLRs might not always be the best predictor.
    """
    n_samples = size(errors, 2)
    is_correct = falses(n_samples)
    total_fails = 0
    
    # Make a progress bar
    # progress = Progress(n_samples, desc="Checking BP solutions: ")
    for i in 1:n_samples
        error_pattern = errors[:, i]
        # Check if any of the proposed recoveries from the layers correctly fixes the error
        #=
        for layer in 1:size(proposed_recoveries, 3)
            proposed_recovery = proposed_recoveries[:, i, layer]
            # Check if the proposed recovery correctly fixes the error
            if mod.(parity_check_matrix_dual * (error_pattern .⊻ proposed_recovery), 2) == zeros(Bool, size(parity_check_matrix_dual, 1))
                is_correct[i] = true
                break
            end
        end
        =#
        # Do: H * (error ⊻ recovery) mod 2 == 0, and check if any column has all zeros, i.e. sums to zero.
        if any(sum(mod.(parity_check_matrix_dual * (error_pattern .⊻ proposed_recoveries[:, i, :]), 2), dims=1) .== 0)
            is_correct[i] = true
        else
            total_fails += 1
        end
        # next!(progress, showvalues = [(:Fails, total_fails)])
    end
    return is_correct
end
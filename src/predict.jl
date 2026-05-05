function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix; batch_size::Int = 1024)::Array{Bool, 3}
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    The samples are processed in batches of `batch_size` to keep GPU memory
    manageable; per-batch posterior tensors are hard-thresholded to Bool
    immediately and written into the output, so only one batch's worth of
    Float32 LLRs lives on the GPU at a time.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `batch_size::Int=1024`: How many samples to push through `forward_pass_gpu` at once. Smaller = less peak memory, more launch overhead.

    Returns:
    - `predicted_recoveries::Array{Bool, 3}`: shape (n_bits × n_samples × n_layers).
    """
    n_total  = size(syndromes, 2)
    n_bits   = bpnn.base.code_n_bits
    n_layers = bpnn.base.n_layers

    predicted_recoveries = falses(n_bits, n_total, n_layers)

    for start in 1:batch_size:n_total
        stop = min(start + batch_size - 1, n_total)
        chunk_synd = syndromes[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)
        chunk_post = forward_pass_gpu(bpnn, chunk_llrs, chunk_synd)
        @views predicted_recoveries[:, start:stop, :] .= (chunk_post .< 0)
    end

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
        if any(sum(mod.(parity_check_matrix_dual * (error_pattern .⊻ proposed_recoveries[:, i, :]), 2), dims=1) .== 0)
            is_correct[i] = true
        else
            total_fails += 1
        end
        # next!(progress, showvalues = [(:Fails, total_fails)])
    end
    return is_correct
end

function predict_and_check_neuralbp(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    errors::BitMatrix;
    batch_size::Int = 1024,
)::BitVector
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    Then check if the predicted recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `batch_size::Int=1024`: How many samples to push through `forward_pass_gpu` at once. Smaller = less peak memory, more launch overhead.
    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.
    
    We check if the recovery at any of the intermediate layers correctly fixes the error, since the final layer's LLRs might not always be the best predictor.
    """
    n_samples = size(syndromes, 2)
    
    H_dual = convert.(Int, bpnn.base.parity_check_matrix)

    # When the GPU option is disabled, we can process all samples at once without batching.
    # This is when `USE_GPU` is set to `false` in `CorrelatedBPDecoderWithCER.jl`.
    # In this case, we will use `forward_pass_with_weights` instead of `forward_pass_gpu`.
    if USE_GPU == false
        posterior_llrs = forward_pass_with_weights(bpnn, repeat(bpnn.base.initial_llrs, 1, n_samples), syndromes)
        proposed_recoveries = posterior_llrs .< 0
        is_correct = check_bp_solutions(H_dual, errors, proposed_recoveries)
        return is_correct
    end

    is_correct = falses(n_samples)
    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)

        # Determine the syndromes, errors, and initial LLRs for the current batch.
        chunk_syndromes  = syndromes[:, start:stop]
        chunk_errors  = errors[:, start:stop]
        chunk_llrs  = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        # Predict the recoveries for the chunk of syndromes using the trained NeuralBP model.
        chunk_posterior_llrs  = forward_pass_gpu(bpnn, chunk_llrs, chunk_syndromes)

        # Hard threshold the posterior LLRs to get proposed recoveries, and check if they correctly fix the errors.
        chunk_recoveries  = Array(chunk_posterior_llrs .< 0) # shape (n_bits, batch_size, n_layers)

        # Check if the proposed recoveries from any of the layers correctly fix the error
        @views is_correct[start:stop] .= check_bp_solutions(H_dual, chunk_errors, chunk_recoveries)
    end

    return is_correct
end
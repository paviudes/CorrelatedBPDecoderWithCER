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

function check_bp_solutions(parity_check_matrix::Matrix{Int}, logicals::Matrix{Int}, errors::BitMatrix, proposed_recoveries::Array{Bool, 3})::BitVector
    """
    Score each sample the way a real decoder is scored, in two steps:

      1. COMMIT (by syndrome — decode-time information). Pick the FIRST layer
         whose recovery clears the syndrome, i.e. the residual `e ⊕ r` commutes
         with the stabilizers: `parity_check_matrix * (e ⊕ r) = 0`. This mimics
         standard BP, which stops at the first iteration whose residual has zero
         syndrome. Only the syndrome (known at decode time) picks the layer.

      2. SCORE (by logicals). The committed residual is a SUCCESS iff it is also
         logically trivial — it commutes with the logical operators:
         `logicals * (e ⊕ r) = 0`. The committed layer already has zero syndrome,
         so only the (few) logical rows are checked here.

    A sample with NO syndrome-clearing layer is a decode failure (there is no OSD
    backstop). We commit to ONE layer by syndrome and score only that layer — NOT
    "any layer that happens to be logically correct", which would hand the decoder
    oracle knowledge of the true error and massively over-count success on small
    codes (few logical cosets).

    Arguments:
    - `parity_check_matrix::Matrix{Int}`: stabilizer checks H — selects the layer.
    - `logicals::Matrix{Int}`: logical operators L — scores the committed layer.
    - `errors::BitMatrix`: one true error pattern per column.
    - `proposed_recoveries::Array{Bool, 3}`: `(n_bits, n_samples, n_layers)` hard
      decisions, one recovery per layer.

    Returns:
    - `is_correct::BitVector`: true for each correctly decoded sample.
    """
    n_samples = size(errors, 2)
    is_correct = falses(n_samples)
    for i in 1:n_samples
        # Residual e ⊕ r for every layer at once: (n_bits, n_layers).
        residuals = errors[:, i] .⊻ proposed_recoveries[:, i, :]
        # Per-layer syndrome weight; zero ⇔ that layer's residual clears the syndrome.
        layer_syndrome_weight = vec(sum(mod.(parity_check_matrix * residuals, 2), dims = 1))
        committed_layer = findfirst(==(0), layer_syndrome_weight)
        if committed_layer === nothing
            continue  # no layer produced a syndrome-valid correction -> failure
        end
        # Score ONLY the committed layer: success iff it is also logically trivial.
        committed_residual = residuals[:, committed_layer]
        logical_syndrome = mod.(logicals * committed_residual, 2)
        if all(logical_syndrome .== 0)
            is_correct[i] = true
        end
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

    A sample is decoded correctly when the FIRST layer whose recovery clears the
    syndrome (the BP stopping condition) also lands in the correct logical coset;
    see `check_bp_solutions`.
    """
    n_samples = size(syndromes, 2)

    # Stabilizer checks H (used to commit to a layer by syndrome) and the logical
    # operators L (used to score that committed layer). `parity_check_matrix_dual`
    # is `vcat(H, logicals)`, so the logical rows are its tail past the H rows.
    parity_check_matrix = convert.(Int, bpnn.base.parity_check_matrix)
    n_checks = size(parity_check_matrix, 1)
    logicals = convert.(Int, bpnn.base.parity_check_matrix_dual[n_checks + 1:end, :])

    # When the GPU is not engaged, process all samples at once without batching.
    # `gpu_active()` is false when USE_GPU=0 at runtime OR no GPU backend is
    # compiled in for this platform — in which case we use
    # `forward_pass_with_weights` instead of `forward_pass_gpu`.
    if !gpu_active()
        posterior_llrs = forward_pass_with_weights(bpnn, repeat(bpnn.base.initial_llrs, 1, n_samples), syndromes)
        proposed_recoveries = Array(posterior_llrs .< 0) # shape (n_bits, n_samples, n_layers)
        is_correct = check_bp_solutions(parity_check_matrix, logicals, errors, proposed_recoveries)
        return is_correct
    end

    println("Using GPU for predictions with batch size = $batch_size. Total samples = $n_samples.")

    is_correct = falses(n_samples)
    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)

        # Determine the syndromes, errors, and initial LLRs for the current batch.
        chunk_syndromes  = syndromes[:, start:stop]
        chunk_errors  = errors[:, start:stop]
        chunk_llrs  = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        # Predict the recoveries for the chunk of syndromes using the trained NeuralBP model.
        chunk_posterior_llrs = forward_pass_gpu(bpnn, chunk_llrs, chunk_syndromes)

        # Hard threshold the posterior LLRs to get proposed recoveries, and check if they correctly fix the errors.
        chunk_recoveries  = Array(chunk_posterior_llrs .< 0) # shape (n_bits, batch_size, n_layers)

        # Commit to the first syndrome-clearing layer, then score its logical coset.
        @views is_correct[start:stop] .= check_bp_solutions(parity_check_matrix, logicals, chunk_errors, chunk_recoveries)
    end

    return is_correct
end

function neuralbp_test_predictions(bpnn::NeuralBP, test_errors_file::String)::BitVector
    """
    Predict the recoveries for the given test syndromes using the trained Neural BP model.
    Test these predictions to see if they match the expected recoveries.
    """
    test_errors = convert.(Bool, readdlm(test_errors_file, Int))
    test_syndromes = convert.(Bool, mod.(bpnn.base.parity_check_matrix * test_errors, 2))
    # start = time()
    is_correct = predict_and_check_neuralbp(bpnn, test_syndromes, test_errors; batch_size=32768)
    # runtime = time() - start
    # println("[", round(runtime, digits=2), "s] elapsed. Predicted recoveries computed and verified.")
    return is_correct
end
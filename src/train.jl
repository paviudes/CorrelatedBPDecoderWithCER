function train_neuralbp!(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    optimizer = OptimiserChain(
        ClipGrad(5.0),    # clip gradient norm at 5.0
        WeightDecay(1e-6), # optional small L2 regularizer
        ADAM(1e-4)        # smaller lr + larger eps for numerical stability
    ),
    n_epochs::Int=10,
    batch_size::Int=32
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    The provided syndromes should be computed from the errors, which are provided as expected recoveries.
    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model to be trained.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    - `initial_llrs::Matrix{Float32}`: Initial LLRs for the bits, to be used as input to the network (default: 1.0 since it is the LLR corresponding to the probability of the bit being 0 equal to 0.9).
    - `optimizer`: The optimizer to use for training (default: ADAM).
    - `n_epochs::Int`: The number of epochs. Each epoch goes through the entire dataset once (default: 10).
    - `batch_size::Int`: The size of each batch for training (default: 32).
    - `loss_function`: The loss function to use for training (default: binary cross-entropy).
    """
    # Batch the training data: syndromes and expected recoveries, into batches.
    # Each batch will be passed through the network as a Matrix.
    # Each batch will be a tuple of (syndromes_batch, expected_recoveries_batch)
    
    # Split an array of 1:n_samples into batches of size batch_size
    n_samples = size(syndromes, 2)
    samples_grouped_by_batch = [
        (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
        for i in 1:ceil(Int, n_samples / batch_size)
    ]
    # println("Samples grouped by batch: ", samples_grouped_by_batch)
    # Create the training dataset as a vector of tuples
    training_dataset = [
        (
            syndromes[1:end, batch_sample_indices],
            expected_recoveries[1:end, batch_sample_indices],
            repeat(bpnn.base.initial_llrs, 1, length(batch_sample_indices))
        )
        for batch_sample_indices in samples_grouped_by_batch
    ]

    # println("Starting training for $n_epochs epochs with batch size $batch_size... with training dataset of shape ", training_dataset)

    # Set the trainable parameters
    opt_state = Flux.setup(optimizer, bpnn)  # create optimiser state (tune lr as needed)

    for epoch in 1:n_epochs
        for (syndromes_train_batch, expected_recoveries_batch, llrs_batch) in training_dataset
            # compute loss and gradients in one shot
            loss, grads = Flux.withgradient(bpnn) do model
                posterior_llrs = model(llrs_batch, syndromes_train_batch)
                # compute_loss_error_from_llrs(posterior_llrs, expected_recoveries_batch, bpnn.parity_check_matrix_dual) #TODO: turn on the additional loss for correlations later.
                compute_loss_including_correlations(
                    posterior_llrs,
                    expected_recoveries_batch,
                    bpnn.base.parity_check_matrix_dual,
                    bpnn.base.connectivity,
                    bpnn.base.correlation_strength,
                    bpnn.base.is_correlated
                )
            end
            # apply update. grads[1] contains gradients for the model
            Flux.update!(opt_state, bpnn, grads[1])
            # println("Epoch $epoch, Batch Loss: $loss")
        end
        # println("Epoch $epoch completed.")
    end
end

function generate_training_data(parity_check_matrix::Matrix{Int}, n_samples::Int, error_probability::Float64)::Tuple{BitMatrix, BitMatrix}
    """
    Generate training data for the NeuralBP model.
    Each sample consists of a random error pattern generated according to the specified error probability.
    The corresponding syndrome is computed using the provided parity-check matrix.
    The error patterns serve as the expected recoveries.

    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `n_samples::Int`: The number of samples to generate.
    - `error_probability::Float64`: The probability of an error occurring on each bit.
    
    Returns:
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    """
    (n_checks, n_bits) = size(parity_check_matrix)
    syndromes = zeros(Bool, n_checks, n_samples)
    expected_recoveries = zeros(Bool, n_bits, n_samples)

    for i in 1:n_samples
        # Generate a random error pattern
        error_pattern = rand(Bool, n_bits) .< error_probability
        expected_recoveries[:, i] = error_pattern

        # Compute the syndrome
        syndrome = mod.(parity_check_matrix * error_pattern, 2)
        syndromes[:, i] = syndrome
    end

    return syndromes, expected_recoveries
end
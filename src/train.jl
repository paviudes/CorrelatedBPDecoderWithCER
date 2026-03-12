function train_neuralbp!(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    optimizer = OptimiserChain(
        ClipGrad(5.0),    # clip gradient norm at 5.0
        WeightDecay(1e-6), # optional small L2 regularizer
        ADAM(1e-2)        # smaller lr + larger eps for numerical stability
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

    # Create progress bar for epochs
    epoch_progress = Progress(n_epochs, desc="Training Epochs: ")
    
    for epoch in 1:n_epochs
        # Create progress bar for batches within each epoch
        batch_progress = Progress(length(training_dataset), desc="Epoch $epoch Batches: ")
        
        # Group batches for parallel processing
        parallel_batch_size = min(4, Threads.nthreads())
        batch_groups = [
            training_dataset[i:min(i+parallel_batch_size-1, end)]
            for i in 1:parallel_batch_size:length(training_dataset)
        ]

        for batch_group in batch_groups
            n_batches_in_group = length(batch_group)
            losses = Vector{Float32}(undef, n_batches_in_group)
            grads_array = Vector{Any}(undef, n_batches_in_group)
            
            # Process batches in parallel
            Threads.@threads for i in 1:n_batches_in_group
                syndromes_batch, expected_recoveries_batch, llrs_batch = batch_group[i]
                n_samples_batch = size(syndromes_batch, 2)
                
                loss, grads = Flux.withgradient(bpnn) do model
                    posterior_llrs = model(llrs_batch, syndromes_batch)
                    compute_layered_loss_including_correlations(
                        posterior_llrs,
                        expected_recoveries_batch,
                        n_samples_batch,
                        bpnn.base.n_layers,
                        bpnn.base.parity_check_matrix_dual,
                        bpnn.base.connectivity,
                        bpnn.base.correlation_strength,
                        bpnn.base.is_correlated #TODO: set explicitly to `false` for excluding correlations, and set to `bpnn.base.is_correlated` for including correlations.
                    )
                end
                
                losses[i] = loss
                grads_array[i] = grads[1]
            end

            # Average losses and gradients
            avg_loss = sum(losses)# / n_batches_in_group
            
            # Average gradients field-wise
            avg_grads = grads_array[1]  # Start with first gradient structure
            for i in 2:n_batches_in_group
                # Add each subsequent gradient to the running sum
                avg_grads = Flux.Optimisers.add!(avg_grads, grads_array[i])
            end
            # # Scale by number of batches to get average
            # avg_grads = Flux.Optimisers.scale!(avg_grads, 1.0 / n_batches_in_group)
            
            # Single parameter update
            Flux.update!(opt_state, bpnn, avg_grads)

            # Update batch progress bar
            ProgressMeter.next!(batch_progress; showvalues = [(:loss, avg_loss)])
        end
        
        # Update epoch progress bar
        ProgressMeter.next!(epoch_progress)
    end
end

function get_grads_for_batch!(
    bpnn::NeuralBP,
    syndromes_batch::BitMatrix,
    expected_recoveries_batch::BitMatrix,
    llrs_batch::Matrix{Float32};
    is_correlated::Bool=false
)::Tuple{Vector{Float32}, Float32}
    """
    Compute the gradients of the loss with respect to the learnable parameters of the NeuralBP model for a given batch of data.
    
    We will do the following for each sample in the minibatch:
    1. Perform a forward pass through the NeuralBP model to compute the posterior LLRs for the given syndromes and initial LLRs.
    2. Compute the Loss using the computed posterior LLRs and the expected recoveries.
    3. Compute the gradients of the Loss with respect to the learnable parameters of the NeuralBP model using backpropagation.

    We will compute all the gradients for the samples in the batch in parallel, and then average the gradients across the batch to get the average gradient for the batch.

    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model for which to compute the gradients.
    - `syndromes_batch::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern for the batch.
    - `expected_recoveries_batch::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome for the batch.
    - `llrs_batch::Matrix{Float32}`: A matrix of initial LLRs for the batch, where each column corresponds to the initial LLRs for a sample in the batch.
    
    Returns:
    - `grads_for_batch::Vector{Float32}`: A vector containing the average gradients of the loss with respect to the learnable parameters of the NeuralBP model for the batch.
    """
    n_samples_batch = size(syndromes_batch, 2)
    grads_for_batch = Vector{Vector{Float32}}(undef, n_samples_batch)
    losses = Vector{Float32}(undef, n_samples_batch)
    Threads.@threads for i in 1:n_samples_batch
        syndrome = syndromes_batch[:, i]
        expected_recovery = expected_recoveries_batch[:, i]
        initial_llrs = llrs_batch[:, i]
        (grads_for_batch[i], losses[i]) = get_grads_for_sample(bpnn, initial_llrs, syndrome, expected_recovery; is_correlated=is_correlated)
    end
    # Average the gradients across the batch
    avg_grads_for_batch = sum(vcat(grads_for_batch...), dims=1) ./ n_samples_batch
    avg_loss_for_batch = sum(losses) / n_samples_batch
    return (avg_grads_for_batch, avg_loss_for_batch)
end

function get_grads_for_sample(
    bpnn::NeuralBP,
    initial_llrs::Vector{Float32},
    syndrome::BitVector,
    expected_recovery::BitVector;
    is_correlated::Bool=false
)::Tuple{Vector{Float32}, Float32}
    """
    Compute the gradients of the loss with respect to the learnable parameters of the NeuralBP model for a single sample.
    
    We will do the following:
    1. Perform a forward pass through the NeuralBP model to compute the posterior LLRs for the given syndrome and initial LLRs.
    2. Compute the Loss using the computed posterior LLRs and the expected recovery.
    3. Compute the gradients of the Loss with respect to the learnable parameters of the NeuralBP model using backpropagation.

    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model for which to compute the gradients.
    - `initial_llrs::Vector{Float32}`: A vector of initial LLRs for the sample, where each element corresponds to the initial LLR for a bit.
    - `syndrome::BitVector`: A vector representing the syndrome corresponding to an error pattern for the sample.
    - `expected_recovery::BitVector`: A vector representing the expected recovery (error pattern) corresponding to the syndrome for the sample.
    
    Returns:
    - `grads_for_sample::Vector{Float32}`: A vector containing the gradients of the loss with respect to the learnable parameters of the NeuralBP model for the sample.
    - `loss::Float32`: The loss value for the sample.
    """

    # start_time = time()

    #=============================================================================
                                    STEP 1: Forward pass
    =============================================================================#

    (posterior_llrs, intermediate_c2v_messages, intermediate_v2c_messages) = bpnn(initial_llrs, syndrome)

    # elapsed_time_forward_pass = time() - start_time
    # println("[", round(elapsed_time_forward_pass, digits=3), " seconds] Completed forward pass.")

    #=============================================================================
                                    STEP 2: Computing Loss
    =============================================================================#
    loss = compute_loss_from_llrs(
        posterior_llrs,
        expected_recovery,
        convert.(Bool, bpnn.base.parity_check_matrix_dual),
        bpnn.loss_weights;
        is_correlated = is_correlated,
        connectivity_edges=bpnn.base.connectivity_edges,
        correlation_strengths=bpnn.base.correlation_strengths
    )

    # elapsed_time_loss = time() - start_time
    # println("[", round(elapsed_time_loss, digits=3), " seconds] Completed loss computation.")

    #=============================================================================
                                    STEP 3: Backward pass
    =============================================================================#
    # Compute the gradients of the Loss with respect to the learnable parameters of the NeuralBP model using backpropagation.
    grads = nachmani_loss_jacobian(
        bpnn,
        expected_recovery,
        intermediate_c2v_messages,
        intermediate_v2c_messages,
        posterior_llrs;
        is_correlated=is_correlated
    )

    # elapsed_time_backward_pass = time() - start_time
    # println("[", round(elapsed_time_backward_pass, digits=3), " seconds] Completed backward pass.")

    #=============================================================================
                                    STEP 4: Convert gradients to a 1D vector
    =============================================================================#
    
    # Pack the gradients into a 1D vector form for the optimizer.
    grad_vector = pack_grads(grads, bpnn)

    return (grad_vector, loss)
end

function train_step!(
    bpnn::NeuralBP,
    initial_llrs::Vector{Float32},
    syndrome::BitVector,
    expected_recovery::BitVector,
    opt_state;
    is_correlated::Bool=false
)
    """
    Implement one round of the training.
    
    Training entails the following steps:
    1. Forward pass: Compute the posterior LLRs for the given syndromes and initial LLRs using the NeuralBP model.
    2. Compute the Loss: Use the computed posterior LLRs and the expected recoveries to compute the Loss using the specified loss function.
    3. Backward pass: Compute the gradients of the Loss with respect to the learnable parameters of the NeuralBP model using backpropagation.
    4. Update parameters: Use the computed gradients to update the learnable parameters of the NeuralBP model using the specified optimizer.

    In step (4) we will use an Optimizer, which takes the variables in a 1D vector form.
    So, we will need to unpack all the learnable parameters of the NeuralBP model into a 1D vector, and also unpack the corresponding gradients into a 1D vector, and then pass them to the optimizer for the parameter update.
    Steps 1-3 will be performed using the `get_grads_for_sample` function.

    After the optimizer updates the parameters in the 1D vector form, we will need to pack them back into the original structure of the learnable parameters of the NeuralBP model.
    The packing is necessary to do forward passes through the NeuralBP model in the next round of training, since the forward pass expects the learnable parameters to be in their original structure.

    We need to profile this function and insert an elapsed time print statement for each of the steps to identify bottlenecks and optimize the training process.
    
    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model to be trained.
    - `syndrome::BitVector`: A vector representing the syndrome corresponding to an error pattern.
    - `expected_recovery::BitVector`: A vector representing the expected recovery (error pattern) corresponding to the syndrome.
    - `optimizer`: The optimizer to use for training (default: ADAM).
    - `is_correlated::Bool`: Whether to consider correlations in the loss computation (default: false).
    """
    
    # start_time = time()

    # Compute the gradients of the Loss with respect to the learnable parameters of the NeuralBP model using backpropagation, and convert them into a 1D vector form for the optimizer.
    (grad_vector, loss) = get_grads_for_sample(bpnn, initial_llrs, syndrome, expected_recovery; is_correlated=is_correlated)
    # Pack the current parameters into a 1D vector form for the optimizer.
    param_vector = pack_params(bpnn)
    
    # Perform the parameter update using the optimizer.
    (opt_state, param_vector) = Optimisers.update(opt_state, param_vector, grad_vector)
    
    # println("Type of param_vector after update: ", typeof(param_vector))
    # println("Updated parameter vector: ", param_vector)

    # Unpack the updated parameters back into the original structure of the learnable parameters of the NeuralBP model.
    unpack_params!(bpnn, param_vector)

    # elapsed_time_update = time() - start_time
    # println("[", round(elapsed_time_update, digits=3), " seconds] Completed parameter update.")
    
    # Debugging: Print the updated parameters after unpacking
    # println("Updated parameters after unpacking: ")
    # println("Weights C2V V2C: ", bpnn.weights_c2v_v2c)
    # println("Weights LLRs: ", bpnn.weights_llrs)
    # println("Weights C2V Readout: ", bpnn.weights_c2v_readout)

    return loss
end

function train!(
    bpnn::NeuralBP, 
    syndromes::BitMatrix, 
    expected_recoveries::BitMatrix; 
    n_epochs::Int=10, 
    optimizer = OptimiserChain(
        ClipGrad(5.0),    # clip gradient norm at 5.0
        WeightDecay(1e-6), # optional small L2 regularizer
        Adam(1e-2)        # smaller lr + larger eps for numerical stability
    ), 
    is_correlated::Bool=false
)
    """
    Train the NeuralBP model for multiple epochs using the provided syndromes and expected recoveries.
    This function iteratively calls `train_step!` for the specified number of epochs, and can optionally shuffle the training data at the beginning of each epoch for better training performance.
    Arguments:
        - `bpnn::NeuralBP`: The NeuralBP model to be trained.
        - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
        - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
        - `n_epochs::Int`: The number of epochs to train for (default: 10).
        - `optimizer`: The optimizer to use for training (default: ADAM).
        - `is_correlated::Bool`: Whether to consider correlations in the loss computation (default: false).
    """
    n_samples = size(syndromes, 2)
    initial_llrs = bpnn.initial_llrs

    # Set the optimizer state
    initial_param_vector = pack_params(bpnn)  # pack initial parameters into a 1D vector
    opt_state = Optimisers.setup(optimizer, initial_param_vector)  # create optimiser state (tune lr as needed)

    # Create progress bar for epochs
    epoch_progress = Progress(n_epochs, desc="Training Epochs: ")
    
    for epoch in 1:n_epochs
        # println("Starting epoch $epoch...")
        
        # Shuffle the samples
        shuffled_indices = randperm(n_samples)
        shuffled_syndromes = syndromes[:, shuffled_indices]
        shuffled_expected_recoveries = expected_recoveries[:, shuffled_indices]
        
        # Create progress bar for sample
        sample_progress = Progress(n_samples, desc="Epoch $epoch: ")
        
        for i in 1:n_samples
            syndrome = shuffled_syndromes[:, i]
            expected_recovery = shuffled_expected_recoveries[:, i]
            loss = train_step!(bpnn, initial_llrs, syndrome, expected_recovery, opt_state; is_correlated=is_correlated)
            # println("Epoch $epoch, Sample $i/$n_samples, Loss: ", loss)
            # Update batch progress bar
            ProgressMeter.next!(sample_progress; showvalues = [(:loss, loss)])
        end

        # Update epoch progress bar
        ProgressMeter.next!(epoch_progress)
    end
end

function train_minibatch!(
    bpnn::NeuralBP, 
    syndromes::BitMatrix, 
    expected_recoveries::BitMatrix; 
    n_epochs::Int=10, 
    optimizer = OptimiserChain(
        ClipGrad(5.0),     # clip gradient norm at 5.0
        WeightDecay(1e-6), # optional small L2 regularizer
        Adam(1e-2)         # smaller lr + larger eps for numerical stability
    ),
    batch_size::Int=32,
    is_correlated::Bool=false
)
    """
    Train samples in mini-batches for multiple epochs using the provided syndromes and expected recoveries.
    This function iteratively calls `get_grads_for_batch!` for each mini-batch of samples to compute the average gradients, and then updates the model parameters using the optimizer.

    Arguments:
        - `bpnn::NeuralBP`: The NeuralBP model to be trained.
        - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
        - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
        - `n_epochs::Int`: The number of epochs to train for (default: 10).
        - `optimizer`: The optimizer to use for training (default: ADAM).
        - `is_correlated::Bool`: Whether to consider correlations in the loss computation (default: false).
    """
    
    n_samples = size(syndromes, 2)

    n_batches = ceil(Int, n_samples / batch_size)
    # Split an array of 1:n_samples into batches of size batch_size
    samples_grouped_by_batch = [
        (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
        for i in 1:n_batches
    ]

    initial_llrs = bpnn.initial_llrs

    # Set the optimizer state
    initial_param_vector = pack_params(bpnn)  # pack initial parameters into a 1D vector
    
    opt_state = Optimisers.setup(optimizer, initial_param_vector)  # create optimiser state (tune lr as needed)

    # Create progress bar for epochs
    epoch_progress = Progress(n_epochs, desc="Training Epochs: ")
    
    for epoch in 1:n_epochs
        # Shuffle the samples
        shuffled_indices = randperm(n_samples)
        shuffled_syndromes = syndromes[:, shuffled_indices]
        shuffled_expected_recoveries = expected_recoveries[:, shuffled_indices]
        
        # Create progress bar for sample
        sample_progress = Progress(n_batches, desc="Epoch $epoch: ")
        
        for batch_samples in samples_grouped_by_batch
            # Compute the average gradients for the batch using `get_grads_for_batch!`,
            # which will perform forward and backward passes for all samples in the batch,
            # and return the average gradients across the batch.
            batch_syndromes = shuffled_syndromes[:, batch_samples]
            batch_expected_recoveries = shuffled_expected_recoveries[:, batch_samples]
            (average_gradients, average_loss) = get_grads_for_batch!(
                bpnn, 
                batch_syndromes, 
                batch_expected_recoveries, 
                repeat(initial_llrs, 1, length(batch_samples));
                is_correlated=is_correlated
            )
            
            # Perform the parameter update using the optimizer.
            param_vector = pack_params(bpnn)  # pack current parameters into a 1D vector
            (opt_state, param_vector) = Optimisers.update(opt_state, param_vector, average_gradients)
            unpack_params!(bpnn, param_vector)  # unpack the updated parameters back into the model

            # println("Epoch $epoch, Batch $i/$n_batches, Loss: ", loss)
            # Update batch progress bar
            ProgressMeter.next!(sample_progress; showvalues = [(:loss, average_loss)])
        end
        
        # Update epoch progress bar
        ProgressMeter.next!(epoch_progress)
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
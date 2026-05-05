function get_loss_value(
    weights_c2v_v2c, # learanble weights for computing m^t_(v→c) from m^(t-1)_(c→v).
    weights_llrs, # learnable weights for m^t_(v→c) from the initial LLRs, and also for computing the posterior LLRs from m^t_(c→v).
    weights_c2v_readout, # learnable weights for computing the readout (posterior LLRs) from m^t_(c→v).
    correlation_importance, # importance of the correlation penalty in the Loss function, to be annealed during training.
    loss_layer_regularizer, # temperature for the smooth minimum approximation when combining losses from different layers, to be annealed during training.
    base, # constant parameters of the model, including the parity-check matrix, connectivity, correlation strengths, etc.
    llrs_batch, # batch of initial LLRs for the bits, to be used as input to the network
    syndromes_batch, # batch of syndromes, to be used as input to the network
    expected_recoveries # batch of expected recoveries (error patterns), to be used for computing the Loss function
)::Float32
    """
    Compute the value of the Loss function for a given batch of data and given values for the weights.
    This version is friendly to Enzyme.jl, which requires a functional approach where the model parameters are passed explicitly to the function computing the loss, rather than being accessed as fields of a struct.
    """
    # Forward pass through the network to get the posterior LLRs
    posterior_llrs = forward_pass_with_weights(
        weights_c2v_v2c,
        weights_llrs,
        weights_c2v_readout,
        base,
        llrs_batch,
        syndromes_batch
    )
    # Compute the total Loss including the correlation penalty
    total_loss = compute_loss_including_correlations(
        posterior_llrs,
        expected_recoveries,
        base.parity_check_matrix_dual,
        base.connectivity,
        base.correlation_strengths,
        base.is_correlated, # Change back to base.is_correlated for the actual implementation, set to `false` for testing without correlations for now.
        correlation_importance, # correlation_importance is passed as a 1-element array to be compatible with Enzyme's autodiff, so we take the first element here.
        loss_layer_regularizer # temperature for the smooth minimum approximation when combining losses from different layers, to be annealed during training.
    )
    return total_loss
end

#=
TODO: This function should be removed eventually since we are no more using Flux.jl.
TODO: We have this commented code to be able to port this optimizer over to the Enzyme.jl version of the training loop.
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

        for b in 1:length(training_dataset)
            # Create a random shuffle ordering of the syndromes and recoveries for each batch.
            shuffled_indices = randperm(size(training_dataset[b][1], 2))

            syndromes_train_batch = training_dataset[b][1][:, shuffled_indices]
            expected_recoveries_batch = training_dataset[b][2][:, shuffled_indices]
            llrs_batch = training_dataset[b][3][:, shuffled_indices]
                
            # compute loss and gradients in one shot
            loss, grads = Flux.withgradient(bpnn) do model
                posterior_llrs = forward_pass(bpnn, llrs_batch, syndromes_train_batch)
                # compute_loss_error_from_llrs(posterior_llrs, expected_recoveries_batch, bpnn.parity_check_matrix_dual) #TODO: turn on the additional loss for correlations later.
                compute_loss_including_correlations(
                    posterior_llrs,
                    expected_recoveries_batch,
                    bpnn.base.parity_check_matrix_dual,
                    bpnn.base.connectivity,
                    bpnn.base.correlation_strengths,
                    bpnn.base.is_correlated, #TODO: set explicitly to `false` for excluding correlations
                    bpnn.weights_loss_layers,
                    bpnn.correlation_importance,
                    smooth_temp
                )
            end
            # apply update. grads[1] contains gradients for the model
            Flux.update!(opt_state, bpnn, grads[1])
            
            # Update batch progress bar with current loss
            ProgressMeter.next!(batch_progress; showvalues = [(:loss, loss)])
        end
        
        # Update epoch progress bar
        ProgressMeter.next!(epoch_progress)
    end
end
=#

function clip_grad!(grad_vector::AbstractArray{Float32}; max_grad_norm::Float32=5f0)
    """
    Clip the gradient in-place to have a maximum L2 norm of `max_grad_norm`.
    """
    grad_norm = sqrt(sum(abs2, grad_vector))
    if grad_norm > max_grad_norm
        grad_vector .*= (max_grad_norm / grad_norm)
    end
    return nothing
end

function train_neuralbp_enzyme!(
    bpnn::NachmaniNeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    learning_rate::Float32 = 1f-4, # learning rate for simple SGD
    n_epochs::Int = 10,
    batch_size::Int = 32,
    max_grad_norm::Float32 = 5f0 # maximum gradient norm for clipping
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    We will use Enzyme.jl for automatic differentiation of the loss function with respect to the model parameters.
    
    The Loss function per layer for training can be expressed in two parts: L_layer = L_syn + α L_corr,
    where
    - L_syn is the part of the Loss that penalizes the model for not producing the correct syndrome, and logical errors.
    - L_corr is the part of the Loss that penalizes the model for not capturing the correlations between qubits.
    - α is a hyperparameter that controls the importance of the correlation penalty (given by `correlation_importance`)

    At the beginning of the training, the model chooses errors that largely violate the syndrome constraints, and here the training is largerly clueless about the error patterns,
    so we want to use the noise dynamics to guide the training by focusing on minimizing L_corr. In other words, we want to start with a large value of α.

    As the training progresses, the model starts to learn to satisfy the syndrome constraints, and here we want to focus more on minimizing L_syn to further improve the performance, so we want to anneal α down to a smaller value.

    Ideally we want BP to stop at any layer that results in the minimum Loss. To achieve this, we use a smooth minimum approximation to combine the Loss values from different layers, which allows the gradients to flow to all layers while still encouraging the model to focus on the layer with the lowest Loss.
    The temperature for the smooth minimum approximation is annealed during training to encourage sharper minima as training progresses.
    """
    base = bpnn.base

    # ---------------------------------
    # Create batches
    # ---------------------------------
    n_samples = size(syndromes, 2)

    samples_grouped_by_batch = [
        (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
        for i in 1:ceil(Int, n_samples / batch_size)
    ]

    training_dataset = [
        (
            syndromes[:, idx],
            expected_recoveries[:, idx],
            repeat(base.initial_llrs, 1, length(idx))
        )
        for idx in samples_grouped_by_batch
    ]
    

    # Annealing schedule parameters for the temperature of the smooth minimum approximation when combining losses from different layers.
    layer_temp_max::Float32 = 5f-0
    layer_temp_min::Float32 = 3f-1
    layer_temp_decay::Float32 = 0.9f0 # decay factor for the annealing schedule of the temperature for the smooth minimum approximation.

    # Annealing schedule parameters for the importance of the correlation penalty in the Loss function.
    correlation_importance_max::Float32 = 1f0
    correlation_importance_min::Float32 = 1f-3
    correlation_importance_decay::Float32 = 0.1f0 # decay factor for the annealing schedule of the correlation importance in the Loss function.

    #=
    # --------------------------
    # Debugging: log the individual losses for samples and epochs.
    n_samples_to_log = n_epochs * length(training_dataset)
    hyperparameters = DataFrame(
        epoch = zeros(Int, n_samples_to_log),
        sample = zeros(Int, n_samples_to_log),
        loss_layer_temp = zeros(Float32, n_samples_to_log),
        correlation_importance = zeros(Float32, n_samples_to_log),
        loss = zeros(Float32, n_samples_to_log),
        min_weight_c2v_v2c = zeros(Float32, n_samples_to_log),
        max_weight_c2v_v2c = zeros(Float32, n_samples_to_log),
        median_weight_c2v_v2c = zeros(Float32, n_samples_to_log),
        min_weight_llrs = zeros(Float32, n_samples_to_log),
        max_weight_llrs = zeros(Float32, n_samples_to_log),
        median_weight_llrs = zeros(Float32, n_samples_to_log),
        min_weight_c2v_readout = zeros(Float32, n_samples_to_log),
        max_weight_c2v_readout = zeros(Float32, n_samples_to_log),
        median_weight_c2v_readout = zeros(Float32, n_samples_to_log)
    )
    # --------------------------
    =#

    # -------------------------
    # Progress bars
    # -------------------------
    # epoch_progress = Progress(n_epochs, desc="Training Epochs: ")

    for epoch in 1:n_epochs
        # batch_progress = Progress(length(training_dataset), desc="Epoch $epoch Batches: ")
        
        bpnn.loss_layer_regularizer[1] = max(layer_temp_min, layer_temp_max * layer_temp_decay^(epoch - 1))
        bpnn.correlation_importance[1] = max(correlation_importance_min, correlation_importance_max * correlation_importance_decay^(epoch - 1))

        for b in 1:length(training_dataset)

            # -------------------------
            # Shuffle batch
            # -------------------------
            shuffled_indices = randperm(size(training_dataset[b][1], 2))

            syndromes_batch = training_dataset[b][1][:, shuffled_indices]
            expected_batch  = training_dataset[b][2][:, shuffled_indices]
            llrs_batch      = training_dataset[b][3][:, shuffled_indices]

            # -------------------------
            # Allocate gradients
            # -------------------------
            grad_w_c2v_v2c = zeros(Float32, length(bpnn.weights_c2v_v2c))
            grad_w_llrs    = zeros(Float32, length(bpnn.weights_llrs))
            grad_w_readout = zeros(Float32, length(bpnn.weights_c2v_readout))
            
            # -------------------------
            # Enzyme autodiff
            # -------------------------
            loss = Enzyme.autodiff(
                Enzyme.Reverse,
                get_loss_value,
                # arguments for which we want gradients:
                Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
                Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
                Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
                # constant arguments:
                Enzyme.Const(bpnn.correlation_importance),
                Enzyme.Const(bpnn.loss_layer_regularizer),
                Enzyme.Const(base),
                Enzyme.Const(llrs_batch),
                Enzyme.Const(syndromes_batch),
                Enzyme.Const(expected_batch)
            )

            #=
            println("Gradients computed for batch $b in epoch $epoch:")
            println("  grad_w_c2v_v2c norm = ", norm(grad_w_c2v_v2c))
            println("  grad_w_llrs norm = ", norm(grad_w_llrs))
            println("  grad_w_readout norm = ", norm(grad_w_readout))
            # println("  grad_w_loss norm = ", norm(grad_w_loss))
            =#

            # -------------------------
            # Gradient clipping
            # -------------------------
            clip_grad!(grad_w_c2v_v2c; max_grad_norm=max_grad_norm)
            clip_grad!(grad_w_llrs; max_grad_norm=max_grad_norm)
            clip_grad!(grad_w_readout; max_grad_norm=max_grad_norm)
            
            # -------------------------
            # Gradient step (simple SGD for now)
            # -------------------------
            @. bpnn.weights_c2v_v2c -= learning_rate * grad_w_c2v_v2c
            @. bpnn.weights_llrs    -= learning_rate * grad_w_llrs
            @. bpnn.weights_c2v_readout -= learning_rate * grad_w_readout
            
            # -------------------------
            # Progress update
            # -------------------------
            current_loss = get_loss_value(
                bpnn.weights_c2v_v2c,
                bpnn.weights_llrs,
                bpnn.weights_c2v_readout,
                bpnn.correlation_importance,
                bpnn.loss_layer_regularizer,
                base,
                llrs_batch,
                syndromes_batch,
                expected_batch
            )
            ProgressMeter.next!(batch_progress; showvalues = [(:loss, current_loss)])

            #=
            # --------------------------------------------------
            # Log the hyperparameters and loss for this batch and epoch for debugging and visualization later.
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :epoch] = epoch
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :sample] = b
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :loss_layer_temp] = bpnn.loss_layer_regularizer[1]
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :correlation_importance] = bpnn.correlation_importance[1]
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :loss] = current_loss
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :min_weight_c2v_v2c] = minimum(bpnn.weights_c2v_v2c)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :max_weight_c2v_v2c] = maximum(bpnn.weights_c2v_v2c)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :median_weight_c2v_v2c] = median(bpnn.weights_c2v_v2c)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :min_weight_llrs] = minimum(bpnn.weights_llrs)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :max_weight_llrs] = maximum(bpnn.weights_llrs)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :median_weight_llrs] = median(bpnn.weights_llrs)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :min_weight_c2v_readout] = minimum(bpnn.weights_c2v_readout)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :max_weight_c2v_readout] = maximum(bpnn.weights_c2v_readout)
            hyperparameters[(epoch - 1) * length(training_dataset) + b, :median_weight_c2v_readout] = median(bpnn.weights_c2v_readout)
            # --------------------------------------------------
            =#
        end

        # ProgressMeter.next!(epoch_progress)
    end

    # Debugging: write the hyperparameters and loss values into a file for visualization later.
    # CSV.write("./../data/15q_Hamm_code_data_10000_train/training_log.csv", hyperparameters)

    return bpnn
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
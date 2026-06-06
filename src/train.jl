function get_loss_value(
    weights_c2v_v2c, # learanble weights for computing m^t_(v→c) from m^(t-1)_(c→v).
    weights_llrs, # learnable weights for m^t_(v→c) from the initial LLRs, and also for computing the posterior LLRs from m^t_(c→v).
    weights_c2v_readout, # learnable weights for computing the readout (posterior LLRs) from m^t_(c→v).
    correlation_importance, # importance of the correlation penalty in the Loss function, to be annealed during training.
    loss_layer_regularizer, # temperature for the smooth minimum approximation when combining losses from different layers, to be annealed during training.
    llr_certainty_importance, # term for ensuring that the LLRs have converged.
    sparsity_regularizer, # term for encouraging sparsity in the LLRs, to be annealed during training.
    warmup_loss_layers, # First number of layers to leave unconstrained in the Loss function.
    base, # constant parameters of the model, including the parity-check matrix, connectivity, correlation strengths, etc.
    llrs_batch, # batch of initial LLRs for the bits, to be used as input to the network
    syndromes_batch, # batch of syndromes, to be used as input to the network
    expected_recoveries # batch of expected recoveries (error patterns), to be used for computing the Loss function
)::Float32
    """
    Compute the value of the Loss function for a given batch of data and given values for the weights.
    This version is friendly to Enzyme.jl, which requires a functional approach where the model parameters are passed explicitly to the function computing the loss, rather than being accessed as fields of a struct.
    """
    # Forward pass through the network to get the posterior LLRs.
    # Always use the CPU path here: Enzyme cannot differentiate through Metal GPU
    # array allocation (MtlArray constructors call task_local_storage via device(),
    # which is non-differentiable). The GPU path is only valid for inference, not AD.
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
        loss_layer_regularizer, # temperature for the smooth minimum approximation when combining losses from different layers, to be annealed during training.
        llr_certainty_importance, # weight for ensuring that the LLRs are converged.
        sparsity_regularizer, # term for encouraging sparsity in the LLRs, to be annealed during training.
        warmup_loss_layers # first number of layers that should be unconstrained since the optimizer doesn't know what the right beliefs are.
    )
    return total_loss
end

function get_individual_loss_values(
    weights_c2v_v2c::Vector{Float32}, # learanble weights for computing m^t_(v→c) from m^(t-1)_(c→v).
    weights_llrs::Vector{Float32}, # learnable weights for m^t_(v→c) from the initial LLRs, and also for computing the posterior LLRs from m^t_(c→v).
    weights_c2v_readout::Vector{Float32}, # learnable weights for computing the readout (posterior LLRs) from m^t_(c→v).
    correlation_importance::Float32, # importance of the correlation penalty in the Loss function, to be annealed during training.
    loss_layer_regularizer::Float32, # temperature for the smooth minimum approximation when combining losses from different layers, to be annealed during training.
    llr_certainty_importance::Float32, # term for ensuring that the LLRs have converged.
    sparsity_importance::Float32, # term for encouraging sparsity in the LLRs, to be annealed during training.
    warmup_loss_layers::Int, # first number of layers that should be excluded from the loss function.
    base::NeuralBPBase, # constant parameters of the model, including the parity-check matrix, connectivity, correlation strengths, etc.
    llrs_batch::Matrix{Float32}, # batch of initial LLRs for the bits, to be used as input to the network
    syndromes_batch::BitMatrix, # batch of syndromes, to be used as input to the network
    expected_recoveries::BitMatrix # batch of expected recoveries (error patterns), to be used for computing the Loss function
)
    """
    Compute the value of the Loss function for a given batch of data and given values for the weights.
    This function is solely for debugging purposes. It provides the loss as a pair of numbers: (loss_syndrome, loss_correlations), where
    loss_syndrome: is the part of the loss function that is responsible for enforcing syndrome contraints.
    loss_correlations: is the part of the loss function that is responsible for encoding correlations.
    """
    # Forward pass through the network to get the posterior LLRs.
    # Always use the CPU path: same reason as in get_loss_value — Metal GPU array
    # allocation is non-differentiable and this function is also used in AD contexts.
    posterior_llrs = forward_pass_with_weights(
        weights_c2v_v2c,
        weights_llrs,
        weights_c2v_readout,
        base,
        llrs_batch,
        syndromes_batch
    )

    parity_check_matrix_dual = base.parity_check_matrix_dual
    connectivity = base.connectivity
    correlation_strengths = base.correlation_strengths
    is_correlated = base.is_correlated

    # Compute the loss from the syndrome and correlation parts individually.
    n_layers::Int = size(posterior_llrs, 3) - warmup_loss_layers
    # Record the syndrome loss, syndrome regularizer, the correlation loss, and the total loss separately per layer.
    losses_per_layer = zeros(Float32, (n_layers, 4))
    for layer in warmup_loss_layers:n_layers
        post = posterior_llrs[:, :, layer]
        # base_loss   = compute_quadratic_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
        base_loss   = compute_sine_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
        llr_reg     = syndrome_loss_regularizer(post) * tanh(layer / n_layers)
        sparse_pen  = sparsity_penalty(post)
        corr_pen    = is_correlated ? compute_additional_loss_from_ising_correlations(post, connectivity, correlation_strengths) : 0f0

        # Record the individual losses
        losses_per_layer[layer - warmup_loss_layers + 1, 1] = base_loss
        losses_per_layer[layer - warmup_loss_layers + 1, 2] = llr_reg
        losses_per_layer[layer - warmup_loss_layers + 1, 3] = corr_pen
        losses_per_layer[layer - warmup_loss_layers + 1, 4] = sparse_pen
        
        # Compute the total loss for the layer
        loss_per_layer = base_loss + 
                         llr_certainty_importance * llr_reg +
                         correlation_importance * corr_pen +
                         sparsity_importance * sparse_pen
        
        losses_per_layer[layer - warmup_loss_layers + 1, 4] = loss_per_layer
    end
    # total_loss = softmin_loss(losses_per_layer[:, 4], loss_layer_regularizer)
    total_loss = linear_ramp_loss(losses_per_layer[:, 4])
    # total_loss = last_layer_only_loss(losses_per_layer[:, 4])
    return (total_loss, losses_per_layer)
end

function compute_hyperparameters(epoch::Int, annealing_schedule::Dict)::Dict{Symbol, Float32}
    """
    Compute the hyperparameters for a given epoch based on the defined annealing schedules in `HP_SCHEDULES`.
    """
    loss_hyperparameters = Dict{Symbol, Float32}()
    for (name, spec) in annealing_schedule
        if spec["direction"] == "down"
            loss_hyperparameters[Symbol(name)] = max(spec["min"], spec["max"] * spec["decay"]^(epoch - 1))
        else  # :up — anneal from min toward max
            loss_hyperparameters[Symbol(name)] = spec["max"] - (spec["max"] - spec["min"]) * spec["decay"]^(epoch - 1)
        end
    end
    return loss_hyperparameters
end

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

# ---------------------------------------------------------------------------
# Debug-logging helpers for train_neuralbp_enzyme!
# These are kept separate so the training loop stays readable and the
# logging concern can be disabled / replaced without touching core logic.
# ---------------------------------------------------------------------------

function init_training_debug_logs(n_samples_to_log::Int)
    """
    Initialize DataFrames for logging hyperparameters, losses, and weight statistics during training.
    The `hp_log` DataFrame logs the hyperparameters and weight statistics for each batch, while the `losses_log` DataFrame logs the individual loss components for each layer and the total loss for each batch.
    """
    hp_log = DataFrame(
        epoch = zeros(Int, n_samples_to_log),
        sample = zeros(Int, n_samples_to_log),
        loss_layer_temp = zeros(Float32, n_samples_to_log),
        correlation_importance = zeros(Float32, n_samples_to_log),
        llr_certainty_importance = zeros(Float32, n_samples_to_log),
        sparsity_importance = zeros(Float32, n_samples_to_log),
        loss = zeros(Float32, n_samples_to_log),
        nan_skip_count = zeros(Int, n_samples_to_log),
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
    losses_log = DataFrame(
        :epoch => zeros(Int, n_samples_to_log),
        :batch => zeros(Int, n_samples_to_log),
        :layers => zeros(Int, n_samples_to_log),
        :base_loss => ["" for _ in 1:n_samples_to_log],
        :syndrome_regularizer => ["" for _ in 1:n_samples_to_log],
        :correlation_penalty => ["" for _ in 1:n_samples_to_log],
        :loss_at_layer => ["" for _ in 1:n_samples_to_log],
        :total_loss => zeros(Float32, n_samples_to_log)
    )
    return hp_log, losses_log
end

function log_batch_debug!(
    hp_log::DataFrame,
    losses_log::DataFrame,
    index::Int,
    epoch::Int,
    b::Int,
    n_layers::Int,
    hp::Dict{Symbol, Float32},
    aggregate_loss::Float32,
    nan_skip_count::Int,
    bpnn::NachmaniNeuralBP,
    individual_losses::Matrix{Float32}
)
    """
    Log the hyperparameters, loss values, and weight statistics for a given batch into the provided DataFrames.
    """
    hp_log[index, :epoch] = epoch
    hp_log[index, :sample] = b
    hp_log[index, :loss_layer_temp] = hp[:loss_layer_temperature]
    hp_log[index, :correlation_importance] = hp[:correlation_importance]
    hp_log[index, :llr_certainty_importance] = hp[:llr_certainty_importance]
    hp_log[index, :sparsity_importance] = hp[:sparsity_importance]
    hp_log[index, :loss] = aggregate_loss
    hp_log[index, :nan_skip_count] = nan_skip_count
    hp_log[index, :min_weight_c2v_v2c] = minimum(bpnn.weights_c2v_v2c)
    hp_log[index, :max_weight_c2v_v2c] = maximum(bpnn.weights_c2v_v2c)
    hp_log[index, :median_weight_c2v_v2c] = median(bpnn.weights_c2v_v2c)
    hp_log[index, :min_weight_llrs] = minimum(bpnn.weights_llrs)
    hp_log[index, :max_weight_llrs] = maximum(bpnn.weights_llrs)
    hp_log[index, :median_weight_llrs] = median(bpnn.weights_llrs)
    hp_log[index, :min_weight_c2v_readout] = minimum(bpnn.weights_c2v_readout)
    hp_log[index, :max_weight_c2v_readout] = maximum(bpnn.weights_c2v_readout)
    hp_log[index, :median_weight_c2v_readout] = median(bpnn.weights_c2v_readout)

    losses_log[index, :epoch] = epoch
    losses_log[index, :batch] = b
    losses_log[index, :layers] = n_layers
    losses_log[index, :base_loss]             = join(["$(individual_losses[l, 1])" for l in 1:n_layers], ",")
    losses_log[index, :syndrome_regularizer]  = join(["$(individual_losses[l, 2])" for l in 1:n_layers], ",")
    losses_log[index, :correlation_penalty]   = join(["$(individual_losses[l, 3])" for l in 1:n_layers], ",")
    losses_log[index, :loss_at_layer]         = join(["$(individual_losses[l, 4])" for l in 1:n_layers], ",")
    losses_log[index, :total_loss] = aggregate_loss
    return nothing
end

function save_training_debug_logs(debugging_logfile::String, hp_log::DataFrame, losses_log::DataFrame)
    CSV.write("$(debugging_logfile).csv", hp_log)
    CSV.write("$(debugging_logfile)_individual_losses.csv", losses_log)
    return nothing
end

# ---------------------------------------------------------------------------

function train_neuralbp_enzyme!(
    bpnn::NachmaniNeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix,
    hyperparameters::Dict;
    debugging_logfile::String=""
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    We use Enzyme.jl for AD of the loss w.r.t. the model parameters, and an
    `Optimisers.jl` optimizer chain (gradient-clip → Adam/AdamW) for updates.

    Annealing intent (see also `compute_hyperparameters`):
    - Early on, the network is clueless and the syndrome objective is uninformative;
      use the channel's correlation prior heavily (large `correlation_importance`).
    - As training progresses, the syndrome term becomes informative; anneal the
      correlation weight down so it becomes a tie-breaker rather than a forcing
      constraint.
    - The per-layer combiner currently uses `linear_ramp_loss` (late layers
      weighted more); the `loss_layer_temperature` hyperparameter is retained
      for compatibility with the soft-min combiner and is annealed even though
      the linear ramp doesn't consume it.

    Robustness against numerical instability:
    - Each batch's gradients are checked for NaN/Inf BEFORE the optimizer step.
      If non-finite, the batch is skipped — no update to weights, no update to
      Adam state. This is the cheapest defense against the optimizer state being
      poisoned by a bad gradient (which would NaN every subsequent batch too).
    - At the start of each epoch, `bpnn` weights and `opt_state` are deep-copied
      into a checkpoint. If `nan_skip_count > max_nan_skips_per_epoch` by the
      end of the epoch, the checkpoint is restored — the epoch is "rolled back"
      and training continues from where it was at the start of the epoch.
    - `adam_eps` is set to 1e-4 by default (vs the typical 1e-8) which removes
      a dominant source of NaN updates in BP-style decoders where some weights'
      gradients can be near-zero for a long time.
    """
    base = bpnn.base
    # Hyperparameters for training
    n_epochs = hyperparameters["n_epochs"]
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    weight_decay = hyperparameters["weight_decay"]
    max_grad_norm = hyperparameters["max_grad_norm"]
    adam_eps = hyperparameters["adam_eps"]
    max_nan_skips_per_epoch = hyperparameters["nanskip"]
    warmup_loss_layers = hyperparameters["warmup_layers"]
    annealing_schedule = Dict(
        key => hyperparameters[key]
        for key in [
            "loss_layer_temperature",
            "correlation_importance",
            "llr_certainty_importance",
            "sparsity_importance"
        ]
    )
    
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

    # --------------------------
    # Debugging: pre-allocate log DataFrames.
    n_samples_to_log = n_epochs * length(training_dataset)
    n_layers = bpnn.base.n_layers - warmup_loss_layers
    hp_log, individual_losses_log = init_training_debug_logs(n_samples_to_log)
    # --------------------------

    # -------------------------
    # Optimizer setup
    # -------------------------
    # Pick Adam vs AdamW based on whether the user specified a non-zero
    # weight_decay. Both wrap inside an OptimiserChain that does gradient-norm
    # clipping first, then the adaptive update.
    inner_opt = if weight_decay > 0f0
        AdamW(learning_rate, (0.9f0, 0.999f0), weight_decay)
    else
        Adam(learning_rate, (0.9f0, 0.999f0), adam_eps)
    end
    opt_rule  = OptimiserChain(ClipGrad(max_grad_norm), inner_opt)
    opt_state = Optimisers.setup(opt_rule, bpnn)

    # -------------------------
    # Progress bars
    # -------------------------
    epoch_progress = Progress(n_epochs, desc="Training Epochs: ")

    for epoch in 1:n_epochs
        batch_progress = Progress(length(training_dataset), desc="Epoch $epoch Batches: ")

        hp = compute_hyperparameters(epoch, annealing_schedule)

        # -------------------------
        # Per-epoch checkpoint — restored at end-of-epoch if too many batches
        # are skipped due to NaN/Inf gradients.
        # -------------------------
        bpnn_checkpoint      = deepcopy(bpnn)
        opt_state_checkpoint = deepcopy(opt_state)
        nan_skip_count       = 0

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
            (_, loss_value) = Enzyme.autodiff(
                Enzyme.ReverseWithPrimal,
                get_loss_value,
                # Arguments for which we want gradients:
                Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
                Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
                Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
                # Constant arguments:
                Enzyme.Const(hp[:correlation_importance]),
                Enzyme.Const(hp[:loss_layer_temperature]),
                Enzyme.Const(hp[:llr_certainty_importance]),
                Enzyme.Const(hp[:sparsity_importance]),
                Enzyme.Const(warmup_loss_layers),
                Enzyme.Const(base),
                Enzyme.Const(llrs_batch),
                Enzyme.Const(syndromes_batch),
                Enzyme.Const(expected_batch)
            )

            # -------------------------
            # NaN / Inf guard — skip the optimizer step entirely if any
            # gradient component is non-finite. This preserves the last
            # known-good Adam state and the last known-good weights.
            # -------------------------
            grads_finite = all(isfinite, grad_w_c2v_v2c) &&
                           all(isfinite, grad_w_llrs)    &&
                           all(isfinite, grad_w_readout)

            if !grads_finite
                nan_skip_count += 1
                @warn "Non-finite gradient at epoch=$epoch batch=$b — skipping update." nan_skip_count
                ProgressMeter.next!(batch_progress; showvalues = [(:loss, NaN32), (:nan_skips, nan_skip_count)])

                # If too many batches have been skipped in this epoch due to NaN/Inf gradients, we break out of the batch loop early to trigger the epoch rollback at the end of the epoch.
                if nan_skip_count > max_nan_skips_per_epoch
                    @warn """
                    Epoch $epoch: $nan_skip_count batches skipped due to non-finite gradients.
                    If this persists, consider:
                    - Lowering `learning_rate`
                    - Raising `adam_eps`
                    - Tightening `max_grad_norm`
                    """
                    break
                end

                continue
            end

            # -------------------------
            # Progress update
            # -------------------------
            (aggregate_loss, individual_losses) = get_individual_loss_values(
                bpnn.weights_c2v_v2c,
                bpnn.weights_llrs,
                bpnn.weights_c2v_readout,
                hp[:correlation_importance],
                hp[:loss_layer_temperature],
                hp[:llr_certainty_importance],
                hp[:sparsity_importance],
                warmup_loss_layers,
                base,
                llrs_batch,
                syndromes_batch,
                expected_batch
            )
            
            # -------------------------
            # Adaptive Gradient Step (Adam / AdamW + ClipGrad)
            # -------------------------
            grads = (
                weights_c2v_v2c     = grad_w_c2v_v2c,
                weights_llrs        = grad_w_llrs,
                weights_c2v_readout = grad_w_readout
            )
            (opt_state, bpnn) = Optimisers.update!(opt_state, bpnn, grads)
            # -------------------------

            ProgressMeter.next!(batch_progress; showvalues = [(:loss, aggregate_loss), (:nan_skips, nan_skip_count)])

            # --------------------------------------------------
            # Debugging: log this batch.
            index = (epoch - 1) * length(training_dataset) + b
            log_batch_debug!(
                hp_log,
                individual_losses_log,
                index,
                epoch,
                b,
                n_layers,
                hp,
                aggregate_loss,
                nan_skip_count,
                bpnn,
                individual_losses
            )
            # --------------------------------------------------
        end

        # -------------------------
        # End-of-epoch rollback if the epoch was unstable.
        # We restore weights in-place (so the caller's `bpnn` reference still
        # points at the rolled-back model) and deep-copy the opt_state back.
        # -------------------------
        if nan_skip_count > max_nan_skips_per_epoch
            bpnn.weights_c2v_v2c     .= bpnn_checkpoint.weights_c2v_v2c
            bpnn.weights_llrs        .= bpnn_checkpoint.weights_llrs
            bpnn.weights_c2v_readout .= bpnn_checkpoint.weights_c2v_readout
            opt_state = deepcopy(opt_state_checkpoint)
        end

        ProgressMeter.next!(epoch_progress; showvalues = [(:nan_skips_this_epoch, nan_skip_count)])
    end

    save_training_debug_logs(debugging_logfile, hp_log, individual_losses_log)

    return bpnn
end

function train_Nachmani_neuralbp(
    base::NeuralBPBase,
    training_errors_file::String,
    hyperparameters::Dict=Dict(); # Hyperparameters for training the Neural BP model
    initial_conditions::Dict=Dict(),
    prefix::String="./../data"
)
    """
    Train a Neural Belief Propagation decoder for the given parity-check matrix.
    The trained model consists of weights (coefficients) for each pair of connected neurons in the neural BP network.
    We will save the weights into a file for later use.
    If this weights file already exists, we will load the weights from the file instead of training a new model.
    """
    
    # Create the models and results directories if they don't exist
    models_dir = "$(prefix)/models"
    if !isdir(models_dir)
        mkdir(models_dir)
    end
    
    # Load the base BP model and the neural BP model with randomly initialized weights.
    if length(initial_conditions) == 0
        initial_conditions = Dict(String, Vector{Float32})(
            "weights_c2v_v2c" => random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale=0.1f0),
            "weights_llrs" => random_values_around_one([base.code_n_bits * base.n_layers]; scale=0.1f0),
            "weights_c2v_readout" => random_values_around_one([base.nb_weights_c2v_readout]; scale=0.1f0)
        )
    end
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=initial_conditions["weights_c2v_v2c"],
        weights_llrs=initial_conditions["weights_llrs"],
        weights_c2v_readout=initial_conditions["weights_c2v_readout"]
    )
    
    # Extract the name of the training file name to include in the weights file name for clarity on what data the model was trained on.
    # We only want the filename without the path and extension.
    # For example, if the training file is `data/hamming/training_data.txt`, we want to extract `training_data`.
    training_source = splitext(basename(training_errors_file))[1]
    
    # Check if the weights file already exists
    n_epochs = hyperparameters["n_epochs"]
    weights_filename =
        "$(models_dir)/neuralbp_weights_" *
        "nlayers_$(base.n_layers)_" *
        "epochs_$(n_epochs)_" *
        "trained_using_$(training_source).json"
    
    if isfile(weights_filename) && !hyperparameters["retrain"]
        # println("Loading existing weights from file: $weights_filename")
        bpnn = load_trained_neuralbp_model(weights_filename, bpnn)
    else
        # Read errors from the training errors file
        expected_recoveries = convert.(Bool, readdlm(training_errors_file, Int))
        # Compute the syndromes for the training errors
        training_syndromes = convert.(Bool, mod.(base.parity_check_matrix * expected_recoveries, 2))
        
        # Train the Neural BP model
        train_neuralbp_enzyme!(
            bpnn, 
            training_syndromes, 
            expected_recoveries, 
            hyperparameters;
            debugging_logfile="$(prefix)/log/debugging_$(training_source)"
        )

        # Save the trained weights to a file
        save_trained_neuralbp_model(weights_filename, bpnn)
    end
    return bpnn
end
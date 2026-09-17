function get_loss_value(
    weights_c2v_v2c, # learnable weights for computing m^t_(v→c) from m^(t-1)_(c→v).
    weights_llrs, # learnable weights for m^t_(v→c) from the initial LLRs, and also for computing the posterior LLRs from m^t_(c→v).
    weights_c2v_readout, # learnable weights for computing the readout (posterior LLRs) from m^t_(c→v).
    coupling_logit, # learnable length-1 vector holding θ; the forward pass uses α = 1/(1+exp(-θ)) ∈ (0,1) (inert for the standard rule).
    loss_layer_temperature, # temperature of the softmin over layers, annealed during training.
    warmup_loss_layers, # first number of layers to leave unconstrained in the loss.
    base, # constant parameters of the model: the parity-check matrices, the check node tables, etc.
    llrs_batch, # batch of initial LLRs for the bits, the input to the network
    syndromes_batch, # batch of syndromes, the input to the network
    expected_recoveries # batch of expected recoveries (error patterns), the target
)::Float32
    """
    The training loss for one batch at the given weights: the forward pass,
    then `compute_loss` — the softmin over scored layers of the base loss.

    Written functionally, with every weight an explicit argument, because that
    is what Enzyme.jl differentiates. The forward pass is always the CPU one:
    Enzyme cannot differentiate through device-array allocation, and the GPU
    path is inference-only.
    """
    syndromes_batch_matrix = Matrix{Bool}(syndromes_batch)
    posterior_llrs = forward_pass_with_weights(
        weights_c2v_v2c,
        weights_llrs,
        weights_c2v_readout,
        coupling_logit,
        base,
        llrs_batch,
        syndromes_batch_matrix
    )
    total_loss = compute_loss(
        posterior_llrs,
        expected_recoveries,
        base.parity_check_matrix_dual,
        loss_layer_temperature,
        warmup_loss_layers
    )
    return total_loss
end

function get_individual_loss_values(
    weights_c2v_v2c::Vector{Float32},
    weights_llrs::Vector{Float32},
    weights_c2v_readout::Vector{Float32},
    coupling_logit::Vector{Float32},
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int,
    base::NeuralBPBase,
    llrs_batch::Matrix{Float32},
    syndromes_batch::BitMatrix,
    expected_recoveries::BitMatrix
)::Tuple{Float32, Vector{Float32}}
    """
    Logging mirror of `get_loss_value`: the total loss and the base loss at
    every scored layer, so the debug log can show which layers the softmin is
    weighting. Same forward pass, same `base_loss_per_layer`, so the two cannot
    disagree.
    """
    posterior_llrs = forward_pass_with_weights(
        weights_c2v_v2c,
        weights_llrs,
        weights_c2v_readout,
        coupling_logit,
        base,
        llrs_batch,
        syndromes_batch
    )
    losses_per_layer::Vector{Float32} = base_loss_per_layer(
        posterior_llrs, expected_recoveries, base.parity_check_matrix_dual, warmup_loss_layers)
    total_loss::Float32 = softmin_loss(losses_per_layer, loss_layer_temperature)
    return (total_loss, losses_per_layer)
end

function compute_hyperparameters(epoch::Int, annealing_schedule::Dict)::Dict{Symbol, Float32}
    """
    Compute the hyperparameters for a given epoch based on the defined annealing schedules in `HP_SCHEDULES`.
    If the annealing schedule for a hyperparameter has both `max` and `min` values set to 0.0, the hyperparameter will be set to 0.0 for all epochs.
    """
    loss_hyperparameters = Dict{Symbol, Float32}()
    for (name, spec) in annealing_schedule
        if spec["max"] == 0.0 && spec["min"] == 0.0
            loss_hyperparameters[Symbol(name)] = 0.0
        else
            if spec["direction"] == "down" # anneal from max toward min
                loss_hyperparameters[Symbol(name)] = max(spec["min"], spec["max"] * spec["decay"]^(epoch - 1))
            else  # "up" — anneal from min toward max
                loss_hyperparameters[Symbol(name)] = spec["max"] - (spec["max"] - spec["min"]) * spec["decay"]^(epoch - 1)
            end
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
    Pre-allocate the two per-batch debug tables. `hp_log` carries the
    hyperparameters in force, the loss, the NaN-skip count, the weight
    statistics and α; `losses_log` carries the base loss at every scored layer.
    """
    hp_log = DataFrame(
        epoch = zeros(Int, n_samples_to_log),
        sample = zeros(Int, n_samples_to_log),
        loss_layer_temp = zeros(Float32, n_samples_to_log),
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
        median_weight_c2v_readout = zeros(Float32, n_samples_to_log),
        coupling_scale = zeros(Float32, n_samples_to_log),
        coupling_logit = zeros(Float32, n_samples_to_log)
    )
    losses_log = DataFrame(
        :epoch => zeros(Int, n_samples_to_log),
        :batch => zeros(Int, n_samples_to_log),
        :layers => zeros(Int, n_samples_to_log),
        :base_loss => ["" for _ in 1:n_samples_to_log],
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
    losses_per_layer::Vector{Float32}
)
    """
    Log the hyperparameters, loss values, and weight statistics for a given batch into the provided DataFrames.
    """
    hp_log[index, :epoch] = epoch
    hp_log[index, :sample] = b
    hp_log[index, :loss_layer_temp] = hp[:loss_layer_temperature]
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
    hp_log[index, :coupling_scale] = effective_coupling_scale(bpnn)
    hp_log[index, :coupling_logit] = bpnn.coupling_logit[1]

    losses_log[index, :epoch] = epoch
    losses_log[index, :batch] = b
    losses_log[index, :layers] = n_layers
    losses_log[index, :base_loss] = join(["$(losses_per_layer[l])" for l in 1:n_layers], ",")
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
    debugging_logfile::String="",
    is_debug::Bool=false,
    is_quiet::Bool=false,
    online_training::Bool=false, # If true, we will generate training samples on the fly instead of reading from a file. However, right now we don't have an implementation for this, so we will simply read a random subset of `batch_size` samples from the training dataset.
    n_gradient_updates_per_epoch::Int=0, # If `online_training` is true, this parameter specifies how many random batches we will generate (or read from the training dataset) for each epoch. If `online_training` is false, this parameter is ignored and we simply use all the batches from the training dataset as usual.
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    We use Enzyme.jl for AD of the loss w.r.t. the model parameters, and an
    `Optimisers.jl` optimizer chain (gradient-clip → Adam/AdamW) for updates.

    The loss is `compute_loss`: the softmin over scored layers of the base loss
    (the residue of e + σ(μ) against [H; L]). Its one annealed hyperparameter,
    `loss_layer_temperature`, follows the "min,max,decay,direction" spec in the
    TOML (see `compute_hyperparameters`) and is annealed DOWN so early training
    averages over layers and late training commits to the best one.

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
    # Whether α (the scale on the CER couplings inside the enriched check node)
    # is learned. Irrelevant for the standard rule, where α is never read; the
    # leaf is frozen there too so that a tanh run never moves it, and a later
    # `check_node = "enriched"` re-test of the same weights file sees α = 1.
    coupling_scale_learnable::Bool =
        Bool(get(hyperparameters, "coupling_scale_learnable", true)) &&
        base.check_node_kind == CHECK_NODE_ENRICHED
    # The softmin temperature is the only annealed hyperparameter.
    annealing_schedule = Dict("loss_layer_temperature" => hyperparameters["loss_layer_temperature"])

    # ---------------------------------
    # Create batches
    # ---------------------------------
    n_samples = size(syndromes, 2)
    
    if (n_gradient_updates_per_epoch == 0)
        # Use the full training dataset, split into batches.
        samples_grouped_by_batch = [
            (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
            for i in 1:ceil(Int, n_samples / batch_size)
        ]
        
        training_dataset = [
            (
                syndromes[:, idx],
                expected_recoveries[:, idx],
            )
            for idx in samples_grouped_by_batch
        ]

        n_gradient_updates_per_epoch = length(training_dataset)
    end

    # --------------------------
    # Debugging: pre-allocate log DataFrames.
    if is_debug
        n_samples_to_log = n_epochs * n_gradient_updates_per_epoch
        n_layers = bpnn.base.n_layers - warmup_loss_layers
        hp_log, individual_losses_log = init_training_debug_logs(n_samples_to_log)
    end
    # --------------------------

    # -------------------------
    # Optimizer setup
    # -------------------------
    # Pick Adam vs AdamW based on whether the user specified a non-zero
    # weight_decay. Both wrap inside an OptimiserChain that does gradient-norm
    # clipping first, then the adaptive update.
    inner_opt::Optimisers.AbstractRule = Adam(learning_rate, (0.9f0, 0.999f0), adam_eps)
    if weight_decay > 0f0
        inner_opt = AdamW(learning_rate, (0.9f0, 0.999f0), weight_decay)
    end
    opt_rule  = OptimiserChain(ClipGrad(max_grad_norm), inner_opt)
    opt_state = Optimisers.setup(opt_rule, bpnn)
    # A frozen leaf ignores its gradient in `Optimisers.update!` AND is exempt
    # from AdamW's weight decay, which would otherwise pull a fixed α toward 0
    # even with a zero gradient.
    if !coupling_scale_learnable
        Optimisers.freeze!(opt_state.coupling_logit)
    end
    # A LEARNED α must not be weight-decayed either. Decay shrinks every
    # parameter toward 0; for the message weights that is the existing
    # regulariser, but for α it is a standing bias toward "no couplings" — a
    # thumb on the scale against the very hypothesis the run is testing. Give
    # α's own leaf a decay-free rule (Adam with the same rate and clipping),
    # leaving every other leaf exactly as before.
    if coupling_scale_learnable && weight_decay > 0f0
        coupling_scale_rule::OptimiserChain = OptimiserChain(
            ClipGrad(max_grad_norm), Adam(learning_rate, (0.9f0, 0.999f0), adam_eps))
        # `opt_state` is a NamedTuple over the model's functor children, so the
        # leaf can be swapped by key without touching the others.
        opt_state = merge(opt_state, (coupling_logit = Optimisers.setup(coupling_scale_rule, bpnn.coupling_logit),))
    end

    if !is_quiet
        n_weights = length(bpnn.weights_c2v_v2c) + length(bpnn.weights_llrs) + length(bpnn.weights_c2v_readout)
        if coupling_scale_learnable
            n_weights += length(bpnn.coupling_logit)
        end
        print_info("Starting training on $(n_samples) samples, split into batches of $(batch_size), with $(n_weights) learnable parameters.")
        if base.check_node_kind == CHECK_NODE_ENRICHED
            print_info("Enriched check node: $(describe_soft_check_tables(base.soft_check_tables)); " *
                       "coupling scale α = $(effective_coupling_scale(bpnn)) (logit $(bpnn.coupling_logit[1])) " *
                       "($(coupling_scale_learnable ? "learnable" : "fixed")).")
        end
    end
    # -------------------------
    # Progress bars
    # -------------------------
    epoch_progress = is_quiet ? nothing : Progress(n_epochs, desc="Training Epochs: ")

    n_applied_updates::Int = 0
    n_rolled_back_epochs::Int = 0
    for epoch in 1:n_epochs
        batch_progress = is_quiet ? nothing : Progress(n_gradient_updates_per_epoch, desc="Epoch $epoch Batches: ")

        hp = compute_hyperparameters(epoch, annealing_schedule)

        # -------------------------
        # Per-epoch checkpoint — restored at end-of-epoch if too many batches
        # are skipped due to NaN/Inf gradients.
        # -------------------------
        bpnn_checkpoint      = deepcopy(bpnn)
        opt_state_checkpoint = deepcopy(opt_state)
        nan_skip_count       = 0

        for b in 1:n_gradient_updates_per_epoch

            if online_training
                # TODO: implement online training by generating a random batch of syndromes and expected recoveries on the fly.
                # For now, we just read a random batch from the training dataset to simulate the online training scenario.
                # n_samples_in_batch = length(samples_grouped_by_batch[b])
                selected_samples = rand(1:n_samples, batch_size)
                syndromes_batch = syndromes[:, selected_samples]
                expected_batch = expected_recoveries[:, selected_samples]
                llrs_batch = repeat(base.initial_llrs, 1, batch_size) # shape (n_bits, batch_size)
            else
                # Use the pre-created batches from the training dataset.
                # -------------------------
                # Shuffle batch
                # -------------------------
                shuffled_indices = randperm(size(training_dataset[b][1], 2))

                syndromes_batch = training_dataset[b][1][:, shuffled_indices]
                expected_batch  = training_dataset[b][2][:, shuffled_indices]
                llrs_batch      = repeat(base.initial_llrs, 1, length(shuffled_indices)) # shape (n_bits, batch_size)
            end

            # -------------------------
            # Allocate gradients
            # -------------------------
            grad_w_c2v_v2c = zeros(Float32, length(bpnn.weights_c2v_v2c))
            grad_w_llrs    = zeros(Float32, length(bpnn.weights_llrs))
            grad_w_readout = zeros(Float32, length(bpnn.weights_c2v_readout))
            # Always Duplicated, even when α is frozen or unused: the optimiser
            # leaf decides whether the gradient is applied, and a fixed
            # signature keeps ONE compiled Enzyme thunk for every configuration.
            grad_coupling_scale::Vector{Float32} = zeros(Float32, length(bpnn.coupling_logit))

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
                Enzyme.Duplicated(bpnn.coupling_logit, grad_coupling_scale),
                # Constant arguments (order MUST match get_loss_value's signature):
                Enzyme.Const(hp[:loss_layer_temperature]),
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
            grads_finite = isfinite(loss_value)          &&
                           all(isfinite, grad_w_c2v_v2c) &&
                           all(isfinite, grad_w_llrs)    &&
                           all(isfinite, grad_w_readout) &&
                           all(isfinite, grad_coupling_scale)

            if !grads_finite
                nan_skip_count += 1
                if !is_quiet
                    @warn "Non-finite gradient at epoch=$epoch batch=$b. Loss = $(loss_value). Skipping update." nan_skip_count
                    ProgressMeter.next!(batch_progress; showvalues = [(:loss, NaN32), (:nan_skips, nan_skip_count)])
                end

                # If too many batches have been skipped in this epoch due to NaN/Inf gradients, we break out of the batch loop early to trigger the epoch rollback at the end of the epoch.
                if nan_skip_count > max_nan_skips_per_epoch
                    if !is_quiet
                        @warn """
                        Epoch $epoch: $nan_skip_count batches skipped due to non-finite gradients.
                        If this persists, consider:
                        - Lowering `learning_rate`
                        - Raising `adam_eps`
                        - Tightening `max_grad_norm`
                        """
                    end
                    break
                end

                continue
            end

            # -------------------------
            # Adaptive Gradient Step (Adam / AdamW + ClipGrad)
            # -------------------------
            grads = (
                weights_c2v_v2c     = grad_w_c2v_v2c,
                weights_llrs        = grad_w_llrs,
                weights_c2v_readout = grad_w_readout,
                coupling_logit      = grad_coupling_scale
            )
            # θ is unconstrained, so the Adam step needs no projection: the
            # logistic link in `forward_pass_with_weights` keeps α in (0, 1)
            # however far θ travels.
            (opt_state, bpnn) = Optimisers.update!(opt_state, bpnn, grads)
            n_applied_updates += 1
            # -------------------------

            # -------------------------
            # Progress update
            # -------------------------
            if !is_quiet
                ProgressMeter.next!(batch_progress; showvalues = [(:loss, loss_value), (:nan_skips, nan_skip_count)])
            end

            # --------------------------------------------------
            # Debugging: log this batch.
            if is_debug
                (aggregate_loss, individual_losses) = get_individual_loss_values(
                    bpnn.weights_c2v_v2c,
                    bpnn.weights_llrs,
                    bpnn.weights_c2v_readout,
                    bpnn.coupling_logit,
                    hp[:loss_layer_temperature],
                    warmup_loss_layers,
                    base,
                    llrs_batch,
                    syndromes_batch,
                    expected_batch
                )
                index = (epoch - 1) * n_gradient_updates_per_epoch + b
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
            end
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
            bpnn.coupling_logit      .= bpnn_checkpoint.coupling_logit
            opt_state = deepcopy(opt_state_checkpoint)
            n_rolled_back_epochs += 1
            # NOT gated on is_quiet: a rolled-back epoch discards its work, and a
            # run where every epoch rolls back ships its initial weights. That
            # once reached a full 20-point cluster sweep unnoticed because every
            # warning sat behind the quiet flag.
            @warn "Epoch $(epoch) ROLLED BACK: $(nan_skip_count) batches had non-finite gradients (limit $(max_nan_skips_per_epoch)). Weights and optimizer state restored to the start of the epoch."
        end

        if !is_quiet
            ProgressMeter.next!(epoch_progress; showvalues = [(:nan_skips_this_epoch, nan_skip_count)])
        end
    end

    if is_debug
        save_training_debug_logs(debugging_logfile, hp_log, individual_losses_log)
    end

    if n_applied_updates == 0
        error("train_neuralbp_enzyme!: not a single gradient update was applied — " *
              "every batch had non-finite gradients and every epoch rolled back " *
              "($(n_rolled_back_epochs) of $(n_epochs)). The weights are still at " *
              "their initial values, and saving them would produce a model " *
              "indistinguishable from an untrained one in every downstream file. " *
              "Refusing. Check the loss terms for Enzyme-incompatible constructs.")
    end
    if n_rolled_back_epochs > 0
        @warn "Training finished with $(n_rolled_back_epochs) of $(n_epochs) epochs rolled back and $(n_applied_updates) applied updates."
    end

    return bpnn
end

function train_Nachmani_neuralbp(
    base::NeuralBPBase,
    training_errors_file::String,
    hyperparameters::Dict=Dict(); # Hyperparameters for training the Neural BP model
    initial_conditions::Dict=Dict(),
    prefix::String="./../data",
    is_debug::Bool=false,
    is_quiet::Bool=false,
    online_training::Bool=false, # If true, we will generate training samples on the fly instead of reading from a file. However, right now we don't have an implementation for this, so we will simply read a random subset of `batch_size` samples from the training dataset.
    n_gradient_updates_per_epoch::Int=0, # If `online_training` is true, this parameter specifies how many random batches we will generate (or read from the training dataset) for each epoch. If `online_training` is false, this parameter is ignored and we simply use all the batches from the training dataset as usual.
    will_test::Bool=false # Whether the caller intends to run predictions after this. Controls ONE console message: a missing weights file is the normal, expected state in a training-only run and worth no comment, but in a test run it means the model we meant to evaluate was absent and is being trained on the fly — which the user needs to know.
)
    """
    Train a Neural Belief Propagation decoder for the given parity-check matrix.
    The trained model consists of weights (coefficients) for each pair of connected neurons in the neural BP network.
    We will save the weights into a file for later use.
    If this weights file already exists, we will load the weights from the file instead of training a new model.
    """
    
    # Seed BEFORE the fallback weight draws below and before any batch sampling
    # inside `train_neuralbp_enzyme!`. No-op when `seed` is unset.
    # `expts/neural_bp_experiments.jl` seeds too, because it draws the initial
    # weights itself, before this function is reached.
    apply_training_seed!(hyperparameters)

    # Create the models and results directories if they don't exist
    models_dir = "$(prefix)/models"
    if !isdir(models_dir)
        mkdir(models_dir)
    end
    
    # Load the base BP model and the neural BP model with randomly initialized weights.
    if length(initial_conditions) == 0
        initial_conditions = Dict{String, Vector{Float32}}(
            "weights_c2v_v2c" => random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale=0.1f0),
            "weights_llrs" => random_values_around_one([base.code_n_bits * base.n_layers]; scale=0.1f0),
            "weights_c2v_readout" => random_values_around_one([base.nb_weights_c2v_readout]; scale=0.1f0)
        )
    end
    # α starts at `coupling_scale_init` (1 = the Bayesian value) unless the
    # caller supplied it explicitly. Read from the hyperparameters here rather
    # than in the script so a caller using the package directly gets the same
    # default.
    initial_coupling_scale::Float32 =
        Float32(get(hyperparameters, "coupling_scale_init", 1.0f0))
    if haskey(initial_conditions, "coupling_scale")
        initial_coupling_scale = Float32(initial_conditions["coupling_scale"][1])
    end
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=initial_conditions["weights_c2v_v2c"],
        weights_llrs=initial_conditions["weights_llrs"],
        weights_c2v_readout=initial_conditions["weights_c2v_readout"],
        coupling_scale=initial_coupling_scale
    )
    # Resuming from a checkpoint restores the TRAINED PARAMETER θ itself, not α.
    # Rebuilding θ from α above costs one logit(logistic(θ)) round trip, exact
    # only to Float32; a resumed run should continue from the θ it stopped at.
    if haskey(initial_conditions, "coupling_logit")
        bpnn.coupling_logit .= initial_conditions["coupling_logit"]
    end

    # Extract the name of the training file name to include in the weights file name for clarity on what data the model was trained on.
    # We only want the filename without the path and extension.
    # For example, if the training file is `data/hamming/training_data.txt`, we want to extract `training_data`.
    training_source = splitext(basename(training_errors_file))[1]
    
    # Check if the weights file already exists
    n_epochs = hyperparameters["n_epochs"]
    # `_no_cer` tag mirrors neural_bp_experiments.jl (and the submit-time
    # preflight) so a no-CER model never overwrites its CER counterpart.
    # `run_tag` does the same job for hyperparameter sweeps: the filename encodes
    # only nlayers/epochs/training_source, so two runs differing ONLY in e.g.
    # `check_node` would otherwise silently share one weights file.
    # `seed_tag` is essential, not cosmetic: without it two runs differing ONLY in
    # seed resolve to the same path, and with `retrain = false` the second would
    # silently load the first's weights — which is precisely the comparison the
    # seed exists to make possible.
    cer_tag = get(hyperparameters, "use_CER", true) ? "" : "_no_cer"
    # NOTE: `single_qubit_rescale` deliberately does NOT contribute a filename
    # tag. Only `run_tag` (and the seed) may extend the name, so the suffix stays
    # something the caller chooses explicitly and the test-side preflight can
    # reconstruct. Put `_sqres0p1` in `run_tag` yourself if you want it.
    run_tag = String(get(hyperparameters, "run_tag", ""))
    seed_tag = seed_tag_for(hyperparameters)
    weights_filename =
        "$(models_dir)/neuralbp_weights_" *
        "nlayers_$(base.n_layers)_" *
        "epochs_$(n_epochs)_" *
        "trained_using_$(training_source)$(cer_tag)$(run_tag)$(seed_tag).json"
    
    if isfile(weights_filename) && !hyperparameters["retrain"]
        # println("Loading existing weights from file: $weights_filename")
        bpnn = load_trained_neuralbp_model(weights_filename, bpnn)
    else
        if will_test
            println("No trained weights at $(weights_filename); training a new model.")
        end
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
            # `cer_tag`/`run_tag`/`seed_tag` all belong here, for the same reason
            # they belong in the weights and results names: without them every
            # point of a sweep writes to ONE debug file and all but the last is
            # lost. A four-point sweep silently left one usable log.
            debugging_logfile="$(prefix)/logs/debugging_$(training_source)$(cer_tag)$(run_tag)$(seed_tag)",
            is_debug=is_debug,
            is_quiet=is_quiet,
            online_training=online_training,
            n_gradient_updates_per_epoch=n_gradient_updates_per_epoch
        )

        # Save the trained weights to a file, recording the seed inside it so the
        # model is self-describing and not only identified by its filename.
        save_trained_neuralbp_model(weights_filename, bpnn; seed=hyperparameter_seed(hyperparameters))
    end
    return bpnn
end
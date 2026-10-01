# =============================================================================
# Loss for the unrolled neural BP decoder.
#
#     total = softmin_over_layers( base_l )
#
# `base_l` is the residue of e + σ(μ_l) against [H; L] at layer l: zero exactly
# when the layer's soft decision clears the syndrome AND lands in the correct
# coset. The softmin over layers, at a temperature annealed down during
# training, lets early training average over layers and late training commit to
# the best one.
#
# That is the whole loss. The auxiliary terms that used to sit beside it — a
# certainty penalty on undecided LLRs, a sparsity prior, and a family of
# correlation rewards fed by the CER couplings, each behind a detached syndrome
# gate — were removed on 2026-09-16. Six correlation forms over 1500 training
# runs never produced a coupling effect through the loss, while the same
# couplings placed inside the check node's forward pass (src/soft_constraints.jl)
# cut the classical failure rate in half with nothing trained. The couplings
# belong in inference; the loss only needs to say what a correct decode is.
# =============================================================================

# Per-check penalty. g(x) = 0 iff x is an even integer, i.e. iff the residual
# error commutes with that generator. Piecewise quadratic with subgradient ±2 at
# the odd integers, so descent is defined everywhere including at the maxima.
@inline floating_modulus(x) = (x - 2 * floor(x / 2))

function smooth_loss(real_syndrome_bit::Float32)::Float32
    """
    g(x) = x²        for 0 ≤ x ≤ 1
         = (2 - x)²  for 1 < x ≤ 2

    on x = s(μ) mod 2. d/dx = 2x and -2(2-x) respectively, so the gradient does
    not vanish at x = 1 and the optimizer is always pushed toward the nearest
    even integer.
    """
    x = floating_modulus(real_syndrome_bit)
    if 0 <= x <= 1
        return x^2
    elseif 1 < x <= 2
        return (2 - x)^2
    else
        error("smooth_loss is only defined for 0 ≤ x ≤ 2")
    end
end

function sine_residue_loss(real_syndrome_bit::Float32)::Float32
    """
    Compute h(x) = |sin (π x / 2)|.

    We will call this on each syndrome bit. We want to regard even syndrome bits as correct, so the loss is zero when x is an even integer.
    """
    residue = abs(sin(pi * real_syndrome_bit / 2))
    return residue
end

function compute_smooth_loss_from_llrs(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix
)::Float32
    """
    Batch-mean residue of the total error against the dual check matrix:

        L(μ, e) = (1/N) ∑_j ∑_i g( ∑_k H^⟂_ik [ e_kj + σ(μ_kj) ] )

    where H^⟂ carries the stabilizer generators and the logical operators, so a
    zero requires both a cleared syndrome and the correct coset.
    """
    n_samples = size(expected_recoveries, 2)
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    # average_loss = sum(@. smooth_loss(commutation_relations_matrix)) / n_samples
    average_loss = sum(@. sine_residue_loss(commutation_relations_matrix)) / n_samples
    return average_loss
end

function softmin_loss(losses_per_layer::AbstractVector{Float32}, temp::Float32)::Float32
    """
    Smooth minimum over layers, T·log(n) BELOW the true minimum at zero spread:

        softmin(L, T) = min(L) - T · log( ∑_l exp( -(L_l - min(L)) / T ) )

    T is annealed down, so early training averages over layers and late training
    commits to the best one. The gradient is the softmax weights, all ≥ 0, so
    lowering the selected layer's loss never pushes up another's.

    The value is monotonically DECREASING in T: at T → 0 it is min(L) exactly,
    and it falls further below min(L) as T grows. Annealing T down therefore
    RAISES the reported number over training even when the decoder is getting
    better, and when many layers have already hit zero loss the number is close
    to -T·log(n_zero_layers) and says more about the schedule than the decoder.
    Read a training-loss trajectory against the temperature schedule, never
    on its own.
    """
    n_layers = length(losses_per_layer)
    if n_layers == 1
        return losses_per_layer[1]
    end
    min_loss = minimum(losses_per_layer)
    aggregate_loss = min_loss - temp * log(sum(exp.(-(losses_per_layer .- min_loss) ./ temp)))
    return aggregate_loss
end

function base_loss_per_layer(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    warmup_loss_layers::Int
)::Vector{Float32}
    """
    `compute_smooth_loss_from_llrs` at every scored layer, i.e. every layer after
    the first `warmup_loss_layers`, in layer order. Shared by the training loss
    and by the per-layer diagnostics so the two can never disagree.
    """
    n_layers::Int = size(posterior_llrs, 3)
    if warmup_loss_layers < 0 || warmup_loss_layers >= n_layers
        throw(ArgumentError(
            "warmup_loss_layers = $(warmup_loss_layers) must lie in [0, n_layers - 1] " *
            "= [0, $(n_layers - 1)]; at least one layer has to be scored."))
    end
    losses::Vector{Float32} = zeros(Float32, n_layers - warmup_loss_layers)
    for layer in (warmup_loss_layers + 1):n_layers
        post::Matrix{Float32} = posterior_llrs[:, :, layer]
        losses[layer - warmup_loss_layers] =
            compute_smooth_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
    end
    return losses
end

# =============================================================================
#   How the scored layers are combined into one number
# =============================================================================
# The three options differ in what they ask of the decoder, and only one of them
# is consistent with a given TEST-TIME readout rule.
#
#   SOFTMIN  min over layers (as T -> 0). Asks for ONE good layer and lets the
#            rest be arbitrary. Measured 2026-10-01: with `warmup_layers = 10`
#            the per-batch argmin sits at layer 11 — the first layer it is
#            allowed to see — in 87% of batches that contain no error, and at a
#            late layer (median 20) in the ones that do. Meanwhile 98.8% of
#            test-time commits happen in layers 1-10. So the layer being
#            optimised and the layer being scored were nearly disjoint sets.
#            This is ANTI-aligned with first-to-clear testing, which needs EVERY
#            layer to be trustworthy because any of them might be the one that
#            commits.
#   LAST     the final layer alone. Consistent with committing at the final
#            layer (`commit_layer_rule = "last"` in predict.jl) — training and
#            testing then score the same layer, and there is no mismatch left to
#            reason about. Gives up early stopping.
#   MEAN     every scored layer, equally. The alignment for first-to-clear
#            testing: pushing all layers toward a correct decode makes whichever
#            one clears first a layer that has been trained.
#
# `softmin` stays the default so every earlier run reproduces exactly.

const LOSS_LAYERS_SOFTMIN::Int = 0
const LOSS_LAYERS_LAST::Int = 1
const LOSS_LAYERS_MEAN::Int = 2

function loss_layer_selection_code(selection_name::AbstractString)::Int
    """
    Resolve the `loss_layer_selection` hyperparameter to its integer code, so a
    typo fails at configuration time rather than inside a training loop.
    """
    normalised_name::String = lowercase(strip(String(selection_name)))
    selection_code::Int = LOSS_LAYERS_SOFTMIN
    if normalised_name == "softmin"
        selection_code = LOSS_LAYERS_SOFTMIN
    elseif normalised_name == "last"
        selection_code = LOSS_LAYERS_LAST
    elseif normalised_name == "mean"
        selection_code = LOSS_LAYERS_MEAN
    else
        throw(ArgumentError(
            "Unknown loss_layer_selection \"$(selection_name)\". Supported: " *
            "\"softmin\" (the historical default, min over layers as T -> 0), " *
            "\"last\" (the final layer alone) and \"mean\" (every scored layer, equally)."))
    end
    return selection_code
end

function loss_layer_selection_name(selection_code::Int)::String
    """
    Inverse of `loss_layer_selection_code`, for recording the mode a run used.
    """
    selection_label::String = "softmin"
    if selection_code == LOSS_LAYERS_LAST
        selection_label = "last"
    elseif selection_code == LOSS_LAYERS_MEAN
        selection_label = "mean"
    elseif selection_code != LOSS_LAYERS_SOFTMIN
        throw(ArgumentError("Unknown loss layer selection code $(selection_code)."))
    end
    return selection_label
end

function combine_layer_losses(
    losses_per_layer::AbstractVector{Float32},
    loss_layer_temperature::Float32,
    loss_layer_selection::Int
)::Float32
    """
    Reduce the per-layer losses to the one number the optimiser sees, by the
    rule `loss_layer_selection` names. The temperature is read only by
    `softmin`; the other two ignore it.
    """
    combined::Float32 = 0.0f0
    if loss_layer_selection == LOSS_LAYERS_SOFTMIN
        combined = softmin_loss(losses_per_layer, loss_layer_temperature)
    elseif loss_layer_selection == LOSS_LAYERS_LAST
        combined = losses_per_layer[end]
    elseif loss_layer_selection == LOSS_LAYERS_MEAN
        combined = sum(losses_per_layer) / Float32(length(losses_per_layer))
    else
        throw(ArgumentError("Unknown loss layer selection code $(loss_layer_selection)."))
    end
    return combined
end

function compute_loss(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int,
    loss_layer_selection::Int = LOSS_LAYERS_SOFTMIN
)::Float32
    """
    Total per-batch loss: the scored layers' base losses, combined by
    `loss_layer_selection` (see `combine_layer_losses`).

        softmin:  total = softmin_T( base_(warmup+1), ..., base_(n_layers) )
        last:     total = base_(n_layers)
        mean:     total = mean( base_(warmup+1), ..., base_(n_layers) )

    `posterior_llrs` is (n_bits × n_samples × n_layers), the readout of every
    layer of the unrolled decoder.

    NOTE on `last`: `warmup_loss_layers` becomes irrelevant, because only the
    final layer is read. It is still honoured for the vector the diagnostics
    log, so a `last` run's per-layer log looks like every other run's.
    """
    losses::Vector{Float32} = base_loss_per_layer(
        posterior_llrs, expected_recoveries, parity_check_matrix_dual, warmup_loss_layers)
    total::Float32 = combine_layer_losses(losses, loss_layer_temperature, loss_layer_selection)
    return total
end

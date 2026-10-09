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

# =============================================================================
#   The per-check residue: which function of the real-valued syndrome bit
# =============================================================================
# Both residues are zero iff x = s(μ) mod 2 is an even integer, i.e. iff the
# soft decision commutes with that generator, and both peak at the odd integers.
# They differ in the GRADIENT at the peak, and that difference is the whole
# reason the second one exists.
#
#   sin_residue   |sin(πx/2)|.  d/dx = (π/2)cos(πx/2) = 0 EXACTLY at x = 1, the
#                 confidently-violated check. Measured 2026-10-08: on every
#                 informative training batch the per-layer loss plateaus at an
#                 exact multiple of 1/20 (0.30 = six violated checks per batch of
#                 20) from layer ~40 onward — BP at a stable fixed point with the
#                 bits saturated, x within 1e-3 of 1. There dL/dμ =
#                 residue'(x)·σ'(μ) is (π/2)cos(πx/2)·σ'(μ) ≈ 1e-6 at |μ| = 8 and
#                 3e-8 at |μ| = 10 — at or below Float32 noise on a 0.3 loss. The
#                 stuck checks sit at a critical point of the loss, and no layer
#                 weighting changes that.
#   smooth_loss   piecewise quadratic, x² for x ≤ 1 and (2−x)² above, with
#                 SUBGRADIENT ±2 at x = 1 toward the nearest even integer. At the
#                 same stuck check dL/dμ ≈ 2·σ'(μ): 7e-4 at |μ| = 8, 9e-5 at
#                 |μ| = 10 — 400 to 3000× the sine's, and above noise until
#                 |μ| ≈ 12. Adam normalises magnitude, so a consistent sign above
#                 noise is exactly what the sine was failing to provide.
#
# `sin_residue` stays the default so every earlier run reproduces exactly.

const BASE_LOSS_SIN_RESIDUE::Int = 0
const BASE_LOSS_SMOOTH::Int = 1

function base_loss_code(base_loss_name::AbstractString)::Int
    """
    Resolve the `base_loss` hyperparameter to its integer code, so a typo fails
    at configuration time rather than inside a training loop.
    """
    normalised_name::String = lowercase(strip(String(base_loss_name)))
    base_loss_selection::Int = BASE_LOSS_SIN_RESIDUE
    if normalised_name == "sin_residue"
        base_loss_selection = BASE_LOSS_SIN_RESIDUE
    elseif normalised_name == "smooth_loss"
        base_loss_selection = BASE_LOSS_SMOOTH
    else
        throw(ArgumentError(
            "Unknown base_loss \"$(base_loss_name)\". Supported: \"sin_residue\" (the " *
            "historical default, |sin(πx/2)| with zero gradient at a violated check) and " *
            "\"smooth_loss\" (piecewise quadratic with subgradient ±2 there)."))
    end
    return base_loss_selection
end

function base_loss_name(base_loss_selection::Int)::String
    """
    Inverse of `base_loss_code`, for recording the residue a run used.
    """
    base_loss_label::String = "sin_residue"
    if base_loss_selection == BASE_LOSS_SMOOTH
        base_loss_label = "smooth_loss"
    elseif base_loss_selection != BASE_LOSS_SIN_RESIDUE
        throw(ArgumentError("Unknown base loss code $(base_loss_selection)."))
    end
    return base_loss_label
end

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
    parity_check_matrix_dual::BitMatrix,
    base_loss_selection::Int = BASE_LOSS_SIN_RESIDUE
)::Float32
    """
    Batch-mean residue of the total error against the dual check matrix:

        L(μ, e) = (1/N) ∑_j ∑_i g( ∑_k H^⟂_ik [ e_kj + σ(μ_kj) ] )

    where H^⟂ carries the stabilizer generators and the logical operators, so a
    zero requires both a cleared syndrome and the correct coset. `g` is the
    residue `base_loss_selection` names: `sine_residue_loss` (default) or
    `smooth_loss`; see the block above for why the choice matters.
    """
    n_samples::Int = size(expected_recoveries, 2)
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    summed_residue::Float32 = 0.0f0
    if base_loss_selection == BASE_LOSS_SIN_RESIDUE
        summed_residue = sum(@. sine_residue_loss(commutation_relations_matrix))
    elseif base_loss_selection == BASE_LOSS_SMOOTH
        summed_residue = sum(@. smooth_loss(commutation_relations_matrix))
    else
        throw(ArgumentError("Unknown base loss code $(base_loss_selection)."))
    end
    average_loss::Float32 = summed_residue / Float32(n_samples)
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
    warmup_loss_layers::Int,
    base_loss_selection::Int = BASE_LOSS_SIN_RESIDUE
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
        losses[layer - warmup_loss_layers] = compute_smooth_loss_from_llrs(
            post, expected_recoveries, parity_check_matrix_dual, base_loss_selection)
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
#            MEASURED 2026-10-08 on real per-layer profiles: a flat mean
#            separates solved from unsolved batches by only 2x, because 100% of a
#            solved batch's mean comes from layers 1-4 (the BP transient: 35,
#            7.5, 1.1, 0.24, 0.12, then the floor). It spends its gradient
#            teaching one-iteration BP to decode. Kept for comparison; do not
#            expect it to train well.
#   RAMP     every scored layer, weighted by a saturating ramp that is zero
#            through `warmup_layers`, rises, and reaches exactly 1 at the last
#            layer:
#
#                w_t = tanh(k * u) / tanh(k),   u = (t - warmup) / (n_layers - warmup)
#
#            and the loss is the w-weighted MEAN of the scored base losses, so its
#            scale matches a single layer's. k is `loss_layer_ramp_sharpness`.
#            Why this and not `last`: once a sample has converged the remaining
#            layers sit at a BP fixed point whose backward Jacobian is
#            contractive, so the last layer's gradient reaching layer t decays
#            like rho^(n_layers - t) — early weights go effectively unconstrained
#            and the per-layer loss rises again after its minimum on about half
#            the batches. A weight on every layer gives each one an
#            un-attenuated gradient from its own term and penalises that rise.
#            Why the warmup: the first ~5 layers are the BP transient and no
#            weight can lower them; any ramp that lets them in is dominated by
#            them (linear 18x separation, cubic 5,800x; with them excluded, any
#            shape is ~110,000x). The transient length is a property of BP
#            convergence, not of n_layers, so warmup 5 should transfer across
#            layer counts. Pairs with `commit_layer_rule = "last"`: the ramp
#            emphasises the last layer and trains against the drift that the
#            last-layer readout pays for. k -> 0 is a linear ramp; k large is a
#            step to uniform-after-warmup.
#
# `softmin` stays the default so every earlier run reproduces exactly.

const LOSS_LAYERS_SOFTMIN::Int = 0
const LOSS_LAYERS_LAST::Int = 1
const LOSS_LAYERS_MEAN::Int = 2
const LOSS_LAYERS_RAMP::Int = 3

# Default sharpness of the ramp. At n_layers = 90 and warmup = 5 this puts w = 0.5
# at layer 21 and w >= 0.9 from layer 47, i.e. the whole second half of the decoder
# is scored at nearly full weight. (At 50 layers: 0.5 at 14, 0.9 from 27.)
const DEFAULT_LOSS_LAYER_RAMP_SHARPNESS::Float32 = 3.0f0

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
    elseif normalised_name == "ramp"
        selection_code = LOSS_LAYERS_RAMP
    else
        throw(ArgumentError(
            "Unknown loss_layer_selection \"$(selection_name)\". Supported: " *
            "\"softmin\" (the historical default, min over layers as T -> 0), " *
            "\"last\" (the final layer alone), \"mean\" (every scored layer, equally) " *
            "and \"ramp\" (every scored layer, weighted by a saturating ramp that is " *
            "0 through warmup_layers and 1 at the last layer)."))
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
    elseif selection_code == LOSS_LAYERS_RAMP
        selection_label = "ramp"
    elseif selection_code != LOSS_LAYERS_SOFTMIN
        throw(ArgumentError("Unknown loss layer selection code $(selection_code)."))
    end
    return selection_label
end

function ramp_layer_weight(scored_layer_index::Int, n_scored_layers::Int,
                           loss_layer_ramp_sharpness::Float32)::Float32
    """
    Weight of the `scored_layer_index`-th SCORED layer (1 = the first layer after
    warmup, `n_scored_layers` = the last layer of the decoder):

        w = tanh(k * u) / tanh(k),   u = scored_layer_index / n_scored_layers

    so w(n_scored_layers) = 1 exactly, w -> 0 as the index -> 0, and the layers
    inside the warmup never reach this function at all (they were dropped by
    `base_loss_per_layer`). k = `loss_layer_ramp_sharpness`. Written against the
    scored index rather than the absolute layer so it needs neither `warmup` nor
    `n_layers`: u is the same number either way.
    """
    if n_scored_layers <= 0
        throw(ArgumentError("ramp_layer_weight: n_scored_layers = $(n_scored_layers) must be positive."))
    end
    if loss_layer_ramp_sharpness <= 0.0f0
        throw(ArgumentError(
            "ramp_layer_weight: loss_layer_ramp_sharpness = $(loss_layer_ramp_sharpness) " *
            "must be positive (k -> 0 is a linear ramp; use a small positive value)."))
    end
    fraction_of_scored_depth::Float32 = Float32(scored_layer_index) / Float32(n_scored_layers)
    weight::Float32 = tanh(loss_layer_ramp_sharpness * fraction_of_scored_depth) /
                      tanh(loss_layer_ramp_sharpness)
    return weight
end

function ramp_loss(losses_per_layer::AbstractVector{Float32},
                   loss_layer_ramp_sharpness::Float32)::Float32
    """
    The `ramp_layer_weight`-weighted MEAN of the scored per-layer losses:

        ramp = sum_i w_i * L_i  /  sum_i w_i

    Dividing by the weight sum keeps the value on the scale of a single layer's
    loss (like `last` and `mean`), so the optimiser hyperparameters carry over.
    The weights are constants with respect to the model, so Enzyme sees a plain
    weighted sum.
    """
    n_scored_layers::Int = length(losses_per_layer)
    weighted_sum::Float32 = 0.0f0
    weight_sum::Float32 = 0.0f0
    for scored_layer_index in 1:n_scored_layers
        weight::Float32 = ramp_layer_weight(scored_layer_index, n_scored_layers, loss_layer_ramp_sharpness)
        weighted_sum += weight * losses_per_layer[scored_layer_index]
        weight_sum += weight
    end
    weighted_mean::Float32 = weighted_sum / weight_sum
    return weighted_mean
end

function combine_layer_losses(
    losses_per_layer::AbstractVector{Float32},
    loss_layer_temperature::Float32,
    loss_layer_selection::Int,
    loss_layer_ramp_sharpness::Float32 = DEFAULT_LOSS_LAYER_RAMP_SHARPNESS
)::Float32
    """
    Reduce the per-layer losses to the one number the optimiser sees, by the
    rule `loss_layer_selection` names. The temperature is read only by
    `softmin` and the sharpness only by `ramp`; the others ignore both.
    """
    combined::Float32 = 0.0f0
    if loss_layer_selection == LOSS_LAYERS_SOFTMIN
        combined = softmin_loss(losses_per_layer, loss_layer_temperature)
    elseif loss_layer_selection == LOSS_LAYERS_LAST
        combined = losses_per_layer[end]
    elseif loss_layer_selection == LOSS_LAYERS_MEAN
        combined = sum(losses_per_layer) / Float32(length(losses_per_layer))
    elseif loss_layer_selection == LOSS_LAYERS_RAMP
        combined = ramp_loss(losses_per_layer, loss_layer_ramp_sharpness)
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
    loss_layer_selection::Int = LOSS_LAYERS_SOFTMIN,
    loss_layer_ramp_sharpness::Float32 = DEFAULT_LOSS_LAYER_RAMP_SHARPNESS,
    base_loss_selection::Int = BASE_LOSS_SIN_RESIDUE
)::Float32
    """
    Total per-batch loss: the scored layers' base losses (the per-check residue
    `base_loss_selection` names, summed over [H; L]), combined by
    `loss_layer_selection` (see `combine_layer_losses`).

        softmin:  total = softmin_T( base_(warmup+1), ..., base_(n_layers) )
        last:     total = base_(n_layers)
        mean:     total = mean( base_(warmup+1), ..., base_(n_layers) )
        ramp:     total = sum_t w_t base_t / sum_t w_t   over t = warmup+1 .. n_layers,
                  w_t = tanh(k u_t)/tanh(k),  u_t = (t - warmup)/(n_layers - warmup)

    `posterior_llrs` is (n_bits × n_samples × n_layers), the readout of every
    layer of the unrolled decoder.

    NOTE on `last`: `warmup_loss_layers` becomes irrelevant, because only the
    final layer is read. It is still honoured for the vector the diagnostics
    log, so a `last` run's per-layer log looks like every other run's.
    NOTE on `ramp`: `warmup_loss_layers` is where the ramp STARTS from zero, so
    it matters here. 5 excludes the BP transient (measured 2026-10-08).
    """
    losses::Vector{Float32} = base_loss_per_layer(
        posterior_llrs, expected_recoveries, parity_check_matrix_dual, warmup_loss_layers,
        base_loss_selection)
    total::Float32 = combine_layer_losses(
        losses, loss_layer_temperature, loss_layer_selection, loss_layer_ramp_sharpness)
    return total
end

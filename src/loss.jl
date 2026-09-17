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
    average_loss = sum(@. smooth_loss(commutation_relations_matrix)) / n_samples
    return average_loss
end

function softmin_loss(losses_per_layer::AbstractVector{Float32}, temp::Float32)::Float32
    """
    Smooth minimum over layers, T·log(n) above the true minimum at zero spread:

        softmin(L, T) = min(L) - T · log( ∑_l exp( -(L_l - min(L)) / T ) )

    T is annealed down, so early training averages over layers and late training
    commits to the best one. The gradient is the softmax weights, all ≥ 0, so
    lowering the selected layer's loss never pushes up another's.
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

function compute_loss(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int
)::Float32
    """
    Total per-batch loss: the softmin over scored layers of the base loss.

        total = softmin_T( base_(warmup+1), ..., base_(n_layers) )

    `posterior_llrs` is (n_bits × n_samples × n_layers), the readout of every
    layer of the unrolled decoder.
    """
    losses::Vector{Float32} = base_loss_per_layer(
        posterior_llrs, expected_recoveries, parity_check_matrix_dual, warmup_loss_layers)
    total::Float32 = softmin_loss(losses, loss_layer_temperature)
    return total
end

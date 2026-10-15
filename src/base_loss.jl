# =============================================================================
#   The base loss: the options for the per-check residue, with their derivatives
# =============================================================================
# The base loss of one layer is the batch-mean residue of the total error against
# the dual check matrix [H; L]:
#
#     L_t(μ) = (1/N) ∑_j ∑_i g( x_ij ),     x = H⊥ ( σ(μ) + e )
#
# with μ the layer's posterior LLRs (n_bits × N), e the true errors, and σ this
# codebase's sigmoid σ(μ) = 1/(1 + e^μ) — the probability that a bit is in error.
# x_ij is the real-valued syndrome bit of row i for sample j; g is the residue,
# zero exactly at the even integers (row satisfied) and peaked at the odd ones.
#
# Each option is a PAIR: the residue g and its derivative g′. The derivative of
# the whole base loss with respect to the posterior LLRs is then, by the chain
# rule through x = H⊥(σ(μ) + e),
#
#     ∂L_t/∂μ_kj = (1/N) σ′(μ_kj) ∑_i H⊥_ik g′(x_ij)
#
# — `compute_base_loss_gradient`. It is the SEED of the hand-written backward
# pass (src/gradient.jl): the only place the loss itself enters the gradient.
#
# Adding an option = a residue function, its derivative, a code below, and one
# branch in each of `compute_base_loss` and `compute_base_loss_gradient`.
#
# `compute_base_loss` was `compute_smooth_loss_from_llrs` in src/loss.jl, moved
# here unchanged (2026-10-09); that name remains as an alias, so every caller and
# the Enzyme-differentiated training loss see the identical computation.
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

# =============================================================================
#   Their derivatives
# =============================================================================
# Each written in the order Enzyme's reverse pass evaluates the corresponding
# forward expression, so the hand-written gradient and Enzyme's agree to a few
# ulp rather than merely to the same mathematics.

function smooth_loss_derivative(real_syndrome_bit::Float32)::Float32
    """
    g′(x) of `smooth_loss`, on x̃ = x mod 2 (floor has zero derivative, so
    dx̃/dx = 1):

        g′ = 2 x̃          for 0 ≤ x̃ ≤ 1
           = -2 (2 - x̃)   for 1 < x̃ ≤ 2

    i.e. 2 × the signed distance to the nearest even integer. At x̃ = 1 the left
    branch is taken (+2), as in `smooth_loss`.
    """
    reduced_bit::Float32 = floating_modulus(real_syndrome_bit)
    derivative::Float32 = 0.0f0
    if 0 <= reduced_bit <= 1
        derivative = 2.0f0 * reduced_bit
    elseif 1 < reduced_bit <= 2
        derivative = -2.0f0 * (2.0f0 - reduced_bit)
    else
        error("smooth_loss_derivative is only defined for 0 ≤ x mod 2 ≤ 2")
    end
    return derivative
end

function sine_residue_derivative(real_syndrome_bit::Float32)::Float32
    """
    g′(x) of `sine_residue_loss`, g = |sin(z)| with z = π x / 2:

        g′ = sign(sin z) · cos z · π/2

    zero at the odd integers (the peaks), a cusp of slope ±π/2 at the even ones.
    """
    angle::Float32 = pi * real_syndrome_bit / 2
    derivative::Float32 = Float32(pi) * (sign(sin(angle)) * cos(angle) / 2)
    return derivative
end

function sigmoid_derivative(x::Float32)::Float32
    """
    d/dx of this codebase's σ(x) = 1/(1 + e^x) (note the sign: σ is the
    probability of an ERROR, decreasing in the LLR):

        σ′(x) = -e^x / (1 + e^x)² = -σ(x) · σ(-x)

    The product form is used because it is accurate everywhere and never NaN:
    -σ(x)(1 - σ(x)) loses all relative accuracy for x ≲ -10 (1 - σ cancels), and
    -e^x/(1 + e^x)², what Enzyme evaluates, overflows to Inf/Inf = NaN beyond
    x ≈ 88.7. Where both are finite they agree to a few ulp for |x| < 44.
    """
    derivative::Float32 = -(sigmoid(x) * sigmoid(-x))
    return derivative
end

# =============================================================================
#   The base loss of one layer, and its gradient with respect to the LLRs
# =============================================================================

function compute_base_loss(
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

function compute_base_loss_gradient(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    base_loss_selection::Int = BASE_LOSS_SIN_RESIDUE
)::Matrix{Float32}
    """
    ∂L/∂μ for `compute_base_loss`, one entry per posterior LLR (n_bits × N):

        ∂L/∂μ_kj = (1/N) · σ′(μ_kj) · ∑_i H⊥_ik g′(x_ij),    x = H⊥ (σ(μ) + e)

    the reverse of the three forward steps: the mean (1/N on every residue), the
    residue (g′ at every real-valued syndrome bit), the product with H⊥ (its
    transpose), and the sigmoid (σ′ at every LLR). The errors e are constants.
    """
    n_samples::Int = size(expected_recoveries, 2)
    e_total_matrix::Matrix{Float32} = sigmoid.(posterior_llrs) .+ expected_recoveries
    commutation_relations_matrix::Matrix{Float32} = parity_check_matrix_dual * e_total_matrix
    residue_derivatives::Matrix{Float32} = similar(commutation_relations_matrix)
    if base_loss_selection == BASE_LOSS_SIN_RESIDUE
        residue_derivatives .= sine_residue_derivative.(commutation_relations_matrix)
    elseif base_loss_selection == BASE_LOSS_SMOOTH
        residue_derivatives .= smooth_loss_derivative.(commutation_relations_matrix)
    else
        throw(ArgumentError("Unknown base loss code $(base_loss_selection)."))
    end
    # d(summed / N)/d(summed) = 1/N, applied to every residue.
    adjoint_per_residue::Float32 = 1.0f0 / Float32(n_samples)
    adjoint_commutation::Matrix{Float32} = residue_derivatives .* adjoint_per_residue
    adjoint_e_total::Matrix{Float32} = transpose(parity_check_matrix_dual) * adjoint_commutation
    gradient::Matrix{Float32} = adjoint_e_total .* sigmoid_derivative.(posterior_llrs)
    return gradient
end

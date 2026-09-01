# =============================================================================
# Loss for the unrolled neural BP decoder.
#
#     total = softmin_over_layers( base_l )
#           + mean_over_layers( mean_over_samples( g_(l,j) · aux_(l,j) ) )
#
# `base` is the residue of e + σ(μ) against [H; L]; it is the only term that
# identifies the correct answer. The auxiliary terms select within its zero set
# and are applied through a detached gate that opens only on samples whose soft
# H-syndrome is already (near-)satisfied. Layer selection sees `base` alone, so a
# layer cannot win by being confident or co-activating instead of clearing the
# syndrome.
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

# -----------------------------------------------------------------------------
# Per-sample quantities. Each returns one value per sample so the gate can be
# applied sample-by-sample before the batch mean is taken.
# -----------------------------------------------------------------------------

function soft_syndrome_weight_per_sample(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix
)::Vector{Float32}
    """
    |s|_j = ∑_checks |sin( (π/2) · [H (e + σ(μ))]_(check, j) )|

    in units of softly broken checks: a satisfied check contributes 0, a fully
    broken one 1. Uses the STABILIZERS ONLY, not [H; L] — the wrong-coset region
    is where the prior has to cooperate with the logical rows of `base`, so it
    must be inside the gate.
    """
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix * e_total_matrix
    syndrome_weights = vec(sum(abs.(sin.((π / 2) .* commutation_relations_matrix)); dims = 1))
    return syndrome_weights
end

function syndrome_gate_per_sample(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix,
    syndrome_gate_threshold::Float32
)::Vector{Float32}
    """
    g_j = 1[ |s|_j < τ ], τ in units of softly broken checks.

    `s .< τ` yields Bools, whose derivative is zero almost everywhere, so no
    gradient flows through the gate. That is required, not incidental: a
    differentiable gate would let the optimizer lower gate×aux by RAISING |s|,
    i.e. by breaking the syndrome to escape a penalty or holding it broken to
    farm a reward.
    """
    s = soft_syndrome_weight_per_sample(posterior_llrs, expected_recoveries, parity_check_matrix)
    gate = Float32.(s .< syndrome_gate_threshold)
    return gate
end

# Integer codes for the certainty penalty. An Int rather than a Symbol or a
# function value because this is passed to Enzyme as a `Const`, and the branch on
# it is lifted OUT of the per-element work below: the three broadcasts are each
# branchless, so the choice costs nothing per element and Enzyme differentiates
# whichever branch was taken exactly as if it were the only one.
const CERTAINTY_PENALTY_ENTROPY::Int     = 1
const CERTAINTY_PENALTY_EXPONENTIAL::Int = 2
const CERTAINTY_PENALTY_HINGE::Int       = 3

function certainty_penalty_code(name::String)::Int
    """
    Map the `certainty_penalty` hyperparameter string onto its integer code.
    Throws on an unknown name rather than silently falling back, so a typo in a
    sweep TOML fails at startup instead of quietly training the default.
    """
    normalised_name::String = lowercase(strip(name))
    codes_by_name::Dict{String, Int} = Dict(
        "entropy"     => CERTAINTY_PENALTY_ENTROPY,
        "exponential" => CERTAINTY_PENALTY_EXPONENTIAL,
        "hinge"       => CERTAINTY_PENALTY_HINGE
    )
    if !haskey(codes_by_name, normalised_name)
        throw(ArgumentError(
            "Unknown certainty_penalty \"$(name)\". Must be one of: " *
            join(sort(collect(keys(codes_by_name))), ", ") * "."
        ))
    end
    code::Int = codes_by_name[normalised_name]
    return code
end

function certainty_per_sample(
    posterior_llrs::Matrix{Float32},
    certainty_penalty_kind::Int,
    certainty_hinge_width::Float32
)::Vector{Float32}
    """
    Per-sample certainty penalty ∑_v f(μ_v). Every choice of f is symmetric in μ,
    maximal at μ = 0 and decaying to 0 as |μ| → ∞, so all of them reward
    certainty; they differ in the force they exert on an undecided qubit.

        ENTROPY      h(σ(μ)), in nats. Max n_bits·log 2 at σ = 0.5. Symmetry
                     forces dh/dμ = 0 exactly at σ = 0.5, so a perfect fractional
                     alias is an unstable equilibrium this term does not actively
                     repair; its force peaks out at |μ| ≈ 2.4 instead.
        EXPONENTIAL  exp(-|μ|). Cusped at 0, so its force is LARGEST precisely
                     where the entropy's is zero.
        HINGE        max(0, 1 - |μ|/w). Constant force 1/w inside the width and
                     none outside, so it repairs aliases without also inflating
                     LLRs that are already decided.
    """
    penalties::Matrix{Float32} = similar(posterior_llrs)
    if certainty_penalty_kind == CERTAINTY_PENALTY_ENTROPY
        penalties = binary_entropy_of_sigmoid.(posterior_llrs)
    elseif certainty_penalty_kind == CERTAINTY_PENALTY_EXPONENTIAL
        penalties = exponential_certainty_penalty.(posterior_llrs)
    elseif certainty_penalty_kind == CERTAINTY_PENALTY_HINGE
        penalties = hinge_certainty_penalty.(posterior_llrs, certainty_hinge_width)
    else
        throw(ArgumentError("Unknown certainty_penalty_kind: $(certainty_penalty_kind)."))
    end
    certainties::Vector{Float32} = vec(sum(penalties; dims = 1))
    return certainties
end

function sparsity_per_sample(posterior_llrs::Matrix{Float32})::Vector{Float32}
    """
    Per-sample predicted error weight ∑_v σ(μ_v). Ordering constraint: α₃ times a
    typical error weight must stay ≲ 1, or a gated solved sample scores worse
    than a failing one.
    """
    weights_of_predictions = vec(sum(sigmoid.(posterior_llrs); dims = 1))
    return weights_of_predictions
end

function ising_correlation_reward_per_sample(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    certainty_threshold::Float32
)::Vector{Float32}
    """
    Co-activation reward over the correlated pairs C, restricted to pairs whose
    BOTH endpoints have decided:

        d_(i,k),j = 1[ min(|μ_(qi,j)|, |μ_(qk,j)|) > c ]
        r_j       = - ( ∑_((qi,qk) ∈ C) d · J_(i,k) σ(μ_(qi,j)) σ(μ_(qk,j)) )
                    / max(1, ∑_((qi,qk) ∈ C) d)

    THE CERTAINTY GATE. Without it the term's gradient,
    ∂/∂μ_i [σ_i σ_k] = σ_i(1-σ_i)σ_k, is maximal at σ_i ≈ 0.5 and vanishes as
    σ → 1: the term would do almost none of its work selecting among decided
    configurations and almost all of it pushing undecided qubits upward, which
    is the opposite of a selector within the solution set. `d` is an indicator
    and therefore detached, for the same reason as the syndrome gate — a smooth
    decidedness weight w(σ) contributes a (dw/dμ)·r term that dominates the
    honest (dr/dμ) term by 4-12x and rewards becoming certain irrespective of
    which way the qubit leans.

    NORMALISATION. Dividing by the number of ACTIVE pairs rather than |C| makes
    r_j a mean over contributing pairs. Under 1/|C| the term is divided by every
    edge in the graph while only the co-flipped ones contribute — for the 72q
    code that is 540 against ≈ 0.19, a dilution of ~2800x that grows with code
    size, so λ would need retuning per code. λ is not comparable across the two
    normalisations.

    r_j ≤ 0 wherever the couplings are positive, so gating it on solved samples
    can never make a solved sample lose to a failing one: rewards are
    ordering-safe by construction.
    """
    n_samples = size(posterior_llrs, 2)
    n_edges = size(connectivity, 1)
    rewards = zeros(Float32, n_samples)
    # BRANCHLESS ON PURPOSE. The gate is Float32 of a comparison — zero derivative,
    # so it is detached exactly like `syndrome_gate_per_sample` — and it MULTIPLIES
    # rather than branching. An earlier version used `continue` plus a division by
    # an Int active-pair count; under Enzyme reverse mode that produced non-finite
    # gradients on every batch, which NaN-skipped every update and rolled back
    # every epoch, silently shipping untrained weights. Straight-line Float32
    # arithmetic is the form with a proven training record.
    @inbounds for j in 1:n_samples
        accumulated_reward::Float32 = 0.0f0
        active_pair_count::Float32 = 0.0f0
        for e in 1:n_edges
            i = connectivity[e, 1]
            k = connectivity[e, 2]
            decided::Float32 = Float32(
                (abs(posterior_llrs[i, j]) > certainty_threshold) &
                (abs(posterior_llrs[k, j]) > certainty_threshold)
            )
            active_pair_count += decided
            accumulated_reward += decided * correlation_strengths[e] *
                                  sigmoid(posterior_llrs[i, j]) * sigmoid(posterior_llrs[k, j])
        end
        rewards[j] = -accumulated_reward / max(1.0f0, active_pair_count)
    end
    return rewards
end

# Integer codes for the correlation term. Same Enzyme reasoning as the certainty
# penalty codes above: passed as a `Const`, branched on OUTSIDE the elementwise
# work, each branch straight-line Float32.
const CORRELATION_FORM_BILINEAR::Int      = 1
const CORRELATION_FORM_LOG_AGREEMENT::Int = 2

function correlation_form_code(name::String)::Int
    """
    Map the `correlation_form` hyperparameter string onto its integer code.
    Throws on an unknown name so a typo in a sweep TOML fails at startup rather
    than quietly training the historical default.
    """
    normalised_name::String = lowercase(strip(name))
    codes_by_name::Dict{String, Int} = Dict(
        "bilinear"      => CORRELATION_FORM_BILINEAR,
        "log_agreement" => CORRELATION_FORM_LOG_AGREEMENT
    )
    if !haskey(codes_by_name, normalised_name)
        throw(ArgumentError(
            "Unknown correlation_form \"$(name)\". Must be one of: " *
            join(sort(collect(keys(codes_by_name))), ", ") * "."
        ))
    end
    code::Int = codes_by_name[normalised_name]
    return code
end

function ising_log_agreement_penalty_per_sample(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    certainty_threshold::Float32,
    agreement_floor::Float32
)::Vector{Float32}
    """
    Weighted negative log-likelihood of pair CONCORDANCE:

        t_i        = tanh(μ_i / 2)                     (= 1 - 2σ(μ_i), the magnetisation)
        A_(i,k),j  = ( 1 + sgn(J) · t_i · t_k ) / 2
        r_j        = ( ∑ d · |J_(i,k)| · (-log max(A, ε)) ) / max(1, ∑ d)

    WHY THIS SHAPE. (1 + t_i t_k)/2 is exactly P(pair agrees) = p_i p_k +
    (1-p_i)(1-p_k). The bilinear form σ_i σ_k has ∂/∂μ_i = σ_i(1-σ_i)σ_k, which
    vanishes at ALL FOUR corners because σ(1-σ) → 0 for either sign of μ: it can
    reward a configuration it likes but exerts no force to REACH it. Here the
    log's 1/(1+t_i t_k) divergence exactly cancels the (1 - t_i²) saturation, so
    the gradient tends to |J| at the two DISCORDANT corners and to 0 at the two
    concordant ones — force precisely where a pair is in the wrong configuration.

    WHY sgn(J) IS INSIDE AND |J| OUTSIDE. With a raw -J log A and J < 0 the term
    is unbounded BELOW: the optimizer wins arbitrarily by driving anti-correlated
    pairs to A → 0, ignoring the syndrome. About a quarter of the measured
    couplings are negative, so that escape is real, not hypothetical. Folding the
    sign into the argument puts the barrier on agreement for J < 0 and on
    disagreement for J > 0, leaving r_j ≥ 0 always: zero where the coupling is
    satisfied, growing where it is violated.

    Not ordering-safe the way the bilinear reward was. That one was ≤ 0, so a
    gated solved sample could never lose to a failing one. This is a PENALTY, so
    a solved sample carrying a violated coupling scores above a solved sample
    without one. That is intended — it is what gives the term force — but it
    means λ has to stay small enough that L1's separation between solved and
    unsolved is not overturned.

    ε (`agreement_floor`) IS MANDATORY. tanh(μ/2) saturates to exactly 1.0f0 in
    Float32 by |μ| ≈ 18, so A hits exactly 0 and log(0) = -Inf, giving a NaN
    gradient, a NaN-skipped batch, a rolled-back epoch and silently untrained
    weights. Clamping is what stops that.
    """
    n_samples = size(posterior_llrs, 2)
    n_edges = size(connectivity, 1)
    penalties = zeros(Float32, n_samples)
    # Branchless, for the reason documented on `ising_correlation_reward_per_sample`.
    @inbounds for j in 1:n_samples
        accumulated_penalty::Float32 = 0.0f0
        active_pair_count::Float32 = 0.0f0
        for e in 1:n_edges
            i = connectivity[e, 1]
            k = connectivity[e, 2]
            decided::Float32 = Float32(
                (abs(posterior_llrs[i, j]) > certainty_threshold) &
                (abs(posterior_llrs[k, j]) > certainty_threshold)
            )
            active_pair_count += decided
            coupling::Float32 = correlation_strengths[e]
            magnetisation_product::Float32 =
                tanh(0.5f0 * posterior_llrs[i, j]) * tanh(0.5f0 * posterior_llrs[k, j])
            concordance::Float32 =
                0.5f0 * (1.0f0 + sign(coupling) * magnetisation_product)
            accumulated_penalty += decided * abs(coupling) *
                                   (-log(max(concordance, agreement_floor)))
        end
        penalties[j] = accumulated_penalty / max(1.0f0, active_pair_count)
    end
    return penalties
end

function correlation_term_per_sample(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    certainty_threshold::Float32,
    correlation_form::Int,
    agreement_floor::Float32
)::Vector{Float32}
    """
    Dispatch to the selected correlation term. BILINEAR is the historical
    co-activation reward (≤ 0); LOG_AGREEMENT is the concordance penalty (≥ 0).
    They differ in sign as well as shape, so λ is NOT comparable between them.
    """
    if correlation_form == CORRELATION_FORM_BILINEAR
        bilinear_rewards::Vector{Float32} = ising_correlation_reward_per_sample(
            posterior_llrs, connectivity, correlation_strengths, certainty_threshold
        )
        return bilinear_rewards
    elseif correlation_form == CORRELATION_FORM_LOG_AGREEMENT
        agreement_penalties::Vector{Float32} = ising_log_agreement_penalty_per_sample(
            posterior_llrs, connectivity, correlation_strengths,
            certainty_threshold, agreement_floor
        )
        return agreement_penalties
    else
        throw(ArgumentError("Unknown correlation_form: $(correlation_form)."))
    end
end

function correlation_gate_open_fraction(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    certainty_threshold::Float32
)::Float32
    """
    Fraction of (pair, sample) slots the certainty gate admits. Diagnostic only,
    never differentiated. At 0 the correlation term never fired and a null result
    says nothing about the couplings; at 1 the gate is not confining anything.
    """
    n_samples = size(posterior_llrs, 2)
    n_edges = size(connectivity, 1)
    if n_edges == 0 || n_samples == 0
        return 0.0f0
    end
    open_count::Float32 = 0.0f0
    @inbounds for j in 1:n_samples
        for e in 1:n_edges
            i = connectivity[e, 1]
            k = connectivity[e, 2]
            open_count += Float32(
                (abs(posterior_llrs[i, j]) > certainty_threshold) &
                (abs(posterior_llrs[k, j]) > certainty_threshold)
            )
        end
    end
    open_fraction::Float32 = open_count / Float32(n_edges * n_samples)
    return open_fraction
end

function compute_loss_including_correlations(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    is_correlated::Bool,
    correlation_weight::Float32,                # λ, overall weight on the correlation term
    loss_layer_temperature::Float32,
    llr_certainty_importance::Float32,
    sparsity_importance::Float32,
    syndrome_gate_threshold::Float32,           # τ, in softly broken checks
    certainty_syndrome_gate_threshold::Float32, # τ₂ for L2 alone; < 0 inherits τ
    correlation_certainty_threshold::Float32,   # c, in LLR units
    certainty_penalty_kind::Int,                # which f in `certainty_per_sample`
    certainty_hinge_width::Float32,             # w, only used by the hinge penalty
    correlation_form::Int,                      # which L3 in `correlation_term_per_sample`
    correlation_agreement_floor::Float32,       # ε, only used by the log-agreement form
    warmup_loss_layers::Int
)::Float32
    """
    Total per-batch loss.

        total = softmin_over_layers( base_l )
              + mean_over_layers( mean_over_samples(
                    g2_(l,j) · α_cert · cert_j
                  + g_(l,j)  · (λ · corr_j + α₃ · sparse_j) ) )

        g_(l,j)  = 1[ soft H-syndrome weight of sample j at layer l < τ  ]
        g2_(l,j) = 1[ soft H-syndrome weight of sample j at layer l < τ₂ ]

    TWO SYNDROME GATES, because τ and the certainty penalty are not independent
    knobs. L2 fires hardest on qubits near μ = 0, but a single qubit at σ ≈ 0.5
    sits in 3 Z-checks (HZ column weight 3) and contributes |sin(π/2 · 0.5)| =
    0.707 to each, driving |s| ≈ 2.1 — four times τ = 0.5. So at a shared gate
    the samples L2 most wants to fix are precisely the ones the gate excludes. A
    narrow hinge measured 0.000 gated contribution across 200 000 layer-samples
    for exactly this reason: not small, identically zero. τ₂ decouples them.

    τ₂ < 0 inherits τ, which reproduces every run made before this split, bit for
    bit. Use τ₂ = 1e6 to let L2 act everywhere while L3 stays confined to solved
    samples. Note a threshold of exactly 0 means "never open" (|s| ≥ 0 always),
    which is how you would switch a term off rather than inherit.

    The auxiliary terms act on the flat solution manifold of `base`, where `base`
    provides no gradient and they are the only forces. `corr_j` carries a second,
    per-pair gate on `c`; see `ising_correlation_reward_per_sample`.

    All hyperparameters are plain `Float32` scalars.
    """
    n_layers = size(posterior_llrs, 3)
    n_scored = n_layers - warmup_loss_layers
    n_samples = size(posterior_llrs, 2)

    base_per_layer      = zeros(Float32, n_scored)
    gated_aux_per_layer = zeros(Float32, n_scored)
    for layer in (warmup_loss_layers + 1):n_layers
        post = posterior_llrs[:, :, layer]
        base_per_layer[layer - warmup_loss_layers] =
            compute_smooth_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)

        # One soft-syndrome pass, two thresholds. The weight is the expensive part
        # (a parity-check matrix multiply), and thresholding it twice costs a
        # comparison, so the split is free rather than doubling the gate cost.
        syndrome_weights = soft_syndrome_weight_per_sample(
            post, expected_recoveries, parity_check_matrix
        )
        effective_certainty_threshold::Float32 = certainty_syndrome_gate_threshold
        if certainty_syndrome_gate_threshold < 0.0f0
            effective_certainty_threshold = syndrome_gate_threshold
        end
        # Detached, exactly as before: comparisons have zero derivative, so the
        # optimizer cannot lower gate x aux by breaking the syndrome.
        gate           = Float32.(syndrome_weights .< syndrome_gate_threshold)
        certainty_gate = Float32.(syndrome_weights .< effective_certainty_threshold)

        cert_j   = certainty_per_sample(post, certainty_penalty_kind, certainty_hinge_width)
        sparse_j = sparsity_per_sample(post)
        certainty_contribution = @. certainty_gate * llr_certainty_importance * cert_j
        if is_correlated
            corr_j = correlation_term_per_sample(
                post, connectivity, correlation_strengths, correlation_certainty_threshold,
                correlation_form, correlation_agreement_floor
            )
            correlation_contribution = @. gate * (correlation_weight * corr_j +
                                                  sparsity_importance * sparse_j)
        else
            correlation_contribution = @. gate * sparsity_importance * sparse_j
        end
        gated_aux_per_layer[layer - warmup_loss_layers] =
            sum(certainty_contribution .+ correlation_contribution) / n_samples
    end

    selection_loss = softmin_loss(base_per_layer, loss_layer_temperature)
    gated_aux_loss = sum(gated_aux_per_layer) / n_scored
    total_loss = selection_loss + gated_aux_loss
    return total_loss
end

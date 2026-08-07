# -----------------------------------------------------------------------------
#   sine_residue(x) = |sin(π x/2)|
#       - The original choice (Eq. 8 of arXiv:1811.07835).
#       - Zeros at even integer values of x.
#       - d/dx vanishes at every integer — both zeros (good) AND maxima (bad),
#         so the loss has plateaus at the worst points (x ≈ 1, 3, ...) where
#         the optimizer receives no gradient signal.
# -----------------------------------------------------------------------------
@inline sine_residue(x) = abs(sin(π * x / 2))

# ------------------------------------------------------------------------------
#   quadratic_residue(x) = (x − 2·round(x/2))² = squared distance from x to
#                          the nearest even integer.
#       - Same zeros as |sin(π x/2)|.
#       - Zeros at even integer values of x.
#       - Piecewise quadratic; d/dx = 2(x − 2k) on each smooth piece, so
#         restoring force scales linearly with distance from the nearest
#         valid solution.
#       - Subgradient ±2 (cusp) at the odd integers — descent is always
#         defined, no plateau at the maxima.
# ------------------------------------------------------------------------------
@inline quadratic_residue(x) = (x - 2 * round(x / 2))^2
# Equivalent floor-based form, in case Enzyme objects to `round`:
#   @inline quadratic_residue(x) = (x - 2 * floor((x + 1) / 2))^2
# Both rely on `round` / `floor` being treated as zero-derivative
# (piecewise constant), which matches their mathematical derivative
# almost everywhere.

function compute_sine_residue_loss_from_llrs(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix
)::Float32
    """
    Original sine-residue loss from Eq. 8 of https://arxiv.org/abs/1811.07835.
    See `compute_quadratic_residue_loss_from_llrs` for the drop-in alternative
    that fixes the gradient-vanishing-at-maxima behaviour.

    L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij [ e_k + σ(μ_k) ] )
    with
        σ(μ_k) = 1 / (1 + exp(μ_k))
        f(x)   = |sin(π x / 2)|
        H^⟂    = rows are stabilizer + logical generators of the code.
    """
    n_samples = size(expected_recoveries, 2)
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    average_loss = sum(@. sine_residue(commutation_relations_matrix)) / n_samples
    return average_loss
end

function compute_quadratic_residue_loss_from_llrs(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix
)::Float32
    """
    Drop-in replacement for `compute_loss_error_from_llrs` that swaps the
    per-check penalty f(x) = |sin(π x / 2)| for the squared distance from x
    to the nearest even integer:

        g(x) = (x - 2 · round(x/2))²

    Both penalties vanish iff x ≡ 0 (mod 2), so the underlying check —
    "residual error has trivial syndrome with respect to H^⟂" — is unchanged.
    The motivation for the swap is purely about the optimization landscape:

    1. Plateau at the maxima. d/dx |sin(πx/2)| = 0 at every integer (both
       zeros AND ones), so syndrome bits that drift to x ≈ 1 sit on a
       plateau and the optimizer can't push them out. g(x) has subgradient
       ±2 at odd integers (cusps), so descent is always defined.
    2. Stronger restoring force near the zeros. d/dx g(x) = 2(x − 2k) on
       each smooth piece — linear in distance from the nearest valid
       solution, vs. the sinusoidal penalty which is shallow near the zeros.

    L(μ, e) = ∑_i  g ( ∑_(jk) H^⟂_ij [ e_k + σ(μ_k) ] )
    with
        σ(μ_k) = 1 / (1 + exp(μ_k))
        g(x)   = (x - 2 · round(x/2))²
        H^⟂    = rows are stabilizer + logical generators of the code.
    """
    n_samples = size(expected_recoveries, 2)
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    average_loss = sum(@. quadratic_residue(commutation_relations_matrix)) / n_samples
    return average_loss
end

function syndrome_loss_regularizer(posterior_llrs::Matrix{Float32})::Float32
    """
    One of the important drawbacks of the Loss function from Eq. 8 of https://arxiv.org/abs/1811.07835 is that it has zero gradient at both even and odd weights of the syndrome.
    For instance, if the posterior LLR is zero for all the bit, then we have σ(μ_k) = 0.5 for all k, but sin(π * H^⟂ * (e + 0.5) / 2) will be zero for all the samples, and hence the Loss will be zero, even though the predicted recoveries are completely wrong.
    To mitigate this issue, we can add a regularizer to the Loss function that penalizes solutions where the posterior probabilities are not close to 0 or 1, i.e. where the LLRs are close to zero.
    This can be achieved by adding a term proportional to the entropy of the predicted probabilities, which encourages the model to make more confident predictions.
    The regularizer can be defined as:
        L_reg(μ) = ∑_v h(σ(μ_v))
    where
        - h(p) = - p log(p) - (1 - p) log(1 - p) is the binary entropy function.
    """
    regularizer = sum(binary_entropy_of_sigmoid.(posterior_llrs))
    return regularizer
end

function compute_additional_loss_from_ising_correlations(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32}
)::Float32
    """
    Ising two-body correlation term — the pair part of the negative log prior.

    For the set `C` of correlated qubit pairs (the rows of `connectivity`), with
    SIGNED log-odds couplings J_(i,k) = log[P00·P11 / (P01·P10)]:

        L_corr(μ) = - (1 / (n_samples · |C|)) ∑_((qi,qk) ∈ C) J_(i,k) · e_(qi) · e_(qk)

    where e_(qi) is approximated by σ(μ_(qi)). This is the co-activation form:
      - J > 0 (correlated pair)      → the loss is LOWERED when both qubits are
                                        flagged, so co-activation is encouraged.
      - J < 0 (anti-correlated pair) → the loss is RAISED when both are flagged,
                                        so simultaneous flips are discouraged.
      - J = 0                        → the pair contributes nothing.

    NOTE: this term is monotonically decreasing in σ wherever J > 0, i.e. it
    always argues for MORE predicted errors and carries no internal counterweight.
    In the full log prior that counterweight is the single-qubit field term, whose
    role here is played by `sparsity_penalty`. The balance between
    `sparsity_importance` (α₃) and `correlation_weight` (α₄) is therefore doing
    real work and must be tuned together — see expts/misc/sweep_correlation_weight.sh.

    There is deliberately NO syndrome gating: the correlation term is a prior on
    the error distribution, so it applies unconditionally rather than being
    switched on/off by whether the current residual already clears the syndrome.
    """
    n_samples = size(posterior_llrs, 2)
    n_edges   = size(connectivity, 1)

    loss = 0.0f0

    @inbounds for j in 1:n_samples
        loss_from_sample::Float32 = 0.0f0
        for e in 1:n_edges
            i = connectivity[e, 1]
            k = connectivity[e, 2]

            μ_i = posterior_llrs[i, j]
            μ_k = posterior_llrs[k, j]

            # sigmoid (stable version optional later)
            σ_i = sigmoid(μ_i)
            σ_k = sigmoid(μ_k)

            loss_from_sample += correlation_strengths[e] * σ_i * σ_k
        end
        loss += loss_from_sample
    end

    # The MINUS is essential: Σ J σ_i σ_k is the (log) prior we want to MAXIMISE,
    # so the loss is its negative. Dropping it inverts the physics — a positive
    # J (correlated pair) would then be pushed APART instead of co-activated.
    #
    # Normalising by n_edges as well as n_samples makes the term a per-edge MEAN,
    # so `correlation_weight` means the same thing regardless of how dense
    # the connectivity graph is.
    correlation_penalty = - loss / (n_samples * n_edges)
    return correlation_penalty
end

function softmin_loss(losses_per_layer::AbstractVector{Float32}, temp::Float32)::Float32
    """
    Compute a smooth approximation to the minimum of the per-layer losses using the softmin function.
    We want to derive a function that approximates the minimum of the losses across layers, but is differentiable and allows for gradient flow to all layers.
    The softmin function is defined as:
    softmin(x) = - temp * log( ∑_i exp(-x_i / temp) )
    where temp is a temperature parameter that controls the smoothness of the approximation.
    As temp approaches zero, the softmin approaches the true minimum, but for larger temp, it provides a smoother approximation that allows for gradient flow to all layers.
    """
    min_loss = minimum(losses_per_layer)
    aggregate_loss = min_loss - temp * log(sum(exp.(-(losses_per_layer .- min_loss) ./ temp)))
    return aggregate_loss
end

function linear_ramp_loss(losses_per_layer::AbstractVector{Float32})::Float32
    """
    Weighted-average loss with linearly increasing emphasis on later layers.

        L = (Σ_t t · L_t) / (Σ_t t)
          = (2 / (n(n+1))) · Σ_t t · L_t

    Layer t = n_layers contributes the most; layer 1 contributes the least
    (weight 1/n the smallest layer). Differs from the soft-min by being
    differentiable everywhere with no temperature parameter to anneal.

    Use this as a drop-in replacement for `softmin_loss(losses_per_layer, τ)`
    when you want late-layer emphasis without the τ-annealing complexity.
    """
    n_layers = length(losses_per_layer)
    aggregate_loss = sum(losses_per_layer[layer] * Float32(tanh(layer / n_layers)) for layer in 1:n_layers)
    return aggregate_loss
end

function last_layer_only_loss(losses_per_layer::AbstractVector{Float32}, _unused::Float32 = 0f0)::Float32
    """
    Simple loss that only considers the last layer's loss, ignoring all earlier layers.
    """
    return losses_per_layer[end]
end

function sparsity_penalty(posterior_llrs::Matrix{Float32})::Float32
    """
    L1 sparsity penalty on the predicted error pattern.

    L_sparse(μ) = (1 / n_samples) * Σ_samples Σ_v σ(μ_v).

    Pushes σ(μ_v) toward 0 — i.e., prefers fewer predicted errors. Combined
    with the syndrome term, the loss minimum corresponds to minimum-weight
    syndrome-satisfying predictions, which is the unique correct decode for
    any error of weight ≤ (d − 1) / 2.

    Intended to be used with a small `sparsity_importance` so that syndrome
    satisfaction dominates and sparsity acts as a tie-breaker.
    """
    n_samples = size(posterior_llrs, 2)
    sparsity_loss = sum(sigmoid.(posterior_llrs)) / n_samples
    return sparsity_loss
end

# =============================================================================
# Syndrome-gated tie-breaker machinery.
#
# Design (see the gating discussion in the project notes):
#   - base_loss (residue against [H; L]) is the only term that identifies the
#     correct answer; its SOFT zero set is however larger than the true
#     solutions (fractional aliases) and flat along the stabilizer coset.
#   - The auxiliary terms (certainty, sparsity, Ising correlation) are
#     selectors WITHIN that zero set. They are therefore applied per sample,
#     multiplied by a DETACHED gate that opens only when the sample's soft
#     H-syndrome is (nearly) satisfied:  g_j = 1[ |s|_j < τ ].
#   - Gate uses H only (not [H; L]): the wrong-coset region — H satisfied but
#     L violated — is exactly where the prior must cooperate with the L-row
#     gradient of base_loss on coset selection.
#   - Ordering constraint: a solved sample's gated-aux loss must stay below
#     the ~1-per-broken-check scale of a failing sample, or the landscape
#     prefers staying broken. Rewards (corr with J>0, value ≤ 0) are safe
#     automatically; penalties (certainty, sparsity) need their per-sample
#     magnitude kept ≲ 1 via their importance weights.
# =============================================================================

function soft_syndrome_weight_per_sample(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix
)::Vector{Float32}
    """
    Per-sample soft syndrome weight of the residual, against the STABILIZER
    generators only (H, not [H; L]):

        |s|_j = Σ_checks |sin( (π/2) · [H (e + σ(μ))]_(check, j) )|

    Units: "softly broken checks" — a satisfied check contributes 0, a fully
    broken one contributes 1. Computed from the SOFT σ(μ), so fractional
    aliases of the residue loss (e.g. σ ≈ 0.5 configurations that make every
    commutation value an even integer) have |s|_j ≈ 0 and count as inside the
    solution set — which is exactly where the tie-breaker terms must act.
    """
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix * e_total_matrix
    syndrome_weights = vec(sum(sine_residue.(commutation_relations_matrix); dims = 1))
    return syndrome_weights
end

function syndrome_gate_per_sample(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix,
    syndrome_gate_threshold::Float32
)::Vector{Float32}
    """
    Detached per-sample gate  g_j = 1[ |s|_j < τ ]  with τ in units of softly
    broken checks (τ = 0.5 ⇒ open below half a broken check; one fully broken
    check ⇒ closed).

    DETACHMENT: `s .< τ` produces Bools, whose derivative w.r.t. s is zero
    almost everywhere, so no gradient flows through the gate — by
    construction, without an explicit stop-gradient. This kills the perverse
    channel where the optimizer lowers gate×aux by RAISING |s| (breaking the
    syndrome to escape a penalty, or holding it broken to farm a reward).
    KEEP the gate an indicator: a smooth gate such as exp(-|s|/s₀) would
    reintroduce that channel unless explicitly detached, which Enzyme does
    not make convenient.
    """
    s = soft_syndrome_weight_per_sample(posterior_llrs, expected_recoveries, parity_check_matrix)
    gate = Float32.(s .< syndrome_gate_threshold)
    return gate
end

function certainty_per_sample(posterior_llrs::Matrix{Float32})::Vector{Float32}
    """
    Per-sample binary entropy Σ_v h(σ(μ_v)). ≈ 0 at binary configurations,
    maximal (n_bits · log 2) at σ = 0.5 — so under the gate this term acts
    almost exclusively on fractional aliases, pushing them to binary.

    Scale: batch-averaged downstream, so a full alias contributes
    ≈ 0.7 · n_bits · llr_certainty_importance to the loss (≈ 0.5 for n = 72
    at α = 0.01) — between a true solution (≈ 0) and one broken check (≈ 1).
    `syndrome_loss_regularizer` in the ungated mode sums over the whole batch
    instead, so the same `llr_certainty_importance` is ~batch_size× stronger
    there.
    """
    certainties = vec(sum(binary_entropy_of_sigmoid.(posterior_llrs); dims = 1))
    return certainties
end

function sparsity_per_sample(posterior_llrs::Matrix{Float32})::Vector{Float32}
    """
    Per-sample predicted error weight Σ_v σ(μ_v); batch-averaged downstream
    (the batch mean equals `sparsity_penalty`, so `sparsity_importance` has
    the same meaning in both modes). Ordering constraint: α₃ · (typical error
    weight) must stay ≲ 1 so a gated solved sample never scores worse than a
    failing one.
    """
    weights_of_predictions = vec(sum(sigmoid.(posterior_llrs); dims = 1))
    return weights_of_predictions
end

function ising_correlation_reward_per_sample(
    posterior_llrs::Matrix{Float32},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32}
)::Vector{Float32}
    """
    Per-sample Ising co-activation reward, per-edge mean (matches the batch
    version's normalization, so `correlation_weight` keeps its meaning):

        r_j = - (1/|C|) Σ_((qi,qk) ∈ C) J_(i,k) σ(μ_(qi,j)) σ(μ_(qk,j))

    r_j ≤ 0 wherever couplings are positive, so gating it on solved samples
    can never make a solved sample lose to a failing one (base ≥ 0 there):
    rewards are ordering-safe automatically.
    """
    n_samples = size(posterior_llrs, 2)
    n_edges = size(connectivity, 1)
    rewards = zeros(Float32, n_samples)
    @inbounds for j in 1:n_samples
        acc::Float32 = 0.0f0
        for e in 1:n_edges
            i = connectivity[e, 1]
            k = connectivity[e, 2]
            acc += correlation_strengths[e] * sigmoid(posterior_llrs[i, j]) * sigmoid(posterior_llrs[k, j])
        end
        rewards[j] = -acc / n_edges
    end
    return rewards
end

function compute_loss_including_correlations(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    is_correlated::Bool,
    correlation_weight::Float32,                # overall weight α₄ on the correlation term
    loss_layer_temperature::Float32,
    llr_certainty_importance::Float32,
    sparsity_importance::Float32,
    syndrome_gate_threshold::Float32,           # τ; ≤ 0 disables the gate
    warmup_loss_layers::Int
)::Float32
    """
    Total per-batch loss. Two modes, chosen by `syndrome_gate_threshold`:

    UNGATED (syndrome_gate_threshold ≤ 0):
      per-layer loss = base + α_cert·llr_reg + α₄·corr + α₃·sparse,
      combined across layers with softmin(loss_layer_temperature).

    GATED (syndrome_gate_threshold > 0):
      total = softmin_over_layers( base_l )
            + mean_over_layers( mean_over_samples( g_(l,j) · aux_(l,j) ) )
      where g_(l,j) = 1[ soft H-syndrome weight of sample j at layer l < τ ]
      (detached — see `syndrome_gate_per_sample`) and
      aux_(l,j) = α_cert·cert_j + α₄·corr_j + α₃·sparse_j (per-sample values).

      The auxiliary terms act only where a sample's soft H-residual is
      (near-)satisfied — the flat solution manifold of base_loss, where base
      provides no gradient and the auxiliary terms are the only forces. Layer
      selection sees base only, so a layer cannot win selection by being
      sparse / confident / co-activating instead of clearing the syndrome
      (softmin over its own vector has gradient = softmax weights ≥ 0, so no
      perverse push onto non-selected layers). `+ T·log(n_layers)` remains
      the softmin offset at zero spread, so loss-content diagnostics apply
      unchanged in both modes.

    All hyperparameters are plain `Float32` scalars (not singleton arrays).
    """
    n_layers = size(posterior_llrs, 3)
    n_scored = n_layers - warmup_loss_layers

    if syndrome_gate_threshold <= 0.0f0
        # ------------------------------------------------------------------
        # UNGATED PATH: auxiliary terms applied unconditionally; softmin over
        # the combined per-layer loss.
        # ------------------------------------------------------------------
        losses_per_layer = zeros(Float32, n_scored)
        for layer in (warmup_loss_layers + 1):n_layers
            post = posterior_llrs[:, :, layer]
            # base_loss   = compute_quadratic_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
            base_loss   = compute_sine_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
            llr_reg     = syndrome_loss_regularizer(post)
            sparse_pen  = sparsity_penalty(post)
            corr_pen::Float32 = 0f0
            if is_correlated
                corr_pen = compute_additional_loss_from_ising_correlations(
                              post, connectivity, correlation_strengths
                          )
            end

            losses_per_layer[layer - warmup_loss_layers] = base_loss +
                                      llr_certainty_importance * llr_reg +
                                      correlation_weight * corr_pen +
                                      sparsity_importance * sparse_pen
        end
        total_loss = softmin_loss(losses_per_layer, loss_layer_temperature)
        # total_loss = linear_ramp_loss(losses_per_layer)
        # total_loss = last_layer_only_loss(losses_per_layer)
        return total_loss
    end

    # ----------------------------------------------------------------------
    # GATED PATH.
    # ----------------------------------------------------------------------
    n_samples = size(posterior_llrs, 2)
    base_per_layer      = zeros(Float32, n_scored)
    gated_aux_per_layer = zeros(Float32, n_scored)
    for layer in (warmup_loss_layers + 1):n_layers
        post = posterior_llrs[:, :, layer]
        base_per_layer[layer - warmup_loss_layers] =
            compute_sine_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)

        gate = syndrome_gate_per_sample(
            post, expected_recoveries, parity_check_matrix, syndrome_gate_threshold
        )
        cert_j   = certainty_per_sample(post)
        sparse_j = sparsity_per_sample(post)
        if is_correlated
            corr_j = ising_correlation_reward_per_sample(post, connectivity, correlation_strengths)
            aux_j = @. llr_certainty_importance * cert_j +
                       correlation_weight * corr_j +
                       sparsity_importance * sparse_j
        else
            aux_j = @. llr_certainty_importance * cert_j + sparsity_importance * sparse_j
        end
        gated_aux_per_layer[layer - warmup_loss_layers] = sum(gate .* aux_j) / n_samples
    end

    selection_loss = softmin_loss(base_per_layer, loss_layer_temperature)
    gated_aux_loss = sum(gated_aux_per_layer) / n_scored
    total_loss = selection_loss + gated_aux_loss
    return total_loss
end
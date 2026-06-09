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
        L_reg(μ) = - ∑_v h(σ(μ_v))
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
    We want to add a term to the Loss function that prefers a correlated error instead of an independent error.
    Right now we want to focus on Ising-type two-body correlations.
    Suppose we have a list of qubit indices that are correlated: (q1, q2), (q3, q4), ... specified by `C`.
    Then we want to add a term to the Loss function that penalizes solutions where the errors at these qubit indices are not correlated.
    For example, if we have an error on q1 but not on q2, we want to penalize that solution. Hence, between q1 and q2, the favoured configurations are
    (0, 0), and (1, 1), while the disfavoured configurations are (0,1) and (1, 0).
    This can be achieved by adding a term proportional to `(e_(q1) - e_(q2))^2` to the Loss function, where `e_(qi)` is the predicted error at qubit `qi`.
    At the end of the day, we want to prioritize solutions that show no errors. So, we don't end up choosing an error solely because it is correlated.
    
    Hence the penalty for violating correlations is:
        L_corr(μ) = exp(-|s|) ∑_((qi, qj) ∈ C)  λ_(i,j) (e_(qi) - e_(qj))^2
    where
        - L(μ, e) is the original Loss function from `compute_loss_error_from_llrs`.
        - λ_(i,j) is a hyperparameter that controls the strength of the correlation penalty for the pair (qi, qj).
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.
        - s is the residualsyndrome of the error, given by: s = H * e_total,
        - H is the parity-check matrix of the dual code,
        - e_total = e_pred + e_expected is the total predicted error.

    Since we want to implement this in a differentiable manner, we can use the fact that:
        (e_(qi) - e_(qj))^2 = (σ(μ_(qi)) - σ(μ_(qj)))^2
    where e_(qi) is approximated by σ(μ_(qi)).

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

            loss_from_sample += correlation_strengths[e] * (σ_i - σ_k)^2
        end
        loss += loss_from_sample
    end

    correlation_penalty = loss / (n_samples * n_edges)
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

function compute_loss_including_correlations(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    is_correlated::Bool,
    correlation_importance::Float32,
    loss_layer_temperature::Float32,
    llr_certainty_importance::Float32,
    sparsity_importance::Float32,
    warmup_loss_layers::Int
)::Float32
    """
    Total per-batch loss combining:
      - base_loss    : quadratic-residue penalty on H^⊥(e + σ(μ))   [syndrome]
      - llr_reg      : β · binary-entropy of σ(μ)                   [commitment]
      - sparse_pen   : L1 norm of σ(μ) per sample                   [low-weight bias]
      - corr_pen     : Σ_(i,j) λ_ij (σ_i − σ_j)²                    [Ising correlation, optional]
    The four per-layer losses are combined across layers via softmin at
    temperature `loss_layer_temperature`.

    All hyperparameters are plain `Float32` scalars (not singleton arrays).
    """
    n_layers = size(posterior_llrs, 3)
    losses_per_layer = zeros(Float32, n_layers - warmup_loss_layers)
    for layer in (warmup_loss_layers + 1):n_layers
        post = posterior_llrs[:, :, layer]
        # base_loss   = compute_quadratic_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
        base_loss   = compute_sine_residue_loss_from_llrs(post, expected_recoveries, parity_check_matrix_dual)
        llr_reg     = syndrome_loss_regularizer(post)
        sparse_pen  = sparsity_penalty(post)
        corr_pen    = is_correlated ? compute_additional_loss_from_ising_correlations(post, connectivity, correlation_strengths) : 0f0

        losses_per_layer[layer - warmup_loss_layers] = base_loss +
                                  llr_certainty_importance * llr_reg +
                                  correlation_importance * corr_pen +
                                  sparsity_importance * sparse_pen
    end
    total_loss = softmin_loss(losses_per_layer, loss_layer_temperature)
    # total_loss = linear_ramp_loss(losses_per_layer)
    # total_loss = last_layer_only_loss(losses_per_layer)
    return total_loss
end
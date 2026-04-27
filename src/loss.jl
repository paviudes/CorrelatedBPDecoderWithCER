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

function compute_loss_error_from_llrs(
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

    Same input/output shape as the original — substitute at the call site
    in `compute_loss_including_correlations` with no other changes.

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

function syndrome_loss_regularizer(posterior_llrs::Matrix{Float32}; β::Float32 = 1f-3)::Float32
    """
    One of the important drawbacks of the Loss function from Eq. 8 of https://arxiv.org/abs/1811.07835 is that it has zero gradient at both even and odd weights of the syndrome.
    For instance, if the posterior LLR is zero for all the bit, then we have σ(μ_k) = 0.5 for all k, but sin(π * H^⟂ * (e + 0.5) / 2) will be zero for all the samples, and hence the Loss will be zero, even though the predicted recoveries are completely wrong.
    To mitigate this issue, we can add a regularizer to the Loss function that penalizes solutions where the posterior probabilities are not close to 0 or 1, i.e. where the LLRs are close to zero.
    This can be achieved by adding a term proportional to the entropy of the predicted probabilities, which encourages the model to make more confident predictions.
    The regularizer can be defined as:
        L_reg(μ) = - β ∑_v h(σ(μ_v))
    where
        - β is a hyperparameter that controls the strength of the regularizer.
        - h(p) = - p log(p) - (1 - p) log(1 - p) is the binary entropy function.
    """
    regularizer = β * sum(binary_entropy_of_sigmoid.(posterior_llrs))
    return regularizer
end

function compute_additional_loss_from_ising_correlations(
    posterior_llrs::Matrix{Float32},
    parity_check_matrix_dual::BitMatrix,
    expected_recoveries::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32}
)::Float32
    """
    We want to add a term to the Loss function that prefers a correlated error instead of an independent error.
    Right now we want to focus on Ising-type two-body correlations.
    Suppose we have a list of qubit indices that are correlated: (q1, q2), (q3, q4), ... specified by `C`.
    Then we want to add a term to the Loss function that penalizes solutions where the errors at these qubit indices are not correlated.
    For example, if we have an error on q1 but not on q2, we want to penalize that solution. Hence, between q1 and q2, the favoured configurations are
    (0, 0), (0, 1) and (1, 1), while the disfavoured configuration is (1, 0).
    This can be achieved by adding a term proportional to `e_(q1) * (1 - e_(q2))` to the Loss function, where `e_(qi)` is the predicted error at qubit `qi`.
    At the end of the day, we want to prioritize solutions that show no errors. So, we don't end up choosing an error solely because it is correlated.
    
    Hence the penalty for violating correlations is:
        L_corr(μ) = exp(-|s|) ∑_((qi, qj) ∈ C)  λ_(i,j) [ e_(qi) * (1 - e_(qj)) ]
    where
        - L(μ, e) is the original Loss function from `compute_loss_error_from_llrs`.
        - λ_(i,j) is a hyperparameter that controls the strength of the correlation penalty for the pair (qi, qj).
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.
        - s is the residualsyndrome of the error, given by: s = H * e_total,
        - H is the parity-check matrix of the dual code,
        - e_total = e_pred + e_expected is the total predicted error.

    Since we want to implement this in a differentiable manner, we can use the fact that:
        e_(qi) * (1 - e_(qj)) = e_(qi) - e_(qi) * e_(qj)
    where e_(qi) is approximated by σ(μ_(qi)).

    So, the Loss function becomes:
        L_total(μ) = L(μ, e) + ∑_((qi, qj) ∈ C) λ_(i,j) [ σ(μ_(qi)) - σ(μ_(qi)) * σ(μ_(qj)) ]
    
    We need to express this in a matrix form for efficient computation.
        L_total(μ) = L(μ, e) + λ .* ( σ(μ(connectivity[:,1]]) - σ(μ[connectivity[:,1]]) .* σ(μ[connectivity[:,2]]) )
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
        # Compute the weight of the residual syndromes: H^⟂ * (e_pred + e_expected).
        # e_residual = @. sigmoid(posterior_llrs[:, j]) + expected_recoveries[:, j]
        # residual_syndrome = Float32.(parity_check_matrix_dual) * e_residual
        # residual_syndrome_weight = sum(sin.(abs.(pi .* residual_syndrome ./ 2)))
        # loss += exp(-residual_syndrome_weight) * loss_from_sample
        loss += loss_from_sample
    end

    correlation_penalty = loss / (n_samples * n_edges)
    return correlation_penalty
end

function compute_loss_including_correlations(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    is_correlated::Bool,
    weights_loss_layers::Vector{Float32},
    correlation_importance::Vector{Float32},
    smooth_temp::Float32 = 1f-3
)::Float32
    """
    Compute the total Loss function including the correlation penalty.
    This function combines `compute_loss_error_from_llrs` and `compute_additional_loss_from_ising_correlations`.
    The total Loss L is a weighted sum of the penalty from failing to satisfy the commutation relations (with the normalizers)
    and the penalty from violating the correlations.
    L_total = L_syndrome + α * L_correlations
    where
        - L_syndrome is the original Loss function from `compute_loss_error_from_llrs`.
        - L_correlations is the additional penalty from violating correlations, computed by `compute_additional_loss_from_ising_correlations`.
        - α is a hyperparameter (`correlation_importance`) that controls the relative importance of the correlation penalty compared to the original Loss.
    """
    total_loss::Float32 = 0.0f0
    syndrome_regularizer_importance = 1f-1
    for layer in 1:size(posterior_llrs, 3)
        base_loss = compute_quadratic_residue_loss_from_llrs(posterior_llrs[:, :, layer], expected_recoveries, parity_check_matrix_dual)
        syndrome_regularizer = syndrome_loss_regularizer(posterior_llrs[:, :, layer])
        if is_correlated
            correlation_penalty = compute_additional_loss_from_ising_correlations(posterior_llrs[:, :, layer], parity_check_matrix_dual, expected_recoveries, connectivity, correlation_strengths)
        else
            correlation_penalty = 0.0f0
        end
        loss_per_layer = base_loss + syndrome_regularizer_importance * syndrome_regularizer + correlation_importance[1] * correlation_penalty
        # total_loss += loss_per_layer * sigmoid(weights_loss_layers[layer])
        total_loss -= smooth_temp * log(sum(exp(-loss_per_layer / smooth_temp))) # Temporary hack to ensure that we only need to focus on a single layer's loss.
    end
    return total_loss
end
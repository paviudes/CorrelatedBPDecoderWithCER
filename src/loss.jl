function compute_loss_error_from_llrs(posterior_llrs::Matrix{Float32}, expected_recoveries::BitMatrix, parity_check_matrix_dual::BitMatrix)::Float32
    """
    Compute a Loss function from the posterior LLRs calculated by the NeuralBP model and the expected recoveries.
    Note that if the posterior LLR is positive, then σ(μ_k) ≈ 0 (no error), else σ(μ_k) ≈ 1 (error).
    The idea is that if the output of the BP decoder, e_pred (≈ σ(μ)) added to the expected recovery (e) commutes with the elements of the dual code, then it is a stabilizer.
    Thus, e_total = e_pred + e_expected should satisfy H^⟂ * e_total = 0, where H^⟂ is the parity-check matrix of the dual code.
    In the context of stabilizer codes, when H is the parity-check matrix specifying the Z-stabilizers, H^⟂ specifies the X-type normalizers.
    
    This motivates the Loss function in Eq. 8 of https://arxiv.org/abs/1811.07835.
    L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij [ e_k + σ(μ_k)])
    where
        - σ(μ_k) = 1 / (1 + exp(μ_k))
        - f(x) = |sin(π x / 2)|
        - H^⟂ is the parity-check matrix of the dual code.
    """
    n_samples = size(expected_recoveries, 2)
    # Compute the average loss over all samples as a Matrix equation.
    e_total_matrix = @. sigmoid(posterior_llrs) + expected_recoveries
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    average_loss = sum(@. abs(sin(π * commutation_relations_matrix / 2))) / n_samples
    return average_loss
end

function compute_additional_loss_from_ising_correlations(posterior_llrs::Matrix{Float32}, connectivity::Matrix{Int}, expected_recoveries::BitMatrix, correlation_strengths::Vector{Float32})::Float32
    """
    We want to add a term to the Loss function that prefers a correlated error instead of an independent error.
    Right now we want to focus on Ising-type two-body correlations.
    Suppose we have a list of qubit indices that are correlated: (q1, q2), (q3, q4), ... specified by `C`.
    Then we want to add a term to the Loss function that penalizes solutions where the errors at these qubit indices are not correlated.
    For example, if we have an error on q1 but not on q2, we want to penalize that solution. Hence, between q1 and q2, the favoured configurations are
    (0, 0), (0, 1) and (1, 1), while the disfavoured configuration is (1, 0).
    This can be achieved by adding a term proportional to `e_(q1) * (1 - e_(q2))` to the Loss function, where `e_(qi)` is the predicted error at qubit `qi`.
    
    Hence the modified Loss function is:
        L_total(μ) = L(μ, e) + ∑_((qi, qj) ∈ C)  λ_(i,j) [ e_(qi) * (1 - e_(qj)) ]
    where
        - L(μ, e) is the original Loss function from `compute_loss_error_from_llrs`.
        - λ_(i,j) is a hyperparameter that controls the strength of the correlation penalty for the pair (qi, qj).
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.

    Since we want to implement this in a differentiable manner, we can use the fact that:
        e_(qi) * (1 - e_(qj)) = e_(qi) - e_(qi) * e_(qj)
    where e_(qi) is approximated by σ(μ_(qi)).

    So, the Loss function becomes:
        L_total(μ) = L(μ, e) + ∑_((qi, qj) ∈ C) λ_(i,j) [ σ(μ_(qi)) - σ(μ_(qi)) * σ(μ_(qj)) ]
    
    We need to express this in a matrix form for efficient computation.
        L_total(μ) = L(μ, e) + λ .* ( σ(μ(connectivity[:,1]]) - σ(μ[connectivity[:,1]]) .* σ(μ[connectivity[:,2]]) )
    """
    n_samples = size(expected_recoveries, 2)
    correlation_strengths_batch = repeat(correlation_strengths, 1, n_samples)
    # Compute the predicted errors from the left part of the connectivity matrix
    e_pred_left = sigmoid.(posterior_llrs[connectivity[1:end, 1], 1:end])
    e_pred_right = sigmoid.(posterior_llrs[connectivity[1:end, 2], 1:end])
    # Compute the correlation penalty term
    # n_edges = size(connectivity, 1)
    # correlation_penalty = sum(@. e_pred_left * (1 - e_pred_right) * correlation_strengths_batch) / (n_samples * n_edges)
    correlation_penalty = sum(@. e_pred_left * (1 - e_pred_right) * correlation_strengths_batch) / (n_samples)
    return correlation_penalty
end

function compute_loss_including_correlations(
    posterior_llrs::Array{Float32, 3},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    is_correlated::Bool,
    weights_loss_layers::Vector{Float32}
)::Float32
    """
    Compute the total Loss function including the correlation penalty.
    This function combines `compute_loss_error_from_llrs` and `compute_additional_loss_from_ising_correlations`.
    """
    total_loss = 0.0f0
    for layer in 1:size(posterior_llrs, 3)
        base_loss = compute_loss_error_from_llrs(posterior_llrs[:, :, layer], expected_recoveries, parity_check_matrix_dual)
        if is_correlated
            correlation_penalty = compute_additional_loss_from_ising_correlations(posterior_llrs[:, :, layer], connectivity, expected_recoveries, correlation_strengths)
        else
            correlation_penalty = 0.0f0
        end
        loss_per_layer = base_loss + correlation_penalty
        total_loss += loss_per_layer * sigmoid(weights_loss_layers[layer])
    end
    return total_loss
end
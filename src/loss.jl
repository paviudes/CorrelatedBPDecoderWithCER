function compute_loss_per_layer(
    posterior_llrs::AbstractVector{Float32}, 
    expected_recovery::BitVector, 
    parity_check_matrix_dual::BitMatrix; 
    is_correlated::Bool=false, 
    connectivity_edges::Matrix{Int}=zeros(Int, 0, 2), 
    correlation_strengths::Vector{Float32}=Float32[]
)::Float32
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

    However, in the presence of correlations, we want to add an additional term L_corr to the Loss function that prefers a correlated error instead of an independent error.

    Recall that the correlations are specified by a correlation graph G = (V, E), where V is the set of vertices corresponding to the bits of the code, and E is the set of edges corresponding to the correlations between the bits.
    Each edge e = (u, v) ∈ E is associated with a correlation weight w_e, which quantifies the strength of the correlation between the two bits connected by the edge.
    
    The correlated part of the Loss function is defined by
        L_corr(μ^t) = ∑_(e=(u, v) ∈ E) w_e * σ(μ^t_u) * (1 - σ(μ^t_v))
    where
        - σ(μ^t_u) is the predicted error probability for bit u at time t,
        - (1 - σ(μ^t_v)) is the predicted probability that there is no error on bit v at time t, and
        - w_e is the correlation weight associated with edge e = (u, v) in the correlation graph.
    """
    # Compute the average loss over all samples as a Matrix equation.
    e_total = sigmoid.(posterior_llrs) + expected_recovery
    commutation_relations = parity_check_matrix_dual * e_total
    iid_loss = sum(map(x -> abs(sin(π * x / 2)), commutation_relations))
    correlated_loss = 0.0f0
    if is_correlated
        for (i, edge) in enumerate(eachrow(connectivity_edges))
            u, v = edge
            w_e = correlation_strengths[i]
            correlated_loss += w_e * sigmoid(posterior_llrs[u]) * (1 - sigmoid(posterior_llrs[v]))
        end
    end
    total_loss = iid_loss + correlated_loss
    return total_loss
end

function compute_loss_from_llrs(
    posterior_llrs_by_layers::Matrix{Float32}, 
    expected_recovery::BitVector, 
    parity_check_matrix_dual::BitMatrix,
    loss_weights::Dict{NTuple{1, Int}, Float32}; 
    is_correlated::Bool=false, 
    connectivity_edges::Matrix{Int}=zeros(Int, 0, 2), 
    correlation_strengths::Vector{Float32}=Float32[]
)::Float32
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

    However, we want to compute the Loss for each layer separately and then average them. This is mentioned in the text following Eq. 8 in the paper.
    Suppose the posterior_llrs is structured as:
    
        posterior_llrs = [μ_s1 | μ_s2 | ... | μ_sK]
    
    where each μ_k is of shape (n bits, n_samples) and K is the number of layers, then for each layer k, we have L_k = ∑_e L(μ_k, e)
    to denote the Loss for layer k. The average Loss is then given by:
        average(L) = (1/K) ∑_(k=1)^K L_k 
    where K is the number of layers.
    """
    loss_by_layers = [
        compute_loss_per_layer(
            posterior_llrs_at_layer,
            expected_recovery,
            parity_check_matrix_dual;
            is_correlated=is_correlated,
            connectivity_edges=connectivity_edges,
            correlation_strengths=correlation_strengths
        )
        for posterior_llrs_at_layer in eachrow(posterior_llrs_by_layers)
    ]
    n_layers = size(posterior_llrs_by_layers, 1)
    average_loss = sum([loss_weights[(layer,)] * loss for (layer, loss) in enumerate(loss_by_layers)]) / n_layers
    return average_loss
end
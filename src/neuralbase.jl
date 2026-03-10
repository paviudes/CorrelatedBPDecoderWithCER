abstract type NeuralBP end

struct NeuralBPBase <: NeuralBP
    """
    Abstract type for Neural Belief Propagation models.

    Structure to represent a layer of the Neural Network that corresponds to unfolded Belief Propagation.

    Key fields:
    - parity_check_matrix: The parity-check matrix defining the code.
    - parity_check_matrix_dual: The dual of the parity-check matrix.
    - connectivity: Matrix defining the connectivity for correlated errors.
    - correlation_strength: Float32 indicating the strength of correlations.
    - is_correlated: Boolean indicating if the error model is correlated.
    - code_n_checks: Number of check nodes in the code.
    - code_n_bits: Number of variable nodes (bits) in the code.
    - initial_llrs: Vector of initial log-likelihood ratios (LLRs).
    - n_layers: Number of layers in the Neural Network.
    - activation_function: Activation function used in the network.
    - inverse_activation_function: Inverse of the activation function.
    """
    parity_check_matrix::BitMatrix
    parity_check_matrix_dual::BitMatrix
    code_n_checks::Int
    code_n_bits::Int
    
    connectivity_edges::Matrix{Int}
    correlation_strengths::Vector{Float32}
    is_correlated::Bool
    
    edges::Vector{Tuple{Int, Int}}  # List of edges (check, vertex)
    neighbors_of_check::Vector{Vector{Int}}  # Neighbors of each check node
    neighbors_of_vertex::Vector{Vector{Int}}  # Neighbors of each variable node
    
    n_layers::Int
    activation_function::Function
    inverse_activation_function::Function 
    derivative_activation_function::Function
    derivative_inverse_activation_function::Function

    function NeuralBPBase(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int},
        n_layers::Int;
        connectivity_edges::Matrix{Int}=Matrix{Int}(undef, 0, 0),
        correlation_strengths::Vector{Float32}=Float32[],
        activation_function::Function=safe_log_tanh,
        inverse_activation_function::Function=safe_atanh_exp,
        derivative_activation_function::Function=safe_cosech, # ∂ log(tanh(x/2)) / ∂x = 1/sinh(x)
        derivative_inverse_activation_function::Function=safe_negative_cosech # ∂ (2 atanh(exp(x))) / ∂x = -1/sinh(x)
    )
        """
        Construct the elements of a `NeuralBPBase` from a given parity-check matrix.
        
        """
        parity_check_matrix = convert.(Bool, copy(parity_check_matrix))
        parity_check_matrix_dual = convert.(Bool, copy(parity_check_matrix_dual))

        # Determine the connectivity between checks and vertices.
        (code_n_checks, code_n_bits) = size(parity_check_matrix)
        edges = Vector{Tuple{Int, Int}}()
        neighbors_of_check = [Int[] for _ in 1:code_n_checks]
        neighbors_of_vertex = [Int[] for _ in 1:code_n_bits]
        for c in 1:code_n_checks
            for v in 1:code_n_bits
                if parity_check_matrix[c, v] == 1
                    push!(edges, (c, v))
                    push!(neighbors_of_check[c], v)
                    push!(neighbors_of_vertex[v], c)
                end
            end
        end

        ## Interpret correlations.
        if (size(connectivity_edges, 1) == 0) || (length(correlation_strengths) == 0)
            is_correlated = false
        else
            is_correlated = true
        end

        return new(
            parity_check_matrix,
            parity_check_matrix_dual,
            code_n_checks,
            code_n_bits,
            connectivity_edges,
            correlation_strengths,
            is_correlated,
            edges,
            neighbors_of_check,
            neighbors_of_vertex,
            n_layers,
            activation_function,
            inverse_activation_function,
            derivative_activation_function,
            derivative_inverse_activation_function
        )
    end
end
abstract type NeuralBP end

struct NeuralBPBase <: NeuralBP
    """
    Abstract type for Neural Belief Propagation models.

    Structure to represent a layer of the Neural Network that corresponds to unfolded Belief Propagation.

    Key fields:
    - parity_check_matrix: The parity-check matrix defining the code.
    - is_correlated: Boolean indicating if the error model is correlated.
    - connectivity: Matrix defining the connectivity for correlated errors.
    - correlation_strength: Float32 indicating the strength of correlations.
    - code_n_checks: Number of check nodes in the code.
    - code_n_bits: Number of variable nodes (bits) in the code.
    - parity_check_matrix_dual: The dual of the parity-check matrix.
    - nb_neurons_per_layer: Number of neurons in each layer of the Neural Network.
    - neuron_to_check_variable: Dictionary mapping neuron indices to (check, variable) pairs.
    - neuron_to_checks: Vector mapping neuron indices to check nodes.
    - neuron_to_bits: Vector mapping neuron indices to variable nodes.
    - adj_initialize_V2C: Adjacency matrix from input layer to V2C layer.
    - adj_V2C_C2V: Adjacency matrix from V2C layer to C2V layer.
    - adj_C2V_V2C: Adjacency matrix from C2V layer to V2C layer.
    - adj_C2V_readout: Adjacency matrix from C2V layer to readout layer.
    - initial_llrs: Vector of initial log-likelihood ratios (LLRs).
    - n_layers: Number of layers in the Neural Network.

    """
    parity_check_matrix::BitMatrix

    is_correlated::Bool
    connectivity::Matrix{Int}
    correlation_strength::Float32
    
    code_n_checks::Int
    code_n_bits::Int
    parity_check_matrix_dual::BitMatrix
    nb_neurons_per_layer::Int
    neuron_to_check_variable::Dict{Int, Tuple{Int, Int}}  # Mapping from neuron index to (check, variable) pair.
    neuron_to_checks::Vector{Int} # Mapping from neuron index to check node.
    neuron_to_bits::Vector{Int} # Mapping from neuron index to variable node.

    # Connectivity: fixed parameters
    adj_initialize_V2C::BitMatrix
    adj_V2C_C2V::BitMatrix
    nb_weights_v2c_c2v::Int
    adj_C2V_V2C::BitMatrix
    nb_weights_c2v_v2c::Int
    adj_C2V_readout::BitMatrix
    nb_weights_c2v_readout::Int

    # Non-zero indices of the adjacency matrices for faster computation
    non_zero_rows_V2C_C2V::Vector{Int}
    non_zero_cols_V2C_C2V::Vector{Int}
    non_zero_rows_C2V_V2C::Vector{Int}
    non_zero_cols_C2V_V2C::Vector{Int}
    non_zero_rows_C2V_readout::Vector{Int}
    non_zero_cols_C2V_readout::Vector{Int}

    # Parameters of the Neural Network
    initial_llrs::Vector{Float32}
    n_layers::Int

    function NeuralBPBase(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int},
        initial_llrs::Vector{Float32},
        n_layers::Int;
        connectivity::Matrix{Int}=Matrix{Int}(undef, 0, 0),
        correlation_strength::Float32=0.0f0
    )
        """
        Construct the elements of a `NeuralBPLayer` from a given parity-check matrix.
        The key elements are
        - The number of neurons in each layer: C2V and V2C: `nb_neurons_per_layer`
        - Dictionary to specify the mapping from a neuron to the corresponding (check, variable) pair.
            - Note that each neuron in the V2C and C2V layers corresponds to an edge in the Tanner graph defined by the parity-check matrix.
            - Each edge connects a variable node to a check node.
            - Thus, we can define a mapping from each neuron index to the corresponding (check, variable) pair.
        - The adjacency matrix connecting the input layer to the V2C layer (`adj_initialize_V2C`).
        - The adjacency matrix connecting the V2C and C2V layers (`adj_V2C_C2V`).
        - The adjacency matrix connecting the C2V and V2C layers (`adj_C2V_V2C`).
        - The adjacency matrix connecting the C2V layer to the readout layer (`adj_V2C_readout`).
        
        These elements can be defined as follows.
        1. Adjacency matrix from input to V2C (`adj_initialize_V2C`):
            This matrix has |V| x |C| rows and |V| columns.
            We have adj_initialize_V2C[i, j] = 
                - define (c, v) from i, and v' = j
                - 1 if H[c, v] == 1 and v == v'. 0 otherwise.
        2. Adjacency matrix from V2C to C2V (`adj_V2C_C2V`):
            This matrix has |V| x |C| rows and |V| x |C| columns.
            We have adj_V2C_C2V[i, j] = 
                - define (c, v) from i
                - define (c', v') from j
                - 1 if H[c, v] == 1 and v != v' and H[c', v'] == 1 and c == c'. 0 otherwise.
        3. Adjacency matrix from C2V to V2C (`adj_C2V_V2C`):
            This matrix has |C| x |V| rows and |C| x |V| columns.
            We have adj_C2V_V2C[i, j] = 
                - define (c, v) from i
                - define (c', v') from j
                - 1 if H[c, v] == 1 and c != c' and H[c', v'] == 1 and v == v'. 0 otherwise.
        4. Adjacency matrix from C2V to readout (`adj_C2V_readout`):
            This matrix has |V| rows and |C| x |V| columns.
            We have adj_C2V_readout[i, j] = 
                - define (c', v') from j, and v = i
                - 1 if H[c', v'] == 1 and v == v'. 0 otherwise.
        
        - Activation functions for the V2C and C2V layers.
        1. For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        2. For the C2V layer, the activation function is f(x) = atanh(exp(x + i π syndrome[c])) where c is the check node corresponding to the neuron.
        """
        parity_check_matrix = convert.(Bool, copy(parity_check_matrix))
        parity_check_matrix_dual = convert.(Bool, copy(parity_check_matrix_dual))

        (code_n_checks, code_n_bits) = size(parity_check_matrix)
        nb_neurons_per_layer = sum(parity_check_matrix)

        ## Interprett correlations.
        if (size(connectivity, 1) == 0)
            is_correlated = false
        else
            is_correlated = true
        end

        ## Define mappings
        # Mapping from neuron index to (check, variable) pair
        neuron_to_check_variable = Dict{Int, Tuple{Int, Int}}()
        neuron_to_checks = Vector{Int}(undef, nb_neurons_per_layer)
        neuron_to_bits = Vector{Int}(undef, nb_neurons_per_layer)
        neuron_index = 1
        for c in 1:code_n_checks
            for v in 1:code_n_bits
                if parity_check_matrix[c, v] == 1
                    neuron_to_check_variable[neuron_index] = (c, v)
                    neuron_to_checks[neuron_index] = c
                    neuron_to_bits[neuron_index] = v
                    neuron_index += 1
                end
            end
        end

        ## Define adjacency matrices
        # 1. Adjacency matrix from the input layer to V2C
        adj_initialize_V2C = zeros(Bool, nb_neurons_per_layer, code_n_bits)
        for i in 1:nb_neurons_per_layer
            (_, v) = neuron_to_check_variable[i]
            adj_initialize_V2C[i, v] = 1
        end

        # 2. Adjacency matrix from V2C to C2V
        adj_V2C_C2V = zeros(Bool, nb_neurons_per_layer, nb_neurons_per_layer)
        for i in 1:nb_neurons_per_layer
            (c, v) = neuron_to_check_variable[i]
            for j in 1:nb_neurons_per_layer
                (c_prime, v_prime) = neuron_to_check_variable[j]
                if c == c_prime && v != v_prime
                    adj_V2C_C2V[i, j] = 1
                end
            end
        end
        # Compute the number of connections from each V2C neuron to C2V neurons
        nb_weights_v2c_c2v = count(x -> x == 1, adj_V2C_C2V)

        # 3. Adjacency matrix from C2V to V2C
        adj_C2V_V2C = zeros(Bool, nb_neurons_per_layer, nb_neurons_per_layer)
        for i in 1:nb_neurons_per_layer
            (c, v) = neuron_to_check_variable[i]
            for j in 1:nb_neurons_per_layer
                (c_prime, v_prime) = neuron_to_check_variable[j]
                if v == v_prime && c != c_prime
                    adj_C2V_V2C[i, j] = 1
                end
            end
        end
        # Compute the number of connections from each C2V neuron to V2C neurons
        nb_weights_c2v_v2c = count(x -> x == 1, adj_C2V_V2C)

        # 4. Adjacency matrix from C2V to readout
        adj_C2V_readout = zeros(Bool, code_n_bits, nb_neurons_per_layer)
        for j in 1:nb_neurons_per_layer
            (_, v_prime) = neuron_to_check_variable[j]
            adj_C2V_readout[v_prime, j] = 1
        end
        # Compute the number of connections from C2V neurons to readout neurons
        nb_weights_c2v_readout = count(x -> x == 1, adj_C2V_readout)

        # Compute the row and column indices of the non-zero elements in each adjacency matrix
        non_zero_V2C_C2V = findall(adj_V2C_C2V .== 1)
        non_zero_rows_V2C_C2V = [i[1] for i in non_zero_V2C_C2V]
        non_zero_cols_V2C_C2V = [i[2] for i in non_zero_V2C_C2V]

        non_zero_C2V_V2C = findall(adj_C2V_V2C .== 1)
        non_zero_rows_C2V_V2C = [i[1] for i in non_zero_C2V_V2C]
        non_zero_cols_C2V_V2C = [i[2] for i in non_zero_C2V_V2C]

        non_zero_C2V_readout = findall(adj_C2V_readout .== 1)
        non_zero_rows_C2V_readout = [i[1] for i in non_zero_C2V_readout]
        non_zero_cols_C2V_readout = [i[2] for i in non_zero_C2V_readout]
        
        return new(
            parity_check_matrix,
            is_correlated,
            connectivity,
            correlation_strength,
            code_n_checks,
            code_n_bits,
            parity_check_matrix_dual,
            nb_neurons_per_layer,
            neuron_to_check_variable,
            neuron_to_checks,
            neuron_to_bits,
            adj_initialize_V2C,
            adj_V2C_C2V,
            nb_weights_v2c_c2v,
            adj_C2V_V2C,
            nb_weights_c2v_v2c,
            adj_C2V_readout,
            nb_weights_c2v_readout,
            non_zero_rows_V2C_C2V,
            non_zero_cols_V2C_C2V,
            non_zero_rows_C2V_V2C,
            non_zero_cols_C2V_V2C,
            non_zero_rows_C2V_readout,
            non_zero_cols_C2V_readout,
            initial_llrs,
            n_layers
        )
    end
end

function add_soft_constraints_to_neuralbpbase(base::NeuralBPBase; connectivity::Matrix{Int}=base.connectivity)::NeuralBPBase
    """
    Create a new NeuralBPBase with updated connectivity for soft constraints.
    We will define a matrix that can be augmented to the parity-check matrix, which encodes the soft constraints.
    The new matrix M is defined as follows.
    - For each pair of variable nodes (i, j) that are connected in the connectivity matrix, we add a row to M with 1s in columns i and j, and 0s elsewhere.
    """
    n_connected_pairs = size(connectivity, 1)
    connectivity_parity_check_matrix = zeros(Bool, n_connected_pairs, base.code_n_bits)
    for pair_index in 1:n_connected_pairs
        (v1, v2) = (connectivity[pair_index, 1], connectivity[pair_index, 2])
        connectivity_parity_check_matrix[pair_index, v1] = 1
        connectivity_parity_check_matrix[pair_index, v2] = 1
    end
    updated_parity_check_matrix = vcat(base.parity_check_matrix, connectivity_parity_check_matrix)
    # Define a new NeuralBPBase with the updated parity-check matrix
    updated_base = NeuralBPBase(
        updated_parity_check_matrix,
        base.parity_check_matrix_dual,
        base.initial_llrs,
        base.n_layers;
        connectivity=connectivity,
        correlation_strength=base.correlation_strength
    )
    return updated_base
end
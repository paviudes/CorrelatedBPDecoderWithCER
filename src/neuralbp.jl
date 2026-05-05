import Zygote: gradient, Params
using Functors: @functor

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
    adj_C2V_V2C::BitMatrix
    adj_C2V_readout::BitMatrix

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

        # 4. Adjacency matrix from C2V to readout
        adj_C2V_readout = zeros(Bool, code_n_bits, nb_neurons_per_layer)
        for j in 1:nb_neurons_per_layer
            (_, v_prime) = neuron_to_check_variable[j]
            adj_C2V_readout[v_prime, j] = 1
        end
        
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
            adj_C2V_V2C,
            adj_C2V_readout,
            initial_llrs,
            n_layers
        )
    end
end

struct StandardNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the standard architecture for Neural Belief Propagation.
    In this variant the number of trainable parameters (weights in the network) is independent of the number of layers.
    """
    base::NeuralBPBase
    weights_v2c_c2v::Matrix{Float32}
    weights_c2v_v2c::Matrix{Float32}
    weights_c2v_readout::Matrix{Float32}
    
    function StandardNeuralBP(
        base::NeuralBPBase;
        weights_v2c_c2v::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_v2c::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_readout::Matrix{Float32}=Matrix{Float32}(undef, 0, 0)
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from V2C to C2V: `weights_v2c_c2v`
        2. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

        """
        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.code_n_bits, base.nb_neurons_per_layer)
        end

        return new(
            base,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function StandardNeuralBP(
        base::NeuralBPBase,
        weights_v2c_c2v::Matrix{Float32},
        weights_c2v_v2c::Matrix{Float32},
        weights_c2v_readout::Matrix{Float32}
    )
        return new(
            base,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor StandardNeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

struct NachmaniNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the Nachmani et al. architecture for Neural Belief Propagation: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.200501.
    In this variant the number of trainable parameters (weights in the network) scales linearly with the number of layers.
    """
    base::NeuralBPBase
    weights_v2c_c2v::Array{Float32, 3}
    weights_c2v_v2c::Array{Float32, 3}
    weights_c2v_readout::Matrix{Float32}

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_v2c_c2v::Array{Float32, 3}=Array{Float32, 3}(undef, 0, 0, 0),
        weights_c2v_v2c::Array{Float32, 3}=Array{Float32, 3}(undef, 0, 0, 0),
        weights_c2v_readout::Matrix{Float32}=Matrix{Float32}(undef, 0, 0)
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from V2C to C2V: `weights_v2c_c2v`
        2. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

        """
        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer, base.n_layers)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer, base.n_layers)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.code_n_bits, base.nb_neurons_per_layer)
        end

        return new(
            base,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        # learnable_parameters::Vector{Symbol},
        weights_v2c_c2v::Array{Float32, 3},
        weights_c2v_v2c::Array{Float32, 3},
        weights_c2v_readout::Matrix{Float32}
    )
        return new(
            base,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NachmaniNeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

# Explicitly tell the Flux framework that the weights are the trainable parameters.
function Flux.trainable(model::NeuralBP)
    return (
        weights_v2c_c2v = model.weights_v2c_c2v,
        weights_c2v_v2c = model.weights_c2v_v2c,
        weights_c2v_readout = model.weights_c2v_readout
    )
end

function (bpnn::StandardNeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
    """
    Forward pass through the Neural Network for Nachmani et al. architecture.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0)).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
    v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.base.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.base.n_layers
        # 1. Forward pass from V2C to C2V layer
        # c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        c2v_neurons = (@. sigmoid(bpnn.weights_c2v_v2c) * bpnn.base.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.base.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        # c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions
        c2v_activated_neurons = 2 .* safe_atanh_exp(c2v_neurons) .* phase_contributions # Use safe version to avoid numerical issues.

        # 2. Forward pass from C2V to V2C layer
        # v2c_neurons = (bpnn.weights_v2c_c2v[1:end, 1:end, iter] .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        v2c_neurons = (@. bpnn.weights_v2c_c2v * bpnn.base.adj_C2V_V2C') * c2v_activated_neurons + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        if (iter == bpnn.base.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    return readout_neurons
end

function (bpnn::NachmaniNeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
    """
    Forward pass through the Neural Network for Nachmani et al. architecture.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0)).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
    v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.base.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.base.n_layers
        # 1. Forward pass from V2C to C2V layer
        # c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        c2v_neurons = (@. sigmoid(bpnn.weights_c2v_v2c[:, :, iter]) * bpnn.base.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.base.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        # c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions
        c2v_activated_neurons = 2 .* safe_atanh_exp(c2v_neurons) .* phase_contributions # Use safe version to avoid numerical issues.

        # 2. Forward pass from C2V to V2C layer
        # v2c_neurons = (bpnn.weights_v2c_c2v[1:end, 1:end, iter] .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        v2c_neurons = (@. bpnn.weights_v2c_c2v[:, :, iter] * bpnn.base.adj_C2V_V2C') * c2v_activated_neurons + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        if (iter == bpnn.base.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    return readout_neurons
end
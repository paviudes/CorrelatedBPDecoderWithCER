import Zygote: gradient, Params
using Functors: @functor

struct NeuralBP
    """
    Structure to represent a layer of the Neural Network that corresponds to unfolded Belief Propagation.
    
    We want to define a Neural Network implementation in Julia.
        - The network consists of N + 2 layers, for some constant N that can be set.
        - The first layer consists of 'n' neurons.
        - There are 'N' middle layers. 
        - Each middle layer consists of two sublayers: V2C and C2V.
        - The sublayer V2C is a set of K_1 neurons.
        - The sublayer C2V is a set of K_2 neurons.
        - The neurons from V2C to C2V are connected using the adjacency matrix: adj_V2C_C2V.
        - The associated weights are weights_v2c_c2v.
        - The neurons from C2V to V2C are connected using an adjacency matrix: adj_C2V_V2C.
        - The associated weights are weights_c2v_v2c.
        - Each neuron in the C2V layer computes an activation function `f_act`.
        - The connectivity from the first layer to V2C is given by the adjacency matrix adj_initialize_V2C.
        - The readout layer consists of 'n' neurons.
        - The connectivity from the final layer's V2C to the readout layer is specified by the adjacency matrix adj_C2V_readout.
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
    
    # Learnable parameters: weights.
    # learnable_parameters::Vector{Symbol}
    weights_v2c_c2v::Array{Float32, 3}
    weights_c2v_v2c::Array{Float32, 3}
    weights_c2v_readout::Matrix{Float32}

    function NeuralBP(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int},
        initial_llrs::Vector{Float32},
        n_layers::Int;
        weights_v2c_c2v::Array{Float32, 3}=Array{Float32, 3}(undef, 0, 0, 0),
        weights_c2v_v2c::Array{Float32, 3}=Array{Float32, 3}(undef, 0, 0, 0),
        weights_c2v_readout::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
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

        ## Setting the learnable parameters to default values.
        1. Weights for the connections from V2C to C2V: `weights_v2c_c2v`
        2. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

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

        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer, n_layers)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer, n_layers)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, code_n_bits, nb_neurons_per_layer)
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
            n_layers,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NeuralBP(
        parity_check_matrix::BitMatrix,
        is_correlated::Bool,
        connectivity::Matrix{Int},
        correlation_strength::Float32,
        code_n_checks::Int,
        code_n_bits::Int,
        parity_check_matrix_dual::BitMatrix,
        nb_neurons_per_layer::Int,
        neuron_to_check_variable::Dict{Int, Tuple{Int, Int}},
        neuron_to_checks::Vector{Int},
        neuron_to_bits::Vector{Int},
        adj_initialize_V2C::BitMatrix,
        adj_V2C_C2V::BitMatrix,
        adj_C2V_V2C::BitMatrix,
        adj_C2V_readout::BitMatrix,
        initial_llrs::Vector{Float32},
        n_layers::Int,
        # learnable_parameters::Vector{Symbol},
        weights_v2c_c2v::Array{Float32, 3},
        weights_c2v_v2c::Array{Float32, 3},
        weights_c2v_readout::Matrix{Float32}
    )
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
            n_layers,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

# Explicitly tell the Flux framework that the weights are the trainable parameters.
function Flux.trainable(model::NeuralBP)
    return (
        weights_v2c_c2v = model.weights_v2c_c2v,
        weights_c2v_v2c = model.weights_c2v_v2c,
        weights_c2v_readout = model.weights_c2v_readout
    )
end

function print_neuralbp_info(bpnn::NeuralBP; io::IO=Base.stdout)
    """
    Print the key information about the NeuralBP model.
    """
    println(io, "NeuralBP Decoder Information:")
    println(io, "----------------------------")
    println(io, "Number of bits (n): ", size(bpnn.parity_check_matrix, 2))
    println(io, "Number of checks (m): ", size(bpnn.parity_check_matrix, 1))
    println(io, "Number of neurons per layer: ", bpnn.nb_neurons_per_layer)
    println(io, "----------------------------")
    # print the input layer to V2C adjacency matrix
    println(io, "Adjacency matrix from input layer to V2C layer ($(size(bpnn.adj_initialize_V2C, 1)) x $(size(bpnn.adj_initialize_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_initialize_V2C)
    println(io, "\n----------------------------")
    # print the V2C to C2V adjacency matrix
    println(io, "Adjacency matrix from V2C layer to C2V layer ($(size(bpnn.adj_V2C_C2V, 1)) x $(size(bpnn.adj_V2C_C2V, 2))):")
    show(io, "text/plain", bpnn.adj_V2C_C2V)
    println(io, "\n----------------------------")
    # print the C2V to V2C adjacency matrix
    println(io, "Adjacency matrix from C2V layer to V2C layer ($(size(bpnn.adj_C2V_V2C, 1)) x $(size(bpnn.adj_C2V_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_V2C)
    println(io, "\n----------------------------")
    # print the C2V to readout adjacency matrix
    println(io, "Adjacency matrix from C2V layer to readout layer ($(size(bpnn.adj_C2V_readout, 1)) x $(size(bpnn.adj_C2V_readout, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_readout)
    println(io, "\n----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Weights from V2C to C2V layer $(size(bpnn.weights_v2c_c2v)):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Weights from C2V to V2C layer $(size(bpnn.weights_c2v_v2c)):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Weights from C2V to readout layer $(size(bpnn.weights_c2v_readout)):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
end

function print_neuralbp_summary(bpnn::NeuralBP; io::IO=Base.stdout, final_llrs::Matrix{Float32}=Float32[])
    """
    Print a summary of training the NeuralBP model.
    Print the fitted parameters of the model.
    1. Weights from V2C to C2V
    2. Weights from C2V to V2C
    3. Weights from C2V to readout
    4. Initial LLRs
    5. (Optional) Final LLRs after training
    """
    println(io, "NeuralBP Training Summary:")
    println(io, "----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Fitted Weights from V2C to C2V layer $(size(bpnn.weights_v2c_c2v)):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Fitted Weights from C2V to V2C layer $(size(bpnn.weights_c2v_v2c)):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Fitted Weights from C2V to readout layer $(size(bpnn.weights_c2v_readout)):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
    # print the initial LLRs
    println(io, "Initial LLRs ($(length(bpnn.initial_llrs))):")
    show(io, "text/plain", bpnn.initial_llrs)
    println(io, "\n----------------------------")
    # print the final LLRs if provided
    if length(final_llrs) > 0
        println(io, "Final LLRs after training ($(size(final_llrs'))):")
        show(io, "text/plain", final_llrs')
        println(io, "\n----------------------------")
    end
    #TODO: print the runtime for training the neural BP model.
end

function load_trained_weights(weights_filename::String, bpnn::NeuralBP)::Dict{String, Any}
    """
    Load the trained weights from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    """
    # Load the weights from the file
    fp = open(weights_filename, "r")
    
    weights_data = JSON.parse(fp)
    
    formatted_weights = Dict{String, Any}()
    
    weights_v2c_c2v = reshape(Float32.(weights_data["weights_v2c_c2v"]), (bpnn.nb_neurons_per_layer, bpnn.nb_neurons_per_layer, bpnn.n_layers))
    formatted_weights["weights_v2c_c2v"] = weights_v2c_c2v
    
    weights_c2v_v2c = reshape(Float32.(weights_data["weights_c2v_v2c"]), (bpnn.nb_neurons_per_layer, bpnn.nb_neurons_per_layer, bpnn.n_layers))
    formatted_weights["weights_c2v_v2c"] = weights_c2v_v2c
    
    weights_c2v_readout = reshape(Float32.(weights_data["weights_c2v_readout"]), (bpnn.code_n_bits, bpnn.nb_neurons_per_layer))
    formatted_weights["weights_c2v_readout"] = weights_c2v_readout
    
    close(fp)
    return formatted_weights
end

function load_trained_neuralbp_model(weights_filename::String, bpnn::NeuralBP)::NeuralBP
    """
    Load a trained version of the NeuralBP model from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    The function will reconstruct the weight matrices from the vectorized versions and create a NeuralBP model with these weights.
    """
    # Load the weights from the file
    weights_data = load_trained_weights(weights_filename, bpnn)

    # Create a new model with the loaded weights
    loaded_bpnn = NeuralBP(
        convert.(Int, bpnn.parity_check_matrix),
        convert.(Int, bpnn.parity_check_matrix_dual),
        bpnn.initial_llrs,
        bpnn.n_layers;
        weights_v2c_c2v=weights_data["weights_v2c_c2v"],
        weights_c2v_v2c=weights_data["weights_c2v_v2c"],
        weights_c2v_readout=weights_data["weights_c2v_readout"],
        connectivity=bpnn.connectivity,
        correlation_strength=bpnn.correlation_strength
    )
    return loaded_bpnn
end

function save_trained_neuralbp_model(weights_filename::String, bpnn::NeuralBP)
    """
    Save the trained version of the NeuralBP model to a file.
    The file will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    """
    # Create a dictionary to store the weights
    weights_data = Dict{String, Any}()
    weights_data["weights_v2c_c2v"] = vec(bpnn.weights_v2c_c2v)
    weights_data["weights_c2v_v2c"] = vec(bpnn.weights_c2v_v2c)
    weights_data["weights_c2v_readout"] = vec(bpnn.weights_c2v_readout)

    # Save the weights to the file
    fp = open(weights_filename, "w")
    JSON.print(fp, weights_data)
    close(fp)
end

function extract_weights_for_BP(bpnn::NeuralBP, layer_index::Int)
    """
    Given the weights in the NeuralBP that were obtained after training, we want to extract the following information to perform standard Belief Propagation.
    1. For each edge from V2C to C2V, extract the weight associated with that edge in the given layer.
    weighted_BP_messages_v2c_c2v[v,c,c,v] = weight associated with the edge from V2C neuron (v,c) to C2V neuron (c,v)
    2. For each edge from C2V to V2C, extract the weight associated with that edge in the given layer.
    weighted_BP_messages_c2v_v2c[c,v,v,c] = weight associated with the edge from C2V neuron (c,v) to V2C neuron (v,c)
    3. For each edge from C2V to a readout bit, extract the weight associated with that edge.
    weighted_BP_messages_c2v_readout[c,v,v] = weight associated with the edge from C2V neuron (c,v) to readout bit v
    4. Return these weights as dictionaries.
    5. Note that the keys in the dictionaries are tuples of indices, and the values are the weights.
    """
    # Initialize the weight dictionaries
    # Extract weights from V2C to C2V
    weighted_BP_messages_v2c_c2v = Dict{Tuple{Int, Int, Int, Int}, Float32}()
    for i in 1:bpnn.nb_neurons_per_layer
        (c, v) = bpnn.neuron_to_check_variable[i]
        for j in 1:bpnn.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.neuron_to_check_variable[j]
            if bpnn.adj_V2C_C2V[i, j] == 1
                weight = bpnn.weights_v2c_c2v[i, j, layer_index]
                weighted_BP_messages_v2c_c2v[(v, c, c_prime, v_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to V2C
    weighted_BP_messages_c2v_v2c = Dict{Tuple{Int, Int, Int, Int}, Float32}()
    for i in 1:bpnn.nb_neurons_per_layer
        (c, v) = bpnn.neuron_to_check_variable[i]
        for j in 1:bpnn.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.neuron_to_check_variable[j]
            if bpnn.adj_C2V_V2C[i, j] == 1
                weight = bpnn.weights_c2v_v2c[i, j, layer_index]
                weighted_BP_messages_c2v_v2c[(c, v, v_prime, c_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to readout
    weighted_BP_messages_c2v_readout = Dict{Tuple{Int, Int, Int}, Float32}()
    for j in 1:bpnn.nb_neurons_per_layer
        (c_prime, v_prime) = bpnn.neuron_to_check_variable[j]
        for v in 1:bpnn.code_n_bits
            if bpnn.adj_C2V_readout[v_prime, j] == 1
                weight = bpnn.weights_c2v_readout[v_prime, j]
                weighted_BP_messages_c2v_readout[(c_prime, v_prime, v)] = weight
            end
        end
    end

    return (weighted_BP_messages_v2c_c2v, weighted_BP_messages_c2v_v2c, weighted_BP_messages_c2v_readout)
end

function extract_weights_for_BP(bpnn::NeuralBP)
    """
    Given the weights in the NeuralBP that were obtained after training, we want to extract the following information to perform standard Belief Propagation.
    1. For each edge from V2C to C2V, extract the weight associated with that edge in each layer.
    2. For each edge from C2V to V2C, extract the weight associated with that edge in each layer.
    3. For each edge from C2V to a readout bit, extract the weight associated with that edge.
    4. Return these weights as lists of dictionaries, where each dictionary corresponds to a layer.
    """
    n_layers = bpnn.n_layers
    weighted_BP_messages_v2c_c2v_layers = Vector{Dict{Tuple{Int, Int, Int, Int}, Float32}}(undef, n_layers)
    weighted_BP_messages_c2v_v2c_layers = Vector{Dict{Tuple{Int, Int, Int, Int}, Float32}}(undef, n_layers)
    weighted_BP_messages_c2v_readout = Dict{Tuple{Int, Int, Int}, Float32}()
    for layer_index in 1:n_layers
        (weighted_BP_messages_v2c_c2v, weighted_BP_messages_c2v_v2c, weighted_BP_messages_c2v_readout_layer) = extract_weights_for_BP(bpnn, layer_index)
        weighted_BP_messages_v2c_c2v_layers[layer_index] = weighted_BP_messages_v2c_c2v
        weighted_BP_messages_c2v_v2c_layers[layer_index] = weighted_BP_messages_c2v_v2c
        # For the readout weights, they are the same for all layers, so we can just store them once.
        if layer_index == 1
            weighted_BP_messages_c2v_readout = weighted_BP_messages_c2v_readout_layer
        end
    end
    return (weighted_BP_messages_v2c_c2v_layers, weighted_BP_messages_c2v_v2c_layers, weighted_BP_messages_c2v_readout)
end

function save_extracted_weights_for_BP(prefix::String, bpnn::NeuralBP)
    """
    Save the extracted weights for Belief Propagation to a file.
    The file will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in separate files where (1) and (2) will be stored as lists of dictionaries (one dictionary per layer), and (3) will be stored as a single dictionary.
    """
    (weighted_BP_messages_v2c_c2v_layers, weighted_BP_messages_c2v_v2c_layers, weighted_BP_messages_c2v_readout) = extract_weights_for_BP(bpnn)
    # Save the weights from V2C to C2V
    fp_v2c_c2v = open("$(prefix)_weights_v2c_c2v.json", "w")
    for layer_index in 1:bpnn.n_layers
        JSON.print(fp_v2c_c2v, weighted_BP_messages_v2c_c2v_layers[layer_index])
    end
    close(fp_v2c_c2v)
    # Save the weights from C2V to V2C
    fp_c2v_v2c = open("$(prefix)_weights_c2v_v2c.json", "w")
    for layer_index in 1:bpnn.n_layers
        JSON.print(fp_c2v_v2c, weighted_BP_messages_c2v_v2c_layers[layer_index])
    end
    close(fp_c2v_v2c)
    # Save the weights from C2V to readout
    fp_c2v_readout = open("$(prefix)_weights_c2v_readout.json", "w")
    JSON.print(fp_c2v_readout, weighted_BP_messages_c2v_readout)
    close(fp_c2v_readout)
end

function load_extracted_weights_for_BP(prefix::String, n_layers::Int)
    """
    Load the extracted weights for Belief Propagation from files.
    The files will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v -- list of dictionaries (one dictionary per layer)
    2. weights_c2v_v2c -- list of dictionaries (one dictionary per layer)
    3. weights_c2v_readout -- single dictionary
    """
    # Load the weights from V2C to C2V
    weighted_BP_messages_v2c_c2v_layers = Vector{Dict{Tuple{Int, Int, Int, Int}, Float32}}(undef, n_layers)
    fp_v2c_c2v = open("$(prefix)_weights_v2c_c2v.json", "r")
    for layer_index in 1:n_layers
        weighted_BP_messages_v2c_c2v_layers[layer_index] = JSON.parse(fp_v2c_c2v)
    end
    close(fp_v2c_c2v)
    # Load the weights from C2V to V2C
    weighted_BP_messages_c2v_v2c_layers = Vector{Dict{Tuple{Int, Int, Int, Int}, Float32}}(undef, n_layers)
    fp_c2v_v2c = open("$(prefix)_weights_c2v_v2c.json", "r")
    for layer_index in 1:n_layers
        weighted_BP_messages_c2v_v2c_layers[layer_index] = JSON.parse(fp_c2v_v2c)
    end
    close(fp_c2v_v2c)
    # Load the weights from C2V to readout
    fp_c2v_readout = open("$(prefix)_weights_c2v_readout.json", "r")
    weighted_BP_messages_c2v_readout = JSON.parse(fp_c2v_readout)
    close(fp_c2v_readout)
    return (weighted_BP_messages_v2c_c2v_layers, weighted_BP_messages_c2v_v2c_layers, weighted_BP_messages_c2v_readout)
end

function (bpnn::NeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
    """
    Forward pass through the Neural Network.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0))).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
    v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.n_layers
        # 1. Forward pass from V2C to C2V layer
        # c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        c2v_neurons = (@. sigmoid(bpnn.weights_c2v_v2c[:, :, iter]) * bpnn.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        # c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions
        c2v_activated_neurons = 2 .* safe_atanh_exp(c2v_neurons) .* phase_contributions # Use safe version to avoid numerical issues.

        # 2. Forward pass from C2V to V2C layer
        # v2c_neurons = (bpnn.weights_v2c_c2v[1:end, 1:end, iter] .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        v2c_neurons = (@. bpnn.weights_v2c_c2v[:, :, iter] * bpnn.adj_C2V_V2C') * c2v_activated_neurons + bpnn.adj_initialize_V2C * initial_llrs_batch
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        if (iter == bpnn.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    return readout_neurons
end

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
    # println("e_total_matrix of shape: ", size(e_total_matrix), ": ", e_total_matrix)
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    average_loss = sum(@. abs(sin(π * commutation_relations_matrix / 2))) / n_samples
    # println("Average Loss (Matrix computation): ", average_loss)
    return average_loss
end

function compute_additional_loss_from_ising_XOR(posterior_llrs::Matrix{Float32}, connectivity::Matrix{Int}, expected_recoveries::BitMatrix, correlation_strength::Float32)::Float32
    """
    We want to add a term to the Loss function that prefers a correlated error instead of an independent error.
    Right now we want to focus on Ising-type two-body correlations.
    Suppose we have a list of qubit indices that are correlated: (q1, q2), (q3, q4), ... specified by `C`.
    Then we want to add a term to the Loss function that penalizes solutions where the errors at these qubit indices are not correlated. For example, if we have an error on q1 but not on q2, we want to penalize that solution.
    This can be achieved by adding a term proportional to `e_(q1) XOR e_(q2)` to the Loss function, where `e_(qi)` is the predicted error at qubit `qi`.
    
    Hence the modified Loss function is:
        L_total(μ) = L(μ, e) + λ * ∑_((qi, qj) ∈ C) [ e_(qi) XOR e_(qj) ]
    where
        - L(μ, e) is the original Loss function from `compute_loss_error_from_llrs`.
        - λ is a hyperparameter that controls the strength of the correlation penalty.
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.

    Since we want to implement this in a differentiable manner, we can use the fact that:
        e_(qi) XOR e_(qj) = e_(qi) + e_(qj) - 2 * e_(qi) * e_(qj)
    where e_(qi) is approximated by σ(μ_(qi)).

    So, the Loss function becomes:
        L_total(μ) = L(μ, e) + λ * ∑_((qi, qj) ∈ C) [ σ(μ_(qi)) + σ(μ_(qj)) - 2 * σ(μ_(qi)) * σ(μ_(qj)) ]
    
    We need to express this in a matrix form for efficient computation.
        L_total(μ) = L(μ, e) + λ * ( σ(μ(connectivity[:,1]]) + σ(μ[connectivity[:,2]]) - 2 * σ(μ[connectivity[:,1]]) .* σ(μ[connectivity[:,2]]) )
    """
    n_samples = size(expected_recoveries, 2)
    # Compute the predicted errors from the left part of the connectivity matrix
    e_pred_left = sigmoid.(posterior_llrs[connectivity[1:end, 1], 1:end])
    e_pred_right = sigmoid.(posterior_llrs[connectivity[1:end, 2], 1:end])
    # Compute the correlation penalty term
    correlation_penalty_matrix = e_pred_left .+ e_pred_right .- 2 .* (e_pred_left .* e_pred_right)
    correlation_penalty = sum(correlation_penalty_matrix) * correlation_strength / n_samples
    return correlation_penalty
end

function compute_additional_loss_from_ising_correlations(posterior_llrs::Matrix{Float32}, connectivity::Matrix{Int}, expected_recoveries::BitMatrix, correlation_strength::Float32)::Float32
    """
    We want to add a term to the Loss function that prefers a correlated error instead of an independent error.
    Right now we want to focus on Ising-type two-body correlations.
    Suppose we have a list of qubit indices that are correlated: (q1, q2), (q3, q4), ... specified by `C`.
    Then we want to add a term to the Loss function that penalizes solutions where the errors at these qubit indices are not correlated.
    For example, if we have an error on q1 but not on q2, we want to penalize that solution. Hence, between q1 and q2, the favoured configurations are
    (0, 0), (0, 1) and (1, 1), while the disfavoured configuration is (1, 0).
    This can be achieved by adding a term proportional to `e_(q1) * (1 - e_(q2))` to the Loss function, where `e_(qi)` is the predicted error at qubit `qi`.
    
    Hence the modified Loss function is:
        L_total(μ) = L(μ, e) + λ * ∑_((qi, qj) ∈ C) [ e_(qi) * (1 - e_(qj)) ]
    where
        - L(μ, e) is the original Loss function from `compute_loss_error_from_llrs`.
        - λ is a hyperparameter that controls the strength of the correlation penalty.
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.

    Since we want to implement this in a differentiable manner, we can use the fact that:
        e_(qi) * (1 - e_(qj)) = e_(qi) - e_(qi) * e_(qj)
    where e_(qi) is approximated by σ(μ_(qi)).

    So, the Loss function becomes:
        L_total(μ) = L(μ, e) + λ * ∑_((qi, qj) ∈ C) [ σ(μ_(qi)) - σ(μ_(qi)) * σ(μ_(qj)) ]
    
    We need to express this in a matrix form for efficient computation.
        L_total(μ) = L(μ, e) + λ * ( σ(μ(connectivity[:,1]]) - σ(μ[connectivity[:,1]]) .* σ(μ[connectivity[:,2]]) )
    """
    n_samples = size(expected_recoveries, 2)
    # Compute the predicted errors from the left part of the connectivity matrix
    e_pred_left = sigmoid.(posterior_llrs[connectivity[1:end, 1], 1:end])
    e_pred_right = sigmoid.(posterior_llrs[connectivity[1:end, 2], 1:end])
    # Compute the correlation penalty term
    # correlation_penalty_matrix = e_pred_left .- (e_pred_left .* e_pred_right)
    # correlation_penalty = sum(correlation_penalty_matrix) * correlation_strength / n_samples
    correlation_penalty = sum(@. e_pred_left * (1 - e_pred_right)) * correlation_strength / n_samples
    return correlation_penalty
end

function compute_loss_including_correlations(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix_dual::BitMatrix,
    connectivity::Matrix{Int},
    correlation_strength::Float32,
    is_correlated::Bool
)::Float32
    """
    Compute the total Loss function including the correlation penalty.
    This function combines `compute_loss_error_from_llrs` and `compute_additional_loss_from_ising_correlations`.
    """
    base_loss = compute_loss_error_from_llrs(posterior_llrs, expected_recoveries, parity_check_matrix_dual)
    if !is_correlated
        return base_loss
    end
    correlation_penalty = compute_additional_loss_from_ising_correlations(posterior_llrs, connectivity, expected_recoveries, correlation_strength)
    total_loss = base_loss + correlation_penalty
    return total_loss
end

function train_neuralbp!(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    optimizer = OptimiserChain(
        ClipGrad(5.0),    # clip gradient norm at 5.0
        WeightDecay(1e-6), # optional small L2 regularizer
        ADAM(1e-4)        # smaller lr + larger eps for numerical stability
    ),
    n_epochs::Int=10,
    batch_size::Int=32
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    The provided syndromes should be computed from the errors, which are provided as expected recoveries.
    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model to be trained.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    - `initial_llrs::Matrix{Float32}`: Initial LLRs for the bits, to be used as input to the network (default: 1.0 since it is the LLR corresponding to the probability of the bit being 0 equal to 0.9).
    - `optimizer`: The optimizer to use for training (default: ADAM).
    - `n_epochs::Int`: The number of epochs. Each epoch goes through the entire dataset once (default: 10).
    - `batch_size::Int`: The size of each batch for training (default: 32).
    - `loss_function`: The loss function to use for training (default: binary cross-entropy).
    """
    # Batch the training data: syndromes and expected recoveries, into batches.
    # Each batch will be passed through the network as a Matrix.
    # Each batch will be a tuple of (syndromes_batch, expected_recoveries_batch)
    
    # Split an array of 1:n_samples into batches of size batch_size
    n_samples = size(syndromes, 2)
    samples_grouped_by_batch = [
        (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
        for i in 1:ceil(Int, n_samples / batch_size)
    ]
    # println("Samples grouped by batch: ", samples_grouped_by_batch)
    # Create the training dataset as a vector of tuples
    training_dataset = [
        (
            syndromes[1:end, batch_sample_indices],
            expected_recoveries[1:end, batch_sample_indices],
            repeat(bpnn.initial_llrs, 1, length(batch_sample_indices))
        )
        for batch_sample_indices in samples_grouped_by_batch
    ]

    # println("Starting training for $n_epochs epochs with batch size $batch_size... with training dataset of shape ", training_dataset)

    # Set the trainable parameters
    opt_state = Flux.setup(optimizer, bpnn)  # create optimiser state (tune lr as needed)

    for epoch in 1:n_epochs
        for (syndromes_train_batch, expected_recoveries_batch, llrs_batch) in training_dataset
            # compute loss and gradients in one shot
            loss, grads = Flux.withgradient(bpnn) do model
                posterior_llrs = model(llrs_batch, syndromes_train_batch)
                # compute_loss_error_from_llrs(posterior_llrs, expected_recoveries_batch, bpnn.parity_check_matrix_dual) #TODO: turn on the additional loss for correlations later.
                compute_loss_including_correlations(
                    posterior_llrs,
                    expected_recoveries_batch,
                    bpnn.parity_check_matrix_dual,
                    bpnn.connectivity,
                    bpnn.correlation_strength,
                    bpnn.is_correlated
                )
            end
            # apply update. grads[1] contains gradients for the model
            Flux.update!(opt_state, bpnn, grads[1])
            # println("Epoch $epoch, Batch Loss: $loss")
        end
        # println("Epoch $epoch completed.")
    end
end

function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix)::BitMatrix
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    
    Returns:
    - `predicted_recoveries::BitMatrix`: A matrix where each column represents the predicted recovery (error pattern) corresponding to the syndrome.
    """
    batch_size = size(syndromes, 2)
    initial_llrs_batch = repeat(bpnn.initial_llrs, 1, batch_size)
    predicted_recoveries_LLRs = bpnn(initial_llrs_batch, syndromes)
    # print_neuralbp_summary(bpnn; final_llrs=predicted_recoveries_LLRs)
    predicted_recoveries = convert.(Bool, (predicted_recoveries_LLRs .< 0))
    return predicted_recoveries
end

function generate_training_data(parity_check_matrix::Matrix{Int}, n_samples::Int, error_probability::Float64)::Tuple{BitMatrix, BitMatrix}
    """
    Generate training data for the NeuralBP model.
    Each sample consists of a random error pattern generated according to the specified error probability.
    The corresponding syndrome is computed using the provided parity-check matrix.
    The error patterns serve as the expected recoveries.

    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `n_samples::Int`: The number of samples to generate.
    - `error_probability::Float64`: The probability of an error occurring on each bit.
    
    Returns:
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    """
    (n_checks, n_bits) = size(parity_check_matrix)
    syndromes = zeros(Bool, n_checks, n_samples)
    expected_recoveries = zeros(Bool, n_bits, n_samples)

    for i in 1:n_samples
        # Generate a random error pattern
        error_pattern = rand(Bool, n_bits) .< error_probability
        expected_recoveries[:, i] = error_pattern

        # Compute the syndrome
        syndrome = mod.(parity_check_matrix * error_pattern, 2)
        syndromes[:, i] = syndrome
    end

    return syndromes, expected_recoveries
end

function check_bp_solutions(parity_check_matrix_dual::Matrix{Int}, errors::BitMatrix, recoveries::BitMatrix)::BitVector
    """
    Check if the provided recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `recoveries::BitMatrix`: A matrix where each row represents the recovery pattern corresponding to the error.

    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.
    """
    n_samples = size(errors, 2)
    is_correct = BitVector(undef, n_samples)

    for i in 1:n_samples
        total_pattern = xor.(errors[1:end, i], recoveries[1:end, i])
        syndrome = mod.(parity_check_matrix_dual * total_pattern, 2)
        is_correct[i] = all(syndrome .== 0)

        # For debugging purposes: if a weight-0 error is not corrected, print details.
        if (sum(errors[1:end, i]) == 0) && (!is_correct[i])
            println("Debug Info: Weight-0 error not corrected for sample $i.")
            println("Error pattern: ", errors[1:end, i])
            println("Recovery pattern: ", recoveries[1:end, i])
            println("Total pattern (error + recovery): ", total_pattern)
            println("Syndrome: ", syndrome)
        end
    end
    return is_correct
end
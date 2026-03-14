function load_trained_weights(weights_filename::String)::Dict{String, Any}
    """
    Load the trained weights from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_c2v_v2c
    2. weights_llrs
    3. weights_c2v_readout
    4. weights_loss_layers
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    """
    # Load the weights from the file
    fp = open(weights_filename, "r")
    
    weights_data = JSON.parse(fp)
    
    formatted_weights = Dict{String, Any}()
    
    weights_c2v_v2c = Float32.(weights_data["weights_c2v_v2c"])
    formatted_weights["weights_c2v_v2c"] = weights_c2v_v2c

    weights_llrs = Float32.(weights_data["weights_llrs"])
    formatted_weights["weights_llrs"] = weights_llrs
    
    weights_c2v_readout = Float32.(weights_data["weights_c2v_readout"])
    formatted_weights["weights_c2v_readout"] = weights_c2v_readout

    weights_loss_layers = Float32.(weights_data["weights_loss_layers"])
    formatted_weights["weights_loss_layers"] = weights_loss_layers
    
    close(fp)
    return formatted_weights
end

#=
function extract_weights_for_BP(bpnn::StandardNeuralBP)::Tuple{
    Dict{Tuple{Int, Int, Int, Int}, Float32}, 
    Dict{Tuple{Int, Int, Int, Int}, Float32}, 
    Dict{Tuple{Int, Int, Int}, Float32}
}
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
    for i in 1:bpnn.base.nb_neurons_per_layer
        (c, v) = bpnn.base.neuron_to_check_variable[i]
        for j in 1:bpnn.base.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
            if bpnn.base.adj_V2C_C2V[i, j] == 1
                weight = bpnn.weights_v2c_c2v[i, j]
                weighted_BP_messages_v2c_c2v[(v, c, c_prime, v_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to V2C
    weighted_BP_messages_c2v_v2c = Dict{Tuple{Int, Int, Int, Int}, Float32}()
    for i in 1:bpnn.base.nb_neurons_per_layer
        (c, v) = bpnn.base.neuron_to_check_variable[i]
        for j in 1:bpnn.base.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
            if bpnn.base.adj_C2V_V2C[i, j] == 1
                weight = bpnn.weights_c2v_v2c[i, j]
                weighted_BP_messages_c2v_v2c[(c, v, v_prime, c_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to readout
    weighted_BP_messages_c2v_readout = Dict{Tuple{Int, Int, Int}, Float32}()
    for j in 1:bpnn.base.nb_neurons_per_layer
        (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
        for v in 1:bpnn.base.code_n_bits
            if bpnn.base.adj_C2V_readout[v_prime, j] == 1
                weight = bpnn.weights_c2v_readout[v_prime, j]
                weighted_BP_messages_c2v_readout[(c_prime, v_prime, v)] = weight
            end
        end
    end

    return (weighted_BP_messages_v2c_c2v, weighted_BP_messages_c2v_v2c, weighted_BP_messages_c2v_readout)
end

function extract_weights_for_BP(bpnn::NachmaniNeuralBP, layer_index::Int)
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
    for i in 1:bpnn.base.nb_neurons_per_layer
        (c, v) = bpnn.base.neuron_to_check_variable[i]
        for j in 1:bpnn.base.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
            if bpnn.base.adj_V2C_C2V[i, j] == 1
                weight = bpnn.weights_v2c_c2v[i, j, layer_index]
                weighted_BP_messages_v2c_c2v[(v, c, c_prime, v_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to V2C
    weighted_BP_messages_c2v_v2c = Dict{Tuple{Int, Int, Int, Int}, Float32}()
    for i in 1:bpnn.base.nb_neurons_per_layer
        (c, v) = bpnn.base.neuron_to_check_variable[i]
        for j in 1:bpnn.base.nb_neurons_per_layer
            (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
            if bpnn.base.adj_C2V_V2C[i, j] == 1
                weight = bpnn.weights_c2v_v2c[i, j, layer_index]
                weighted_BP_messages_c2v_v2c[(c, v, v_prime, c_prime)] = weight
            end
        end
    end
    # Extract weights from C2V to readout
    weighted_BP_messages_c2v_readout = Dict{Tuple{Int, Int, Int}, Float32}()
    for j in 1:bpnn.base.nb_neurons_per_layer
        (c_prime, v_prime) = bpnn.base.neuron_to_check_variable[j]
        for v in 1:bpnn.base.code_n_bits
            if bpnn.base.adj_C2V_readout[v_prime, j] == 1
                weight = bpnn.weights_c2v_readout[v_prime, j]
                weighted_BP_messages_c2v_readout[(c_prime, v_prime, v)] = weight
            end
        end
    end

    return (weighted_BP_messages_v2c_c2v, weighted_BP_messages_c2v_v2c, weighted_BP_messages_c2v_readout)
end

function extract_weights_for_BP(bpnn::NachmaniNeuralBP)
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

function save_extracted_weights_for_BP(prefix::String, bpnn::StandardNeuralBP)
    """
    Save the extracted weights for Belief Propagation to a file.
    The file will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_c2v_v2c
    2. weights_llrs
    3. weights_c2v_readout
    4. weights_loss_layers
    They will be stored in separate files where (1) and (2) will be stored as lists of dictionaries (one dictionary per layer), and (3) and (4) will be stored as a single dictionary.
    """
    (weighted_BP_messages_v2c_c2v_layers, weighted_BP_messages_c2v_v2c_layers, weighted_BP_messages_c2v_readout) = extract_weights_for_BP(bpnn)
    
    # Save the weights from C2V to V2C
    fp_c2v_v2c = open("$(prefix)_weights_c2v_v2c.json", "w")
    JSON.print(fp_c2v_v2c, weighted_BP_messages_c2v_v2c_layers)
    close(fp_c2v_v2c)
    
    # Save the weights for the LLRs
    fp_llrs = open("$(prefix)_weights_llrs.json", "w")
    JSON.print(fp_llrs, bpnn.weights_llrs)
    close(fp_llrs)
    
    # Save the weights from C2V to readout
    fp_c2v_readout = open("$(prefix)_weights_c2v_readout.json", "w")
    JSON.print(fp_c2v_readout, weighted_BP_messages_c2v_readout)
    close(fp_c2v_readout)

    # Save the weights for the loss layers
    fp_loss_layers = open("$(prefix)_weights_loss_layers.json", "w")
    JSON.print(fp_loss_layers, bpnn.weights_loss_layers)
    close(fp_loss_layers)
end

function save_extracted_weights_for_BP(prefix::String, bpnn::NachmaniNeuralBP)
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
    for layer_index in 1:bpnn.base.n_layers
        JSON.print(fp_v2c_c2v, weighted_BP_messages_v2c_c2v_layers[layer_index])
    end
    close(fp_v2c_c2v)
    # Save the weights from C2V to V2C
    fp_c2v_v2c = open("$(prefix)_weights_c2v_v2c.json", "w")
    for layer_index in 1:bpnn.base.n_layers
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
=#
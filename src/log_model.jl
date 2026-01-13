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
    # Create a NachmaniNeuralBP model with the loaded weights
    loaded_bpnn = NachmaniNeuralBP(
        bpnn.base,
        weights_v2c_c2v=weights_data["weights_v2c_c2v"],
        weights_c2v_v2c=weights_data["weights_c2v_v2c"],
        weights_c2v_readout=weights_data["weights_c2v_readout"]
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
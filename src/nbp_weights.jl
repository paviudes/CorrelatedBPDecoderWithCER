function load_trained_weights(weights_filename::String)::Dict{String, Any}
    """
    Load the trained weights from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_c2v_v2c
    2. weights_llrs
    3. weights_c2v_readout
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

    close(fp)
    return formatted_weights
end
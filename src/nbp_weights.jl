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

    # α for the enriched check node. Absent from every model file written
    # before the enriched rule existed, and inert for the standard rule, so a
    # missing key means the Bayesian default α = 1 and old files stay readable.
    coupling_scale::Vector{Float32} = Float32.(get(weights_data, "coupling_scale", [1.0]))
    formatted_weights["coupling_scale"] = coupling_scale
    # The trained parameter itself, when the file is new enough to carry it.
    if haskey(weights_data, "coupling_logit")
        formatted_weights["coupling_logit"] = Float32.(weights_data["coupling_logit"])
    end
    # The layer schedule on α, [T₀, log(w - W_MIN)]. Absent from every file
    # written before the schedule existed; `load_trained_neuralbp_model` then
    # keeps the caller's defaults, which the constant schedule never reads.
    if haskey(weights_data, "coupling_schedule")
        formatted_weights["coupling_schedule"] = Float32.(weights_data["coupling_schedule"])
    end
    formatted_weights["coupling_schedule_kind"] =
        String(get(weights_data, "coupling_schedule_kind", "constant"))
    # Which check-node rule the weights were trained under; files older than
    # the enriched rule were necessarily trained with the standard one.
    formatted_weights["check_node"] = String(get(weights_data, "check_node", "tanh"))

    close(fp)
    return formatted_weights
end
function load_base_BP_model(parity_check_matrix_file::String, logicals_file::String, n_hidden_layers::Int; correlation_strengths_file::String="", use_cer::Bool=true)
    """
    Load the base BP model from the parity check matrix and logical operators files.
    The parity check matrix file is a text file where each line corresponds to a row of the parity check matrix, and the entries are separated by spaces.
    The logical operators file is a text file where each line corresponds to a logical operator, and the entries are separated by spaces.
    The function will read these files, construct the parity check matrix and logical operators, and return a NeuralBPBase model.

    `use_cer` (default true) controls whether correlated-error-rate priors are
    used. When false, the `correlation_strengths_file` is IGNORED even if present
    (as if the correlated_weights/ folder did not exist): single-qubit priors
    default to p=0.1 and the connectivity/correlations are left empty, so the
    base's `is_correlated` is false and the correlation loss term is dropped.
    """
    # Load the parity check matrix
    parity_check_matrix = readdlm(parity_check_matrix_file, Int)
    n_bits = size(parity_check_matrix, 2)
    # Load the logical operators
    logicals = readdlm(logicals_file, Int)
    # Construct the dual parity check matrix
    dual_parity_check_matrix = vcat(parity_check_matrix, logicals)
    # If the `correlation_strengths_file` is provided (and CER is enabled), parse
    # the correlation strengths and connectivity matrix from the file. Otherwise
    # (no file, or use_cer=false) use empty values + preset p=0.1 priors.
    if use_cer
        (connectivity_matrix, correlation_strengths, single_qubit_error_rates) = parse_cer_data(correlation_strengths_file; verbose=false)
        initial_llrs = zeros(Float32, n_bits)
        for qubit in 1:n_bits
            if haskey(single_qubit_error_rates, qubit)
                p = single_qubit_error_rates[qubit]
                initial_llrs[qubit] = log((1-p)/p)
            else
                initial_llrs[qubit] = log(9) # Default to p=0.1 for qubits not specified in the file
            end
        end
    else
        connectivity_matrix = zeros(Int, 0, 0)
        correlation_strengths = Float32[]
        initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1
    end
    # Construct the NeuralBPBase model
    base = NeuralBPBase(
        parity_check_matrix,
        dual_parity_check_matrix,
        initial_llrs,
        n_hidden_layers;
        connectivity=connectivity_matrix,
        correlation_strengths=correlation_strengths,
    )
    return base
end

function load_trained_neuralbp_model(weights_filename::String, bpnn::NachmaniNeuralBP)::NachmaniNeuralBP
    """
    Load a trained version of the NeuralBP model from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_c2v_v2c
    2. weights_llrs
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    The function will reconstruct the weight matrices from the vectorized versions and create a NeuralBP model with these weights.
    """
    # Load the weights from the file
    weights_data = load_trained_weights(weights_filename)
    # Create a NachmaniNeuralBP model with the loaded weights
    loaded_bpnn = NachmaniNeuralBP(
        bpnn.base,
        weights_c2v_v2c=weights_data["weights_c2v_v2c"],
        weights_llrs=weights_data["weights_llrs"],
        weights_c2v_readout=weights_data["weights_c2v_readout"]
    )
    return loaded_bpnn
end

function save_trained_neuralbp_model(weights_filename::String, bpnn::NeuralBP)
    """
    Save the trained version of the NeuralBP model to a file.
    The file will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_c2v_v2c
    2. weights_llrs
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    """
    # Create a dictionary to store the weights
    weights_data = Dict{String, Any}()
    weights_data["weights_c2v_v2c"] = vec(bpnn.weights_c2v_v2c)
    weights_data["weights_llrs"] = vec(bpnn.weights_llrs)
    weights_data["weights_c2v_readout"] = vec(bpnn.weights_c2v_readout)
    # Save the weights to the file
    fp = open(weights_filename, "w")
    JSON.print(fp, weights_data)
    close(fp)
end
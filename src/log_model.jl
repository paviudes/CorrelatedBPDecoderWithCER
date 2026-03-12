function load_NBP(base::NeuralBPBase, weights_directory::String)::NachmaniNeuralBP
    """
    Load an already trained NeuralBP model from file.
    
    The Neural BP model is deinfed by the following properties. Each of these properties are stored in separate CSV files.

    1. NeuralBPBase -- the base structure containing the parity-check matrix and other properties.
    These are stored in `base_filename` and read by `load_BP_base`

    2. initial_llrs -- the initial log-likelihood ratios (LLRs) for the variable nodes.
    These are stored in `<weights_directory>/initial_llrs.csv`.
    The file has two columns:
        - v: variable node index (1 to n_bits)
        - llr: initial LLR value (Float32)

    3. weights_c2v_v2c -- the weights appearing in the vertex -> check messages.
    This is a list of dictionaries of the form Vector{Dict{Tuple{Int, Int, Int, Int}, Float32}} with `n_layers` entries.
    The CSV file containing the weights is formatted as follows:
        Each row corresponds to a single weight.
        The columns are: t, v, c, c', v', weight
        where
        - t: layer index (1 to n_layers)
        - v: variable node index (1 to n_bits)
        - c: check node index (1 to n_checks)
        - c': neighboring check node index (1 to n_checks)
    4. weights_llrs -- the weights appearing in the vertex -> check messages. This is a vector of integers.
    
    5. weights_c2v_readout -- the weights appearing in the LLR calculations in front of the check --> vertex messages.
    This is a dictionary of the form Dict{Tuple{Int, Int, Int}, Float32}
    The CSV file containing the weights is formatted as follows:
        Each row corresponds to a single weight.
        The columns are: t, v, c, v, weight
        where
        - t: layer index (1 to n_layers)
        - v: variable node index (1 to n_bits)
        - c: check node index (1 to n_checks)
    """
    # Load the initial LLRs
    initial_llrs_filepath = joinpath(weights_directory, "initial_llrs.csv")
    # Read the CSV file as an array
    initial_llrs_df = CSV.read(initial_llrs_filepath, DataFrame)
    initial_llrs = convert(Vector{Float32}, initial_llrs_df.llr)

    # Load the weights from C2V to V2C
    weights_c2v_v2c_filepath = joinpath(weights_directory, "weights_c2v_v2c.csv")
    weights_c2v_v2c_df = CSV.read(weights_c2v_v2c_filepath, DataFrame)
    weights_c2v_v2c = Dict{NTuple{5,Int}, Float32}()
    for row in eachrow(weights_c2v_v2c_df)
        t = row.t
        v = row.v
        c = row.c
        cp = row.cp
        vp = row.vp
        weight = row.weight
        weights_c2v_v2c[(t, v, c, cp, vp)] = weight
    end
    
    # Load the weights for LLRs
    weights_llrs_filepath = joinpath(weights_directory, "weights_llrs.csv")
    weights_llrs_df = CSV.read(weights_llrs_filepath, DataFrame)
    weights_llrs = Dict{NTuple{2,Int}, Float32}()
    for row in eachrow(weights_llrs_df)
        t = row.t
        v = row.v
        weight = row.weight
        weights_llrs[(t, v)] = weight
    end

    # Load the weights from C2V to readout
    weights_c2v_readout_filepath = joinpath(weights_directory, "weights_c2v_readout.csv")
    weights_c2v_readout_df = CSV.read(weights_c2v_readout_filepath, DataFrame)
    weights_c2v_readout = Dict{NTuple{4,Int}, Float32}()
    for row in eachrow(weights_c2v_readout_df)
        t = row.t
        v = row.v
        c = row.c
        vp = row.vp
        weight = row.weight
        weights_c2v_readout[(t, v, c, vp)] = weight
    end

    # Load the loss weights
    loss_weights_filepath = joinpath(weights_directory, "loss_weights.csv")
    loss_weights_df = CSV.read(loss_weights_filepath, DataFrame)
    loss_weights = Dict{NTuple{1, Int}, Float32}()
    for row in eachrow(loss_weights_df)
        layer = row.layer
        weight = row.weight
        loss_weights[(layer,)] = weight
    end
    
    # Construct the NachmaniNeuralBP model
    bpnn = NachmaniNeuralBP(
        base,
        initial_llrs;
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout,
        loss_weights=loss_weights
    )
    return bpnn
end

function save_NBP(directory::String, bpnn::NachmaniNeuralBP)
    """
    Save the extracted weights for Belief Propagation to files in a directory.
    We need to store four properties, each in a separate file:
    1. initial_llrs -- stored in `<directory>/initial_llrs.csv`
    2. weights_c2v_v2c -- stored in `<directory>/weights_c2v_v2c.csv`
    3. weights_llrs -- stored in `<directory>/weights_llrs.csv`
    4. weights_c2v_readout -- stored in `<directory>/weights_c2v_readout.csv`
    5. loss_weights -- stored in `<directory>/loss_weights.csv`
    """
    # Create the directory if it does not exist
    if !isdir(directory)
        mkpath(directory)
    end
    # Save the initial LLRs
    initial_llrs_filepath = joinpath(directory, "initial_llrs.csv")
    initial_llrs_df = DataFrame(v=Int[], llr=Float32[])
    for (v, llr) in enumerate(bpnn.initial_llrs)
        push!(initial_llrs_df, (v=v, llr=llr))
    end
    CSV.write(initial_llrs_filepath, initial_llrs_df)

    # Save the weights from V2C to C2V
    weights_c2v_v2c_filepath = joinpath(directory, "weights_c2v_v2c.csv")
    weights_c2v_v2c_df = DataFrame(t=Int[], v=Int[], c=Int[], cp=Int[], vp=Int[], weight=Float32[])
    for ((t, v, c, cp, vp), weight) in bpnn.weights_c2v_v2c
        push!(weights_c2v_v2c_df, (t=t, v=v, c=c, cp=cp, vp=vp, weight=weight))
    end
    CSV.write(weights_c2v_v2c_filepath, weights_c2v_v2c_df)

    # Save the weights for LLRs
    weights_llrs_filepath = joinpath(directory, "weights_llrs.csv")
    weights_llrs_df = DataFrame(t=Int[], v=Int[], weight=Float32[])
    for ((t, v), weight) in bpnn.weights_llrs
        push!(weights_llrs_df, (t=t, v=v, weight=weight))
    end
    CSV.write(weights_llrs_filepath, weights_llrs_df)

    # Save the weights from C2V to readout
    weights_c2v_readout_filepath = joinpath(directory, "weights_c2v_readout.csv")
    weights_c2v_readout_df = DataFrame(t=Int[], v=Int[], c=Int[], vp=Int[], weight=Float32[])
    for ((t, v, c, vp), weight) in bpnn.weights_c2v_readout
        push!(weights_c2v_readout_df, (t=t, v=v, c=c, vp=vp, weight=weight))
    end
    CSV.write(weights_c2v_readout_filepath, weights_c2v_readout_df)

    # Save the loss weights
    loss_weights_filepath = joinpath(directory, "loss_weights.csv")
    loss_weights_df = DataFrame(layer=Int[], weight=Float32[])
    for ((layer,), weight) in bpnn.loss_weights
        push!(loss_weights_df, (layer=layer, weight=weight))
    end
    CSV.write(loss_weights_filepath, loss_weights_df)
end
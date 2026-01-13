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
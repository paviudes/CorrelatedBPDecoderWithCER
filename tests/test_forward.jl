using SparseArrays
using DelimitedFiles
using CorrelatedBPDecoderWithCER

@inline function load_neural_BP_model()::NeuralBP
    """
    Load a Neural BP model from a file.
    """
    # Define the parity-check matrix
    example_name = "hamming"
    prefix = "./../data/$(example_name)"
    # Read from the files `data/<example_name>/HX.txt` and `data/<example_name>/LX.txt`
    H = readdlm("$(prefix)/HX.txt", Int)
    # To load the dual matrix, load the logical operators LX and append it to H to form H_dual
    logicals = readdlm("$(prefix)/LX.txt", Int)
    H_dual = vcat(H, logicals)
    n_bits = size(H, 2)
    
    # Number of layers (rounds of BP)
    n_layers = 3
    
    # Initialize the NeuralBP model with random weights around 1.0.
    base = NeuralBPBase(
        H,
        H_dual,
        zeros(Float32, n_bits), # default initial LLRs corresponding to p=0.5
        n_layers
    )
    
    #=
    weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * n_layers]; scale=0.01f0)
    weights_llrs = random_values_around_one([n_bits * n_layers]; scale=0.01f0)
    weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale=0.01f0)
    =#

    # Set all weights to 1.0 for testing, since that corresponds to standard BP.
    weights_c2v_v2c = ones(Float32, base.nb_weights_c2v_v2c * n_layers)
    weights_llrs = ones(Float32, n_bits * n_layers)
    weights_c2v_readout = ones(Float32, base.nb_weights_c2v_readout)
    bpnn = NachmaniNeuralBP(
        base,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_llrs=weights_llrs,
        weights_c2v_readout=weights_c2v_readout
    )

    return bpnn

end

function test_forward_propagation()
    """
    We want to test the forward propagation of the NeuralBP model.
    We will define a small parity-check matrix, a syndrome and initial LLRs.
    We will then perform a forward pass through the network and print the output.
    There are two ways to forward propagate through the network. One is an efficient in-place version that uses pre-allocated arrays to store intermediate results,
    and the other is a functional version that constructs new arrays at each step. We will test both versions and check that they give the same output.
    """
    # Load the NeuralBP model with predefined weights.
    bpnn = load_neural_BP_model()

    # Define a syndrome to be a random binary vector of size equal to the number of rows of H
    syndrome = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    syndromes_batch = repeat(convert.(Bool, syndrome), 1, 1)  # single sample

    # Define initial LLRs batch
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1
    initial_llrs_batch = repeat(initial_llrs, 1, 1) # Initial LLRs corresponding to p=0.1

    # Perform a forward pass
    output_llrs_inplace_version = bpnn(initial_llrs_batch, syndromes_batch)
    output_llrs_functional_version = forward_pass_with_weights(bpnn, initial_llrs_batch, syndromes_batch)
    
    # Check if the outputs match
    println("Syndrome: ", syndrome)
    if all(isapprox.(output_llrs_inplace_version, output_llrs_functional_version, atol=1e-6))
        println("Forward pass outputs from both versions match, and they produce the LLRS: ", output_llrs_inplace_version, ".")
    else
        println("Forward pass outputs from both versions do not match.")
        # print the posterior LLRs from both versions, at each layer, to see where they start to differ.
        for layer in 1:n_layers
            println("Layer ", layer, ":")
            # Check if the outputs match at this layer
            if all(isapprox.(output_llrs_inplace_version[:, :, layer], output_llrs_functional_version[:, :, layer], atol=1e-6))
                println("LLRs at layer ", layer, " match: ", output_llrs_inplace_version[:, :, layer])
            else
                println("LLRs at layer ", layer, " do not match.")
                println("In-place version LLRs: ", output_llrs_inplace_version[:, :, layer])
                println("Functional version LLRs: ", output_llrs_functional_version[:, :, layer])
            end
            println("----------------------------------------------")
        end
    end

    # Run the standard BP decoder for `n_layers` iterations to get the expected LLRs after `n_layers` iterations.
    n_iterations = n_layers
    (final_llrs_standard_bp, _) = run_bp("SumProduct", H, size(H, 1) + 1, syndrome, convert.(Float64, initial_llrs), n_iterations; verbose=false)

    # Check if the final LLRs from the functional form of the Neural BP match the expected LLRs from the standard BP after `n_layers` iterations.
    if all(isapprox.(output_llrs_functional_version[:, :, n_layers], final_llrs_standard_bp, atol=1e-6))
        println("Final LLRs from the functional version of Neural BP after $(n_layers) iterations match the expected values from standard BP:", output_llrs_functional_version[:, :, n_layers])
    else
        println("Final LLRs from the functional version of Neural BP after $(n_layers) iterations do not match the expected values from standard BP.")
        println("Expected: ", final_llrs_standard_bp)
        println("Got: ", output_llrs_functional_version[:, :, n_layers])
    end
end

function test_forward_comparison_gpu(;
    atol::Float32=1f-4,
    rtol::Float32=1f-4
)
    """
    Compare the GPU forward pass output to a CPU reference implementation.
    """
    # Load the NeuralBP model with predefined weights.
    bpnn = load_neural_BP_model()
    n_bits = bpnn.base.code_n_bits

    # Define a syndrome to be a random binary vector of size equal to the number of rows of H
    syndrome = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    syndromes_batch = repeat(convert.(Bool, syndrome), 1, 1)  # single sample

    # Define initial LLRs batch
    initial_llrs = convert.(Float32, log(9)) .* ones(Float32, n_bits) # Initial LLRs corresponding to p=0.1
    initial_llrs_batch = repeat(initial_llrs, 1, 1) # Initial LLRs corresponding to p=0.1
    
    cpu_posteriors = forward_pass(bpnn, initial_llrs_batch, syndromes_batch)
    gpu_posteriors = forward_pass_gpu(bpnn, initial_llrs_batch, syndromes_batch)

    is_match = isapprox(cpu_posteriors, gpu_posteriors; atol=atol, rtol=rtol)
    if is_match
        println("GPU output matches CPU within tol (atol=$atol, rtol=$rtol).")
    else
        diff = abs.(cpu_posteriors .- gpu_posteriors)
        println("GPU vs CPU mismatch.")
        println("  max abs diff:  ", maximum(diff))
        println("  mean abs diff: ", sum(diff) / length(diff))
        println("  CPU range:     [", minimum(cpu_posteriors), ", ", maximum(cpu_posteriors), "]")
        println("  GPU range:     [", minimum(gpu_posteriors), ", ", maximum(gpu_posteriors), "]")
    end
    return nothing
end
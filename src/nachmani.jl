import Zygote: gradient, Params
using Functors: @functor

@enum NachmaniLearnableParameterGroup begin
    C2V_V2C
    LLRS
    C2V_READOUT
end

mutable struct NachmaniNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the Nachmani et al. architecture for Neural Belief Propagation: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.200501.
    In this variant the number of trainable parameters (weights in the network) scales linearly with the number of layers.
    """
    base::NeuralBPBase
    initial_llrs::Vector{Float32}

    # learnable parameters
    weights_c2v_v2c::Dict{NTuple{5,Int}, Float32} # The keys are tuples of the form (layer, vertex, check, check_prime, vertex) representing W^t_(v,c ; c',v).
    weights_llrs::Dict{NTuple{2,Int}, Float32} # Keys are tuples of the form (layer, vertex) representing b^t_v.
    weights_c2v_readout::Dict{NTuple{4,Int}, Float32} # Keys are tuples of the form (layer, vertex, check, vertex) representing W^t_(v;c,v).

    # Vectorization map, so that we can easily convert all the learable parameters to a single 1D vector for optimization.
    vmap_c2v_v2c::Dict{NTuple{5,Int}, Int}
    vmap_llrs::Dict{NTuple{2,Int}, Int}
    vmap_c2v_readout::Dict{NTuple{4,Int}, Int}
    # Inverse Vectorization maps, so that we can easily convert the 1D vector of parameters back to the structured format.
    invmap::Dict{Int, Tuple{NachmaniLearnableParameterGroup, Tuple}}
    
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        initial_llrs::Vector{<:Real};
        weights_c2v_v2c::Dict{NTuple{5,Int}, Float32}=Dict{NTuple{5,Int}, Float32}(),
        weights_llrs::Dict{NTuple{2,Int}, Float32}=Dict{NTuple{2,Int}, Float32}(),
        weights_c2v_readout::Dict{NTuple{4,Int}, Float32}=Dict{NTuple{4,Int}, Float32}()
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        2. Weights for the channel LLRs at each iteration: `weights_llrs`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

        """
        variance::Float32 = 0.01f0
        # We will initialize the learnable parameters to Gaussian random values around 1, if they are not explicitly provided.
        if (length(weights_c2v_v2c) == 0)
            weights_c2v_v2c = Dict{NTuple{5,Int}, Float32}()
            for (check, vertex) in base.edges
                for check_prime in base.neighbors_of_vertex[vertex]
                    if check_prime != check
                        for layer in 1:base.n_layers-1
                            # define the weight W^(t)_(v,c ; c',v)
                            if haskey(weights_c2v_v2c, (layer, vertex, check, check_prime, vertex)) == false
                                weights_c2v_v2c[(layer, vertex, check, check_prime, vertex)] = 1.0f0 + variance * randn(Float32)
                            end
                        end
                    end
                end
            end
        end
        if (length(weights_llrs) == 0)
            weights_llrs = Dict{NTuple{2,Int}, Float32}()
            for layer in 1:base.n_layers
                for vertex in 1:base.code_n_bits
                    # define the weight b^t_v
                    if haskey(weights_llrs, (layer, vertex)) == false
                        weights_llrs[(layer, vertex)] = 1.0f0 + variance * randn(Float32)
                    end
                end
            end
        end
        if (length(weights_c2v_readout) == 0)
            weights_c2v_readout = Dict{NTuple{4,Int}, Float32}()
            for (check, vertex) in base.edges
                # define the weight W^(t)_(v;c,v)
                for layer in 1:base.n_layers
                    if haskey(weights_c2v_readout, (layer, vertex, check, vertex)) == false
                        weights_c2v_readout[(layer, vertex, check, vertex)] = 1.0f0 + variance * randn(Float32)
                    end
                end
            end
        end
        
        return new(
            base,
            Float32.(initial_llrs), # Convert the initial LLRs to Float32.
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout
        )
    end
end

function build_vectorization_maps!(bpnn::NachmaniNeuralBP)
    """
    Build the vectorization maps for the learnable parameters in the NachmaniNeuralBP model.
    This allows us to easily convert the structured learnable parameters to a single 1D vector for optimization, and vice versa.
    """
    bpnn.invmap = Dict{Int, Tuple{NachmaniLearnableParameterGroup, Tuple}}()

    # Build the vectorization map for weights_c2v_v2c
    bpnn.vmap_c2v_v2c = Dict{NTuple{5,Int}, Int}()
    count::Int = 1
    for key in keys(bpnn.weights_c2v_v2c)
        bpnn.vmap_c2v_v2c[key] = count
        bpnn.invmap[count] = (C2V_V2C, key)
        count += 1
    end

    # Build the vectorization map for weights_llrs
    bpnn.vmap_llrs = Dict{NTuple{2,Int}, Int}()
    for key in keys(bpnn.weights_llrs)
        bpnn.vmap_llrs[key] = count
        bpnn.invmap[count] = (LLRS, key)
        count += 1
    end

    # Build the vectorization map for weights_c2v_readout
    bpnn.vmap_c2v_readout = Dict{NTuple{4,Int}, Int}()
    for key in keys(bpnn.weights_c2v_readout)
        bpnn.vmap_c2v_readout[key] = count
        bpnn.invmap[count] = (C2V_READOUT, key)
        count += 1
    end
    return nothing
end

function pack_params(bpnn::NachmaniNeuralBP)::Vector{Float32}
    """
    Pack the learnable parameters of the NachmaniNeuralBP model into a single 1D vector for optimization.
    The order of the parameters in the vector is determined by the vectorization maps built in `build_vectorization_maps!`.
    """
    n_params = length(bpnn.invmap)
    param_vector = zeros(Float32, n_params)
    for param_index in 1:n_params
        param_group, tuple_key = bpnn.invmap[param_index]
        if param_group == C2V_V2C
            param_vector[param_index] = bpnn.weights_c2v_v2c[tuple_key]
        elseif param_group == LLRS
            param_vector[param_index] = bpnn.weights_llrs[tuple_key]
        elseif param_group == C2V_READOUT
            param_vector[param_index] = bpnn.weights_c2v_readout[tuple_key]
        else
            error("Unknown parameter group: $param_group")
        end
    end
    return param_vector
end

function pack_grads(jacobian::Dict{NachmaniLearnableParameterGroup, Dict{NTuple, Float32}}, bpnn::NachmaniNeuralBP)::Vector{Float32}
    """
    Pack the gradients of the learnable parameters of the NachmaniNeuralBP model into a single 1D vector for optimization.
    The input `jacobian` is a dictionary containing the gradients for each group of learnable parameters, 
    The order of the gradients in the vector is determined by the vectorization maps built in `build_vectorization_maps!`.
    """
    n_params = length(bpnn.invmap)
    grad_vector = zeros(Float32, n_params)
    for param_index in 1:n_params
        param_group, tuple_key = bpnn.invmap[param_index]
        if param_group == C2V_V2C
            grad_vector[param_index] = jacobian[C2V_V2C][tuple_key]
        elseif param_group == LLRS
            grad_vector[param_index] = jacobian[LLRS][tuple_key]
        elseif param_group == C2V_READOUT
            grad_vector[param_index] = jacobian[C2V_READOUT][tuple_key]
        else
            error("Unknown parameter group: $param_group")
        end
    end
    return grad_vector
end

function unpack_params!(bpnn::NachmaniNeuralBP, param_vector::Vector{Float32})
    """
    Unpack the 1D vector of parameters back into the structured format of the NachmaniNeuralBP model.
    The order of the parameters in the vector is determined by the vectorization maps built in `build_vectorization_maps!`.
    This function modifies the `bpnn` object in-place.
    """
    n_params = length(param_vector)
    for param_index in 1:n_params
        param_group, tuple_key = bpnn.invmap[param_index]
        if param_group == C2V_V2C
            bpnn.weights_c2v_v2c[tuple_key] = param_vector[param_index]
        elseif param_group == LLRS
            bpnn.weights_llrs[tuple_key] = param_vector[param_index]
        elseif param_group == C2V_READOUT
            bpnn.weights_c2v_readout[tuple_key] = param_vector[param_index]
        else
            error("Unknown parameter group: $param_group")
        end
    end
    return nothing
end

function reset_to_standard_BP(bpnn::NachmaniNeuralBP)
    """
    Set the weights in the NachmaniNeuralBP model such that the model behaves like standard Belief Propagation.
    This is done by setting all of the following weights to 1.0:
        - weights_c2v_v2c
        - weights_llrs
        - weights_c2v_readout
    """
    # Set weights_c2v_v2c to 1.0
    reset_weights_c2v_v2c = bpnn.weights_c2v_v2c
    for key in keys(reset_weights_c2v_v2c)
        reset_weights_c2v_v2c[key] = 1.0f0
    end
    # Set weights_llrs to 1.0
    reset_weights_llrs = bpnn.weights_llrs
    for key in keys(reset_weights_llrs)
        reset_weights_llrs[key] = 1.0f0
    end
    # Set weights_c2v_readout to 1.0
    reset_weights_c2v_readout = bpnn.weights_c2v_readout
    for key in keys(reset_weights_c2v_readout)
        reset_weights_c2v_readout[key] = 1.0f0
    end

    # Return a new NachmaniNeuralBP model with the standard settings
    standard_bp_model = NachmaniNeuralBP(
        bpnn.base,
        bpnn.initial_llrs;
        weights_c2v_v2c=reset_weights_c2v_v2c,
        weights_llrs=reset_weights_llrs,
        weights_c2v_readout=reset_weights_c2v_readout
    )
    return standard_bp_model
end

# function pack()
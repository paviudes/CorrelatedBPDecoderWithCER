import Zygote: gradient, Params
using Functors: @functor

struct NachmaniNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the Nachmani et al. architecture for Neural Belief Propagation: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.200501.
    In this variant the number of trainable parameters (weights in the network) scales linearly with the number of layers.
    """
    base::NeuralBPBase
    weights_c2v_v2c::Vector{Float32}
    weights_llrs::Vector{Float32}
    weights_c2v_readout::Vector{Float32}
    weights_loss_layers::Vector{Float32}

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_llrs::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_loss_layers::Vector{Float32}=Vector{Float32}(undef, 0)
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        2. Weights for the connections from C2V to V2C: `weights_llrs`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`
        4. Weights for the loss at each layer: `weights_loss_layers`

        """
        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_weights_c2v_v2c * base.n_layers)
        end
        if (size(weights_llrs, 1) == 0)
            weights_llrs = randn(Float32, base.code_n_bits * base.n_layers)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.nb_weights_c2v_readout)
        end
        if (size(weights_loss_layers, 1) == 0)
            weights_loss_layers = randn(Float32, base.n_layers)
        end

        return new(
            base,
            # learnable_parameters,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            weights_loss_layers
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        weights_c2v_v2c::Vector{Float32},
        weights_llrs::Vector{Float32},
        weights_c2v_readout::Vector{Float32},
        weights_loss_layers::Vector{Float32}
    )
        return new(
            base,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            weights_loss_layers
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NachmaniNeuralBP (weights_c2v_v2c, weights_llrs, weights_c2v_readout, weights_loss_layers)

function (bpnn::NachmaniNeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)::Array{Float32, 3}
    """
    Forward pass through the Neural Network for Nachmani et al. architecture.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0)).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    v2c_activated_neurons = safe_log_tanh(v2c_neurons) # Use safe version to avoid numerical issues.

    # Initialize the intermediate LLRs
    posterior_llrs = Vector{Matrix{Float32}}()

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    # selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
    
    # Accumulate all the posterior LLRs. This is because a for loop where we populate the LLRs cannot be acceptable for Zygote.
    v2c_state = v2c_activated_neurons
    posterior_llrs_tuple = ()

    for layer in 1:bpnn.base.n_layers
        v2c_state, llr = compute_llr_at_layer_t(
            bpnn,
            v2c_state,
            syndromes_batch,
            initial_llrs_batch,
            layer
        )

        posterior_llrs_tuple = (posterior_llrs_tuple..., llr)
    end

    posterior_llrs = collect(posterior_llrs_tuple)
    posterior_llrs_tensor = stack(posterior_llrs; dims=3)

    return posterior_llrs_tensor
end

function compute_llr_at_layer_t(bpnn::NachmaniNeuralBP, v2c_activated_neurons_previous_layer::Matrix{ComplexF32}, syndromes_batch::BitMatrix, initial_llrs_batch::AbstractMatrix{<:Real}, layer::Int)::Tuple{Matrix{ComplexF32}, Matrix{Float32}}
    """
    Compute the LLRs at layer `t` given the activated neurons at the V2C layer and the syndromes, at layer `t-1`.

    We have to compute two forward passes:
    1. Computing messages m_(c->v) from messages m_(v->c) and the syndrome information.
       This is given by
       a(m_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m_(v'->c))
       where a(m) is the activation function for the C2V layer, given by a(m) = log(tanh(m/2))
    2. Computing messages m_(v->c) from messages m_(c->v) and the initial LLRs.
       This is given by
       m_(v->c) = b_v l_v + ∑_(c' ∈ N(v) - c) m_(c'->v) W^t_(v,c;c',v)
    
    Finally, we compute the LLRs at layer `t` from the messages m_(c->v) and the initial LLRs using the readout weights:
    μ^t_v = l_v + ∑_(c ∈ N(v)) m_(c->v) W^t_(v; c,v)

    """
    #=
    1. We want to implement the message passing rule:
    a(m_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m_(v'->c))
    =#
    c2v_activated_neurons = bpnn.base.adj_V2C_C2V * v2c_activated_neurons_previous_layer + im * Float32(π) * syndromes_batch[bpnn.base.neuron_to_checks, 1:end]
    # Apply the activation function for C2V layer.
    c2v_neurons = safe_atanh_exp(c2v_activated_neurons) # Use safe version to avoid numerical issues.
    
    #=
    2. We want to implement the message passing rule:
    m_(v->c) = b_v l_v + ∑_(c' ∈ N(v) - c) m_(c'->v) W^t_(v,c';v,c)
    =#
    weights_start = (layer - 1) * bpnn.base.nb_weights_c2v_v2c + 1
    weights_end = layer * bpnn.base.nb_weights_c2v_v2c
    extended_weights_c2v_v2c = sparse(bpnn.base.non_zero_rows_C2V_V2C, bpnn.base.non_zero_cols_C2V_V2C, bpnn.weights_c2v_v2c[weights_start:weights_end], bpnn.base.nb_neurons_per_layer, bpnn.base.nb_neurons_per_layer) |> Matrix
    # Compute the weights for the LLRs at layer `t`.
    weights_start = (layer - 1) * bpnn.base.code_n_bits + 1
    weights_end = layer * bpnn.base.code_n_bits
    weights_llrs = bpnn.weights_llrs[weights_start:weights_end]
    # Compute all the messages m_(v->c) using matrix operations.
    v2c_neurons = (extended_weights_c2v_v2c .* bpnn.base.adj_C2V_V2C) * c2v_neurons + (bpnn.base.adj_initialize_V2C * (initial_llrs_batch .* weights_llrs))

    # Apply the activation functions for V2C layer.
    v2c_activated_neurons = safe_log_tanh(v2c_neurons) # Use safe version to avoid numerical issues.

    # 3. Compute the LLRs at layer `t` from the messages m_(c->v) and the initial LLRs using the readout weights:
    extended_weights_c2v_readout = sparse(bpnn.base.non_zero_rows_C2V_readout, bpnn.base.non_zero_cols_C2V_readout, bpnn.weights_c2v_readout, bpnn.base.code_n_bits, bpnn.base.nb_neurons_per_layer) |> Matrix
    posteriod_llr_at_layer_t = (weights_llrs .* initial_llrs_batch) .+ (extended_weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_neurons

    return (v2c_activated_neurons, posteriod_llr_at_layer_t)
end
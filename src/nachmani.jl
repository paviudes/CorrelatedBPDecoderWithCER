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

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_llrs::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0),
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        2. Weights for the connections from C2V to V2C: `weights_llrs`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`
        4. Importance of the correlation penalty in the Loss function: `correlation_importance`
        5. Temperature for the smooth minimum approximation when combining losses from different layers: `loss_layer_regularizer`
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

        return new(
            base,
            # learnable_parameters,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        weights_c2v_v2c::Vector{Float32},
        weights_llrs::Vector{Float32},
        weights_c2v_readout::Vector{Float32},
    )
        return new(
            base,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NachmaniNeuralBP (weights_c2v_v2c, weights_llrs, weights_c2v_readout)


function c2v_to_v2c_with_weights!(
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    messages_v2c,
    messages_c2v,
    weighted_channel_llrs,
    weights_llr_layer,
    weights_c2v_v2c_layer,
    initial_llrs_batch,
    base
)
    """
    Similar to `c2v_to_v2c!`, but with the weights for the current layer passed as arguments, so that it can be used with Enzyme.jl.
    """
    # update sparse values instead of rebuilding structure
    # weight_matrix_v2c.nzval .= weights_c2v_v2c_layer

    # compute: ∑_(c' ∈ N(v) - c) w^(t-1)_(v,c;c',v) m^(t-1)_(c'->v)
    sparse_multiply!(
        messages_v2c,
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_c2v_v2c_layer,
        messages_c2v
    )
    
    # channel contribution
    # @. weighted_channel_llrs = weights_llr_layer * initial_llrs_batch
    @inbounds for j in axes(initial_llrs_batch, 2) # also use @simd to tell the compiler that this loop can be vectorized, since there are no data dependencies between iterations
        for i in axes(initial_llrs_batch, 1)
            weighted_channel_llrs[i, j] = weights_llr_layer[i, j] * initial_llrs_batch[i, j]
        end
    end
    
    mul!(messages_v2c, base.adj_initialize_V2C, weighted_channel_llrs, 1f0, 1f0)
    # apply the activation function a(x) = log(tanh(x/2)) split into magnitude + sign
    safe_log_tanh_split!(activated_m_v2c_magnitudes, activated_m_v2c_signs, messages_v2c)
    return nothing
end





function readout_with_weights!(
    posterior_llrs,
    messages_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base
)
    """
    Similar to `readout!`, but with the weights for the current layer passed as arguments, so that it can be used with Enzyme.jl.
    Compute the LLRs at the readout layer given the activated neurons at the C2V layer and the initial LLRs, for a single layer.
    μ^t_v = b^(t)_v l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^t_(v; c,v) is a weight.
    """
    #=
    weight_matrix_readout.nzval .= weights_readout
    
    # Compute ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    mul!(posterior_llrs, weight_matrix_readout, messages_c2v)
    =#
    # Compute ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    sparse_multiply!(
        posterior_llrs,
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        weights_readout,
        messages_c2v
    )

    # Add the weighted channel LLR contribution, i.e b^(t)_v l_v
    @inbounds for j in axes(posterior_llrs, 2)
        for i in axes(posterior_llrs, 1)
            posterior_llrs[i, j] += weights_llrs[i] * channel_llrs[i, j]
        end
    end
    return nothing
end

@inline function get_layer_weights(
    weights_c2v_v2c,
    weights_llrs,
    base,
    layer,
    nsamples
)
    """
    Slice the weights relevant for the current layer.
    """
    weights_start = (layer - 1) * base.nb_weights_c2v_v2c + 1
    weights_end   = layer * base.nb_weights_c2v_v2c

    weights_messages = @view weights_c2v_v2c[weights_start:weights_end]

    weights_llr_layer = weights_llrs[
        (layer - 1)*base.code_n_bits + 1 : layer*base.code_n_bits
    ] .* ones(Float32, 1, nsamples)

    return weights_messages, weights_llr_layer
end

function compute_layer_with_weights!(
    # Intermediate messages
    messages_c2v,
    activated_m_c2v_magnitudes,
    activated_m_c2v_signs,
    messages_v2c,
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    # buffers for contributions to the messages and readout
    weighted_channel_llrs,
    posterior_llrs,
    # inputs
    syndromes_batch,
    initial_llrs_batch,
    # Explicit learanable weights as arguments for Enzyme.jl compatibility
    weights_c2v_v2c,
    weights_llrs,
    weights_c2v_readout,
    # constant arguments
    base,
    layer,
    nsamples
)
    """
    Compute one layer forward transition in the Neural BP model (one iteration of BP).
    Same as compute_layer!, but uses explicit weights instead of bpnn.
    This version is for the in-place version of the forward pass, with explicit weight arguments, so that it's friendly for Enzyme.jl.
    """
    # ---- Slice the weights relevant for the current layer ----
    weights_c2v_v2c_layer, weights_llr_layer = get_layer_weights(weights_c2v_v2c, weights_llrs, base, layer, nsamples)

    # nsamples = size(initial_llrs_batch, 2)
    # -------------------------
    # 1. C2V → V2C
    # -------------------------
    c2v_to_v2c_with_weights!(
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        messages_v2c,
        messages_c2v,
        weighted_channel_llrs,
        weights_llr_layer,
        weights_c2v_v2c_layer,
        initial_llrs_batch,
        base
    )
    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    v2c_to_c2v!(
        messages_c2v,
        activated_m_c2v_magnitudes,
        activated_m_c2v_signs,
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base
    )
    # -------------------------
    # 3. Readout
    # -------------------------
    readout_with_weights!(
        posterior_llrs,
        messages_c2v,
        weights_c2v_readout,
        weights_llr_layer,
        initial_llrs_batch,
        base
    )
    return nothing
end

function (bpnn::NachmaniNeuralBP)(
    initial_llrs_batch::AbstractMatrix{<:Real},
    syndromes_batch::BitMatrix
)
    """
    Forward pass through the Neural Network for Nachmani et al. architecture, returning the LLRs at each layer.
    """

    base = bpnn.base
    n_batches = size(initial_llrs_batch, 2)
    neurons_per_layer = base.nb_neurons_per_layer

    # Buffers
    messages_c2v      = zeros(Float32, neurons_per_layer, n_batches)

    # C2V buffers (magnitude + sign)
    m_c2v_magnitudes  = similar(messages_c2v)
    m_c2v_signs       = falses(size(messages_c2v))

    # V2C buffers
    messages_v2c      = similar(messages_c2v)

    # V2C activated (magnitude + sign)
    m_v2c_magnitudes  = similar(messages_v2c)
    m_v2c_signs       = falses(size(messages_v2c))

    # Channel contribution buffer
    weighted_channel_llrs = similar(initial_llrs_batch)
    # this will carry weights_llrs .* channel_llrs to avoid recomputing

    posterior_llrs_layer = zeros(Float32, base.code_n_bits, n_batches)

    # Sparse templates
    weight_matrix_v2c = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        zeros(Float32, length(base.non_zero_rows_C2V_V2C)),
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )

    weight_matrix_readout = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        zeros(Float32, length(base.non_zero_rows_C2V_readout)),
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    posterior_llrs = Zygote.Buffer(zeros(Float32, base.code_n_bits, n_batches, base.n_layers))

    for layer in 1:base.n_layers
        compute_layer!(
            messages_c2v,
            m_c2v_magnitudes,
            m_c2v_signs,
            messages_v2c,
            m_v2c_magnitudes,
            m_v2c_signs,
            weighted_channel_llrs,
            posterior_llrs_layer,
            syndromes_batch,
            initial_llrs_batch,
            bpnn,
            layer,
            weight_matrix_v2c,
            weight_matrix_readout
        )

        posterior_llrs[:, :, layer] = posterior_llrs_layer
    end

    return copy(posterior_llrs)
end

function forward_pass_with_weights(
    weights_c2v_v2c,
    weights_llrs,
    weights_c2v_readout,
    base,
    initial_llrs_batch,
    syndromes_batch
)
    """
    Forward pass with explicit weights as arguments, so that it's friendly for Enzyme.jl.
    """
    n_samples = size(initial_llrs_batch, 2)
    neurons_per_layer = base.nb_neurons_per_layer

    # Define the messages m^t_(c→v), where at t=0, they are initialized to zeros.
    messages_c2v = zeros(Float32, neurons_per_layer, n_samples)
    messages_v2c = zeros(Float32, neurons_per_layer, n_samples)

    # Buffers for the activated magnitudes and signs at C2V and V2C layers
    activated_m_c2v_magnitudes = similar(messages_c2v)
    activated_m_c2v_signs = falses(size(messages_c2v))
    activated_m_v2c_magnitudes = similar(messages_c2v)
    activated_m_v2c_signs = falses(size(messages_c2v))

    # Buffer for the weighted channel LLRs
    weighted_channel_llrs = Matrix{Float32}(undef, size(initial_llrs_batch))

    # Storing the posterior LLRs at each layer.
    posterior_llrs_layer = zeros(Float32, base.code_n_bits, n_samples)

    # posterior LLRs for all layers, as a 3D tensor: (n_bits × n_samples × n_layers)
    posterior_llrs = zeros(Float32, base.code_n_bits, n_samples, base.n_layers)

    # Forward pass through layers
    for layer in 1:base.n_layers
        compute_layer_with_weights!(
            messages_c2v,
            activated_m_c2v_magnitudes,
            activated_m_c2v_signs,
            messages_v2c,
            activated_m_v2c_magnitudes,
            activated_m_v2c_signs,
            weighted_channel_llrs,
            posterior_llrs_layer,
            syndromes_batch,
            initial_llrs_batch,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            base,
            layer,
            n_samples,
            #weight_matrix_v2c,
            #weight_matrix_readout
        )
        posterior_llrs[:, :, layer] .= posterior_llrs_layer
    end
    return posterior_llrs
end

function forward_pass_with_weights(
    bpnn::NachmaniNeuralBP,
    initial_llrs_batch,
    syndromes_batch
)
    """
    Only for testing purposes: forward pass with explicit weights as arguments, so that it's friendly for Enzyme.jl.
    """
    return forward_pass_with_weights(
        bpnn.weights_c2v_v2c,
        bpnn.weights_llrs,
        bpnn.weights_c2v_readout,
        bpnn.base,
        initial_llrs_batch,
        syndromes_batch
    )
end
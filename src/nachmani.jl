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

function v2c_to_c2v!(
    messages_c2v,                            # output (final LLRs)
    activated_m_c2v_magnitudes,       # buffer: aggregated magnitudes
    activated_m_c2v_signs,            # buffer: aggregated signs
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    syndromes_batch,
    base
)
    """
    Compute the messages from check nodes to variable nodes (C2V) given the messages from variable nodes to check nodes (V2C) and the syndromes, for a single layer.

    We have to compute the following for each edge (c, v):

        a(m^t_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m^t_(v'->c))

    where
    - s_c is the syndrome bit corresponding to check node c
    - N(c) is the set of variable nodes connected to check node c.

    We then have to apply the inverse activation function to get the messages m^t_(c->v) from a(m^t_(c->v)): 
        m^t_(c->v) = a^(-1)(a(m^t_(c->v)))
    where a^(-1)(x) = 2 * atanh(exp(x)).

    Instead of representing a(m^t_(c->v)) as a complex number, which needs to be exponentiated, we note that
    exp(i π s_c + ∑ a(m^t_(v'->c))) = exp(∑ a(m^t_(v'->c))) * (-1)^(s_c)
                                    = exp(∑ |a(m^t_(v'->c))|) * (-1)^(s_c + parity of signs of incoming messages)

    So we will
    - magnitudes sum linearly
    - combine the signs with the syndromes via XOR (parity)

    Final output:
        exp(x) = exp(magnitude) * (-1)^sign
    """

    # -------------------------
    # 1. Magnitude aggregation (same as before, but real)
    # -------------------------
    mul!(activated_m_c2v_magnitudes, base.adj_V2C_C2V, activated_m_v2c_magnitudes)

    # -------------------------
    # 2. Set the signs to the parity of the incoming signs, XOR with syndrome
    # -------------------------
    xor_affine!(activated_m_c2v_signs, base.adj_V2C_C2V, activated_m_v2c_signs, syndromes_batch[base.neuron_to_checks, :])
    
    # -------------------------
    # 3. Activation: 2 * atanh(exp(x))
    # -------------------------
    safe_atanh_exp_signed!(
        messages_c2v,
        activated_m_c2v_magnitudes,
        activated_m_c2v_signs
    )

    return nothing
end

function v2c_to_c2v(
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    syndromes_batch,
    base
)
    """
    Functional (Zygote-friendly) V2C → C2V.
    """

    # Magnitudes: standard sparse matmul
    activated_m_c2v_magnitudes = base.adj_V2C_C2V * activated_m_v2c_magnitudes

    # Signs: accumulate parity via adjacency
    # Convert Bool → Int for summation
    signs_int = Int.(activated_m_v2c_signs)

    parity = base.adj_V2C_C2V * signs_int

    # mod 2 for XOR
    activated_m_c2v_signs = mod.(parity, 2) .== 1

    # add syndrome
    activated_m_c2v_signs = xor.(activated_m_c2v_signs, syndromes_batch[base.neuron_to_checks, :])

    # Compute the messages from check to vertex by applying the inverse activation function safely
    m_c2v = safe_atanh_exp_signed(activated_m_c2v_magnitudes, activated_m_c2v_signs)

    return m_c2v
end

function c2v_to_v2c!(
    activated_m_v2c_magnitudes,        # output: magnitudes (log-domain)
    activated_m_v2c_signs,             # output: sign bits
    messages_v2c,                   # buffer (pre-activation real values)
    weighted_channel_llrs,   # buffer (same size as channel_llrs)
    messages_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base,
    weight_matrix
)
    """
    Compute the messages from variable nodes to check nodes (V2C) given the messages from check nodes to variable nodes (C2V) and the initial LLRs, for a single layer.

    m^(t)_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)

    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^(t-1)_(v,c;c',v) is a weight.

    Instead of returning complex values, we represent:
        log(tanh(x/2)) = magnitude + i π * sign

    as:
    - magnitude: log(tanh(|x|/2))
    - sign:      1_{x < 0}
    """

    # update sparse values instead of rebuilding structure
    weight_matrix.nzval .= weights_messages

    # compute: ∑_(c' ∈ N(v) - c) ...
    mul!(messages_v2c, weight_matrix .* base.adj_C2V_V2C, messages_c2v_previous)

    # ---- channel contribution ----
    # weighted_channel_llrs = weights_llrs .* channel_llrs
    @. weighted_channel_llrs = weights_llrs * channel_llrs

    # m_v2c += base_projection * weighted_channel_llrs
    mul!(messages_v2c, base.adj_initialize_V2C, weighted_channel_llrs, 1f0, 1f0)

    # apply the activation function a(x) = log(tanh(x/2)) split into magnitude + sign
    safe_log_tanh_split!(activated_m_v2c_magnitudes, activated_m_v2c_signs, messages_v2c)

    return nothing
end

function c2v_to_v2c_with_weights!(
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    messages_v2c,
    messages_c2v,
    weighted_channel_llrs,
    weights_llr_layer,
    weights_c2v_v2c_layer,
    initial_llrs_batch,
    base,
    weight_matrix_v2c
)
    """
    Similar to `c2v_to_v2c!`, but with the weights for the current layer passed as arguments, so that it can be used with Enzyme.jl.
    """
    # update sparse values instead of rebuilding structure
    weight_matrix_v2c.nzval .= weights_c2v_v2c_layer

    # compute: ∑_(c' ∈ N(v) - c) ...
    mul!(messages_v2c, weight_matrix_v2c .* base.adj_C2V_V2C, messages_c2v)
    # channel contribution
    @. weighted_channel_llrs = weights_llr_layer * initial_llrs_batch
    mul!(messages_v2c, base.adj_initialize_V2C, weighted_channel_llrs, 1f0, 1f0)
    # apply the activation function a(x) = log(tanh(x/2)) split into magnitude + sign
    safe_log_tanh_split!(activated_m_v2c_magnitudes, activated_m_v2c_signs, messages_v2c)
    return nothing
end

function c2v_to_v2c(
    m_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base
)
    """
    Functional (Zygote-friendly) version of C2V → V2C.
    """

    weight_matrix = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_messages,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )

    # Compute \sum_(c' ∈ N(v) - c) W^(t-1)_(v,c;c',v) m^(t-1)_(c'->v)
    m_v2c = (weight_matrix .* base.adj_C2V_V2C) * m_c2v_previous
    # Compute b^(t)_v l_v contribution
    scaled_llrs = base.adj_initialize_V2C * (weights_llrs .* channel_llrs)
    m_v2c = m_v2c .+ scaled_llrs
    
    # apply activation
    activated_m_v2c_magnitudes, activated_m_v2c_signs =
        safe_log_tanh_split(m_v2c)

    return (activated_m_v2c_magnitudes, activated_m_v2c_signs)
end

function readout!(
    posterior_llrs,
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base,
    weight_matrix
)
    """
    Compute the LLRs at the readout layer given the activated neurons at the C2V layer and the initial LLRs, for a single layer.
    μ^t_v = b^(t)_v l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^t_(v; c,v) is a weight.
    """

    weight_matrix.nzval .= weights_readout

    mul!(posterior_llrs, weight_matrix .* base.adj_C2V_readout, m_c2v)

    @. posterior_llrs += weights_llrs * channel_llrs

    return nothing
end

function readout_with_weights!(
    posterior_llrs,
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base,
    weight_matrix
)
    """
    Similar to `readout!`, but with the weights for the current layer passed as arguments, so that it can be used with Enzyme.jl.
    """
    weight_matrix.nzval .= weights_readout

    mul!(posterior_llrs, weight_matrix .* base.adj_C2V_readout, m_c2v)

    @. posterior_llrs += weights_llrs * channel_llrs

    return nothing
end

function readout(
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base
)
    """
    Functional (Zygote-friendly) version of readout.
    """

    # build sparse weight matrix (no nzval mutation)
    weight_matrix = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        weights_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    # message contribution
    posterior_llrs =
        (weight_matrix .* base.adj_C2V_readout) * m_c2v

    # channel contribution
    posterior_llrs =
        posterior_llrs .+ (weights_llrs .* channel_llrs)

    return posterior_llrs
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

function compute_layer!(
    messages_c2v,
    activated_m_c2v_magnitudes,         # buffer: C2V magnitudes
    activated_m_c2v_signs,              # buffer: C2V signs
    messages_v2c,
    activated_m_v2c_magnitudes,         # buffer: V2C magnitudes
    activated_m_v2c_signs,              # buffer: V2C signs
    weighted_channel_llrs,    # buffer: weights_llrs .* channel_llrs
    posterior_llrs,
    syndromes_batch,
    initial_llrs_batch,
    bpnn,
    layer,
    weight_matrix_v2c,
    weight_matrix_readout
)
    """
    Compute the LLRs at layer `t` given the activated neurons at the V2C layer and the syndromes, at layer `t-1`.

    We have to compute two forward passes:
    1. m^t_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)
    2. a(m^t_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m^t_(v'->c))

    Instead of complex numbers:
    - we store magnitudes and signs separately
    - signs are propagated via XOR (parity)

    We compute the LLRs at layer `t` from the messages m^t_(c->v) and the initial LLRs using the readout weights:
    μ^t_v = l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    """

    base = bpnn.base
    nsamples = size(initial_llrs_batch, 2)

    # ---- Slice the weights relevant for the current layer ----
    weights_messages, weights_llr = get_layer_weights(bpnn.weights_c2v_v2c, bpnn.weights_llrs, base, layer, nsamples)

    # -------------------------
    # 1. C2V → V2C
    # -------------------------
    c2v_to_v2c!(
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        messages_v2c,
        weighted_channel_llrs,
        messages_c2v,
        weights_llr,
        weights_messages,
        initial_llrs_batch,
        base,
        weight_matrix_v2c
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
    readout!(
        posterior_llrs,
        messages_c2v,
        bpnn.weights_c2v_readout,
        weights_llr,
        initial_llrs_batch,
        base,
        weight_matrix_readout
    )

    return nothing
end

function compute_layer(
    messages_c2v,
    syndromes_batch,
    initial_llrs_batch,
    bpnn,
    layer
)
    """
    Functional (Zygote-friendly) version of one layer.
    """

    base = bpnn.base
    nsamples = size(initial_llrs_batch, 2)

    # ---- weights ----
    weights_start = (layer - 1) * base.nb_weights_c2v_v2c + 1
    weights_end   = layer * base.nb_weights_c2v_v2c

    weights_messages = @view bpnn.weights_c2v_v2c[weights_start:weights_end]

    weights_llr = bpnn.weights_llrs[
        (layer - 1)*base.code_n_bits + 1 : layer*base.code_n_bits
    ] .* ones(Float32, 1, nsamples)

    # -------------------------
    # 1. C2V → V2C
    # -------------------------
    m_v2c_magnitudes, m_v2c_signs =
        c2v_to_v2c(
            messages_c2v,
            weights_llr,
            weights_messages,
            initial_llrs_batch,
            base
        )

    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    messages_c2v_new =
        v2c_to_c2v(
            m_v2c_magnitudes,
            m_v2c_signs,
            syndromes_batch,
            base
        )

    # -------------------------
    # 3. Readout
    # -------------------------
    weight_matrix_readout = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        bpnn.weights_c2v_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    posterior_llrs =
        (weights_llr .* initial_llrs_batch) .+
        (weight_matrix_readout .* base.adj_C2V_readout) * messages_c2v_new

    return (messages_c2v_new, posterior_llrs)
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
    nsamples,
    # matrix templates for in-place operations
    weight_matrix_v2c,
    weight_matrix_readout
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
        base,
        weight_matrix_v2c
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
        base,
        weight_matrix_readout
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

function forward_pass(bpnn, initial_llrs_batch, syndromes_batch)
    """
    Functional version of the forward pass: bpnn(initial_llrs_batch, syndromes_batch), returning the LLRs at each layer as a vector of matrices.
    """

    base = bpnn.base
    n_batches = size(initial_llrs_batch, 2)

    messages_c2v = zeros(Float32, base.nb_neurons_per_layer, n_batches)

    posterior_list = Vector{Matrix{Float32}}(undef, base.n_layers)

    for layer in 1:base.n_layers
        messages_c2v, posterior_llrs =
            compute_layer(
                messages_c2v,
                syndromes_batch,
                initial_llrs_batch,
                bpnn,
                layer
            )

        posterior_list[layer] = posterior_llrs
    end

    # Stack into 3D tensor: (n_bits × n_batches × n_layers)
    posterior_llrs_all = cat(posterior_list...; dims=3)

    return posterior_llrs_all
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
    weighted_channel_llrs = similar(initial_llrs_batch)

    # Storing the posterior LLRs at each layer.
    posterior_llrs_layer = zeros(Float32, base.code_n_bits, n_samples)

    # Sparse templates for weight matrices
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
            weight_matrix_v2c,
            weight_matrix_readout
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
function v2c_to_c2v!(
    messages_c2v,                     # output (final LLRs)
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
    # compute: ∑_(c' ∈ N(v) - c) w^(t-1)_(v,c;c',v) m^(t-1)_(c'->v)
    sparse_multiply!(
        messages_v2c,
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_c2v_v2c_layer,
        messages_c2v
    )
    # channel contribution
    # Precompute weighted_channel_llrs = weights_llr_layer * initial_llrs_batch
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
    coupling_scale,
    # constant arguments
    base,
    layer,
    nsamples
)
    """
    Compute one layer forward transition in the Neural BP model (one iteration of BP).
    Same as compute_layer!, but uses explicit weights instead of bpnn.
    This version is for the in-place version of the forward pass, with explicit weight arguments, so that it's friendly for Enzyme.jl.

    `coupling_scale` is a length-1 vector holding α ITSELF (already through the
    logistic link — see `forward_pass_with_weights`), the scale on the CER
    couplings inside the enriched check node. Read only when
    `base.check_node_kind == CHECK_NODE_ENRICHED`; the standard rule ignores it.
    """
    # ---- Slice the weights relevant for the current layer ----
    weights_c2v_v2c_layer, weights_llr_layer = get_layer_weights(weights_c2v_v2c, weights_llrs, base, layer, nsamples)

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
    # Enriched check node: the standard rule above has filled every row; the
    # kernel overwrites the rows of checks that carry CER couplings, using the
    # RAW v2c messages of this layer (not their log-tanh activations).
    if base.check_node_kind == CHECK_NODE_ENRICHED
        apply_enriched_checks!(
            messages_c2v,
            messages_v2c,
            syndromes_batch,
            coupling_scale,
            base.soft_check_tables
        )
    end
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

function forward_pass_with_weights(
    weights_c2v_v2c,
    weights_llrs,
    weights_c2v_readout,
    coupling_logit,
    coupling_schedule,
    base,
    initial_llrs_batch,
    syndromes_batch
)
    """
    Forward pass with explicit weights as arguments, so that it's friendly for Enzyme.jl.

    `coupling_logit` is the length-1 vector holding θ, the UNCONSTRAINED trained
    parameter. The logistic link α = 1/(1 + exp(-θ)) is applied exactly HERE,
    once per forward pass rather than once per layer, and α is what travels down
    to the check-node kernel. Putting the link at this level is what keeps α
    inside (0, 1) for free while leaving the kernel able to take any α — which
    is why `α = 0` is still expressible and still reproduces the tanh rule.

    `coupling_schedule` is the length-2 vector [T₀, log(w - W_MIN)] of the layer
    schedule. Under `COUPLING_SCHEDULE_STEP` each layer t receives
    α_t = α · d(t) with d the logistic step of `coupling_schedule_damping`;
    under `COUPLING_SCHEDULE_CONSTANT` every layer receives α and the vector is
    never read. The damping is evaluated per layer inside the differentiated
    region, so the gradient with respect to [T₀, ρ] is exact.

    Both conversions are inside the differentiated region, so Enzyme carries the
    chains θ -> α -> messages -> loss and [T₀, ρ] -> d(t) -> messages -> loss,
    and the reported gradients are with respect to the parameters Adam updates.
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

    # θ -> α, once. Ignored by the standard check node, but computed
    # unconditionally so the differentiated code path does not branch on it.
    linked_coupling_scale::Float32 = coupling_scale_from_logit(coupling_logit[1])

    # Forward pass through layers
    for layer in 1:base.n_layers
        # α_t for this layer. A fresh length-1 vector per layer rather than one
        # buffer mutated in place: the kernel reads it after this point and the
        # reverse pass needs each layer's value, and a new allocation gives
        # Enzyme nothing to disambiguate. Ninety 4-byte allocations per forward
        # pass is not a cost worth an aliasing question.
        layer_damping::Float32 = 1.0f0
        if base.coupling_schedule_kind == COUPLING_SCHEDULE_STEP
            layer_damping = coupling_schedule_damping(coupling_schedule, layer)
        end
        layer_coupling_scale::Vector{Float32} = Float32[linked_coupling_scale * layer_damping]
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
            layer_coupling_scale,
            base,
            layer,
            n_samples
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
    Forward pass of a model, unpacking its weights into the explicit-weight version above.
    """
    return forward_pass_with_weights(
        bpnn.weights_c2v_v2c,
        bpnn.weights_llrs,
        bpnn.weights_c2v_readout,
        bpnn.coupling_logit,
        bpnn.coupling_schedule,
        bpnn.base,
        initial_llrs_batch,
        syndromes_batch
    )
end
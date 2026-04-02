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

#=
function v2c_to_c2v(
    m_v2c_activated,
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
    """
    m_c2v_activated = base.adj_V2C_C2V * m_v2c_activated + im * Float32(π) .* syndromes_batch[base.neuron_to_checks, :]
    # apply the activation function a(x) = 2 * atanh(exp(x))
    m_c2v = safe_atanh_exp(m_c2v_activated)
    return m_c2v
end

function c2v_to_v2c(
    m_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base_projection,
    base
)
    """
    Compute the messages from variable nodes to check nodes (V2C) given the messages from check nodes to variable nodes (C2V) and the initial LLRs, for a single layer.
    m^(t)_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)
    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^(t-1)_(v,c;c',v) is a weight.
    """
    # compute: ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)
    weight_matrix = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_messages,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )
    # println("Matrix sizes:\nbase_projection: ", size(base_projection), "\nweights_llrs .* channel_llrs: ", size(weights_llrs .* channel_llrs), "\nweight_matrix .* base.adj_C2V_V2C: ", size(weight_matrix .* base.adj_C2V_V2C), "\nm_c2v_previous: ", size(m_c2v_previous))
    m_v2c = base_projection * (weights_llrs .* channel_llrs) .+ (weight_matrix .* base.adj_C2V_V2C) * m_c2v_previous
    # apply the activation function a(x) = log(tanh(x/2))
    m_v2c_activated = safe_log_tanh(m_v2c)
    return m_v2c_activated
end

function readout(
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base
)
    """
    Compute the LLRs at the readout layer given the activated neurons at the C2V layer and the initial LLRs, for a single layer.
    μ^t_v = b^(t)_v l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^t_(v; c,v) is a weight.
    """
    weight_matrix = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        weights_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )
    posterior_llrs = (weights_llrs .* channel_llrs) .+ (weight_matrix .* base.adj_C2V_readout) * m_c2v

    return posterior_llrs
end

function compute_layer!(
    messages_c2v,
    syndromes_batch,
    initial_llrs_batch,
    base_projection,
    bpnn,
    layer
)
    """
    Compute the LLRs at layer `t` given the activated neurons at the V2C layer and the syndromes, at layer `t-1`.

    We have to compute two forward passes:
    1. m^t_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)
    2. a(m^t_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m^t_(v'->c))
    
    We compute the LLRs at layer `t` from the messages m^t_(c->v) and the initial LLRs using the readout weights:
    μ^t_v = l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)

    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^(t-1)_(v,c;c',v) is a weight for the message from check node c' to variable node v in layer t-1
    - W^t_(v; c,v) is a weight for the message from check node c to variable node v in layer t
    - s_c is the syndrome bit corresponding to check node c
    - N(c) is the set of variable nodes connected to check node c.
    - a(x) is the activation function for the messages from check nodes to variable nodes, defined as a(x) = 2 * atanh(exp(x)).
    - b^(t)_v is a weight for the initial LLR corresponding to variable node v in layer t.
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
    messages_v2c_activated = c2v_to_v2c(messages_c2v, weights_llr, weights_messages, initial_llrs_batch, base_projection, base)
    
    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    messages_c2v = v2c_to_c2v(messages_v2c_activated, syndromes_batch, base)

    # -------------------------
    # 3. Readout
    # -------------------------
    posterior_llrs = readout(messages_c2v, bpnn.weights_c2v_readout, weights_llr, initial_llrs_batch, base)

    return posterior_llrs
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

    # Buffers to store the messages and the posteriod LLRs at each layer.
    messages_c2v = zeros(Float32, neurons_per_layer, n_batches)
    posterior_llrs = Zygote.Buffer(zeros(Float32, base.code_n_bits, n_batches, base.n_layers))

    for layer in 1:base.n_layers
        posterior_llrs[:, :, layer] = compute_layer!(
            messages_c2v,
            syndromes_batch,
            initial_llrs_batch,
            base.adj_initialize_V2C,
            bpnn,
            layer
        )
    end

    return copy(posterior_llrs)
end
=#

function v2c_to_c2v!(
    m_c2v,                 # output (final)
    m_c2v_activated,       # buffer (same size)
    m_v2c_activated,
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
    """

    # 1. m_c2v_activated = A * m_v2c_activated
    mul!(m_c2v_activated, base.adj_V2C_C2V, m_v2c_activated)

    # 2. add syndrome term in-place (avoid allocating new matrix)
    @views @. m_c2v_activated += im * Float32(π) * syndromes_batch[base.neuron_to_checks, :]

    # 3. apply the activation function a(x) = 2 * atanh(exp(x)) in-place
    safe_atanh_exp!(m_c2v, m_c2v_activated)

    return nothing
end

function c2v_to_v2c!(
    m_v2c_activated,   # output
    m_v2c,             # buffer
    scaled_llrs,           # NEW buffer (same size as channel_llrs)
    m_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base_projection,
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
    """

    # update sparse values instead of rebuilding structure
    weight_matrix.nzval .= weights_messages

    # compute: ∑_(c' ∈ N(v) - c) ...
    mul!(m_v2c, weight_matrix .* base.adj_C2V_V2C, m_c2v_previous)

    # ---- channel contribution ----
    @. scaled_llrs = weights_llrs * channel_llrs

    # m_v2c += base_projection * tmp_llr
    mul!(m_v2c, base_projection, scaled_llrs, 1f0, 1f0)

    # apply the activation function a(x) = log(tanh(x/2))
    safe_log_tanh!(m_v2c_activated, m_v2c)

    return nothing
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

function compute_layer!(
    messages_c2v,
    m_c2v_activated,
    messages_v2c,
    messages_v2c_activated,
    scaled_llrs,                # ✅ NEW BUFFER
    posterior_llrs,
    syndromes_batch,
    initial_llrs_batch,
    base_projection,
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
    
    We compute the LLRs at layer `t` from the messages m^t_(c->v) and the initial LLRs using the readout weights:
    μ^t_v = l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
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
    c2v_to_v2c!(
        messages_v2c_activated,
        messages_v2c,
        scaled_llrs,
        messages_c2v,
        weights_llr,
        weights_messages,
        initial_llrs_batch,
        base_projection,
        base,
        weight_matrix_v2c
    )

    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    v2c_to_c2v!(
        messages_c2v,
        m_c2v_activated,
        messages_v2c_activated,
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
    messages_c2v           = zeros(Float32, neurons_per_layer, n_batches)
    m_c2v_activated        = Matrix{ComplexF32}(undef, size(messages_c2v))

    messages_v2c           = similar(messages_c2v)
    messages_v2c_activated = Matrix{ComplexF32}(undef, size(messages_v2c))

    scaled_llrs            = similar(initial_llrs_batch) # this will carry llr weights times channel llrs, to avoid recomputing this product multiple times.

    posterior_llrs_layer   = zeros(Float32, base.code_n_bits, n_batches)

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
            m_c2v_activated,
            messages_v2c,
            messages_v2c_activated,
            scaled_llrs,
            posterior_llrs_layer,
            syndromes_batch,
            initial_llrs_batch,
            base.adj_initialize_V2C,
            bpnn,
            layer,
            weight_matrix_v2c,
            weight_matrix_readout
        )

        posterior_llrs[:, :, layer] = posterior_llrs_layer
    end

    return copy(posterior_llrs)
end
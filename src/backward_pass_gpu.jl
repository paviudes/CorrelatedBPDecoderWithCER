# backward_pass_gpu.jl
# ----------------------------------------------------------------------------
# GPU-side backward pass, the mirror of forward_gpu.jl: the same dense algebra as
# backward_pass_with_weights.jl on device arrays (`ArrayT` is Metal.MtlArray on
# Apple, CUDA.CuArray on Linux, Array when neither backend is present, as chosen
# in forward_gpu.jl). Every function is tested against its CPU counterpart.
#
# Everything the backward multiplies by is uploaded once, in `GPUBackwardState`,
# exactly as `GPUState` holds the forward's matrices: the transposed adjacencies
# and the one-hot selectors (functions of the code), and the transposed weight
# matrices (functions of the model, so the state is rebuilt whenever the weights
# change — once per training step, like the forward state).
#
# Weight gradients come out FLAT, in the weight vector's own order, from two
# one-hot selectors per weighted pair (the output and the input edge of pair k):
# W̄_k = ∑_samples (S_out m̄)_k (S_in m)_k. Only matrix products and broadcasts, so
# nothing indexes a device array element by element.
# ----------------------------------------------------------------------------

struct GPUBackwardState
    # Dense transposes on the device. `adj_V2C_C2V` is symmetric, so its
    # transpose equals the forward's matrix, and `adj_initialize_V2C'` equals
    # `adj_C2V_readout`; both are materialised as transposes anyway so the
    # backward never depends on those facts.
    adj_V2C_C2V_transposed        :: ArrayT{Float32, 2}   # (nb_neurons × nb_neurons)
    adj_initialize_V2C_transposed :: ArrayT{Float32, 2}   # (n_bits × nb_neurons)
    # W^{t-1} transposed, one dense array per layer: entry (c2v edge, v2c edge)
    # holds the weight of the pair, so `W_transposed * m̄_v2c` is rule 2̄.
    W_msg_transposed_layers       :: Vector{ArrayT{Float32, 2}}
    # The readout's weighted matrix, transposed: entry (c2v edge, bit) holds W^ro.
    W_readout_transposed          :: ArrayT{Float32, 2}   # (nb_neurons × n_bits)
    # One-hot selectors: row k picks the output (v2c) edge / the input (c2v) edge
    # of weighted pair k, and the bit / the c2v edge of readout weight k.
    pair_output_selector          :: ArrayT{Float32, 2}   # (n_pairs × nb_neurons)
    pair_input_selector           :: ArrayT{Float32, 2}   # (n_pairs × nb_neurons)
    readout_bit_selector          :: ArrayT{Float32, 2}   # (n_readout × n_bits)
    readout_edge_selector         :: ArrayT{Float32, 2}   # (n_readout × nb_neurons)
    nb_neurons                    :: Int
    n_bits                        :: Int
    n_layers                      :: Int
end

function one_hot_selector(selected_columns::Vector{Int}, n_columns::Int)::Matrix{Float32}
    """
    A (length(selected_columns) × n_columns) matrix whose row k is 1 at
    `selected_columns[k]` and 0 elsewhere, so `selector * X` gathers those rows
    of X with a matrix product.
    """
    selector::Matrix{Float32} = zeros(Float32, length(selected_columns), n_columns)
    for (row, column) in enumerate(selected_columns)
        selector[row, column] = 1.0f0
    end
    return selector
end

function build_gpu_backward_state(
    base::NeuralBPBase,
    weights_c2v_v2c::AbstractVector{<:Real},
    weights_c2v_readout::AbstractVector{<:Real}
)::GPUBackwardState
    """
    Upload the transposed adjacency matrices, the transposed, masked per-layer
    weight matrices and readout matrix, and the one-hot selectors the backward
    pass multiplies by.
    """
    adj_V2C_C2V_transposed::ArrayT{Float32, 2} = _to_dense_gpu(transpose(base.adj_V2C_C2V))
    adj_initialize_V2C_transposed::ArrayT{Float32, 2} = _to_dense_gpu(transpose(base.adj_initialize_V2C))
    # The forward scatters weight k to (rows[k], cols[k]); the transpose scatters
    # it to (cols[k], rows[k]). Same weight vector, index vectors swapped.
    W_transposed_cpu::Array{Float32, 3} = _scatter_layer_weights(
        Vector{Int}(base.non_zero_cols_C2V_V2C),
        Vector{Int}(base.non_zero_rows_C2V_V2C),
        weights_c2v_v2c,
        base.n_layers,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer,
    )
    @inbounds for layer in 1:base.n_layers
        @views W_transposed_cpu[:, :, layer] .*= transpose(base.adj_C2V_V2C)
    end
    W_msg_transposed_layers::Vector{ArrayT{Float32, 2}} =
        [ArrayT{Float32}(Matrix{Float32}(W_transposed_cpu[:, :, layer])) for layer in 1:base.n_layers]
    readout_transposed_cpu::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, base.code_n_bits)
    @inbounds for k in eachindex(base.non_zero_rows_C2V_readout)
        readout_transposed_cpu[base.non_zero_cols_C2V_readout[k], base.non_zero_rows_C2V_readout[k]] =
            Float32(weights_c2v_readout[k])
    end
    state::GPUBackwardState = GPUBackwardState(
        adj_V2C_C2V_transposed,
        adj_initialize_V2C_transposed,
        W_msg_transposed_layers,
        _to_device(readout_transposed_cpu),
        _to_device(one_hot_selector(Vector{Int}(base.non_zero_rows_C2V_V2C), base.nb_neurons_per_layer)),
        _to_device(one_hot_selector(Vector{Int}(base.non_zero_cols_C2V_V2C), base.nb_neurons_per_layer)),
        _to_device(one_hot_selector(Vector{Int}(base.non_zero_rows_C2V_readout), base.code_n_bits)),
        _to_device(one_hot_selector(Vector{Int}(base.non_zero_cols_C2V_readout), base.nb_neurons_per_layer)),
        base.nb_neurons_per_layer,
        base.code_n_bits,
        base.n_layers,
    )
    return state
end

function build_gpu_backward_state(bpnn::NachmaniNeuralBP)::GPUBackwardState
    """
    The state for a model: its base and its current message and readout weights.
    """
    state::GPUBackwardState = build_gpu_backward_state(bpnn.base, bpnn.weights_c2v_v2c, bpnn.weights_c2v_readout)
    return state
end

function release_gpu_backward_state!(state::GPUBackwardState)::Nothing
    """
    Return the state's device buffers to the allocator now rather than at the
    next garbage collection (see `release_device_array!`).
    """
    release_device_array!(state.adj_V2C_C2V_transposed)
    release_device_array!(state.adj_initialize_V2C_transposed)
    for layer_weights in state.W_msg_transposed_layers
        release_device_array!(layer_weights)
    end
    release_device_array!(state.W_readout_transposed)
    release_device_array!(state.pair_output_selector)
    release_device_array!(state.pair_input_selector)
    release_device_array!(state.readout_bit_selector)
    release_device_array!(state.readout_edge_selector)
    return nothing
end

# ----------------------------------------------------------------------------
# Rule 1̄ and rule 2̄
# ----------------------------------------------------------------------------

function backward_v2c_to_c2v_gpu(
    adjoint_activated_m_c2v::ArrayT{Float32, 2},   # Ā_{c->v}, (nb_neurons × batch)
    state::GPUBackwardState
)::ArrayT{Float32, 2}
    """
    `backward_v2c_to_c2v!` on the device: Ā_{v'->c} = ∑_{v ∈ N(c) - v'} Ā_{c->v},
    i.e. the transpose of `adj_V2C_C2V` applied to the c2v adjoints. Returns a new
    (nb_neurons × batch) array, in the functional style of `_compute_layer_gpu`.
    """
    adjoint_activated_m_v2c::ArrayT{Float32, 2} = state.adj_V2C_C2V_transposed * adjoint_activated_m_c2v
    return adjoint_activated_m_v2c
end

function backward_c2v_to_v2c_gpu(
    adjoint_messages_v2c::ArrayT{Float32, 2},   # m̄^t_{v->c}, (nb_neurons × batch)
    state::GPUBackwardState,
    layer::Int
)::ArrayT{Float32, 2}
    """
    `backward_c2v_to_v2c!` on the device: m̄^{t-1}_{c'->v} = ∑_{c ∈ N(v) - c'}
    W^{t-1}_{(v,c;c',v)} m̄^t_{v->c}, the transposed weight matrix of `layer`
    applied to the v2c adjoints. Returns a new (nb_neurons × batch) array.
    """
    adjoint_messages_c2v_previous::ArrayT{Float32, 2} = state.W_msg_transposed_layers[layer] * adjoint_messages_v2c
    return adjoint_messages_c2v_previous
end

function weight_gradients_c2v_to_v2c_gpu(
    adjoint_messages_v2c::ArrayT{Float32, 2},    # m̄^t_{v->c},     (nb_neurons × batch)
    messages_c2v_previous::ArrayT{Float32, 2},   # m^{t-1}_{c'->v}, (nb_neurons × batch), from the forward
    initial_llrs_batch::ArrayT{Float32, 2},      # l_v,             (n_bits × batch)
    state::GPUBackwardState
)::Tuple{ArrayT{Float32, 1}, ArrayT{Float32, 1}}
    """
    `weight_gradients_c2v_to_v2c!` on the device, for one layer, both FLAT:

        W̄_k  = ∑_samples m̄^t[rows[k]] · m^{t-1}[cols[k]]        (n_pairs,)
        b̄_v  = ∑_samples l_v · ∑_{c ∈ N(v)} m̄^t_{v->c}           (n_bits,), this rule's share

    the first as the selected rows of the two matrices multiplied and summed over
    the batch. Returned, not accumulated: the caller writes them into its
    per-layer gradient columns.
    """
    selected_outputs::ArrayT{Float32, 2} = state.pair_output_selector * adjoint_messages_v2c
    selected_inputs::ArrayT{Float32, 2} = state.pair_input_selector * messages_c2v_previous
    weight_gradient::ArrayT{Float32, 1} = vec(sum(selected_outputs .* selected_inputs; dims = 2))
    adjoint_channel::ArrayT{Float32, 2} = state.adj_initialize_V2C_transposed * adjoint_messages_v2c
    llr_weight_gradient::ArrayT{Float32, 1} = vec(sum(adjoint_channel .* initial_llrs_batch; dims = 2))
    return (weight_gradient, llr_weight_gradient)
end

# ----------------------------------------------------------------------------
# The readout
# ----------------------------------------------------------------------------

function backward_readout_gpu(
    adjoint_posterior_llrs::ArrayT{Float32, 2},   # μ̄_v, (n_bits × batch)
    state::GPUBackwardState
)::ArrayT{Float32, 2}
    """
    `backward_readout!` on the device: the readout's share of m̄_{c->v},
    W^ro_{(v;c,v)} · μ̄_v, i.e. the transposed readout matrix applied to μ̄.
    Returns it; the caller adds it to rule 2̄'s share from the layer above.
    """
    adjoint_messages_c2v::ArrayT{Float32, 2} = state.W_readout_transposed * adjoint_posterior_llrs
    return adjoint_messages_c2v
end

function weight_gradients_readout_gpu(
    adjoint_posterior_llrs::ArrayT{Float32, 2},   # μ̄^t,            (n_bits × batch)
    messages_c2v::ArrayT{Float32, 2},             # m^t_{c->v},      (nb_neurons × batch), from the forward
    initial_llrs_batch::ArrayT{Float32, 2},       # l_v,             (n_bits × batch)
    state::GPUBackwardState
)::Tuple{ArrayT{Float32, 1}, ArrayT{Float32, 1}}
    """
    `weight_gradients_readout!` on the device, for one layer:

        W̄^ro_k = ∑_samples μ̄[bit_k] · m[edge_k]      (n_readout,), this layer's share
        b̄_v    = ∑_samples μ̄_v · l_v                 (n_bits,),    the readout's share

    Returned, not accumulated: W^ro is shared by every layer, so the caller sums
    its gradient over layers; b^t collects this share and rule 2's.
    """
    selected_bits::ArrayT{Float32, 2} = state.readout_bit_selector * adjoint_posterior_llrs
    selected_edges::ArrayT{Float32, 2} = state.readout_edge_selector * messages_c2v
    readout_gradient::ArrayT{Float32, 1} = vec(sum(selected_bits .* selected_edges; dims = 2))
    llr_weight_gradient::ArrayT{Float32, 1} = vec(sum(adjoint_posterior_llrs .* initial_llrs_batch; dims = 2))
    return (readout_gradient, llr_weight_gradient)
end

# ----------------------------------------------------------------------------
# The two activation conversions: the CPU's scalar factors, broadcast
# ----------------------------------------------------------------------------

function backward_atanh_exp_signed_gpu(
    adjoint_messages_c2v::ArrayT{Float32, 2},   # m̄_{c->v},               (nb_neurons × batch)
    magnitudes_c2v::ArrayT{Float32, 2},         # M_{c->v} from the forward
    signs_c2v::ArrayT{Float32, 2}               # 0/1: parity of the incoming signs XOR syndrome
)::ArrayT{Float32, 2}
    """
    `backward_atanh_exp_signed!` on the device: `atanh_exp_signed_adjoint` at every
    entry, the sign being the forward GPU path's 0/1 Float32 encoding.
    """
    adjoint_magnitudes_c2v::ArrayT{Float32, 2} =
        atanh_exp_signed_adjoint.(adjoint_messages_c2v, magnitudes_c2v, signs_c2v .!= 0.0f0)
    return adjoint_magnitudes_c2v
end

function backward_log_tanh_split_gpu(
    adjoint_magnitudes_v2c::ArrayT{Float32, 2},   # ∂L/∂ log tanh(|m_{v->c}|/2), (nb_neurons × batch)
    messages_v2c::ArrayT{Float32, 2}              # RAW m_{v->c} from the forward
)::ArrayT{Float32, 2}
    """
    `backward_log_tanh_split!` on the device: `log_tanh_split_adjoint` at every entry.
    """
    adjoint_messages_v2c::ArrayT{Float32, 2} = log_tanh_split_adjoint.(adjoint_magnitudes_v2c, messages_v2c)
    return adjoint_messages_v2c
end

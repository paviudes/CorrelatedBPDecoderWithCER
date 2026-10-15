# =============================================================================
#   The gradient of the training loss on the GPU
# =============================================================================
# `compute_loss_and_gradients_gpu` takes the same arguments as
# `compute_loss_and_gradients` (src/gradient.jl) and `get_loss_value`, and returns
# the same `LossAndGradients`, with the message passing in both directions on the
# device (`ArrayT`: Metal on Apple, CUDA on Linux, plain Arrays without a backend).
# The three stages are the CPU wrapper's:
#
#   1. FORWARD WITH A TAPE, `forward_pass_gpu_with_tape`: the testing forward's own
#      layer function (`_compute_layer_gpu_values`), keeping each layer's raw v2c
#      messages, check-node magnitudes and signs, and outgoing c2v messages on the
#      device, and the posteriors μᵗ on the host.
#   2. THE LOSS AND ITS SEED, on the host, from μᵗ: `base_loss_per_layer`,
#      `combine_layer_losses` and `compute_base_loss_gradient` — the CPU path's
#      functions on the GPU's posteriors. Only μᵗ crosses to the host, and only
#      the seed ∂L/∂μᵗ comes back, one layer at a time.
#   3. THE REVERSE SWEEP on the device, step for step the CPU one, with the GPU
#      mirrors of every rule adjoint (src/backward_pass_gpu.jl). The weight
#      gradients collect in device matrices, one column per layer, and come to
#      the host once at the end.
#
# The posteriors alone give the loss, but not the gradient: the reverse sweep
# multiplies by derivative factors evaluated at each layer's MESSAGES (a′ at the
# v2c messages, (a⁻¹)′ at the check-node magnitudes, the incoming c2v messages in
# the weight read-off). μᵗ = R m_c2vᵗ + bᵗ ⊙ l is n_bits numbers per sample made
# from nb_neurons messages, so the messages cannot be recovered from it; the tape
# keeps them.
#
# Scope: the standard (tanh) check node, as on the CPU; an enriched base is
# refused. α is never read, so its gradient is exactly zero.
# =============================================================================

struct GPUForwardTape
    # Per layer t = 1..T, on the device. `incoming_messages_c2v[t]` is m_c2vᵗ⁻¹
    # (zeros at t = 1) and is the same array as `outgoing_messages_c2v[t - 1]`.
    incoming_messages_c2v :: Vector{ArrayT{Float32, 2}}
    messages_v2c          :: Vector{ArrayT{Float32, 2}}   # raw m_v2cᵗ
    magnitudes_c2v        :: Vector{ArrayT{Float32, 2}}   # Mᵗ
    signs_c2v             :: Vector{ArrayT{Float32, 2}}   # 0/1, parity XOR syndrome
    outgoing_messages_c2v :: Vector{ArrayT{Float32, 2}}   # m_c2vᵗ
    # μᵗ for every layer, on the HOST, (n_bits × n_samples × n_layers): what
    # `forward_pass_gpu` returns, so the loss functions take it as is.
    posterior_llrs        :: Array{Float32, 3}
end

function forward_pass_gpu_with_tape(
    gpu_state::GPUState,
    initial_llrs_gpu::ArrayT{Float32, 2}
)::GPUForwardTape
    """
    The GPU forward, layer for layer `_forward_pass_gpu_chunk` (same layer
    function, same order), keeping each layer's values for the backward.
    Standard (tanh) check node only.
    """
    n_samples::Int = size(initial_llrs_gpu, 2)
    n_layers::Int = gpu_state.n_layers
    if gpu_state.check_node_kind == CHECK_NODE_ENRICHED
        throw(ArgumentError("forward_pass_gpu_with_tape: standard (tanh) check node only."))
    end
    if n_samples != gpu_state.n_samples
        throw(DimensionMismatch(
            "forward_pass_gpu_with_tape: state was built for $(gpu_state.n_samples) samples " *
            "but was given $(n_samples)."))
    end

    incoming_messages_c2v::Vector{ArrayT{Float32, 2}} = Vector{ArrayT{Float32, 2}}(undef, n_layers)
    messages_v2c::Vector{ArrayT{Float32, 2}} = Vector{ArrayT{Float32, 2}}(undef, n_layers)
    magnitudes_c2v::Vector{ArrayT{Float32, 2}} = Vector{ArrayT{Float32, 2}}(undef, n_layers)
    signs_c2v::Vector{ArrayT{Float32, 2}} = Vector{ArrayT{Float32, 2}}(undef, n_layers)
    outgoing_messages_c2v::Vector{ArrayT{Float32, 2}} = Vector{ArrayT{Float32, 2}}(undef, n_layers)
    posterior_3d_gpu::ArrayT{Float32, 3} = ArrayT{Float32}(zeros(Float32, gpu_state.n_bits, n_samples, n_layers))

    messages_c2v::ArrayT{Float32, 2} = ArrayT{Float32}(zeros(Float32, gpu_state.nb_neurons, n_samples))
    for layer in 1:n_layers
        incoming_messages_c2v[layer] = messages_c2v
        layer_values::GPULayerValues = _compute_layer_gpu_values(messages_c2v, initial_llrs_gpu, gpu_state, layer)
        messages_v2c[layer] = layer_values.messages_v2c
        magnitudes_c2v[layer] = layer_values.magnitudes_c2v
        signs_c2v[layer] = layer_values.signs_c2v
        outgoing_messages_c2v[layer] = layer_values.messages_c2v
        @views posterior_3d_gpu[:, :, layer] .= layer_values.posterior_llrs
        messages_c2v = layer_values.messages_c2v
    end
    posterior_llrs::Array{Float32, 3} = Array(posterior_3d_gpu)
    release_device_array!(posterior_3d_gpu)

    tape::GPUForwardTape = GPUForwardTape(
        incoming_messages_c2v,
        messages_v2c,
        magnitudes_c2v,
        signs_c2v,
        outgoing_messages_c2v,
        posterior_llrs,
    )
    return tape
end

function release_gpu_forward_tape!(tape::GPUForwardTape)::Nothing
    """
    Free the tape's device arrays. `incoming_messages_c2v[t]` is
    `outgoing_messages_c2v[t - 1]`, so only the first incoming array is its own.
    """
    release_device_array!(tape.incoming_messages_c2v[1])
    for layer in eachindex(tape.messages_v2c)
        release_device_array!(tape.messages_v2c[layer])
        release_device_array!(tape.magnitudes_c2v[layer])
        release_device_array!(tape.signs_c2v[layer])
        release_device_array!(tape.outgoing_messages_c2v[layer])
    end
    return nothing
end

function compute_loss_and_gradients_gpu(
    weights_c2v_v2c::Vector{Float32},
    weights_llrs::Vector{Float32},
    weights_c2v_readout::Vector{Float32},
    coupling_logit::Vector{Float32},
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int,
    loss_layer_selection::Int,
    loss_layer_ramp_sharpness::Float32,
    base_loss_selection::Int,
    base::NeuralBPBase,
    llrs_batch::Matrix{Float32},
    syndromes_batch::AbstractMatrix{Bool},
    expected_recoveries::BitMatrix
)::LossAndGradients
    """
    `compute_loss_and_gradients` with the forward and the reverse sweep on the
    device: same arguments in the same order, same result. See the block above
    for the three stages; standard (tanh) check node only.

    Builds the forward and backward states from the weights it is given and
    frees them, with the tape, before returning, so a training loop can call it
    once per step without device memory piling up behind the garbage collector.
    """
    if base.check_node_kind == CHECK_NODE_ENRICHED
        throw(ArgumentError(
            "compute_loss_and_gradients_gpu: the hand-written backward pass does not cover " *
            "the enriched check node. Use Enzyme (get_loss_value) for check_node = \"enriched\"."))
    end
    n_layers::Int = base.n_layers
    n_bits::Int = base.code_n_bits
    n_samples::Int = size(llrs_batch, 2)
    n_pairs_per_layer::Int = base.nb_weights_c2v_v2c
    n_readout_weights::Int = length(weights_c2v_readout)

    # ---- 1. forward, with the tape ----------------------------------------------
    syndromes_bits::BitMatrix = BitMatrix(syndromes_batch)
    gpu_state::GPUState = build_gpu_state(
        base, weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_logit, syndromes_bits)
    backward_state::GPUBackwardState = build_gpu_backward_state(base, weights_c2v_v2c, weights_c2v_readout)
    initial_llrs_gpu::ArrayT{Float32, 2} = ArrayT{Float32}(llrs_batch)
    tape::GPUForwardTape = forward_pass_gpu_with_tape(gpu_state, initial_llrs_gpu)

    # ---- 2. the loss, and how much each scored layer counts in it (host) ---------
    losses_per_layer::Vector{Float32} = base_loss_per_layer(
        tape.posterior_llrs, expected_recoveries, base.parity_check_matrix_dual,
        warmup_loss_layers, base_loss_selection)
    total_loss::Float32 = combine_layer_losses(
        losses_per_layer, loss_layer_temperature, loss_layer_selection, loss_layer_ramp_sharpness)
    layer_loss_weights::Vector{Float32} = combine_layer_losses_gradient(
        losses_per_layer, loss_layer_temperature, loss_layer_selection, loss_layer_ramp_sharpness)

    # ---- 3. the reverse sweep (device) ---------------------------------------------
    # One column per layer, in the weight vectors' own layout once flattened.
    gradient_weights_c2v_v2c_device::ArrayT{Float32, 2} = ArrayT{Float32}(zeros(Float32, n_pairs_per_layer, n_layers))
    gradient_weights_llrs_device::ArrayT{Float32, 2} = ArrayT{Float32}(zeros(Float32, n_bits, n_layers))
    gradient_weights_readout_device::ArrayT{Float32, 1} = ArrayT{Float32}(zeros(Float32, n_readout_weights))

    # m̄_c2vᵗ: rule 2̄'s share from layer t+1 (zero at t = T).
    adjoint_messages_c2v_from_above::ArrayT{Float32, 2} =
        ArrayT{Float32}(zeros(Float32, base.nb_neurons_per_layer, n_samples))
    for layer in n_layers:-1:1
        # The seed, on the host: zero inside the warmup, c_t · ∂L_t/∂μᵗ after it.
        seed::Matrix{Float32} = zeros(Float32, n_bits, n_samples)
        if layer > warmup_loss_layers
            seed = layer_loss_weights[layer - warmup_loss_layers] .* compute_base_loss_gradient(
                tape.posterior_llrs[:, :, layer], expected_recoveries,
                base.parity_check_matrix_dual, base_loss_selection)
        end
        adjoint_posterior_llrs::ArrayT{Float32, 2} = ArrayT{Float32}(seed)

        # Readout: m̄_c2vᵗ = (rule 2̄ from above) + Rᵀ μ̄ᵗ, and its weight gradients
        # (W̄^ro, summed over layers, and the readout's share of b̄ᵗ).
        adjoint_messages_c2v::ArrayT{Float32, 2} =
            adjoint_messages_c2v_from_above .+ backward_readout_gpu(adjoint_posterior_llrs, backward_state)
        readout_gradients::Tuple{ArrayT{Float32, 1}, ArrayT{Float32, 1}} = weight_gradients_readout_gpu(
            adjoint_posterior_llrs, tape.outgoing_messages_c2v[layer], initial_llrs_gpu, backward_state)
        gradient_weights_readout_device .+= readout_gradients[1]

        # Out of the check node, across it (rule 1̄), into it.
        adjoint_magnitudes_c2v::ArrayT{Float32, 2} = backward_atanh_exp_signed_gpu(
            adjoint_messages_c2v, tape.magnitudes_c2v[layer], tape.signs_c2v[layer])
        adjoint_magnitudes_v2c::ArrayT{Float32, 2} = backward_v2c_to_c2v_gpu(adjoint_magnitudes_c2v, backward_state)
        adjoint_messages_v2c::ArrayT{Float32, 2} =
            backward_log_tanh_split_gpu(adjoint_magnitudes_v2c, tape.messages_v2c[layer])

        # The variable node: its parameter gradients (W̄ᵗ, and rule 2's share of
        # b̄ᵗ), then rule 2̄ for the layer below.
        layer_gradients::Tuple{ArrayT{Float32, 1}, ArrayT{Float32, 1}} = weight_gradients_c2v_to_v2c_gpu(
            adjoint_messages_v2c, tape.incoming_messages_c2v[layer], initial_llrs_gpu, backward_state)
        @views gradient_weights_c2v_v2c_device[:, layer] .= layer_gradients[1]
        @views gradient_weights_llrs_device[:, layer] .= layer_gradients[2] .+ readout_gradients[2]
        if layer > 1
            adjoint_messages_c2v_from_above = backward_c2v_to_v2c_gpu(adjoint_messages_v2c, backward_state, layer)
        end
    end

    gradient_weights_c2v_v2c::Vector{Float32} = vec(Array(gradient_weights_c2v_v2c_device))
    gradient_weights_llrs::Vector{Float32} = vec(Array(gradient_weights_llrs_device))
    gradient_weights_c2v_readout::Vector{Float32} = Array(gradient_weights_readout_device)

    release_gpu_forward_tape!(tape)
    release_gpu_state!(gpu_state)
    release_gpu_backward_state!(backward_state)
    release_device_array!(initial_llrs_gpu)
    release_device_array!(gradient_weights_c2v_v2c_device)
    release_device_array!(gradient_weights_llrs_device)
    release_device_array!(gradient_weights_readout_device)

    result::LossAndGradients = LossAndGradients(
        total_loss,
        gradient_weights_c2v_v2c,
        gradient_weights_llrs,
        gradient_weights_c2v_readout,
        zeros(Float32, length(coupling_logit)),
    )
    return result
end

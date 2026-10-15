# =============================================================================
#   The gradient of the training loss, by a hand-written backward pass
# =============================================================================
# `compute_loss_and_gradients` takes the same arguments as `get_loss_value`
# (src/train.jl) and returns what
#
#     Enzyme.autodiff(ReverseWithPrimal, get_loss_value, Duplicated(weights...)..., Const(...)...)
#
# returns: the loss and its gradient with respect to the four parameter vectors.
# Enzyme stays the oracle (tests/test_gradient.jl); this exists so that training
# needs no automatic differentiation, and so that the same sweep runs on a GPU,
# where Enzyme cannot follow device allocation: `compute_loss_and_gradients_gpu`
# (src/gradient_gpu.jl) is this function step for step on device arrays.
#
# The loss reaches the weights only through the posterior LLRs μᵗ of the scored
# layers, so
#
#     dL/dW = ∑_t  (∂L/∂μᵗ) · (dμᵗ/dW)
#
# and the computation has three stages:
#
#   1. FORWARD WITH A TAPE. The forward pass, layer for layer the one
#      `forward_pass_with_weights` runs (same functions, same buffers, so the
#      values are bit-identical), keeping what the backward needs per layer: the
#      incoming and outgoing c2v messages, the raw v2c messages, the check-node
#      magnitudes and signs, and μᵗ.
#   2. THE SEED. ∂L/∂μᵗ = c_t · ∂L_t/∂μᵗ: the layer's weight in the combination
#      (`combine_layer_losses_gradient`, src/loss.jl) times the base loss's own
#      derivative (`compute_base_loss_gradient`, src/base_loss.jl). Zero inside
#      the warmup.
#   3. THE REVERSE SWEEP, t = T down to 1, each step the adjoint of one forward
#      step (src/backward_pass_with_weights.jl):
#
#        m̄_c2vᵗ  = Rᵀ μ̄ᵗ + (rule 2̄ of layer t+1)          backward_readout!
#        W̄^ro, b̄ᵗ  += readout read-offs                       weight_gradients_readout!
#        M̄ᵗ      = (a⁻¹)′ ⊙ m̄_c2vᵗ                            backward_atanh_exp_signed!
#        mag̅ᵗ    = Sᵀ M̄ᵗ                       (rule 1̄)       backward_v2c_to_c2v!
#        m̄_v2cᵗ  = a′ ⊙ mag̅ᵗ                                  backward_log_tanh_split!
#        W̄ᵗ, b̄ᵗ  += rule-2 read-offs                          weight_gradients_c2v_to_v2c!
#        m̄_c2vᵗ⁻¹ = Wᵗᵀ m̄_v2cᵗ                 (rule 2̄)       backward_c2v_to_v2c!
#
# Scope: the standard (tanh) check node. The enriched check node's adjoint, and
# with it the gradient of α (`coupling_logit`), is not written; an enriched base
# is refused rather than given a wrong gradient. Under the tanh node α is never
# read, so its gradient is exactly zero — as Enzyme reports it.
#
# Prototyped in float64 against central finite differences for every layer rule,
# residue, warmup and temperature before being written here (2026-10-09).
# =============================================================================

struct ForwardTape
    # Per layer t = 1..T. `incoming_messages_c2v[t]` is m_c2vᵗ⁻¹ (zeros at t = 1);
    # `outgoing_messages_c2v[t]` is m_c2vᵗ, and is the same array as
    # `incoming_messages_c2v[t + 1]` — stored once, referenced twice.
    incoming_messages_c2v    :: Vector{Matrix{Float32}}   # (nb_neurons × n_samples) each
    messages_v2c             :: Vector{Matrix{Float32}}   # raw m_v2cᵗ
    activated_c2v_magnitudes :: Vector{Matrix{Float32}}   # Mᵗ = adj_V2C_C2V · log tanh(|m_v2cᵗ|/2)
    activated_c2v_signs      :: Vector{BitMatrix}         # parity of the incoming signs XOR syndrome
    outgoing_messages_c2v    :: Vector{Matrix{Float32}}   # m_c2vᵗ
    # μᵗ for every layer, (n_bits × n_samples × n_layers) — the layout
    # `forward_pass_with_weights` returns, so the loss functions take it as is.
    posterior_llrs           :: Array{Float32, 3}
end

struct LossAndGradients
    # The loss, and its gradient with respect to each parameter vector, in that
    # vector's own layout — what Enzyme writes into the `Duplicated` shadows.
    loss                :: Float32
    weights_c2v_v2c     :: Vector{Float32}
    weights_llrs        :: Vector{Float32}
    weights_c2v_readout :: Vector{Float32}
    coupling_logit      :: Vector{Float32}
end

function forward_pass_with_tape(
    weights_c2v_v2c::Vector{Float32},
    weights_llrs::Vector{Float32},
    weights_c2v_readout::Vector{Float32},
    coupling_logit::Vector{Float32},
    base::NeuralBPBase,
    initial_llrs_batch::Matrix{Float32},
    syndromes_batch::Matrix{Bool}
)::ForwardTape
    """
    `forward_pass_with_weights`, step for step — the same buffers, the same
    α, the same `compute_layer_with_weights!` — additionally copying
    each layer's intermediate values into a `ForwardTape`. The posterior LLRs are
    therefore bit-identical to the untaped forward pass; a test asserts it.

    Memory: four (nb_neurons × n_samples) arrays and one BitMatrix per layer —
    about 3.5 MB at 216 edges, 20 samples, 50 layers.
    """
    n_samples::Int = size(initial_llrs_batch, 2)
    neurons_per_layer::Int = base.nb_neurons_per_layer
    n_layers::Int = base.n_layers

    messages_c2v::Matrix{Float32} = zeros(Float32, neurons_per_layer, n_samples)
    messages_v2c::Matrix{Float32} = zeros(Float32, neurons_per_layer, n_samples)
    activated_m_c2v_magnitudes::Matrix{Float32} = similar(messages_c2v)
    activated_m_c2v_signs::BitMatrix = falses(size(messages_c2v))
    activated_m_v2c_magnitudes::Matrix{Float32} = similar(messages_c2v)
    activated_m_v2c_signs::BitMatrix = falses(size(messages_c2v))
    weighted_channel_llrs::Matrix{Float32} = Matrix{Float32}(undef, size(initial_llrs_batch))
    posterior_llrs_layer::Matrix{Float32} = zeros(Float32, base.code_n_bits, n_samples)
    posterior_llrs::Array{Float32, 3} = zeros(Float32, base.code_n_bits, n_samples, n_layers)

    incoming_messages_c2v::Vector{Matrix{Float32}} = Vector{Matrix{Float32}}(undef, n_layers)
    tape_messages_v2c::Vector{Matrix{Float32}} = Vector{Matrix{Float32}}(undef, n_layers)
    tape_c2v_magnitudes::Vector{Matrix{Float32}} = Vector{Matrix{Float32}}(undef, n_layers)
    tape_c2v_signs::Vector{BitMatrix} = Vector{BitMatrix}(undef, n_layers)
    outgoing_messages_c2v::Vector{Matrix{Float32}} = Vector{Matrix{Float32}}(undef, n_layers)

    linked_coupling_scale::Float32 = coupling_scale_from_logit(coupling_logit[1])
    previous_messages_c2v::Matrix{Float32} = zeros(Float32, neurons_per_layer, n_samples)
    for layer in 1:n_layers
        coupling_scale_vector::Vector{Float32} = Float32[linked_coupling_scale]
        # `compute_layer_with_weights!` overwrites `messages_c2v` (its input) with
        # its output, so the input is the previous layer's stored output.
        incoming_messages_c2v[layer] = previous_messages_c2v
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
            coupling_scale_vector,
            base,
            layer,
            n_samples
        )
        posterior_llrs[:, :, layer] .= posterior_llrs_layer
        tape_messages_v2c[layer] = copy(messages_v2c)
        tape_c2v_magnitudes[layer] = copy(activated_m_c2v_magnitudes)
        tape_c2v_signs[layer] = copy(activated_m_c2v_signs)
        outgoing_messages_c2v[layer] = copy(messages_c2v)
        previous_messages_c2v = outgoing_messages_c2v[layer]
    end
    tape::ForwardTape = ForwardTape(
        incoming_messages_c2v,
        tape_messages_v2c,
        tape_c2v_magnitudes,
        tape_c2v_signs,
        outgoing_messages_c2v,
        posterior_llrs,
    )
    return tape
end

function compute_loss_and_gradients(
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
    The training loss and its gradient with respect to `weights_c2v_v2c`,
    `weights_llrs`, `weights_c2v_readout` and `coupling_logit`, without
    automatic differentiation. Arguments as `get_loss_value`, in the same
    order. See the block above for the three stages; standard (tanh) check
    node only.
    """
    if base.check_node_kind == CHECK_NODE_ENRICHED
        throw(ArgumentError(
            "compute_loss_and_gradients: the hand-written backward pass does not cover the " *
            "enriched check node (its adjoint, and the gradient of α). Use Enzyme " *
            "(get_loss_value) for check_node = \"enriched\"."))
    end
    n_layers::Int = base.n_layers
    n_bits::Int = base.code_n_bits
    n_samples::Int = size(llrs_batch, 2)
    n_neurons::Int = base.nb_neurons_per_layer
    n_pairs_per_layer::Int = base.nb_weights_c2v_v2c

    # ---- 1. forward, with the tape ----------------------------------------------
    syndromes_matrix::Matrix{Bool} = Matrix{Bool}(syndromes_batch)
    tape::ForwardTape = forward_pass_with_tape(
        weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_logit,
        base, llrs_batch, syndromes_matrix)

    # ---- 2. the loss, and how much each scored layer counts in it -----------------
    losses_per_layer::Vector{Float32} = base_loss_per_layer(
        tape.posterior_llrs, expected_recoveries, base.parity_check_matrix_dual,
        warmup_loss_layers, base_loss_selection)
    total_loss::Float32 = combine_layer_losses(
        losses_per_layer, loss_layer_temperature, loss_layer_selection, loss_layer_ramp_sharpness)
    layer_loss_weights::Vector{Float32} = combine_layer_losses_gradient(
        losses_per_layer, loss_layer_temperature, loss_layer_selection, loss_layer_ramp_sharpness)

    # ---- 3. the reverse sweep ------------------------------------------------------
    gradient_weights_c2v_v2c::Vector{Float32} = zeros(Float32, length(weights_c2v_v2c))
    gradient_weights_llrs::Vector{Float32} = zeros(Float32, length(weights_llrs))
    gradient_weights_c2v_readout::Vector{Float32} = zeros(Float32, length(weights_c2v_readout))

    # m̄_c2vᵗ: holds rule 2̄'s share from layer t+1 on entry to step t (zero at
    # t = T), then the readout's share is added.
    adjoint_messages_c2v::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)
    adjoint_posterior_llrs::Matrix{Float32} = zeros(Float32, n_bits, n_samples)
    adjoint_magnitudes_c2v::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)
    adjoint_magnitudes_v2c::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)
    adjoint_messages_v2c::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)

    for layer in n_layers:-1:1
        weights_layer::AbstractVector{Float32} =
            view(weights_c2v_v2c, (layer - 1) * n_pairs_per_layer + 1 : layer * n_pairs_per_layer)
        gradient_weights_layer::AbstractVector{Float32} =
            view(gradient_weights_c2v_v2c, (layer - 1) * n_pairs_per_layer + 1 : layer * n_pairs_per_layer)
        gradient_llr_layer::AbstractVector{Float32} =
            view(gradient_weights_llrs, (layer - 1) * n_bits + 1 : layer * n_bits)

        # The seed: zero inside the warmup, c_t · ∂L_t/∂μᵗ after it.
        fill!(adjoint_posterior_llrs, 0.0f0)
        if layer > warmup_loss_layers
            base_loss_gradient::Matrix{Float32} = compute_base_loss_gradient(
                tape.posterior_llrs[:, :, layer], expected_recoveries,
                base.parity_check_matrix_dual, base_loss_selection)
            adjoint_posterior_llrs .= layer_loss_weights[layer - warmup_loss_layers] .* base_loss_gradient
        end

        # Readout: m̄_c2vᵗ += Rᵀ μ̄ᵗ, and the readout's parameter gradients.
        backward_readout!(adjoint_messages_c2v, adjoint_posterior_llrs, weights_c2v_readout, base)
        weight_gradients_readout!(
            gradient_weights_c2v_readout, gradient_llr_layer, adjoint_posterior_llrs,
            tape.outgoing_messages_c2v[layer], llrs_batch, base)

        # Out of the check node, across it (rule 1̄), into it.
        backward_atanh_exp_signed!(
            adjoint_magnitudes_c2v, adjoint_messages_c2v,
            tape.activated_c2v_magnitudes[layer], tape.activated_c2v_signs[layer])
        backward_v2c_to_c2v!(adjoint_magnitudes_v2c, adjoint_magnitudes_c2v, base)
        backward_log_tanh_split!(adjoint_messages_v2c, adjoint_magnitudes_v2c, tape.messages_v2c[layer])

        # The variable node: its parameter gradients, then rule 2̄, which leaves
        # m̄_c2vᵗ⁻¹ — rule 2̄'s share for the next step — in `adjoint_messages_c2v`.
        weight_gradients_c2v_to_v2c!(
            gradient_weights_layer, gradient_llr_layer, adjoint_messages_v2c,
            tape.incoming_messages_c2v[layer], llrs_batch, base)
        backward_c2v_to_v2c!(adjoint_messages_c2v, adjoint_messages_v2c, weights_layer, base)
    end

    result::LossAndGradients = LossAndGradients(
        total_loss,
        gradient_weights_c2v_v2c,
        gradient_weights_llrs,
        gradient_weights_c2v_readout,
        zeros(Float32, length(coupling_logit)),
    )
    return result
end

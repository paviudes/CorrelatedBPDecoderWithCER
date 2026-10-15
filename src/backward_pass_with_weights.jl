# backward_pass_with_weights.jl
# ----------------------------------------------------------------------------
# The hand-written backward pass of the forward pass in forward_pass_weights.jl,
# built one rule at a time. Every function here is the ADJOINT of one forward
# step: given ∂L/∂(that step's outputs) it returns ∂L/∂(that step's inputs), L
# being the scalar training loss. The adjoint of a linear step is its transpose,
# and the transposes already exist in `NeuralBPBase`: `adj_V2C_C2V` and
# `adj_C2V_V2C` are symmetric, and `adj_initialize_V2C'` is `adj_C2V_readout`.
#
# Notation, per edge and per sample: m is a raw message, A = a(m) its activated
# value (a(x) = log tanh(x/2) in the magnitude/sign form the forward uses), and a
# bar is ∂L/∂ of that quantity — the same shape as the quantity itself.
#
# Step one of the construction is the reversed DATA FLOW: the forward rules'
# sums, over the same neighbourhoods, from outputs back to inputs, with the
# constant inputs (syndrome, channel LLR) dropped. Step two, the derivative
# factors between the raw and activated adjoints and the weight gradients, is
# written separately.
# ----------------------------------------------------------------------------

function backward_v2c_to_c2v!(
    adjoint_activated_m_v2c::AbstractMatrix{Float32},   # output: Ā_{v'->c}, (nb_neurons × n_samples)
    adjoint_activated_m_c2v::AbstractMatrix{Float32},   # input:  Ā_{c->v},  (nb_neurons × n_samples)
    base::NeuralBPBase
)::Nothing
    """
    The adjoint of the check-node aggregation in `v2c_to_c2v!`, in the activated
    domain, within one layer. The forward rule, for every output edge (c->v):

        A_{c->v} = iπ s_c + ∑_{v' ∈ N(c) - v} A_{v'->c}

    where we're using the short-hand notation: A_{c->v} = a(m_{c->v}); 'a' being the activation function a(x) = log tanh(x/2).

    Differentiating it with respect to one input A_{v'->c} gives 1 for every
    output of the same check with v ≠ v' and 0 otherwise; L depends on A_{v'->c}
    only through those outputs, so the chain rule gives

        Ā_{v'->c} = ∑_{v ∈ N(c) - v'} Ā_{c->v}

    the same leave-one-out sum run from outputs back to inputs, i.e. multiplication
    by the transpose of `adj_V2C_C2V`. The syndrome term is a constant and
    contributes nothing. The factors (a⁻¹)′ on the way in and a′(m_{v'->c}) on the
    way out, which turn raw adjoints into activated ones and back, are not part of
    this step.
    """
    mul!(adjoint_activated_m_v2c, transpose(base.adj_V2C_C2V), adjoint_activated_m_c2v)
    return nothing
end
function backward_c2v_to_v2c!(
    adjoint_messages_c2v_previous::Matrix{Float32},   # output: m̄^{t-1}_{c'->v}, (nb_neurons × n_samples)
    adjoint_messages_v2c::Matrix{Float32},            # input:  m̄^t_{v->c},      (nb_neurons × n_samples)
    weights_c2v_v2c_layer::AbstractVector{Float32},   # W^{t-1}: this layer's slice, one weight per (v,c;c',v) pair
    base::NeuralBPBase
)::Nothing
    """
    The adjoint of the variable-node rule in `c2v_to_v2c_with_weights!`, raw
    domain, from layer t back to layer t-1. The forward rule, for every output
    edge (v->c):

        m^t_{v->c} = b^t_v l_v + ∑_{c' ∈ N(v) - c} W^{t-1}_{(v,c;c',v)} m^{t-1}_{c'->v}

    Differentiating it with respect to one input m^{t-1}_{c'->v} gives the
    weight W_{(v,c;c',v)} for every output of the same variable with c ≠ c' and 0
    otherwise, so the chain rule gives

        m̄^{t-1}_{c'->v} = ∑_{c ∈ N(v) - c'} W^{t-1}_{(v,c;c',v)} · m̄^t_{v->c}

    the same leave-one-out sum with the same weights, each read from the other
    end of its pair: the transpose of the sparse weight matrix, which is
    `sparse_multiply!` with `rows` and `cols` swapped and the weight vector
    unchanged. The channel term b l_v is a constant input of this rule — nothing
    flows back through it; it reappears as the gradient of b in
    `weight_gradients_c2v_to_v2c!`.
    """
    sparse_multiply!(
        adjoint_messages_c2v_previous,
        base.non_zero_cols_C2V_V2C,
        base.non_zero_rows_C2V_V2C,
        weights_c2v_v2c_layer,
        adjoint_messages_v2c
    )
    return nothing
end

function weight_gradients_c2v_to_v2c!(
    gradient_weights_c2v_v2c_layer::AbstractVector{Float32},   # W̄^{t-1}: accumulated, one entry per (v,c;c',v) pair
    gradient_weights_llr_layer::AbstractVector{Float32},       # b̄^t: accumulated, one entry per bit (this rule's share)
    adjoint_messages_v2c::Matrix{Float32},                     # m̄^t_{v->c},    (nb_neurons × n_samples)
    messages_c2v_previous::Matrix{Float32},                    # m^{t-1}_{c'->v}, (nb_neurons × n_samples), from the forward
    initial_llrs_batch::Matrix{Float32},                       # l_v,             (n_bits × n_samples)
    base::NeuralBPBase
)::Nothing
    """
    The derivatives of the same rule with respect to its parameters. Each weight
    appears in exactly one place in the whole network — the output edge (v->c) of
    this layer multiplying m^{t-1}_{c'->v} — so the chain rule stops there:

        W̄^{t-1}_{(v,c;c',v)} = ∑_samples  m̄^t_{v->c} · m^{t-1}_{c'->v}

    an outer product of the edge adjoint and the stored forward message, summed
    over the batch, read off at the nonzeros (rows[k], cols[k]). The channel
    weight b^t_v feeds every output edge of variable v, each through l_v:

        b̄^t_v  +=  ∑_samples  l_v · ∑_{c ∈ N(v)} m̄^t_{v->c}

    i.e. `adj_initialize_V2C'` applied to the edge adjoints, then the LLR. Both
    ACCUMULATE: b^t_v is also used by the readout at layer t, whose share is added
    by the readout's own adjoint; the caller zeroes the gradient buffers once per
    batch.
    """
    n_samples::Int = size(adjoint_messages_v2c, 2)
    n_pairs::Int = length(base.non_zero_rows_C2V_V2C)
    @inbounds for k in 1:n_pairs
        output_edge::Int = base.non_zero_rows_C2V_V2C[k]
        input_edge::Int = base.non_zero_cols_C2V_V2C[k]
        batch_sum::Float32 = 0.0f0
        @simd for j in 1:n_samples
            batch_sum += adjoint_messages_v2c[output_edge, j] * messages_c2v_previous[input_edge, j]
        end
        gradient_weights_c2v_v2c_layer[k] += batch_sum
    end
    # ∑_{c ∈ N(v)} m̄_{v->c}, per bit and sample: the transpose of the edge <- bit map.
    adjoint_channel::Matrix{Float32} = zeros(Float32, base.code_n_bits, n_samples)
    mul!(adjoint_channel, transpose(base.adj_initialize_V2C), adjoint_messages_v2c)
    @inbounds for bit in 1:base.code_n_bits
        batch_sum_llr::Float32 = 0.0f0
        @simd for j in 1:n_samples
            batch_sum_llr += adjoint_channel[bit, j] * initial_llrs_batch[bit, j]
        end
        gradient_weights_llr_layer[bit] += batch_sum_llr
    end
    return nothing
end

# ----------------------------------------------------------------------------
# The two activation conversions around rule 1̄, and the readout.
# ----------------------------------------------------------------------------
# The forward clamps tanh(|m|/2) and exp(M) into [eps, 1 - eps]; where a clamp is
# active the forward value is a constant, so the derivative there is exactly 0 —
# what Enzyme computes by differentiating the same `ifelse`. Replicated here so
# the two gradients agree; the clamps sit where messages saturate, i.e. where the
# unclamped derivative would be ~1e-7 anyway.
#
# The factors are evaluated in the order Enzyme's reverse pass evaluates them,
# and with the derivative forms it uses (1/cosh² for tanh′, 1/(1 - y²) for
# atanh′). That matters near saturation: 1 - tanh² and 1/cosh² are the same
# function but differ by up to 25% in Float32 next to the clamp (measured on a
# Float32 emulation, 2026-10-09).
#
# Each factor is one scalar function, used by the CPU loop below and broadcast
# over device arrays by the GPU version (backward_pass_gpu.jl), so the two paths
# share a single definition. Their descriptions are comments rather than the
# usual in-body string, since these bodies are compiled into GPU kernels.

# ∂L/∂m for one entry of `safe_log_tanh_split!`, given ∂L/∂(its magnitude):
#
#     d magnitude / d m = sign(m) · (1/2) · sech²(|m|/2) / tanh(|m|/2)    (unclamped)
#                       = 0                                                (clamped)
@inline function log_tanh_split_adjoint(adjoint_magnitude::Float32, message::Float32)::Float32
    half_magnitude::Float32 = abs(message) * 0.5f0
    tanh_value::Float32 = tanh(half_magnitude)
    adjoint_message::Float32 = 0.0f0
    if tanh_value >= eps(Float32) && tanh_value <= 1.0f0 - eps(Float32)
        cosh_value::Float32 = cosh(half_magnitude)
        adjoint_tanh::Float32 = adjoint_magnitude / tanh_value
        adjoint_half_magnitude::Float32 = adjoint_tanh / (cosh_value * cosh_value)
        adjoint_message = sign(message) * (adjoint_half_magnitude * 0.5f0)
    end
    return adjoint_message
end

# ∂L/∂M for one entry of `safe_atanh_exp_signed!`, given ∂L/∂(its output m):
# m = 2 atanh(y), y = ±clamp(exp(M), eps, 1 - eps), so
#
#     d m / d M = 2 · (±1) · exp(M) / (1 - y²)    (unclamped)
#               = 0                                (clamped)
@inline function atanh_exp_signed_adjoint(adjoint_message::Float32, magnitude::Float32, is_negative::Bool)::Float32
    exponential::Float32 = exp(magnitude)
    adjoint_magnitude::Float32 = 0.0f0
    if exponential >= eps(Float32) && exponential <= 1.0f0 - eps(Float32)
        signed_value::Float32 = is_negative ? -exponential : exponential
        adjoint_signed_value::Float32 = (2.0f0 * adjoint_message) / (1.0f0 - signed_value * signed_value)
        adjoint_exponential::Float32 = is_negative ? -adjoint_signed_value : adjoint_signed_value
        adjoint_magnitude = adjoint_exponential * exponential
    end
    return adjoint_magnitude
end

function backward_log_tanh_split!(
    adjoint_messages_v2c::Matrix{Float32},      # output: m̄_{v->c},                    (nb_neurons × n_samples)
    adjoint_magnitudes_v2c::Matrix{Float32},    # input:  ∂L/∂ log tanh(|m_{v->c}|/2), (nb_neurons × n_samples)
    messages_v2c::Matrix{Float32}               # m_{v->c} from the forward,          (nb_neurons × n_samples)
)::Nothing
    """
    The adjoint of `safe_log_tanh_split!`, the activation a(m) on the way INTO the
    check node. Its real part is the magnitude log(clamp(tanh(|m|/2), eps, 1 - eps));
    the sign part is piecewise constant in m and carries no derivative, so the
    factor is `log_tanh_split_adjoint`: 1/sinh(m) where unclamped, large for small
    |m| and small for large |m|, and 0 where clamped. OVERWRITES the output.
    """
    @inbounds for i in eachindex(messages_v2c)
        adjoint_messages_v2c[i] = log_tanh_split_adjoint(adjoint_magnitudes_v2c[i], messages_v2c[i])
    end
    return nothing
end

function backward_atanh_exp_signed!(
    adjoint_magnitudes_c2v::Matrix{Float32},    # output: ∂L/∂M_{c->v},           (nb_neurons × n_samples)
    adjoint_messages_c2v::Matrix{Float32},      # input:  m̄_{c->v},               (nb_neurons × n_samples)
    magnitudes_c2v::Matrix{Float32},            # M_{c->v} from the forward,     (nb_neurons × n_samples)
    signs_c2v::AbstractMatrix{Bool}             # its sign (parity XOR syndrome), (nb_neurons × n_samples)
)::Nothing
    """
    The adjoint of `safe_atanh_exp_signed!`, the inverse activation on the way OUT
    of the check node, entry by entry through `atanh_exp_signed_adjoint`. The
    sign is a constant here, exactly as in rule 1̄. OVERWRITES the output.
    """
    @inbounds for i in eachindex(magnitudes_c2v)
        adjoint_magnitudes_c2v[i] = atanh_exp_signed_adjoint(adjoint_messages_c2v[i], magnitudes_c2v[i], signs_c2v[i])
    end
    return nothing
end

function backward_readout!(
    adjoint_messages_c2v::Matrix{Float32},          # ACCUMULATED: m̄_{c->v} += ...,     (nb_neurons × n_samples)
    adjoint_posterior_llrs::Matrix{Float32},        # μ̄_v from the loss,                (n_bits × n_samples)
    weights_c2v_readout::AbstractVector{Float32},   # W^ro, one weight per (v; c,v)
    base::NeuralBPBase
)::Nothing
    """
    The adjoint of `readout_with_weights!` with respect to the c2v messages. The
    forward rule, for every bit v:

        μ_v = b_v l_v + ∑_{c ∈ N(v)} W^ro_{(v;c,v)} m_{c->v}

    so  m̄_{c->v} += W^ro_{(v;c,v)} · μ̄_v  — the transpose of the readout's
    weighted matrix (same weight vector, indices swapped). ACCUMULATES: the c2v
    messages of layer t are also the input of layer t+1, whose rule 2̄ has already
    written its share into `adjoint_messages_c2v`.
    """
    n_samples::Int = size(adjoint_posterior_llrs, 2)
    @inbounds for k in eachindex(base.non_zero_rows_C2V_readout)
        bit::Int = base.non_zero_rows_C2V_readout[k]
        edge::Int = base.non_zero_cols_C2V_readout[k]
        readout_weight::Float32 = weights_c2v_readout[k]
        @simd for j in 1:n_samples
            adjoint_messages_c2v[edge, j] += readout_weight * adjoint_posterior_llrs[bit, j]
        end
    end
    return nothing
end

function weight_gradients_readout!(
    gradient_weights_c2v_readout::AbstractVector{Float32},   # W̄^ro, ACCUMULATED over layers (shared by all of them)
    gradient_weights_llr_layer::AbstractVector{Float32},     # b̄^t, ACCUMULATED (rule 2 adds its share)
    adjoint_posterior_llrs::Matrix{Float32},                 # μ̄^t,                (n_bits × n_samples)
    messages_c2v::Matrix{Float32},                           # m^t_{c->v} forward, (nb_neurons × n_samples)
    initial_llrs_batch::Matrix{Float32},                     # l_v,                (n_bits × n_samples)
    base::NeuralBPBase
)::Nothing
    """
    The derivatives of the readout with respect to its parameters:

        W̄^ro_{(v;c,v)} += ∑_samples μ̄_v · m_{c->v}
        b̄^t_v          += ∑_samples μ̄_v · l_v

    The readout weights are ONE vector shared by every layer, so their gradient
    sums over layers; b^t is the same per-layer channel weight rule 2 uses, so
    its gradient collects both shares.
    """
    n_samples::Int = size(adjoint_posterior_llrs, 2)
    @inbounds for k in eachindex(base.non_zero_rows_C2V_readout)
        bit::Int = base.non_zero_rows_C2V_readout[k]
        edge::Int = base.non_zero_cols_C2V_readout[k]
        batch_sum::Float32 = 0.0f0
        @simd for j in 1:n_samples
            batch_sum += adjoint_posterior_llrs[bit, j] * messages_c2v[edge, j]
        end
        gradient_weights_c2v_readout[k] += batch_sum
    end
    @inbounds for bit in axes(adjoint_posterior_llrs, 1)
        batch_sum_llr::Float32 = 0.0f0
        @simd for j in 1:n_samples
            batch_sum_llr += adjoint_posterior_llrs[bit, j] * initial_llrs_batch[bit, j]
        end
        gradient_weights_llr_layer[bit] += batch_sum_llr
    end
    return nothing
end

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
    # The LOGIT of α, the scale on the CER couplings inside the enriched check
    # node. What the forward pass uses is α = 1/(1 + exp(-θ)) ∈ (0, 1); what is
    # stored and trained is the unconstrained θ, so Adam never has to be
    # projected and α can never leave its physical range (see
    # `coupling_scale_from_logit` in soft_constraints.jl). A length-1 vector,
    # not a scalar, so Enzyme and Optimisers treat it like the other weight
    # vectors. Inert when `base.check_node_kind == CHECK_NODE_TANH`.
    coupling_logit::Vector{Float32}
    # The layer schedule on α, stored as [T₀, ρ]: the layer at which the
    # couplings are half off, and log(w - W_MIN) for the width w of the roll-off
    # (see `coupling_schedule_damping` in soft_constraints.jl). The forward pass
    # uses α_t = α · d(t) when `base.coupling_schedule_kind` is the step
    # schedule, and ignores this vector entirely for the constant one — carried
    # then, like `coupling_logit` on a tanh model, so one struct layout and one
    # Enzyme signature serve every configuration. Always length 2.
    coupling_schedule::Vector{Float32}

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_llrs::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0),
        coupling_scale::Float32=1.0f0,
        coupling_schedule_layer::Float32=Float32(base.n_layers),
        coupling_schedule_width::Float32=3.0f0,
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        2. Weights for the connections from C2V to V2C: `weights_llrs`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`
        4. Scale α on the CER couplings in the enriched check node: `coupling_scale`
           (default 1), given in HUMAN UNITS — the fraction of the measured coupling
           strength to use. It is stored as its logit and trained there; read it back
           with `effective_coupling_scale`.
        5. The layer schedule on α, in HUMAN UNITS: `coupling_schedule_layer` is the
           layer T₀ at which the couplings are half off (default: the last layer, so
           that a step schedule left at its default barely differs from constant),
           `coupling_schedule_width` the width w of the roll-off in layers. Stored as
           [T₀, log(w - W_MIN)] and trained there; read back with
           `effective_coupling_schedule`. Ignored under the constant schedule.
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
        coupling_logit::Vector{Float32} = Float32[logit_from_coupling_scale(coupling_scale)]
        coupling_schedule::Vector{Float32} =
            coupling_schedule_parameters(coupling_schedule_layer, coupling_schedule_width)

        return new(
            base,
            # learnable_parameters,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_logit,
            coupling_schedule
        )
    end

    # Internal constructor for Functors reconstruction with all field values.
    # Takes the LOGIT and the stored schedule parameters, unlike the keyword
    # constructor above which takes α, T₀ and w: this is the path Optimisers and
    # the weights-file loader use, and both carry the stored parameters rather
    # than the human-facing ones.
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        weights_c2v_v2c::Vector{Float32},
        weights_llrs::Vector{Float32},
        weights_c2v_readout::Vector{Float32},
        coupling_logit::Vector{Float32},
        coupling_schedule::Vector{Float32},
    )
        if length(coupling_logit) != 1
            throw(ArgumentError("coupling_logit must be a length-1 vector, got length $(length(coupling_logit))."))
        end
        if length(coupling_schedule) != 2
            throw(ArgumentError("coupling_schedule must be a length-2 vector [T₀, log(w - W_MIN)], got length $(length(coupling_schedule))."))
        end
        return new(
            base,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_logit,
            coupling_schedule
        )
    end
end

# Make NeuralBP work with Functors by only making the weight vectors children.
# `coupling_logit` and `coupling_schedule` are children too, so the optimiser
# sees them; when either is not meant to be learned the training loop FREEZES
# its optimiser leaf instead (see `train_neuralbp_enzyme!`), which keeps this
# declaration static.
@functor NachmaniNeuralBP (weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_logit, coupling_schedule)

function effective_coupling_scale(bpnn::NachmaniNeuralBP)::Float32
    """
    The α the forward pass actually uses: the logistic of the stored logit.
    Everything that reports, logs or saves α for a human should go through this
    rather than reading the raw parameter.

    Under the step schedule this is the α BEFORE the layer damping, i.e. the
    value at layers well before T₀; see `effective_coupling_scale_at_layer` for
    the α a specific layer sees.
    """
    alpha::Float32 = coupling_scale_from_logit(bpnn.coupling_logit[1])
    return alpha
end

function effective_coupling_schedule(bpnn::NachmaniNeuralBP)::Tuple{Float32, Float32}
    """
    The schedule in HUMAN UNITS: (T₀, w), the layer at which the couplings are
    half off and the width of the roll-off in layers. Everything that reports,
    logs or saves the schedule should go through this rather than reading the
    stored [T₀, ρ].
    """
    step_layer::Float32 = bpnn.coupling_schedule[1]
    step_width::Float32 = coupling_schedule_width_from_parameter(bpnn.coupling_schedule[2])
    return (step_layer, step_width)
end

function effective_coupling_scale_at_layer(bpnn::NachmaniNeuralBP, layer::Int)::Float32
    """
    The α layer `layer` actually uses: α itself under the constant schedule,
    α · d(layer) under the step schedule. This is the single definition of α_t;
    both forward passes are checked against it.
    """
    alpha::Float32 = effective_coupling_scale(bpnn)
    damping::Float32 = 1.0f0
    if bpnn.base.coupling_schedule_kind == COUPLING_SCHEDULE_STEP
        damping = coupling_schedule_damping(bpnn.coupling_schedule, layer)
    end
    layer_alpha::Float32 = alpha * damping
    return layer_alpha
end

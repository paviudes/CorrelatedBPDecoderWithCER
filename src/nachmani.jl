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

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_llrs::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0),
        coupling_scale::Float32=1.0f0,
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

        return new(
            base,
            # learnable_parameters,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_logit
        )
    end

    # Internal constructor for Functors reconstruction with all field values.
    # Takes the LOGIT, unlike the keyword constructor above which takes α: this
    # is the path Optimisers and the weights-file loader use, and both carry the
    # stored parameter rather than the human-facing one.
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        weights_c2v_v2c::Vector{Float32},
        weights_llrs::Vector{Float32},
        weights_c2v_readout::Vector{Float32},
        coupling_logit::Vector{Float32},
    )
        if length(coupling_logit) != 1
            throw(ArgumentError("coupling_logit must be a length-1 vector, got length $(length(coupling_logit))."))
        end
        return new(
            base,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_logit
        )
    end
end

# Make NeuralBP work with Functors by only making the weight vectors children.
# `coupling_logit` is a child too, so the optimiser sees it; when α is not meant
# to be learned the training loop FREEZES its optimiser leaf instead (see
# `train_neuralbp_enzyme!`), which keeps this declaration static.
@functor NachmaniNeuralBP (weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_logit)

function effective_coupling_scale(bpnn::NachmaniNeuralBP)::Float32
    """
    The α the forward pass actually uses: the logistic of the stored logit.
    Everything that reports, logs or saves α for a human should go through this
    rather than reading the raw parameter.
    """
    alpha::Float32 = coupling_scale_from_logit(bpnn.coupling_logit[1])
    return alpha
end

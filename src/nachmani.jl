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
    # α: the scale on the CER couplings inside the enriched check node (see
    # soft_constraints.jl). A length-1 vector, not a scalar, so that Enzyme and
    # Optimisers treat it exactly like the other weight vectors. Inert when
    # `base.check_node_kind == CHECK_NODE_TANH`. α = 1 is the Bayesian value.
    coupling_scale::Vector{Float32}

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_llrs::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0),
        coupling_scale::Vector{Float32}=Float32[1.0f0],
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        2. Weights for the connections from C2V to V2C: `weights_llrs`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`
        4. Scale α on the CER couplings in the enriched check node: `coupling_scale` (default 1)
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
        if length(coupling_scale) != 1
            throw(ArgumentError("coupling_scale must be a length-1 vector holding α, got length $(length(coupling_scale))."))
        end

        return new(
            base,
            # learnable_parameters,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_scale
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        weights_c2v_v2c::Vector{Float32},
        weights_llrs::Vector{Float32},
        weights_c2v_readout::Vector{Float32},
        coupling_scale::Vector{Float32},
    )
        return new(
            base,
            weights_c2v_v2c,
            weights_llrs,
            weights_c2v_readout,
            coupling_scale
        )
    end
end

# Make NeuralBP work with Functors by only making the weight vectors children.
# `coupling_scale` is a child too, so the optimiser sees it; when it is not
# meant to be learned the training loop FREEZES its optimiser leaf instead
# (see `train_neuralbp_enzyme!`), which keeps this declaration static.
@functor NachmaniNeuralBP (weights_c2v_v2c, weights_llrs, weights_c2v_readout, coupling_scale)

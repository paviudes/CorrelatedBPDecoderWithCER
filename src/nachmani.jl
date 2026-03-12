import Zygote: gradient, Params
using Functors: @functor

struct NachmaniNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the Nachmani et al. architecture for Neural Belief Propagation: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.200501.
    In this variant the number of trainable parameters (weights in the network) scales linearly with the number of layers.
    """
    base::NeuralBPBase
    weights_v2c_c2v::Vector{Float32}
    weights_c2v_v2c::Vector{Float32}
    weights_c2v_readout::Vector{Float32}

    function NachmaniNeuralBP(
        base::NeuralBPBase;
        weights_v2c_c2v::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_v2c::Vector{Float32}=Vector{Float32}(undef, 0),
        weights_c2v_readout::Vector{Float32}=Vector{Float32}(undef, 0)
    )
        """
        Define the NeuralBP model.
        ## Set the learnable parameters to default values.
        1. Weights for the connections from V2C to C2V: `weights_v2c_c2v`
        2. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

        """
        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, base.nb_weights_v2c_c2v * base.n_layers)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_weights_c2v_v2c * base.n_layers)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.nb_weights_c2v_readout)
        end

        return new(
            base,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NachmaniNeuralBP(
        base::NeuralBPBase,
        # learnable_parameters::Vector{Symbol},
        weights_v2c_c2v::Vector{Float32},
        weights_c2v_v2c::Vector{Float32},
        weights_c2v_readout::Vector{Float32}
    )
        return new(
            base,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NachmaniNeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

function (bpnn::NachmaniNeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)::Array{3, Float32}
    """
    Forward pass through the Neural Network for Nachmani et al. architecture.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0)).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
    v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

    # Initialize the intermediate LLRs
    posterior_llrs = zeros(Float32, bpnn.base.n_layers, bpnn.base.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.base.n_layers
        # 1. Forward pass from V2C to C2V layer
        weights_start = (iter - 1) * bpnn.base.nb_weights_v2c_c2v + 1
        weights_end = iter * bpnn.base.nb_weights_v2c_c2v
        extended_weights_v2c_c2v = sparse(bpnn.base.non_zero_rows_V2C_C2V, bpnn.base.non_zero_cols_V2C_C2V, bpnn.weights_v2c_c2v[weights_start:weights_end], bpnn.base.nb_neurons_per_layer, bpnn.base.nb_neurons_per_layer) |> Matrix
        c2v_neurons = (@. sigmoid(extended_weights_v2c_c2v) * bpnn.base.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.base.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        # c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions
        c2v_activated_neurons = 2 .* safe_atanh_exp(c2v_neurons) .* phase_contributions # Use safe version to avoid numerical issues.

        # 2. Forward pass from C2V to V2C layer
        weights_start = (iter - 1) * bpnn.base.nb_weights_c2v_v2c + 1
        weights_end = iter * bpnn.base.nb_weights_c2v_v2c
        extended_weights_c2v_v2c = sparse(bpnn.base.non_zero_rows_C2V_V2C, bpnn.base.non_zero_cols_C2V_V2C, bpnn.weights_c2v_v2c[weights_start:weights_end], bpnn.base.nb_neurons_per_layer, bpnn.base.nb_neurons_per_layer) |> Matrix
        v2c_neurons = (@. extended_weights_c2v_v2c * bpnn.base.adj_C2V_V2C') * c2v_activated_neurons + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        # 3. Forward pass from final V2C layer to readout layer
        extended_weights_c2v_readout = sparse(bpnn.base.non_zero_rows_C2V_readout, bpnn.base.non_zero_cols_C2V_readout, bpnn.weights_c2v_readout, bpnn.base.code_n_bits, bpnn.base.nb_neurons_per_layer) |> Matrix
        posterior_llrs[iter, :, :] = initial_llrs_batch .+ (extended_weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_activated_neurons
    end
    return posterior_llrs
end
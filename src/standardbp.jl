import Zygote: gradient, Params
using Functors: @functor
using SparseArrays: sparse

struct StandardNeuralBP <: NeuralBP
    """
    Subtype of NeuralBP implementing the standard architecture for Neural Belief Propagation.
    In this variant the number of trainable parameters (weights in the network) is independent of the number of layers.
    """
    base::NeuralBPBase
    weights_v2c_c2v::Vector{Float32}
    weights_c2v_v2c::Vector{Float32}
    weights_c2v_readout::Vector{Float32}
    
    function StandardNeuralBP(
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
            weights_v2c_c2v = randn(Float32, base.nb_weights_v2c_c2v)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_weights_c2v_v2c)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.nb_weights_c2v_readout)
        end

        return new(
            base,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function StandardNeuralBP(
        base::NeuralBPBase,
        weights_v2c_c2v::Vector{Float32},
        weights_c2v_v2c::Vector{Float32},
        weights_c2v_readout::Vector{Float32}
    )
        return new(
            base,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor StandardNeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

function (bpnn::StandardNeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
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

    ## Define the extended weights matrix, that can be multiplied with the adjacency matrices.
    extended_weights_v2c_c2v = sparse(bpnn.base.non_zero_rows_V2C_C2V, bpnn.base.non_zero_cols_V2C_C2V, bpnn.weights_v2c_c2v, bpnn.base.nb_neurons_per_layer, bpnn.base.nb_neurons_per_layer) |> Matrix
    extended_weights_c2v_v2c = sparse(bpnn.base.non_zero_rows_C2V_V2C, bpnn.base.non_zero_cols_C2V_V2C, bpnn.weights_c2v_v2c, bpnn.base.nb_neurons_per_layer, bpnn.base.nb_neurons_per_layer) |> Matrix
    extended_weights_c2v_readout = sparse(bpnn.base.non_zero_rows_C2V_readout, bpnn.base.non_zero_cols_C2V_readout, bpnn.weights_c2v_readout, bpnn.base.code_n_bits, bpnn.base.nb_neurons_per_layer) |> Matrix

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
    v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.base.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.base.n_layers
        # 1. Forward pass from V2C to C2V layer
        # c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        c2v_neurons = (@. sigmoid(extended_weights_c2v_v2c) * bpnn.base.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.base.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        # c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions
        c2v_activated_neurons = 2 .* safe_atanh_exp(c2v_neurons) .* phase_contributions # Use safe version to avoid numerical issues.

        # 2. Forward pass from C2V to V2C layer
        # v2c_neurons = (bpnn.weights_v2c_c2v[1:end, 1:end, iter] .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        v2c_neurons = (@. extended_weights_v2c_c2v * bpnn.base.adj_C2V_V2C') * c2v_activated_neurons + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        if (iter == bpnn.base.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (extended_weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    return readout_neurons
end

struct SoftStandardBP <: NeuralBP
    """
    This is a variant of the Standard Neural BP where we have soft constraints to encourage correlated error patterns.
    For each edge in the connectivity matrix, we will add a soft check that encourages the corresponding variable nodes to have similar LLRs.
    For example, if variable nodes v1 and v2 are connected in the connectivity matrix, we will add a check s_k that is connected to both v1 and v2.

    The messages passed to and from these soft checks are as follows.
    1. From soft check s_k to variable node v1:
    - m_(s_k -> v1) = m_(v2 -> s_k).
    - The activation function for the soft check neuron is: f(x) = log( (exp(x) + e^-w) / (e^-w * exp(x) + 1) ), where 'w' is the weight that will be learned during training.
    - Note that when w -> ∞, this activation function reduces to f(x) = x, which corresponds to a hard equality constraint. Similarly for w -> 0, we get f(x) = 0, which corresponds to no constraint.
    
    2. From variable node v1 to soft check s_k:
    - m_(v1 -> s_k) = LLR(v1) + sum of all incoming messages to v1 except from s_k.
    - There is no special activation function for this message; it is the identity function.
    
    Hence, we will have additional C2V and V2C neurons corresponding to the messages passed for these soft checks.
    - The neuron m_(s_k -> v1) will be connected to only to m_(v2 -> s_k).
    - The neuron m_(v1 -> s_k) will be connected to all the C2V neurons corresponding to incoming messages to v1, except m_(s_k -> v1).

    To make the construction of the adjacency matrices easier, we will augment the parity-check matrix with additional rows corresponding to the soft checks.
    For each row of the connectivity matrix, we will add a new row to the parity-check matrix with 1s in the columns corresponding to the variable nodes connected by that edge.
    This augmented parity-check matrix will be used to construct the adjacency matrices and other elements of the NeuralBPBase.

    We will store the index in the V2C layer where the soft constraint messages start, so that during the forward pass we can separately address the soft constraint messages from the regular BP messages.
    """
    base::NeuralBPBase
    soft_neurons_starting_index::Int # Index where soft constraint messages start.
    c2v_checks_indices::Vector{Int} # An index mapping from each soft constraint neuron in C2V layer to the corresponding soft check index.
    
    # For convenience we will split the adjacency matrices in the base into hard and soft parts during the forward pass.
    adj_V2C_C2V_hard::Matrix{Int}
    adj_V2C_C2V_soft::Matrix{Int}
    adj_C2V_V2C_hard::Matrix{Int}
    adj_C2V_V2C_soft::Matrix{Int}

    # Weights for regular BP messages between variable and check nodes
    weights_v2c_c2v::Matrix{Float32}
    weights_c2v_v2c::Matrix{Float32}
    weights_c2v_readout::Matrix{Float32}
    # Weights for soft constraint messages
    weights_soft_checks::Vector{Float32}
    
    function SoftStandardBP(
        base::NeuralBPBase;
        weights_v2c_c2v::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_v2c::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_readout::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_soft_checks::Vector{Float32}=Vector{Float32}(undef, 0)
    )
        """
        Define the SoftStandardNeuralBP model.
        """
        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, base.nb_neurons_per_layer, base.nb_neurons_per_layer)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, base.code_n_bits, base.nb_neurons_per_layer)
        end
        if (length(weights_soft_checks) == 0)
            n_soft_checks = size(base.connectivity, 1)
            weights_soft_checks = randn(Float32, n_soft_checks)
        end
        # The starting index of the soft neurons in the V2C layer is one more than the total number of regular BP neurons.
        soft_neurons_starting_index = base.nb_neurons_per_layer + 1
        # Refedine the base by adding the soft constraints to the parity-check matrix.
        base_with_soft_constraints = add_soft_constraints_to_neuralbpbase(base)
        
        # Split the adjacency matrices in the base into hard and soft parts.
        adj_V2C_C2V_hard = base_with_soft_constraints.adj_V2C_C2V[1:base.nb_neurons_per_layer, 1:base.nb_neurons_per_layer]
        adj_V2C_C2V_soft = base_with_soft_constraints.adj_V2C_C2V[base.nb_neurons_per_layer+1:end, base.nb_neurons_per_layer+1:end]
        adj_C2V_V2C_hard = base_with_soft_constraints.adj_C2V_V2C[1:base.nb_neurons_per_layer, 1:base.nb_neurons_per_layer]
        adj_C2V_V2C_soft = base_with_soft_constraints.adj_C2V_V2C[base.nb_neurons_per_layer+1:end, base.nb_neurons_per_layer+1:end]
        
        # Mapping between soft constraint neurons in C2V layer to the corresponding soft check index.
        c2v_checks_indices = Vector{Int}(undef, size(base_with_soft_constraints.nb_neurons_per_layer))
        for neuron_index in 1:base_with_soft_constraints.nb_neurons_per_layer
            if neuron_index >= soft_neurons_starting_index
                # This is a soft constraint neuron.
                # The corresponding soft check index is given by:
                c2v_checks_indices[neuron_index] = neuron_index - base.nb_neurons_per_layer
            else
                # This is a regular BP neuron.
                c2v_checks_indices[neuron_index] = -1 # Indicate that this is not a soft check neuron.
            end
        end

        # Create a SoftStandardBP instance
        return new(
            base_with_soft_constraints,
            soft_neurons_starting_index,
            c2v_checks_indices,
            adj_V2C_C2V_hard,
            adj_V2C_C2V_soft,
            adj_C2V_V2C_hard,
            adj_C2V_V2C_soft,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout,
            weights_soft_checks
        )   
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function SoftStandardBP(
        base::NeuralBPBase,
        soft_neurons_starting_index::Int,
        c2v_checks_indices::Vector{Int},
        adj_V2C_C2V_hard::Matrix{Int},
        adj_V2C_C2V_soft::Matrix{Int},
        adj_C2V_V2C_hard::Matrix{Int},
        adj_C2V_V2C_soft::Matrix{Int},
        weights_v2c_c2v::Matrix{Float32},
        weights_c2v_v2c::Matrix{Float32},
        weights_c2v_readout::Matrix{Float32},
        weights_soft_checks::Vector{Float32}
    )
        return new(
            base,
            soft_neurons_starting_index,
            c2v_checks_indices,
            adj_V2C_C2V_hard,
            adj_V2C_C2V_soft,
            adj_C2V_V2C_hard,
            adj_C2V_V2C_soft,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout,
            weights_soft_checks
        )
    end
end

function (bpnn::SoftStandardBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
    """
    Forward pass through the Neural Network for Standard Neural Belief Propagation with soft constraints.
    The forward pass is similar to that of StandardNeuralBP, with additional steps to handle the soft constraint messages.

    For the hard constraint messages, we follow the same steps as in StandardNeuralBP.
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

    For the soft constraint messages, we have to modify steps in 2.
    2a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` but weight 1.
    2b. Apply the soft activation function on the c2v neurons:
        f(x) = log( (exp(x) + e^-w) / (e^-w * exp(x) + 1) ), where 'w' is the weight corresponding to the soft check c.
    2c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` but weight 1 if C is a soft check.
    2d. No activation function for the soft constraint messages in V2C layer.
    """
    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.base.adj_initialize_V2C * initial_llrs_batch

    ## Apply the activation function elemenwise to all neurons in `v2c`
    # For the hard constaint neurons, we apply the standard activation function: log(tanh(abs(x)/2))
    v2c_neurons_hard = v2c_neurons[1:bpnn.soft_neurons_starting_index-1]
    v2c_activated_neurons_hard = safe_log_tanh(abs.(v2c_neurons_hard))
    # For the soft constraint neurons, there is no activation function; identity.
    v2c_neurons_soft = v2c_neurons[bpnn.soft_neurons_starting_index:end]
    v2c_activated_neurons_soft = v2c_neurons_soft # Identity activation function.
    # Combine the activated neurons back
    v2c_activated_neurons = vcat(v2c_activated_neurons_hard, v2c_activated_neurons_soft)

    # 2. For N iterations:
    # For each message computed in C2V, determine all the V2C neurons that contribute to it.
    # This is because any V2C neuron that is negative contributes a phase flip.
    # We only need to do this for the hard constraint messages.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.base.adj_V2C_C2V_hard)]
        
    for iter in 1:bpnn.base.n_layers
        # 1. Forward pass from V2C to C2V layer
        #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.
        c2v_neurons_hard = (@. sigmoid(bpnn.weights_c2v_v2c) * bpnn.base.adj_V2C_C2V_hard') * v2c_activated_neurons_hard
        c2v_neurons_soft = bpnn.base.adj_V2C_C2V_soft' * v2c_activated_neurons_soft # Weight is 1 for soft constraint messages.

        # Compute the number of negative messages (in `v2c_neurons_hard`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons_hard)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.base.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for the hard constraints in the C2V layer.
        c2v_activated_neurons_hard = 2 .* safe_atanh_exp(c2v_neurons_hard) .* phase_contributions
        # For the soft constraint neurons, apply the soft activation function.
        c2v_activated_neurons_soft = safe_soft_activation(c2v_neurons_soft)

        # 2. Forward pass from C2V to V2C layer
        # v2c_neurons = (@. bpnn.weights_v2c_c2v * bpnn.base.adj_C2V_V2C') * c2v_activated_neurons + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        v2c_neurons_hard = (@. bpnn.weights_v2c_c2v * bpnn.base.adj_C2V_V2C_hard') * c2v_activated_neurons_hard + bpnn.base.adj_C2V_V2C_soft' * c2v_activated_neurons_soft + bpnn.base.adj_initialize_V2C * initial_llrs_batch
        v2c_neurons_soft = (@. bpnn.weights_v2c_c2v * bpnn.base.adj_C2V_V2C_hard') 
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        # v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))
        v2c_activated_neurons = safe_log_tanh(abs.(v2c_neurons)) # Use safe version to avoid numerical issues.

        if (iter == bpnn.base.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.base.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    return readout_neurons
end
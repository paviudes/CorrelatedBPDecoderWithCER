struct NeuralBP
    """
    Structure to represent a layer of the Neural Network that corresponds to unfolded Belief Propagation.
    
    We want to define a Neural Network implementation in Julia.
        - The network consists of N + 2 layers, for some constant N that can be set.
        - The first layer consists of 'n' neurons.
        - There are 'N' middle layers. 
        - Each middle layer consists of two sublayers: V2C and C2V.
        - The sublayer V2C is a set of K_1 neurons.
        - The sublayer C2V is a set of K_2 neurons.
        - The neurons from V2C to C2V are connected using the adjacency matrix: adj_V2C_C2V.
        - The associated weights are weights_v2c_c2v.
        - The neurons from C2V to V2C are connected using an adjacency matrix: adj_C2V_V2C.
        - The associated weights are weights_c2v_v2c.
        - Each neuron in the C2V layer computes an activation function `f_act`.
        - The connectivity from the first layer to V2C is given by the adjacency matrix adj_initialize_V2C.
        - The readout layer consists of 'n' neurons.
        - The connectivity from the final layer's V2C to the readout layer is specified by the adjacency matrix adj_C2V_readout.
    """
    parity_check_matrix::BitMatrix
    code_n_checks::Int
    code_n_bits::Int
    parity_check_matrix_dual::BitMatrix
    nb_neurons_per_layer::Int
    neuron_to_check_variable::Dict{Int, Tuple{Int, Int}}  # Mapping from neuron index to (check, variable) pair.

    # Connectivity: fixed parameters
    adj_initialize_V2C::BitMatrix
    adj_V2C_C2V::BitMatrix
    adj_C2V_V2C::BitMatrix
    adj_C2V_readout::BitMatrix

    # Learnable parameters: weights.
    weights_v2c_c2v::Matrix{Float32}
    weights_c2v_v2c::Matrix{Float32}
    weights_c2v_readout::Matrix{Float32}

    function NeuralBP(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int};
        weights_v2c_c2v::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_v2c::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
        weights_c2v_readout::Matrix{Float32}=Matrix{Float32}(undef, 0, 0)
    )
        """
        Construct the elements of a `NeuralBPLayer` from a given parity-check matrix.
        The key elements are
        - The number of neurons in each layer: C2V and V2C: `nb_neurons_per_layer`
        - Dictionary to specify the mapping from a neuron to the corresponding (check, variable) pair.
            - Note that each neuron in the V2C and C2V layers corresponds to an edge in the Tanner graph defined by the parity-check matrix.
            - Each edge connects a variable node to a check node.
            - Thus, we can define a mapping from each neuron index to the corresponding (check, variable) pair.
        - The adjacency matrix connecting the input layer to the V2C layer (`adj_initialize_V2C`).
        - The adjacency matrix connecting the V2C and C2V layers (`adj_V2C_C2V`).
        - The adjacency matrix connecting the C2V and V2C layers (`adj_C2V_V2C`).
        - The adjacency matrix connecting the C2V layer to the readout layer (`adj_V2C_readout`).
        
        These elements can be defined as follows.
        1. Adjacency matrix from input to V2C (`adj_initialize_V2C`):
            This matrix has |V| x |C| rows and |V| columns.
            We have adj_initialize_V2C[i, j] = 
                - define (c, v) from i, and v' = j
                - 1 if H[c, v] == 1 and v == v'. 0 otherwise.
        2. Adjacency matrix from V2C to C2V (`adj_V2C_C2V`):
            This matrix has |V| x |C| rows and |V| x |C| columns.
            We have adj_V2C_C2V[i, j] = 
                - define (c, v) from i
                - define (c', v') from j
                - 1 if H[c, v] == 1 and v != v' and H[c', v'] == 1 and c == c'. 0 otherwise.
        3. Adjacency matrix from C2V to V2C (`adj_C2V_V2C`):
            This matrix has |C| x |V| rows and |C| x |V| columns.
            We have adj_C2V_V2C[i, j] = 
                - define (c, v) from i
                - define (c', v') from j
                - 1 if H[c, v] == 1 and c != c' and H[c', v'] == 1 and v == v'. 0 otherwise.
        4. Adjacency matrix from C2V to readout (`adj_C2V_readout`):
            This matrix has |V| rows and |C| x |V| columns.
            We have adj_C2V_readout[i, j] = 
                - define (c', v') from j, and v = i
                - 1 if H[c', v'] == 1 and v == v'. 0 otherwise.

        ## Setting the learnable parameters to default values.
        1. Weights for the connections from V2C to C2V: `weights_v2c_c2v`
        2. Weights for the connections from C2V to V2C: `weights_c2v_v2c`
        3. Weights for the connections from C2V to readout: `weights_c2v_readout`

        - Activation functions for the V2C and C2V layers.
        1. For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        2. For the C2V layer, the activation function is f(x) = atanh(exp(x + i π syndrome[c])) where c is the check node corresponding to the neuron.

        """
        parity_check_matrix = convert.(Bool, copy(parity_check_matrix))
        parity_check_matrix_dual = convert.(Bool, copy(parity_check_matrix_dual))

        (code_n_checks, code_n_bits) = size(parity_check_matrix)
        nb_neurons_per_layer = sum(parity_check_matrix)

        ## Define mappings
        # Mapping from neuron index to (check, variable) pair
        neuron_to_check_variable = Dict{Int, Tuple{Int, Int}}()
        neuron_index = 1
        for c in 1:code_n_checks
            for v in 1:code_n_bits
                if parity_check_matrix[c, v] == 1
                    neuron_to_check_variable[neuron_index] = (c, v)
                    neuron_index += 1
                end
            end
        end

        ## Define adjacency matrices
        # 1. Adjacency matrix from the input layer to V2C
        adj_initialize_V2C = zeros(Bool, nb_neurons_per_layer, code_n_bits)
        for i in 1:nb_neurons_per_layer
            (_, v) = neuron_to_check_variable[i]
            adj_initialize_V2C[i, v] = 1
        end

        # 2. Adjacency matrix from V2C to C2V
        adj_V2C_C2V = zeros(Bool, nb_neurons_per_layer, nb_neurons_per_layer)
        for i in 1:nb_neurons_per_layer
            (c, v) = neuron_to_check_variable[i]
            for j in 1:nb_neurons_per_layer
                (c_prime, v_prime) = neuron_to_check_variable[j]
                if c == c_prime && v != v_prime
                    adj_V2C_C2V[i, j] = 1
                end
            end
        end

        # 3. Adjacency matrix from C2V to V2C
        adj_C2V_V2C = zeros(Bool, nb_neurons_per_layer, nb_neurons_per_layer)
        for i in 1:nb_neurons_per_layer
            (c, v) = neuron_to_check_variable[i]
            for j in 1:nb_neurons_per_layer
                (c_prime, v_prime) = neuron_to_check_variable[j]
                if v == v_prime && c != c_prime
                    adj_C2V_V2C[i, j] = 1
                end
            end
        end

        # 4. Adjacency matrix from C2V to readout
        adj_C2V_readout = zeros(Bool, code_n_bits, nb_neurons_per_layer)
        for j in 1:nb_neurons_per_layer
            (_, v_prime) = neuron_to_check_variable[j]
            adj_C2V_readout[v_prime, j] = 1
        end

        # The parameters (weights) that are not explicitly fixed are considered learnable.
        # We will also initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        learnable_parameters = Vector{Symbol}()
        if (size(weights_v2c_c2v, 1) == 0)
            push!(learnable_parameters, :weights_v2c_c2v)
            weights_v2c_c2v = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            push!(learnable_parameters, :weights_c2v_v2c)
            weights_c2v_v2c = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
        end
        if (size(weights_c2v_readout, 1) == 0)
            push!(learnable_parameters, :weights_c2v_readout)
            weights_c2v_readout = randn(Float32, code_n_bits, nb_neurons_per_layer)
        end

        return new(
            parity_check_matrix,
            code_n_checks,
            code_n_bits,
            parity_check_matrix_dual,
            nb_neurons_per_layer,
            neuron_to_check_variable,
            adj_initialize_V2C,
            adj_V2C_C2V,
            adj_C2V_V2C,
            adj_C2V_readout,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Functors.@functor NeuralBP  # makes the weights trainable.

# Emplicitly tell the Flux framework that the weights are the trainable parameters.
function Flux.trainable(model::NeuralBP)
    return Tuple(getfield(model, pname) for pname in model.learnable_parameters)
end

function print_neuralbp_info(bpnn::NeuralBP; io::IO=Base.stdout)
    """
    Print the key information about the NeuralBP model.
    """
    println(io, "NeuralBP Decoder Information:")
    println(io, "----------------------------")
    println(io, "Number of bits (n): ", size(bpnn.parity_check_matrix, 2))
    println(io, "Number of checks (m): ", size(bpnn.parity_check_matrix, 1))
    println(io, "Number of neurons per layer: ", bpnn.nb_neurons_per_layer)
    println(io, "----------------------------")
    # print the input layer to V2C adjacency matrix
    println(io, "Adjacency matrix from input layer to V2C layer ($(size(bpnn.adj_initialize_V2C, 1)) x $(size(bpnn.adj_initialize_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_initialize_V2C)
    println(io, "\n----------------------------")
    # print the V2C to C2V adjacency matrix
    println(io, "Adjacency matrix from V2C layer to C2V layer ($(size(bpnn.adj_V2C_C2V, 1)) x $(size(bpnn.adj_V2C_C2V, 2))):")
    show(io, "text/plain", bpnn.adj_V2C_C2V)
    println(io, "\n----------------------------")
    # print the C2V to V2C adjacency matrix
    println(io, "Adjacency matrix from C2V layer to V2C layer ($(size(bpnn.adj_C2V_V2C, 1)) x $(size(bpnn.adj_C2V_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_V2C)
    println(io, "\n----------------------------")
    # print the C2V to readout adjacency matrix
    println(io, "Adjacency matrix from C2V layer to readout layer ($(size(bpnn.adj_C2V_readout, 1)) x $(size(bpnn.adj_C2V_readout, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_readout)
    println(io, "\n----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Weights from V2C to C2V layer ($(size(bpnn.weights_v2c_c2v, 1)) x $(size(bpnn.weights_v2c_c2v, 2))):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Weights from C2V to V2C layer ($(size(bpnn.weights_c2v_v2c, 1)) x $(size(bpnn.weights_c2v_v2c, 2))):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Weights from C2V to readout layer ($(size(bpnn.weights_c2v_readout, 1)) x $(size(bpnn.weights_c2v_readout, 2))):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
end

function (bpnn::NeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix; n_layers::Int=5)
    """
    Forward pass through the Neural Network.
    We have four main steps:
    1. Forward pass from input layer to V2C layer. This is done using the adjacency matrix `adj_initialize_V2C` and the corresponding weights.
    2. For N iterations:
        a. Forward pass from V2C to C2V layer using `adj_V2C_C2V` and weights.
        b. Apply activation functions for C2V layer:
            f(x) = atanh(exp(x)) * (-1)^syndrome[c] * ∏_(v' ∈ N(c) - v)  (-1)^(δ(m_(v' -> c) < 0))).
        For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        c. Forward pass from C2V to V2C layer using `adj_C2V_V2C` and weights.
        d. Apply activation functions for V2C layer: f(x) = log(tanh(x/2)).
    3. Forward pass from final V2C layer to readout layer using `adj_C2V_readout` and weights.
    4. Return the output of the readout layer.
    """

    println("NeuralBP Forward Pass: n_layers =", n_layers, ", each with ", bpnn.nb_neurons_per_layer, " neurons.")
    println("Neuron state of size ", size(initial_llrs_batch), ": ", initial_llrs_batch)
    
    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.adj_initialize_V2C * initial_llrs_batch

    println("v2c_neurons after input layer: size ", size(v2c_neurons), ": ", v2c_neurons)

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))

    println("v2c_activated_neurons after input layer: size ", size(v2c_activated_neurons), ": ", v2c_activated_neurons)

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    for iter in 1:n_layers
        # 1. Forward pass from V2C to C2V layer
        c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        # Apply the activation functions for C2V layer.
        # Note that since `c2v_neurons` is a matrix, we need to apply the i-th activation function to the i-th row of the matrix.
        c2v_activated_neurons = zeros(Float32, size(c2v_neurons))
        for i in 1:bpnn.nb_neurons_per_layer
            (c, _) = bpnn.neuron_to_check_variable[i]
            
            # Compute the total number of messages (in `v2c_activated_neurons`) which are negative for the check node `c`.
            selected_v2c = findall(x -> x == 1, bpnn.adj_V2C_C2V[:, i])
            
            # In every column of `v2c_neurons`, compute the number of negative messages.
            negative_messages = sum(v2c_neurons[selected_v2c, :] .< 0, dims=1)
            
            # Compute the phase contribution from the syndrome and the negative messages.
            phase_contribution = mod.(syndromes_batch[c, 1:end] .+ negative_messages, 2)
            
            c2v_activated_neurons[i, 1:end] = 2 * atanh.(exp.(c2v_neurons[i, 1:end])) .* ((-1) .^ phase_contribution)
        end

        # 2. Forward pass from C2V to V2C layer
        v2c_neurons = (bpnn.weights_v2c_c2v .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        #== Debug: Compute v2c_neurons using explicit loops and compare with matrix multiplication.
        v2c_neurons = zeros(Float32, bpnn.nb_neurons_per_layer, size(initial_llrs_batch, 2))
        for i in 1:bpnn.nb_neurons_per_layer
            (_, v) = bpnn.neuron_to_check_variable[i]
            # The neurons corresponding to `v2c messages` are sum of messages from all connected `c2v neurons`, plus the initial llr for the variable node.
            selected_c2v = findall(x -> x == 1, bpnn.adj_C2V_V2C[:, i])
            v2c_neurons[i, 1:end] = sum(c2v_activated_neurons[selected_c2v, 1:end], dims=1) .+ initial_llrs_batch[v, 1:end]
            # v2c_neurons[i, 1:end] = sum(bpnn.weights_c2v_v2c[selected_c2v, i]' * c2v_activated_neurons[selected_c2v, 1:end], dims=1) .+ initial_llrs_batch[v, 1:end]
        end
        v2c_neurons_matmul = bpnn.adj_C2V_V2C' * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        println("v2c_neurons of size ", size(v2c_neurons), ": ")
        show(stdout, "text/plain", v2c_neurons)
        println()
        println("v2c_neurons_matmul of size ", size(v2c_neurons_matmul), ": ")
        show(stdout, "text/plain", v2c_neurons_matmul)
        println()
        # Check if the two matrices are close.
        if !all(isapprox.(v2c_neurons, v2c_neurons_matmul; atol=1e-5))
            error("v2c_neurons computed using explicit loops and matrix multiplication do not match.")
        end
        ==#
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))

        if (iter == n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            println("Readout adjacency matrix size: ", size(bpnn.adj_C2V_readout))
            show(stdout, "text/plain", bpnn.adj_C2V_readout)
            println()
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    
    return readout_neurons
end

function sigmoid(x::Float32)::Float32
    # Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
    return 1.0f0 / (1.0f0 + exp(x))
end

function compute_loss_error_from_llrs(posterior_llrs::Matrix{Float32}, expected_recoveries::BitMatrix, parity_check_matrix_dual::BitMatrix)::Float64
    """
    Compute a Loss function from the posterior LLRs calculated by the NeuralBP model and the expected recoveries.
    Note that if the posterior LLR is positive, then σ(μ_k) ≈ 0 (no error), else σ(μ_k) ≈ 1 (error).
    The idea is that if the output of the BP decoder, e_pred (≈ σ(μ)) added to the expected recovery (e) commutes with the elements of the dual code, then it is a stabilizer.
    Thus, e_total = e_pred + e_expected should satisfy H^⟂ * M * e_total = 0, where M is the symplectic matrix.
    
    This motivates the Loss function in Eq. 8 of https://arxiv.org/abs/1811.07835.
    L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij M_(jk) [ e_k + σ(μ_k)])
    where
        - σ(μ_k) = 1 / (1 + exp(μ_k))
        - f(x) = |sin(π x / 2)|
        - M = [0 I ; I 0] is the symplectic matrix
        - H^⟂ is the parity-check matrix of the dual code.
    """
    (n_bits, n_samples) = size(expected_recoveries)
    loss_samples = zeros(Float32, n_samples)
    for s in 1:n_samples
        e_expected = convert.(Float32, expected_recoveries[:, s])
        e_pred::Vector{Float32} = (posterior_llrs[:, s] .< 0)
        e_total = e_expected + e_pred

        # Since M is the symplectic matrix, we can compute M * e_total as follows:
        # If e_total = [e_X; e_Z], then M * e_total = [e_Z; e_X]
        half_n = n_bits ÷ 2
        e_X = e_total[1:half_n]
        e_Z = e_total[half_n+1:end]
        M_e_total = vcat(e_Z, e_X)

        # Compute H^⟂ * M * e_total
        commutation_relations = parity_check_matrix_dual * M_e_total

        # For each bit of the commutation relations, compute the loss contribution.
        loss_sample = 0.0
        for i in 1:size(commutation_relations::Vector{Float32}, 1)
            loss_sample += abs(sin(π * commutation_relations[i] / 2))
        end
        loss_samples[s] = loss_sample
    end

    average_loss = sum(loss_samples) / n_samples
    return average_loss
end

function train_neuralbp!(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    initial_llrs::Matrix{Float32}=ones(Float32, size(expected_recoveries)),
    optimizer::Function=ADAM,
    n_epochs::Int=10,
    batch_size::Int=32
)
    """
    Train the NeuralBP model using the provided syndromes and expected recoveries.
    The provided syndromes should be computed from the errors, which are provided as expected recoveries.
    Arguments:
    - `bpnn::NeuralBP`: The NeuralBP model to be trained.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    - `initial_llrs::Matrix{Float32}`: Initial LLRs for the bits, to be used as input to the network (default: 1.0 since it is the LLR corresponding to the probability of the bit being 0 equal to 0.9).
    - `optimizer`: The optimizer to use for training (default: ADAM).
    - `n_epochs::Int`: The number of epochs. Each epoch goes through the entire dataset once (default: 10).
    - `batch_size::Int`: The size of each batch for training (default: 32).
    - `loss_function`: The loss function to use for training (default: binary cross-entropy).
    """
    # Batch the training data: syndromes and expected recoveries, into batches.
    # Each batch will be passed through the network as a Matrix.
    # Each batch will be a tuple of (syndromes_batch, expected_recoveries_batch)
    
    # Split an array of 1:n_samples into batches of size batch_size
    n_samples = size(syndromes, 2)
    samples_grouped_by_batch = [
        (i-1) * batch_size + 1 : min(i * batch_size, n_samples)
        for i in 1:ceil(Int, n_samples / batch_size)
    ]
    # Create the training dataset as a vector of tuples
    training_dataset = [
        (
            syndromes[:, batch_sample_indices],
            expected_recoveries[:, batch_sample_indices],
            repeat(initial_llrs[:, 1:1], 1, length(batch_sample_indices))
        )
        for batch_sample_indices in samples_grouped_by_batch
    ]

    for epoch in 1:n_epochs
        for (syndromes_train_batch, expected_recoveries_batch, llrs_batch) in training_dataset
            #=
            # This code is depricated since we are using the Flux.gradient function.
            gs = Flux.gradient(Flux.params(bpnn)) do
                posterior_llrs_pred = bpnn(llrs_batch, syndromes_train_batch)
                loss = compute_loss_error_from_llrs(posterior_llrs_pred, expected_recoveries_batch, bpnn.parity_check_matrix_dual)
                return loss
            end
            =#
            posterior_llrs_pred = bpnn(llrs_batch, syndromes_train_batch)
            batch_loss = (bpnn::NeuralBP) -> compute_loss_error_from_llrs(
                posterior_llrs_pred,
                expected_recoveries_batch,
                bpnn.parity_check_matrix_dual
            )
            # Compute gradients
            grads = gradient(batch_loss, bpnn)
            Flux.Optimise.update!(optimizer, Flux.params(bpnn), grads)
        end
        println("Epoch $epoch completed.")
    end
end

function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix)::BitMatrix
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    
    Returns:
    - `predicted_recoveries::BitMatrix`: A matrix where each column represents the predicted recovery (error pattern) corresponding to the syndrome.
    """
    predicted_recoveries_LLRs = bpnn(syndromes)
    predicted_recoveries = convert.(Bool, (predicted_recoveries_LLRs .> 0))
    return predicted_recoveries
end

function generate_training_data(parity_check_matrix::Matrix{Int}, n_samples::Int, error_probability::Float64)::Tuple{BitMatrix, BitMatrix}
    """
    Generate training data for the NeuralBP model.
    Each sample consists of a random error pattern generated according to the specified error probability.
    The corresponding syndrome is computed using the provided parity-check matrix.
    The error patterns serve as the expected recoveries.

    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `n_samples::Int`: The number of samples to generate.
    - `error_probability::Float64`: The probability of an error occurring on each bit.
    
    Returns:
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `expected_recoveries::BitMatrix`: A matrix where each column represents the expected recovery (error pattern) corresponding to the syndrome.
    """
    (n_checks, n_bits) = size(parity_check_matrix)
    syndromes = zeros(Bool, n_checks, n_samples)
    expected_recoveries = zeros(Bool, n_bits, n_samples)

    for i in 1:n_samples
        # Generate a random error pattern
        error_pattern = rand(Bool, n_bits) .< error_probability
        expected_recoveries[:, i] = error_pattern

        # Compute the syndrome
        syndrome = mod.(parity_check_matrix * error_pattern, 2)
        syndromes[:, i] = syndrome
    end

    return syndromes, expected_recoveries
end
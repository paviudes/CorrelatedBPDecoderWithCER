import Zygote: gradient, Params
using Functors: @functor

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
    neuron_to_checks::Vector{Int} # Mapping from neuron index to check node.
    neuron_to_bits::Vector{Int} # Mapping from neuron index to variable node.

    # Connectivity: fixed parameters
    adj_initialize_V2C::BitMatrix
    adj_V2C_C2V::BitMatrix
    adj_C2V_V2C::BitMatrix
    adj_C2V_readout::BitMatrix

    # Parameters of the Neural Network
    initial_llrs::Vector{Float32}
    n_layers::Int
    
    # Learnable parameters: weights.
    # learnable_parameters::Vector{Symbol}
    weights_v2c_c2v::Matrix{Float32}
    weights_c2v_v2c::Matrix{Float32}
    weights_c2v_readout::Matrix{Float32}

    function NeuralBP(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int},
        initial_llrs::Vector{Float32},
        n_layers::Int;
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
        neuron_to_checks = Vector{Int}(undef, nb_neurons_per_layer)
        neuron_to_bits = Vector{Int}(undef, nb_neurons_per_layer)
        neuron_index = 1
        for c in 1:code_n_checks
            for v in 1:code_n_bits
                if parity_check_matrix[c, v] == 1
                    neuron_to_check_variable[neuron_index] = (c, v)
                    neuron_to_checks[neuron_index] = c
                    neuron_to_bits[neuron_index] = v
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

        # We will initialize the learnable parameters to Gaussian random values, if they are not explicitly provided.
        if (size(weights_v2c_c2v, 1) == 0)
            weights_v2c_c2v = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
        end
        if (size(weights_c2v_v2c, 1) == 0)
            weights_c2v_v2c = randn(Float32, nb_neurons_per_layer, nb_neurons_per_layer)
        end
        if (size(weights_c2v_readout, 1) == 0)
            weights_c2v_readout = randn(Float32, code_n_bits, nb_neurons_per_layer)
        end

        return new(
            parity_check_matrix,
            code_n_checks,
            code_n_bits,
            parity_check_matrix_dual,
            nb_neurons_per_layer,
            neuron_to_check_variable,
            neuron_to_checks,
            neuron_to_bits,
            adj_initialize_V2C,
            adj_V2C_C2V,
            adj_C2V_V2C,
            adj_C2V_readout,
            initial_llrs,
            n_layers,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end

    # Internal constructor for Flux/Functors reconstruction with all field values
    function NeuralBP(
        parity_check_matrix::BitMatrix,
        code_n_checks::Int,
        code_n_bits::Int,
        parity_check_matrix_dual::BitMatrix,
        nb_neurons_per_layer::Int,
        neuron_to_check_variable::Dict{Int, Tuple{Int, Int}},
        neuron_to_checks::Vector{Int},
        neuron_to_bits::Vector{Int},
        adj_initialize_V2C::BitMatrix,
        adj_V2C_C2V::BitMatrix,
        adj_C2V_V2C::BitMatrix,
        adj_C2V_readout::BitMatrix,
        initial_llrs::Vector{Float32},
        n_layers::Int,
        # learnable_parameters::Vector{Symbol},
        weights_v2c_c2v::Matrix{Float32},
        weights_c2v_v2c::Matrix{Float32},
        weights_c2v_readout::Matrix{Float32}
    )
        return new(
            parity_check_matrix,
            code_n_checks,
            code_n_bits,
            parity_check_matrix_dual,
            nb_neurons_per_layer,
            neuron_to_check_variable,
            neuron_to_checks,
            neuron_to_bits,
            adj_initialize_V2C,
            adj_V2C_C2V,
            adj_C2V_V2C,
            adj_C2V_readout,
            initial_llrs,
            n_layers,
            # learnable_parameters,
            weights_v2c_c2v,
            weights_c2v_v2c,
            weights_c2v_readout
        )
    end
end

# Make NeuralBP work with Functors by only making the weight matrices children
@functor NeuralBP (weights_v2c_c2v, weights_c2v_v2c, weights_c2v_readout)

# Explicitly tell the Flux framework that the weights are the trainable parameters.
function Flux.trainable(model::NeuralBP)
    return (
        weights_v2c_c2v = model.weights_v2c_c2v,
        weights_c2v_v2c = model.weights_c2v_v2c,
        weights_c2v_readout = model.weights_c2v_readout
    )
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

function print_neuralbp_summary(bpnn::NeuralBP; io::IO=Base.stdout, final_llrs::Matrix{Float32}=Float32[])
    """
    Print a summary of training the NeuralBP model.
    Print the fitted parameters of the model.
    1. Weights from V2C to C2V
    2. Weights from C2V to V2C
    3. Weights from C2V to readout
    4. Initial LLRs
    5. (Optional) Final LLRs after training
    """
    println(io, "NeuralBP Training Summary:")
    println(io, "----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Fitted Weights from V2C to C2V layer ($(size(bpnn.weights_v2c_c2v, 1)) x $(size(bpnn.weights_v2c_c2v, 2))):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Fitted Weights from C2V to V2C layer ($(size(bpnn.weights_c2v_v2c, 1)) x $(size(bpnn.weights_c2v_v2c, 2))):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Fitted Weights from C2V to readout layer ($(size(bpnn.weights_c2v_readout, 1)) x $(size(bpnn.weights_c2v_readout, 2))):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
    # print the initial LLRs
    println(io, "Initial LLRs ($(length(bpnn.initial_llrs))):")
    show(io, "text/plain", bpnn.initial_llrs)
    println(io, "\n----------------------------")
    # print the final LLRs if provided
    if length(final_llrs) > 0
        println(io, "Final LLRs after training ($(size(final_llrs'))):")
        show(io, "text/plain", final_llrs')
        println(io, "\n----------------------------")
    end
end

function load_trained_neuralbp_model(weights_filename::String, bpnn::NeuralBP)::NeuralBP
    """
    Load a trained version of the NeuralBP model from a file.
    The file should contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    The function will reconstruct the weight matrices from the vectorized versions and create a NeuralBP model with these weights.
    """
    # Load the weights from the file
    fp = open(weights_filename, "r")
    weights_data = JSON.parse(fp)
    weights_v2c_c2v = reshape(Float32.(weights_data["weights_v2c_c2v"]), (bpnn.nb_neurons_per_layer, bpnn.nb_neurons_per_layer))
    weights_c2v_v2c = reshape(Float32.(weights_data["weights_c2v_v2c"]), (bpnn.nb_neurons_per_layer, bpnn.nb_neurons_per_layer))
    weights_c2v_readout = reshape(Float32.(weights_data["weights_c2v_readout"]), (bpnn.code_n_bits, bpnn.nb_neurons_per_layer))
    close(fp)

    # Create a new model with the loaded weights
    loaded_bpnn = NeuralBP(
        convert.(Int, bpnn.parity_check_matrix),
        convert.(Int, bpnn.parity_check_matrix_dual),
        bpnn.initial_llrs,
        bpnn.n_layers;
        weights_v2c_c2v=weights_v2c_c2v,
        weights_c2v_v2c=weights_c2v_v2c,
        weights_c2v_readout=weights_c2v_readout
    )
    return loaded_bpnn
end

function save_trained_neuralbp_model(weights_filename::String, bpnn::NeuralBP)
    """
    Save the trained version of the NeuralBP model to a file.
    The file will contain the weights that specify the forward pass of the NeuralBP model.
    These weights are:
    1. weights_v2c_c2v
    2. weights_c2v_v2c
    3. weights_c2v_readout
    They will be stored in a dictionary with the corresponding keys. The values will be vectorized versions of the weight matrices.
    """
    # Create a dictionary to store the weights
    weights_data = Dict{String, Any}()
    weights_data["weights_v2c_c2v"] = vec(bpnn.weights_v2c_c2v)
    weights_data["weights_c2v_v2c"] = vec(bpnn.weights_c2v_v2c)
    weights_data["weights_c2v_readout"] = vec(bpnn.weights_c2v_readout)

    # Save the weights to the file
    fp = open(weights_filename, "w")
    JSON.print(fp, weights_data)
    close(fp)
end

function (bpnn::NeuralBP)(initial_llrs_batch::AbstractMatrix{<:Real}, syndromes_batch::BitMatrix)
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

    # 1. Forward pass from input layer to V2C layer
    v2c_neurons = bpnn.adj_initialize_V2C * initial_llrs_batch

    # Apply the activation function elemenwise to all neurons in `v2c_input`
    v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))

    # Initialize the readout neurons
    readout_neurons = zeros(Float32, bpnn.code_n_bits, size(initial_llrs_batch, 2))

    # 2. For N iterations:
    # Precompute the selected V2C neurons for each C2V neuron.
    selected_v2c = [findall(col .== 1) for col in eachcol(bpnn.adj_V2C_C2V)]
    # println("Selected V2C neurons for all C2V neurons: ", selected_v2c)
        
    for iter in 1:bpnn.n_layers
        # 1. Forward pass from V2C to C2V layer
        # c2v_neurons = (bpnn.weights_c2v_v2c .* bpnn.adj_V2C_C2V') * v2c_activated_neurons
        c2v_neurons = (sigmoid.(bpnn.weights_c2v_v2c) .* bpnn.adj_V2C_C2V') * v2c_activated_neurons #TODO: check if sigmoid is needed. Without the sigmoid, the training is unstable, but the algorithm is more faithful to BP.

        # Compute the number of negative messages (in `v2c_activated_neurons`) in the expression for each C2V neuron.
        # For each neuron in the C2V layer, we need to compute the number of negative messages from the corresponding V2C neurons that are connected to it.
        n_negative_messages = hcat([[count(v2c_neurons_in_batch[rows] .< 0) for rows in selected_v2c] for v2c_neurons_in_batch in eachcol(v2c_neurons)]...)

        # Compute the phase contribution from the syndrome and the negative messages.
        phase_contributions = (-1) .^ (syndromes_batch[bpnn.neuron_to_checks, 1:end] .+ n_negative_messages)

        # Apply the activation functions for C2V layer.
        c2v_activated_neurons = 2 * atanh.(exp.(c2v_neurons)) .* phase_contributions

        #=
        # Note that since `c2v_neurons` is a matrix, we need to apply the i-th activation function to the i-th row of the matrix.
        c2v_activated_neurons = zeros(Float32, size(c2v_neurons))
        for i in 1:bpnn.nb_neurons_per_layer
            (c, _) = bpnn.neuron_to_check_variable[i]
            
            println("Number of negative messages incoming to neuron $i at iteration $iter: ", n_negative_messages[i, 1:end])

            println("Syndrome for check node $c at iteration $iter: ", syndromes_batch[c, 1:end])
            
            # Compute the phase contribution from the syndrome and the negative messages.
            # phase_contribution = (-1) .^ (syndromes_batch[c, 1:end] .+ n_negative_messages[i, 1:end])

            if (any(c2v_neurons[i, 1:end] .> 0))
                DomainError("Invalid value encountered in atanh for C2V neuron $i during iteration $iter. atanh(exp(x)) is only defined for x <= 0. We have x = $(c2v_neurons[i, 1:end]).")
            end
            println("Phase contribution for neuron $i at iteration $iter: ", phase_contributions[i, 1:end])
            println("C2V neuron values before activation for neuron $i at iteration $iter: ", c2v_neurons[i, 1:end])
            c2v_activated_neurons[i, 1:end] = 2 * atanh.(exp.(c2v_neurons[i, 1:end])) .* phase_contributions[i, 1:end]
        end
        =#

        # 2. Forward pass from C2V to V2C layer
        v2c_neurons = (bpnn.weights_v2c_c2v .* bpnn.adj_C2V_V2C') * c2v_activated_neurons .+ (bpnn.adj_initialize_V2C * initial_llrs_batch)
        
        # Apply the activation function for V2C layer.
        # Since the activation function is the same for all neurons, we can apply it elementwise.
        v2c_activated_neurons = log.(tanh.(abs.(v2c_neurons) ./ 2))

        if (iter == bpnn.n_layers)
            # 3. Forward pass from final V2C layer to readout layer
            readout_neurons = initial_llrs_batch .+ (bpnn.weights_c2v_readout .* bpnn.adj_C2V_readout) * c2v_activated_neurons
        end
    end
    
    return readout_neurons
end

# Define the sigmoid function: σ(x) = 1 / (1 + exp(x))
sigmoid(x::T) where T <: Number = 1.0f0 / (1.0f0 + exp(x))

function compute_loss_error_from_llrs(posterior_llrs::Matrix{Float32}, expected_recoveries::BitMatrix, parity_check_matrix_dual::BitMatrix)::Float64
    """
    Compute a Loss function from the posterior LLRs calculated by the NeuralBP model and the expected recoveries.
    Note that if the posterior LLR is positive, then σ(μ_k) ≈ 0 (no error), else σ(μ_k) ≈ 1 (error).
    The idea is that if the output of the BP decoder, e_pred (≈ σ(μ)) added to the expected recovery (e) commutes with the elements of the dual code, then it is a stabilizer.
    Thus, e_total = e_pred + e_expected should satisfy H^⟂ * e_total = 0, where H^⟂ is the parity-check matrix of the dual code.
    In the context of stabilizer codes, when H is the parity-check matrix specifying the Z-stabilizers, H^⟂ specifies the X-type normalizers.
    
    This motivates the Loss function in Eq. 8 of https://arxiv.org/abs/1811.07835.
    L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij [ e_k + σ(μ_k)])
    where
        - σ(μ_k) = 1 / (1 + exp(μ_k))
        - f(x) = |sin(π x / 2)|
        - H^⟂ is the parity-check matrix of the dual code.
    """
    n_samples = size(expected_recoveries, 2)
    # Compute the average loss over all samples as a Matrix equation.
    e_pred_matrix = sigmoid.(posterior_llrs)
    e_total_matrix = convert.(Float32, expected_recoveries) .+ e_pred_matrix
    # println("e_total_matrix of shape: ", size(e_total_matrix), ": ", e_total_matrix)
    commutation_relations_matrix = parity_check_matrix_dual * e_total_matrix
    loss_matrix = sum(abs.(sin.(π .* commutation_relations_matrix ./ 2)), dims=1)
    average_loss = sum(loss_matrix) / n_samples
    # println("Average Loss (Matrix computation): ", average_loss)
    return average_loss
end

function train_neuralbp!(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    expected_recoveries::BitMatrix;
    optimizer=ADAM(1e-3), #TODO: unable to place a datatype without avoiding errors.
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
    # println("Samples grouped by batch: ", samples_grouped_by_batch)
    # Create the training dataset as a vector of tuples
    training_dataset = [
        (
            syndromes[1:end, batch_sample_indices],
            expected_recoveries[1:end, batch_sample_indices],
            repeat(bpnn.initial_llrs, 1, length(batch_sample_indices))
        )
        for batch_sample_indices in samples_grouped_by_batch
    ]

    # println("Starting training for $n_epochs epochs with batch size $batch_size... with training dataset of shape ", training_dataset)

    # Set the trainable parameters
    opt_state = Flux.setup(optimizer, bpnn)  # create optimiser state (tune lr as needed)

    for epoch in 1:n_epochs
        for (syndromes_train_batch, expected_recoveries_batch, llrs_batch) in training_dataset
            # compute loss and gradients in one shot
            loss, grads = Flux.withgradient(bpnn) do model
                posterior_llrs = model(llrs_batch, syndromes_train_batch)
                compute_loss_error_from_llrs(posterior_llrs, expected_recoveries_batch, bpnn.parity_check_matrix_dual)
            end
            # apply update. grads[1] contains gradients for the model
            Flux.update!(opt_state, bpnn, grads[1])
            # println("Epoch $epoch, Batch Loss: $loss")
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
    batch_size = size(syndromes, 2)
    initial_llrs_batch = repeat(bpnn.initial_llrs, 1, batch_size)
    predicted_recoveries_LLRs = bpnn(initial_llrs_batch, syndromes)
    # print_neuralbp_summary(bpnn; final_llrs=predicted_recoveries_LLRs)
    predicted_recoveries = convert.(Bool, (predicted_recoveries_LLRs .< 0))
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

function check_bp_solutions(parity_check_matrix_dual::Matrix{Int}, errors::BitMatrix, recoveries::BitMatrix)::BitVector
    """
    Check if the provided recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `parity_check_matrix::Matrix{Int}`: The parity-check matrix defining the code.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `recoveries::BitMatrix`: A matrix where each row represents the recovery pattern corresponding to the error.

    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.
    """
    n_samples = size(errors, 2)
    is_correct = BitVector(undef, n_samples)

    for i in 1:n_samples
        total_pattern = xor.(errors[1:end, i], recoveries[1:end, i])
        syndrome = mod.(parity_check_matrix_dual * total_pattern, 2)
        is_correct[i] = all(syndrome .== 0)

        # For debugging purposes: if a weight-0 error is not corrected, print details.
        if (sum(errors[1:end, i]) == 0) && (!is_correct[i])
            println("Debug Info: Weight-0 error not corrected for sample $i.")
            println("Error pattern: ", errors[1:end, i])
            println("Recovery pattern: ", recoveries[1:end, i])
            println("Total pattern (error + recovery): ", total_pattern)
            println("Syndrome: ", syndrome)
        end
    end
    return is_correct
end
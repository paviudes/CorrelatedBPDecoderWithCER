abstract type NeuralBP end

struct NeuralBPBase <: NeuralBP
    """
    Abstract type for Neural Belief Propagation models.

    Structure to represent a layer of the Neural Network that corresponds to unfolded Belief Propagation.

    Key fields:
    - parity_check_matrix: The parity-check matrix defining the code.
    - is_correlated: Boolean indicating if the error model is correlated.
    - connectivity: Matrix defining the connectivity for correlated errors.
    - correlation_strength: Float32 indicating the strength of correlations.
    - code_n_checks: Number of check nodes in the code.
    - code_n_bits: Number of variable nodes (bits) in the code.
    - parity_check_matrix_dual: The dual of the parity-check matrix.
    - nb_neurons_per_layer: Number of neurons in each layer of the Neural Network.
    - neuron_to_check_variable: Dictionary mapping neuron indices to (check, variable) pairs.
    - neuron_to_checks: Vector mapping neuron indices to check nodes.
    - neuron_to_bits: Vector mapping neuron indices to variable nodes.
    - adj_initialize_V2C: Adjacency matrix from input layer to V2C layer.
    - adj_V2C_C2V: Adjacency matrix from V2C layer to C2V layer.
    - adj_C2V_V2C: Adjacency matrix from C2V layer to V2C layer.
    - adj_C2V_readout: Adjacency matrix from C2V layer to readout layer.
    - initial_llrs: Vector of initial log-likelihood ratios (LLRs).
    - n_layers: Number of layers in the Neural Network.

    """
    parity_check_matrix::BitMatrix

    is_correlated::Bool
    connectivity::Matrix{Int}
    correlation_strengths::Vector{Float32}
    
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
    nb_weights_v2c_c2v::Int
    adj_C2V_V2C::BitMatrix
    nb_weights_c2v_v2c::Int
    adj_C2V_readout::BitMatrix
    nb_weights_c2v_readout::Int

    # Non-zero indices of the adjacency matrices for faster computation
    non_zero_rows_V2C_C2V::Vector{Int}
    non_zero_cols_V2C_C2V::Vector{Int}
    non_zero_rows_C2V_V2C::Vector{Int}
    non_zero_cols_C2V_V2C::Vector{Int}
    non_zero_rows_C2V_readout::Vector{Int}
    non_zero_cols_C2V_readout::Vector{Int}

    # Parameters of the Neural Network
    initial_llrs::Vector{Float32}
    n_layers::Int

    function NeuralBPBase(
        parity_check_matrix::Matrix{Int},
        parity_check_matrix_dual::Matrix{Int},
        initial_llrs::Vector{Float32},
        n_layers::Int;
        connectivity::Matrix{Int}=Matrix{Int}(undef, 0, 0),
        correlation_strengths::Vector{Float32}=Float32[]
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
            So: a(m_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m_(v'->c))
            is given by
                a(m_(c->v)) = adj_V2C_C2V[(c,v), (v', c)] * a(m_(v'->c)) + i π s_c
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
        
        - Activation functions for the V2C and C2V layers.
        1. For the V2C layer, irrespective of the particular neuron, the activation function is f(x) = log(tanh(x/2))
        2. For the C2V layer, the activation function is f(x) = atanh(exp(x + i π syndrome[c])) where c is the check node corresponding to the neuron.
        """
        parity_check_matrix = convert.(Bool, copy(parity_check_matrix))
        parity_check_matrix_dual = convert.(Bool, copy(parity_check_matrix_dual))

        (code_n_checks, code_n_bits) = size(parity_check_matrix)
        nb_neurons_per_layer = sum(parity_check_matrix)

        ## Interprett correlations.
        if (size(connectivity, 1) == 0)
            is_correlated = false
        else
            is_correlated = true
        end

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
        # Compute the number of connections from each V2C neuron to C2V neurons
        nb_weights_v2c_c2v = count(x -> x == 1, adj_V2C_C2V)

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
        # Compute the number of connections from each C2V neuron to V2C neurons
        nb_weights_c2v_v2c = count(x -> x == 1, adj_C2V_V2C)

        # 4. Adjacency matrix from C2V to readout
        adj_C2V_readout = zeros(Bool, code_n_bits, nb_neurons_per_layer)
        for j in 1:nb_neurons_per_layer
            (_, v_prime) = neuron_to_check_variable[j]
            adj_C2V_readout[v_prime, j] = 1
        end
        # Compute the number of connections from C2V neurons to readout neurons
        nb_weights_c2v_readout = count(x -> x == 1, adj_C2V_readout)

        # Compute the row and column indices of the non-zero elements in each adjacency matrix
        non_zero_V2C_C2V = findall(adj_V2C_C2V .== 1)
        non_zero_rows_V2C_C2V = [i[1] for i in non_zero_V2C_C2V]
        non_zero_cols_V2C_C2V = [i[2] for i in non_zero_V2C_C2V]

        non_zero_C2V_V2C = findall(adj_C2V_V2C .== 1)
        non_zero_rows_C2V_V2C = [i[1] for i in non_zero_C2V_V2C]
        non_zero_cols_C2V_V2C = [i[2] for i in non_zero_C2V_V2C]

        non_zero_C2V_readout = findall(adj_C2V_readout .== 1)
        non_zero_rows_C2V_readout = [i[1] for i in non_zero_C2V_readout]
        non_zero_cols_C2V_readout = [i[2] for i in non_zero_C2V_readout]
        
        return new(
            parity_check_matrix,
            is_correlated,
            connectivity,
            correlation_strengths,
            code_n_checks,
            code_n_bits,
            parity_check_matrix_dual,
            nb_neurons_per_layer,
            neuron_to_check_variable,
            neuron_to_checks,
            neuron_to_bits,
            adj_initialize_V2C,
            adj_V2C_C2V,
            nb_weights_v2c_c2v,
            adj_C2V_V2C,
            nb_weights_c2v_v2c,
            adj_C2V_readout,
            nb_weights_c2v_readout,
            non_zero_rows_V2C_C2V,
            non_zero_cols_V2C_C2V,
            non_zero_rows_C2V_V2C,
            non_zero_cols_C2V_V2C,
            non_zero_rows_C2V_readout,
            non_zero_cols_C2V_readout,
            initial_llrs,
            n_layers
        )
    end
end

function parse_cer_data(correlation_strengths_file::String; verbose::Bool = true)::Tuple{Matrix{Int}, Vector{Float32}, Dict{Int, Float32}}
    """
    Parse the correlation strengths and connectivity from a file.
    Each line of the file should be one of the two formats:
    index : value
    where index is a qubit index (integer) and value is the corresponding correlation strength (float), or
    (index1, index2) : value
    where index1 and index2 are the indices of two qubits (integers) specifying an edge between two qubits, and `value` is the weight for that edge.
    For instance, here are a few lines of the file:
    1 : 0.1
    (1, 29) : 0.005851149427861176
    2 : 0.2
    (1, 2) : 0.0063269667097774155
    (1, 30) : 0.006204408265565164
    (2, 3) : 0.006110318845201687
    4 : 0.05
    (2, 1) : 0.0063269667097774155
    (2, 31) : 0.005576402801319799
    (3, 2) : 0.006110318845201688
    6 : 0.15

    We need to parse this file to extract
    - the connectivity matrix: two-column matrix where each row corresponds to an edge between two qubits
    - the correlation strengths: vector where each element corresponds to the weight of the edge between the two qubits (same order as the rows of the connectivity matrix)
    - single qubit error rates: dictionary where each key is a qubit index and the value is the error rate of that qubit
    
    Parse the correlation strengths and connectivity from a file.
    Each line of the file should be one of the two formats:
    index : value
    where index is a qubit index (integer) and value is the corresponding correlation strength (float), or
    (index1, index2) : value
    where index1 and index2 are the indices of two qubits (integers) specifying an edge between two qubits, and `value` is the weight for that edge.
    
    For instance, here are a few lines of the file:
    1 : 0.1
    (1, 29) : 0.005851149427861176
    2 : 0.2
    (1, 2) : 0.0063269667097774155
    (1, 30) : 0.006204408265565164
    (2, 3) : 0.006110318845201687
    4 : 0.05
    (2, 1) : 0.0063269667097774155
    (2, 31) : 0.005576402801319799
    (3, 2) : 0.006110318845201688
    6 : 0.15

    We need to parse this file to extract
    - the connectivity matrix: two-column matrix where each row corresponds to an edge between two qubits
    - the correlation strengths: vector where each element corresponds to the weight of the edge between the two qubits (same order as the rows of the connectivity matrix)
    - single qubit error rates: dictionary where each key is a qubit index and the value is the error rate of that qubit
    """
    min_error_rate::Float32 = 1f-6  # minimum error rate to avoid issues with log(0) in the decoder

    # Graceful no-CER path: if the file is missing, warn and return empty
    # structures so the caller can proceed without correlation data instead of
    # crashing on `open`.
    if !isfile(correlation_strengths_file)
        verbose && @warn "parse_cer_data: CER file not present at " *
            "$(correlation_strengths_file); returning empty connectivity + " *
            "marginals (no correlation data)."
        return (Matrix{Int}(undef, 0, 2), Vector{Float32}(undef, 0), Dict{Int, Float32}())
    end

    connectivity = Vector{Tuple{Int, Int}}(undef, 0)
    correlation_strengths = Vector{Float32}(undef, 0)
    single_qubit_error_rates = Dict{Int, Float32}()

    # Counters for the end-of-function summary.
    n_two_qubit_unparsed::Int    = 0   # "(i,j): v"-shaped lines that failed to parse
    n_single_qubit_unparsed::Int = 0   # "i: v"-shaped lines that failed to parse
    n_unrecognized::Int          = 0   # non-empty lines matching neither shape

    # Whitespace-insensitive around ',' and ':'. The previous strict extraction
    # regex `\((\d+), (\d+)\) : ([\d\.]+)` demanded exactly one space after the
    # comma and around the colon, so files written as e.g. "(2,18): 0.000953"
    # passed the loose detection but failed extraction and were dropped SILENTLY.
    # `\s*` everywhere removes that gap; the shape is anchored end-to-end so a
    # malformed value can't be partially captured (it's flagged instead).
    #
    # The value pattern accepts an optional sign, a plain decimal, AND scientific
    # notation — needed because the weak edges are written in exponential form
    # (e.g. "(1,4): 1.9e-05", "(13,15): 9e-06").
    value = raw"[-+]?[\d.]+(?:[eE][-+]?\d+)?"
    single_qubit_regex::Regex = Regex("^\\s*(\\d+)\\s*:\\s*($value)\\s*\$")
    two_qubit_regex::Regex    = Regex("^\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*:\\s*($value)\\s*\$")

    open(correlation_strengths_file, "r") do io
        for raw_line in eachline(io)
            line = strip(raw_line)
            isempty(line) && continue

            if startswith(line, "(")
                # (b): two-qubit "(i, j) : v" line.
                m = match(two_qubit_regex, line)
                if m !== nothing
                    qubit1 = parse(Int, m.captures[1])
                    qubit2 = parse(Int, m.captures[2])
                    two_qubit_error_rate = parse(Float32, m.captures[3])
                    push!(connectivity, (qubit1, qubit2))
                    push!(correlation_strengths, max(two_qubit_error_rate, min_error_rate))
                else
                    n_two_qubit_unparsed += 1
                    if verbose
                        @warn "parse_cer_data: could not parse two-qubit line: $(raw_line)" maxlog = 5
                    end
                end
            elseif occursin(':', line)
                # (a): single-qubit "i : v" line.
                m = match(single_qubit_regex, line)
                if m !== nothing
                    qubit = parse(Int, m.captures[1])
                    error_rate = parse(Float32, m.captures[2])
                    single_qubit_error_rates[qubit] = max(error_rate, min_error_rate)
                else
                    n_single_qubit_unparsed += 1
                    if verbose
                        @warn "parse_cer_data: could not parse single-qubit line: $(raw_line)" maxlog = 5
                    end
                end
            else
                n_unrecognized += 1
                if verbose
                    @warn "parse_cer_data: line matches no expected format, skipped: $(raw_line)" maxlog = 5
                end
            end
        end
    end

    connectivity_matrix::Matrix{Int} = Matrix{Int}(undef, 0, 2)
    if !isempty(connectivity)
        connectivity_matrix = hcat([c[1] for c in connectivity], [c[2] for c in connectivity])
    end

    # Every qubit that appears in an edge needs a single-qubit rate; any that
    # has no explicit "i : v" line is "not found" and set to the default floor.
    n_single_qubit_defaulted::Int = 0
    for (qubit1, qubit2) in connectivity
        for qubit in (qubit1, qubit2)
            if !haskey(single_qubit_error_rates, qubit)
                single_qubit_error_rates[qubit] = min_error_rate
                n_single_qubit_defaulted += 1
            end
        end
    end

    # ---- End-of-function summary (gated by `verbose`). --------------------
    if verbose
        @info "parse_cer_data: parsed $(length(correlation_strengths)) two-qubit " *
              "marginals and $(length(single_qubit_error_rates)) single-qubit rates " *
              "from $(correlation_strengths_file)."
    end
    if n_two_qubit_unparsed > 0
        @warn "parse_cer_data: $(n_two_qubit_unparsed) two-qubit marginal(s) were not found (pair-shaped lines that failed to parse)."
    end
    if n_single_qubit_defaulted > 0
        @warn "parse_cer_data: $(n_single_qubit_defaulted) single-qubit marginal(s) were not found and set to the default (min_error_rate = $(min_error_rate))."
    end
    if n_single_qubit_unparsed > 0 || n_unrecognized > 0
        @warn "parse_cer_data: skipped $(n_single_qubit_unparsed) unparseable single-qubit line(s) and $(n_unrecognized) unrecognized line(s)."
    end

    return (connectivity_matrix, correlation_strengths, single_qubit_error_rates)
end

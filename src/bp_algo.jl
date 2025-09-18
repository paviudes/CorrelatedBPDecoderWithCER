using JSON
using LinearAlgebra

mutable struct BPSettings
    """
    Settings for the Belief Propagation algorithm.
    """
    n_iterations_of_BP::Int
    rounds_per_BP::Int
    llr_convergence_threshold::Float64
    llr_confidence_threshold::Float64
    weight_soft_constraint::Float64
    current_iteration::Int
    converged::Bool
    error::Vector{Int}
    syndrome::Vector{Int}
    initial_probabilities::Vector{Float64}
    final_probabilities::Vector{Float64}
    recovery_hard_decision::Vector{Int}
    verbose::Bool
    runtime::Float64
    is_decoder_failure::Bool
    function BPSettings(n_iterations_of_BP::Int, rounds_per_BP::Int, llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=2.0, weight_soft_constraint::Float64=0.5, current_iteration::Int=0, converged::Bool=false, error::Vector{Int}=zeros(Int, 0), syndrome::Vector{Int}=zeros(Int, 0), initial_probabilities::Vector{Float64}=zeros(Float64, 0), final_probabilities::Vector{Float64}=zeros(Float64, 0), verbose::Bool=false, runtime::Float64=0.0, is_decoder_failure::Bool=false)
        recovery_hard_decision = get_recovery_using_hard_decision(compute_llrs_from_probabilities(final_probabilities); confidence=0.0)
        new(n_iterations_of_BP, rounds_per_BP, llr_convergence_threshold, llr_confidence_threshold, weight_soft_constraint, current_iteration, converged, error, syndrome, initial_probabilities, final_probabilities, recovery_hard_decision, verbose, runtime, is_decoder_failure)
    end
end

function print_bp_settings(bpset::BPSettings; io::IO=stdout)
    """
    Print the BPSettings in a readable format.
    """
    println(io, "Belief Propagation Settings:")
    println(io, "Number of BP Iterations: ", bpset.n_iterations_of_BP)
    println(io, "Rounds per BP Iteration: ", bpset.rounds_per_BP)
    println(io, "LLR Convergence Threshold: ", bpset.llr_convergence_threshold)
    println(io, "LLR Confidence Threshold: ", bpset.llr_confidence_threshold)
    println(io, "Weight of Soft Constraint: ", bpset.weight_soft_constraint)
    println(io, "Current Iteration: ", bpset.current_iteration)
    println(io, "Converged: ", bpset.converged)
    println(io, "Error: ", bpset.error)
    println(io, "Syndrome: ", bpset.syndrome)
    println(io, "Initial Probabilities: ", bpset.initial_probabilities)
    println(io, "Final Probabilities: ", bpset.final_probabilities)
    println(io, "Recovery Hard Decision: ", bpset.recovery_hard_decision)
    println(io, "Verbose: ", bpset.verbose)
    println(io, "Runtime (seconds): ", bpset.runtime)
    println(io, "Is Decoder Failure: ", bpset.is_decoder_failure)
end

function save_bp_settings(bpset::BPSettings, filename::String)
    """
    Save the BPSettings to a JSON file.
    """
    bpset_dict = Dict(
        "n_iterations_of_BP" => bpset.n_iterations_of_BP,
        "rounds_per_BP" => bpset.rounds_per_BP,
        "llr_convergence_threshold" => bpset.llr_convergence_threshold,
        "llr_confidence_threshold" => bpset.llr_confidence_threshold,
        "weight_soft_constraint" => bpset.weight_soft_constraint,
        "current_iteration" => bpset.current_iteration,
        "converged" => bpset.converged,
        "error" => bpset.error,
        "syndrome" => bpset.syndrome,
        "initial_probabilities" => bpset.initial_probabilities,
        "final_probabilities" => bpset.final_probabilities,
        "recovery_hard_decision" => bpset.recovery_hard_decision,
        "verbose" => bpset.verbose,
        "runtime" => bpset.runtime,
        "is_decoder_failure" => bpset.is_decoder_failure
    )
    open(filename, "w") do io
        JSON.print(io, bpset_dict)
    end
end

function load_bp_settings(filename::String)::BPSettings
    """
    Load the BPSettings from a JSON file.
    """
    bpset_dict = open(filename, "r") do io
        JSON.parse(io)
    end
    return BPSettings(
        bpset_dict["n_iterations_of_BP"],
        bpset_dict["rounds_per_BP"],
        bpset_dict["llr_convergence_threshold"],
        bpset_dict["llr_confidence_threshold"],
        bpset_dict["weight_soft_constraint"],
        bpset_dict["current_iteration"],
        bpset_dict["converged"],
        bpset_dict["error"],
        bpset_dict["syndrome"],
        bpset_dict["initial_probabilities"],
        bpset_dict["final_probabilities"],
        bpset_dict["verbose"],
        bpset_dict["runtime"],
        bpset_dict["is_decoder_failure"]
    )
end

function get_message_from_check_to_vertex(G::TannerGraph, check::Int, vertex::Int, syndrome_bit::Int, messages_v2c::Matrix{Float64})
	"""
	Compute the message sent from a check node to a variable node in a Tanner graph.
	The check node sends a message to the variable node based on the messages it received from all other connected variable nodes.
	The message indicates the likelihood of the variable node being in a certain state (0 or 1) given the parity-check constraint.
	m_{c->v} = 2 * atanh(∏_{v' ∈ N(c) - {v}} tanh(m_{v'->c}/2))
	"""

	# Compute the product of the tanh messages from the neighboring variable nodes
	product = 1.0
	for incident_vertex in G.check_neighbors[check]
        if (incident_vertex != vertex)
		    product *= tanh(messages_v2c[incident_vertex, check] / 2)
        end
	end

	# Compute the message from the check node to the variable node
    message = 2 * atanh(product)
    if syndrome_bit == 1
        message = -message
    end
	return message
end

function get_message_from_vertex_to_check(G::TannerGraph, vertex::Int, check::Int, initial_llr::Vector{Float64}, messages_c2v::Matrix{Float64})
	"""
	Compute the message sent from a variable node to a check node in a Tanner graph.
	The variable node sends a message to the check node based on the received log-likelihood ratio (LLR) and the messages it received from all other connected check nodes.
	The message indicates the likelihood of the variable node being in a certain state (0 or 1) given the information from the channel and other checks.
	m_{v->c} = LLR(v) + ∑_{c' ∈ N(v) - {c}} m_{c'->v}
	"""
	# Compute the sum of the messages from the neighboring check nodes
	sum_messages = 0.0
	for incident_check in G.vertex_neighbors[vertex]
        if (incident_check != check)
		    sum_messages += messages_c2v[incident_check, vertex]
        end
	end

	# Compute the message from the variable node to the check node
	return initial_llr[vertex] + sum_messages
end

function compute_llrs_from_probabilities(probabilities::Vector{Float64})::Vector{Float64}
    """
    Compute the log-likelihood ratios (LLRs) from the given probabilities.
    The LLR for a bit is defined as:
    LLR = log(P(bit=0) / P(bit=1))
    """
    llrs = zeros(Float64, length(probabilities))
    for i in 1:length(probabilities)
        p0 = probabilities[i]
        p1 = 1.0 - p0
        llrs[i] = log(p0 / p1)
    end
    return llrs
end

function compute_probabilities_from_llrs(llrs::Vector{Float64})::Vector{Float64}
    """
    Compute the probabilities from the given log-likelihood ratios (LLRs).
    The probability for a bit being 0 is defined as:
    P(bit=0) = 1 / (1 + exp(-LLR))
    """
    probabilities = zeros(Float64, length(llrs))
    for i in 1:length(llrs)
        probabilities[i] = 1.0 / (1.0 + exp(-llrs[i]))
    end
    return probabilities
end

function soft_message_from_check_to_vertex(G::TannerGraph, check::Int, vertex::Int, messages_v2c::Matrix{Float64}, weight_soft_constraint::Float64)
    """
    Pass a message from a check node representing a soft constraint to a variable node in the Tanner graph.
    The difference between a soft constraint and a hard constraint is that a soft constraint allows for some uncertainty or probability in the satisfaction of the constraint.
    The soft constraint in this case is a two-bit parity-check constraint whose satisfaction enforces correlation between the two bits. This is useful for decoding correlated errors.
    
    Suppose the check is

    f_k = x_i + x_j (mod 2)

    where x_i and x_j are the two variable nodes connected to the check node f_k. The check node f_k enforces that the sum of x_i and x_j is even (0), in other words, x_i and x_j are correlated.

    The message from the check node to a variable node (say x_i) is:
    m_{f_k -> x_i} = log( (α + (1-α) exp(- m_{v_j -> f_k}) ) / (α exp(- m_{v_j -> f_k})) + (1 - α))

    For the sake of numerical stability, we can rewrite this as:
    m_{f_k -> x_i} = 2 atanh (tanh(J) * tanh(m_{v_j -> f_k}/2))
    where J = 1/2 ln(α / (1 - α)).

    where m_{v_j -> f_k} is the message from the other variable node (x_j) to the check node (f_k), and α is a parameter that controls the strength of the correlation. α = 0 is uncorrelated while α = 1 is fully correlated.

    The message from the vertex to the check node remains the same as in the hard constraint case.
    """
    neighbor = setdiff(G.check_neighbors[check], [vertex])[1]  # The other variable node connected to the check node
    ising_coupling = 0.5 * log(weight_soft_constraint / (1 - weight_soft_constraint))
    # Using the numerically stable form
    message_llr_from_check_to_vertex = 2 * atanh(tanh(ising_coupling) * tanh(messages_v2c[neighbor, check] / 2))
    return message_llr_from_check_to_vertex
end

function bp_round(G::TannerGraph, llrs_from_previous_round::Vector{Float64}, syndrome::Vector{Int}, messages_v2c::Matrix{Float64}, weight_soft_constraint::Float64)
    """
    Perform one round of belief propagation on the Tanner graph G with the given initial log-likelihood ratios (LLRs).
    """
    # Initialize messages
    # messages_v2c = zeros(Float64, G.nv, G.nc)  # Messages from variable nodes to check nodes
    messages_c2v = zeros(Float64, G.nc, G.nv)  # Messages from check nodes to variable nodes

    # Pass messages from check nodes to variable nodes
    for c in 1:G.nc
        if c < G.soft_constraint_start
            # Hard constraint
            for v in G.check_neighbors[c]
                messages_c2v[c, v] = get_message_from_check_to_vertex(G, c, v, syndrome[c], messages_v2c)
            end
        else
            # Soft constraint
            for v in G.check_neighbors[c]
                messages_c2v[c, v] = soft_message_from_check_to_vertex(G, c, v, messages_v2c, weight_soft_constraint)
            end
        end
    end

    # Pass messages from variable nodes to check nodes
    for v in 1:G.nv
        for c in G.vertex_neighbors[v]
            messages_v2c[v, c] = get_message_from_vertex_to_check(G, v, c, llrs_from_previous_round, messages_c2v)
        end
    end

    # Compute the updated LLRs for each variable node
    updated_llrs = zeros(Float64, G.nv)
    for v in 1:G.nv
        sum_messages = sum(messages_c2v[c, v] for c in G.vertex_neighbors[v])
        updated_llrs[v] = llrs_from_previous_round[v] + sum_messages
    end

    return (messages_v2c, updated_llrs)
end

function bp_initialize(G::TannerGraph, llrs::Vector{Float64})
    """
    Initialize the belief propagation algorithm by computing the initial log-likelihood ratios (LLRs)
    from the given probabilities and initializing the messages going from the vertices to the checks.
    The messages from the vertices to the checks are initialized to the initial LLRs.
    Note: nothing special is done for soft constraints at this stage; they are handled differently only during passing messages from checks to vertices.
    """
    messages_v2c = zeros(Float64, G.nv, G.nc)
    for v in 1:G.nv
        messages_v2c[v, G.vertex_neighbors[v]] .= llrs[v]
    end
    return messages_v2c
end

function get_recovery_using_hard_decision(llrs::Vector{Float64}; confidence::Float64=0.0)::Vector{Int}
    """
    Get the recovery vector using hard decision based on the log-likelihood ratios (LLRs).
    A bit is decided to be 0 if its LLR is greater than the confidence threshold, and 1 otherwise.
    """
    recovery = [llr >= confidence ? 0 : 1 for llr in llrs]
    return recovery
end

function should_bp_stop(G::TannerGraph, old_llrs::Vector{Float64}, new_llrs::Vector{Float64}, syndrome::Vector{Int}, llr_convergence_threshold::Float64, llr_confidence_threshold::Float64; verbose::Bool=false, io::IO=stdout)::Bool
    """
    Check if the belief propagation algorithm should be stopped.
    We will make a hard decision based on the LLRs and check if the decoded codeword satisfies all parity-check equations. This excludes the soft constraints added to handle correlations.
    If the maximum change in LLRs is below the threshold, we consider the algorithm to have converged.
    """
    stop = true
    # Make a hard decision based on the LLRs
    hard_decision_recovery = get_recovery_using_hard_decision(new_llrs; confidence=0.0)
    # Check if the decoded codeword satisfies all parity-check equations
    for c in 1:G.soft_constraint_start - 1
        parity_check = sum(hard_decision_recovery[v] for v in G.check_neighbors[c]) % 2
        if parity_check != syndrome[c]
            stop = false
            break
        end
    end
    if (stop == true)
        if verbose
            println(io, "Converged based on parity-check satisfaction.")
        end
        # Check if the LLRs have reached the threshold for confidence.
        if get_confident_bits(new_llrs, llr_confidence_threshold) == collect(1:G.nv)
            if verbose
                println(io, "Converged based on LLR confidence: All |LLR| >= ", llr_confidence_threshold)
            end
            stop = true
        else
            stop = false
        end
        # Check for convergence based on LLR changes
        max_change = maximum(abs.(new_llrs .- old_llrs))
        if max_change <= llr_convergence_threshold
            if verbose
                println(io, "Converged based on LLR change: ", max_change, " <= ", llr_convergence_threshold)
            end
            stop = true
        end
    end
    return stop
end

function get_confident_bits(llrs::Vector{Float64}, confidence_threshold::Float64)::Vector{Int}
    """
    Get the indices of bits that have LLRs exceeding the confidence threshold.
    """
    confident_bits = findall(abs.(llrs) .>= confidence_threshold)
    return confident_bits
end

function run_bp(parity_check_matrix::Matrix{Int}, soft_constraint_start::Int, syndrome::Vector{Int}, initial_llrs::Vector{Float64}, max_interations::Int; llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=2.0, weight_soft_constraint::Float64=0.5, verbose::Bool=false, io::IO=stdout)::Tuple{Vector{Float64}, Int}
    """
    Run the belief propagation algorithm on the given parity-check matrix and syndrome.
    Returns the final log-likelihood ratios (LLRs) after running BP.
    """
    # Remove any zeros rows from the parity-check matrix and corresponding syndrome bits
    non_zero_rows = findall(row -> any(x -> x != 0, row), eachrow(parity_check_matrix))
    parity_check_matrix_non_trivial = parity_check_matrix[non_zero_rows, :]
    syndrome_non_trivial = syndrome[non_zero_rows]
    if verbose
        println(io, "Running BP on parity-check matrix of size ", size(parity_check_matrix_non_trivial), " with syndrome ", syndrome_non_trivial)
        println(io, "--------------------------------------------------------")
        println(io, "BP Round 0:\nLLRs = ", initial_llrs)
        println(io, "--------------------------------------------------------")
    end
    # Create the Tanner graph from the parity-check matrix
    n_hard_constraints = count(row -> any(x -> x != 0, row), eachrow(parity_check_matrix[1:soft_constraint_start-1, :])) # We need to know where the hard constraints end, in the non-trivial parity check matrix.
    soft_constraint_start = n_hard_constraints + 1
    G = TannerGraph(parity_check_matrix_non_trivial, soft_constraint_start)
    # Initialize messages
    messages_v2c = bp_initialize(G, initial_llrs)
    old_llrs = copy(initial_llrs)
    current_iteration = 1
    stop = false
    while ((stop == false) && (current_iteration <= max_interations))
        (messages_v2c, new_llrs) = bp_round(G, old_llrs, syndrome_non_trivial, messages_v2c, weight_soft_constraint)
        if verbose
            println(io, "--------------------------------------------------------")
            println(io, "BP Round $(current_iteration):\nLLRs = ", new_llrs)
            println(io, "--------------------------------------------------------")
        end
        stop = should_bp_stop(G, old_llrs, new_llrs, syndrome, llr_convergence_threshold, llr_confidence_threshold; verbose=verbose, io=io)
        # Update old LLRs for the next iteration
        old_llrs = new_llrs
        # Increment the iteration counter
        current_iteration += 1
    end
    return (old_llrs, current_iteration-1)
end

function classical_belief_propagation_decoder(C::ClassicalCode, error::Vector{Int}, syndrome::Vector{Int}, initial_probabilities::Vector{Float64}, rounds_per_BP::Int, n_iterations_of_BP::Int; llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=2.0, weight_soft_constraint::Float64=0.5, verbose::Bool=false, io::IO=stdout)::BPSettings
    """
    We want to run BP iteratively, and after each iteration, we want to check if any bits have reached a certain confidence threshold in their LLRs.
    1. After each iteration of BP, we will identify the bits whose LLRs exceed the confidence threshold. These are considered "confident" bits, denoted by the set F. We will fix their values by making a hard decision.
    2. We can then permute the columns of the parity-check matrix to move the confident bits to the front. This helps in reducing the problem size for subsequent iterations.
    3. Suppose H = [H_F | H_U] where H_F corresponds to the confident bits and H_U to the uncertain bits. We will focus our BP updates on the submatrix H_U in the next iteration.
    4. We can now split the error as e = [e_F | e_U] where e_F are the errors on the confident bits (which we can fix using hard decision) and e_U are the errors on the uncertain bits.
    5. The syndrome can be updated accordingly: s = H * e^T = H_F * e_F^T + H_U * e_U^T. Since we have fixed e_F, we can compute the contribution of H_F * e_F^T and add it from the syndrome to get a new syndrome for the uncertain bits.
    6. We will repeat this until either all bits are confident or we reach the maximum number of iterations.
    
    Here is the pseudocode for the modified BP algorithm:
        1. Initialize the LLRs based on the initial probabilities: LLR_i = log(P(bit_i=0)/P(bit_i=1)).
        2. For i in 1 to n_iterations:
            a. F = { i | |LLR_i| >= llr_confidence_threshold } (the set of confident bits).
            b. U = {1...n} - F (the set of uncertain bits).
            c. H_F = H[:, F], H_U = H[:, U].
            d. Make a hard decision on the bits in F: recovery_F = {0 if LLR_i >= 0 else 1 for i in F}.
            e. Update the syndrome: s_U = syndrome - H_F * recovery_F^T (mod 2).
            f. Run one iteration of BP on the current parity-check matrix H_U and syndrome and obtain the LLRs.
            g. Update LLR_i for i in U based on the BP output.
        3. After exiting the loop, make a hard decision on the remaining bits in U.
    """
    parity_check_matrix = C.H
    start_time = time()
    n_iterations_total = 0
    llrs = compute_llrs_from_probabilities(initial_probabilities)
    for i in 1:n_iterations_of_BP
        frozen_bits = get_confident_bits(llrs, llr_confidence_threshold)
        
        if length(frozen_bits) == length(llrs)
            break # All bits are confident, we can stop
        end
        
        if length(frozen_bits) == 0
            # No confident bits, we cannot update the syndrome
            # We will run BP on the entire matrix yet again
            syndrome_contribution_frozen = zeros(Int, size(parity_check_matrix, 1))
            recovery_F = Int[]
        else
            H_F = parity_check_matrix[:, frozen_bits]
            recovery_F = get_recovery_using_hard_decision(llrs[frozen_bits]; confidence=0.0)
            syndrome_contribution_frozen = mod.(H_F * recovery_F, 2)
        end
        # println("syndrome_contribution_frozen: ", syndrome_contribution_frozen)

        uncertain_bits = setdiff(1:length(llrs), frozen_bits)
        H_U = parity_check_matrix[:, uncertain_bits]
        s_U = mod.(syndrome .+ syndrome_contribution_frozen, 2)
        
        #=
        println("Running BP on uncertain bits (U): ", uncertain_bits)
        println("LLRs for uncertain bits (U): ", llrs[uncertain_bits])
        println("Syndrome for uncertain bits (U): ", s_U)
        println("Parity-check matrix for uncertain bits (H_U):\n", H_U)
        =#

        if verbose
            println(io, "================ Batch $(i) of running BP on $(length(uncertain_bits)) bits ================")
            # println(io, "LLRs: ", llrs)
            # println(io, "------------------------------------------")
        end

        (new_llrs_U, current_iteration) = run_bp(
            H_U, # Parity-check matrix for uncertain bits
            C.r + 1, # The index where the soft constraints start in H_U
            s_U, # Syndrome for uncertain bits
            llrs[uncertain_bits], # Initial LLRs for uncertain bits
            rounds_per_BP; # Number of rounds of BP to run
            llr_convergence_threshold=llr_convergence_threshold,
            llr_confidence_threshold=llr_confidence_threshold,
            weight_soft_constraint=weight_soft_constraint, # Weight for the soft constraint
            verbose=verbose,
            io=io
        )
        llrs[uncertain_bits] .= new_llrs_U

        # println("Updated LLRs for uncertain bits (U): ", llrs[uncertain_bits])

        if (verbose)
            println(io, "Confident bits (F): ", frozen_bits)
            println(io, "Recovery on confident bits: ", recovery_F)
            println(io, "Uncertain bits (U): ", setdiff(1:length(llrs), frozen_bits))
            println(io, "Syndrome for uncertain bits: ", s_U)
        end

        n_iterations_total += current_iteration - 1
    end

    final_llrs = deepcopy(llrs)
    final_probabilities = compute_probabilities_from_llrs(final_llrs)
    runtime = time() - start_time
    if verbose
        println(io, "Final probabilities after $(n_iterations_total) iterations is ", final_probabilities)
    end
    # println("Loading all properties into BPSettings struct... in $(rounds_per_BP * n_iterations_of_BP) rounds of BP")
    # Load all the properties into the BPSettings struct
    bpset = BPSettings(n_iterations_of_BP, rounds_per_BP, llr_convergence_threshold, llr_confidence_threshold, weight_soft_constraint, n_iterations_total, false, error, syndrome, initial_probabilities, final_probabilities, verbose, runtime)
    bpset.recovery_hard_decision = get_recovery_using_hard_decision(final_llrs; confidence=0.0)
    bpset.is_decoder_failure = is_decoder_failure(error, bpset.recovery_hard_decision, C.L)
    return bpset
end

function is_decoder_failure(error::Vector{Int}, recovery::Vector{Int}, logical_operators::Matrix{Int})::Bool
    """
    Check if there is a logical error after applying the recovery to the error.
    A logical error occurs if the combined error (error + recovery) anti-commutes with any of the logical operators.
    """
    residual_error = mod.(error .+ recovery, 2)
    conjugate_components = mod.(logical_operators * residual_error, 2)
    # If any of the conjugate components is 1, there is a logical error
    if any(conjugate_components .== 1)
        return true  # Logical error detected
    end
    return false  # No logical error
end

function quantum_belief_propagation_decoder(Q::QuantumCode, error::Vector{Int}, initial_probabilities::Vector{Float64}, rounds_per_BP::Int, n_iterations_of_BP::Int; llr_convergence_threshold::Float64=1e-6, llr_confidence_threshold::Float64=2.0, weight_soft_constraint::Float64=0.5, verbose::Bool=false, io::IO=stdout)::BPSettings
    """
    Run the belief propagation for a quantum CSS code.
    We will essentially run the classical BP on both the X and Z components separately.
    """
    start_time = time()

    (error_X, error_Z) = separate_error_components(error)
    (syndrome_X, syndrome_Z) = measure_syndrome_quantum_code(Q, error)

    syndrome = vcat(syndrome_X, syndrome_Z)

    # println("Shape of the Parity Check Matrix for X part: ", size(Q.CX.H))
    # println("Shape of the syndrome for the X part: ", size(syndrome_X))

    # Run the classical BP decoder on the X part
    if (verbose)
        println(io, "-----------------------------------------")
        println(io, "Error Z: ", error_Z)
        println(io, "Syndrome X: ", syndrome_X)
        println(io, "Decoding X part:")
    end
    bpset_X = classical_belief_propagation_decoder(Q.CX, error_Z, syndrome_X, initial_probabilities, rounds_per_BP, n_iterations_of_BP; llr_convergence_threshold=llr_convergence_threshold, llr_confidence_threshold=llr_confidence_threshold, weight_soft_constraint=weight_soft_constraint, verbose=verbose, io=io)
    if (verbose)
        print_bp_settings(bpset_X; io=io)
    end
    
    # Run the classical BP decoder on the Z part
    if (verbose)
        println(io, "-----------------------------------------")
        println(io, "Error X: ", error_X)
        println(io, "Syndrome Z: ", syndrome_Z)
        println(io, "Decoding Z part:")
    end
    bpset_Z = classical_belief_propagation_decoder(Q.CZ, error_X, syndrome_Z, initial_probabilities, rounds_per_BP, n_iterations_of_BP; llr_convergence_threshold=llr_convergence_threshold, llr_confidence_threshold=llr_confidence_threshold, weight_soft_constraint=weight_soft_constraint, verbose=verbose, io=io)
    if (verbose)
        print_bp_settings(bpset_Z; io=io)
    end
    
    runtime = time() - start_time
    
    # Load all the properties into the BPSettings struct
    bpset = BPSettings(
        rounds_per_BP,
        n_iterations_of_BP,
        llr_convergence_threshold,
        llr_confidence_threshold,
        weight_soft_constraint,
        bpset_X.current_iteration + bpset_Z.current_iteration,
        false,
        error,
        syndrome,
        initial_probabilities,
        zeros(length(initial_probabilities)),
        verbose,
        runtime
    )
    bpset.converged = bpset_X.converged && bpset_Z.converged
    bpset.recovery_hard_decision = join_error_components(bpset_X.recovery_hard_decision, bpset_Z.recovery_hard_decision)
    bpset.is_decoder_failure = bpset_X.is_decoder_failure || bpset_Z.is_decoder_failure

    if (verbose)
        if (bpset.is_decoder_failure)
            println(io, "Decoder failed to correct the error.")
        else
            println(io, "Decoder successfully corrected the error.")
        end
        println(io, "-----------------------------------------")
    end

    return bpset
end
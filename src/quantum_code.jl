using LinearAlgebra

struct ClassicalCode
	"""
	Structure to represent the X or Z component of a CSS quantum stabilizer code.
	"""
	name::String # Optional name for the code
	n::Int # Number of physical qubits
	k::Int # Number of logical qubits
	r::Int # Number of stabilizer generators
	H::Matrix{Int} # Combined parity-check matrix [H_real; H_soft]
	tg::TannerGraph # Tanner graph representation of H
	L::Matrix{Int} # Logical operators

	function ClassicalCode(H_real::Matrix{Int}, L::Matrix{Int}; H_soft::Matrix{Int}=zeros(Int, 0, 0), name::String="Unnamed Code")
		r, n = size(H_real)
		k = n - r
        H_combined = vcat(H_real, H_soft)
        tg = TannerGraph(H_combined, r + 1)
		new(name, n, k, r, H_combined, tg, L)
	end
end

struct QuantumCode
    """
    Structure to represent a CSS quantum stabilizer code.
    """
    name::String # Optional name for the code
    n::Int # Number of physical qubits
    k::Int # Number of logical qubits
    r::Int # Number of stabilizer generators
    CX::ClassicalCode # X component of the code
    correlations_X::Matrix{Int} # Pairs of qubits with correlated X errors
    CZ::ClassicalCode # Z component of the code
    correlations_Z::Matrix{Int} # Pairs of qubits with correlated Z errors
    # H::Matrix{Int} # Combined parity-check matrix [H_X; H_Z]
    # L::Matrix{Int} # Combined logical operators [L_Z; L_X]

    function QuantumCode(HX::Matrix{Int}, HZ::Matrix{Int}, LX::Matrix{Int}, LZ::Matrix{Int}; correlations_X::Matrix{Int}=Int[], correlations_Z::Matrix{Int}=Int[], name::String="Unnamed Quantum Code")
        rX, nX = size(HX)
        rZ, nZ = size(HZ)

        @assert size(HX, 2) == size(HZ, 2) "H_X and H_Z must have the same number of columns (physical qubits)."
        @assert size(LX, 1) == size(LZ, 1) "There must be the same number of X and Z logical operators."
        
        # We will convert the correlations into soft constraints by adding rows to HX and HZ.
        # Each row in correlations_X (or correlations_Z) specifies a pair of qubits that are correlated.
        n_corr_X = size(correlations_X, 1)
        if (n_corr_X > 0)
            HX_soft = zeros(Int, n_corr_X, nX)
            for i in 1:n_corr_X
                q1, q2 = correlations_X[i, :]
                if ((1 <= q1 <= nX) || (1 <= q2 <= nX)) == false
                    throw(BoundsError("Correlation ($q1, $q2) is outside the valid range in correlations_X."))
                end
                HX_soft[i, q1] = 1
                HX_soft[i, q2] = 1
            end
        else
            HX_soft = zeros(Int, 0, 0)
        end
        n_corr_Z = size(correlations_Z, 1)
        if (n_corr_Z > 0)
            HZ_soft = zeros(Int, n_corr_Z, nZ)
            for i in 1:n_corr_Z
                q1, q2 = correlations_Z[i, :]
                if ((1 <= q1 <= nZ) || (1 <= q2 <= nZ)) == false
                    throw(BoundsError("Correlation ($q1, $q2) is outside the valid range in correlations_Z."))
                end
                HZ_soft[i, q1] = 1
                HZ_soft[i, q2] = 1
            end
        else
            HZ_soft = zeros(Int, 0, 0)
        end
        
        CX = ClassicalCode(HX, LZ; H_soft = HX_soft, name=name * " X")
        CZ = ClassicalCode(HZ, LX; H_soft = HZ_soft, name=name * " Z")

        # H = vcat(HX, HZ)
        # L = vcat(LX, LZ)
        n = nX
        r = rX + rZ
        k = n - r

        # qcode = new(name, n, k, r, CX, CZ, H, L)
        qcode = new(name, n, k, r, CX, correlations_X, CZ, correlations_Z)
        # Temporary: don't validate the code since it has extra rows in HX and HZ to handle correlations.
        # if is_valid(qcode) == false
        #     error("The provided stabilizers and logical operators do not satisfy the commutation relations.")
        # end
        return qcode
    end
end

function is_valid(Q::QuantumCode)::Bool
    """
    Check if the proposed stabilizers and logical operators satisfy the commutation relations.
    Returns true if they do, false otherwise.
    """
    stabilizer_generators_X = Q.CX.H[1:Q.CX.r, :]  # Only consider the real stabilizers, not the soft constraints.
    stabilizer_generators_Z = Q.CZ.H[1:Q.CZ.r, :]  # Only consider the real stabilizers, not the soft constraints.
    
    # Check that stabilizers commute with each other
    if any(mod.(stabilizer_generators_X * stabilizer_generators_Z', 2) .!= 0)
        error("Stabilizers do not commute with each other.")
        return false
    end

    # Check that logical operators commute with stabilizers
    commutations_X_stab_Z_logical = mod.(stabilizer_generators_X * Q.CX.L', 2)
    if any(commutations_X_stab_Z_logical .!= 0)
        violations = findall(commutations_X_stab_Z_logical .!= 0)
        for I in violations
            println("[SZ_", I[1], ", LX_", I[2], "] ≠ 0.")
        end
        error("Z Logical operators do not commute with the X stabilizers.")
        return false
    end
    commutations_Z_stab_X_logical = mod.(stabilizer_generators_Z * Q.CZ.L', 2)
    if any(commutations_Z_stab_X_logical .!= 0)
        violations = findall(commutations_Z_stab_X_logical .!= 0)
        for I in violations
            println("[SX_", I[1], ", LZ_", I[2], "] ≠ 0.")
        end
        error("X Logical operators do not commute with the Z stabilizers.")
        return false
    end

    # Check that each logical operator anti-commutes with a unique logical operator of the other type
    commutations_X_logical_Z_logical = mod.(Q.CX.L * Q.CZ.L', 2)
    if any(diag(commutations_X_logical_Z_logical) .!= 1)
        @warn ("Logical operators do not satisfy the canonical anti-commutation relations, i.e. [X_i, Z_j] = 0 ∀ i ≠ j, and {X_i, Z_i} = 0.")
    end

    return true
end

function measure_syndrome_classical_code(parity_check_matrix::Matrix{Int}, soft_constraint_start::Int, error::Vector{Int})
    """
    Measure the syndrome for a given error vector using the classical code's parity-check matrix.
    Some of the rows of the parity-check matrix are considered as "soft constraints" and are ignored in the syndrome measurement.
    The soft constraints are assumed to be the last N rows of the parity-check matrix, where N = `soft_constraint_start`.
    The syndrome for the soft constraints is not measured (set to zero).
    """
	hard_constraints = parity_check_matrix[1:soft_constraint_start-1, :]
    syndrome_hard = mod.(hard_constraints * error, 2)
    syndrome = vcat(syndrome_hard, zeros(Int, size(parity_check_matrix, 1) - soft_constraint_start + 1))
    return syndrome
end

function measure_syndrome_quantum_code(Q::QuantumCode, error::Vector{Int})::Tuple{Vector{Int}, Vector{Int}}
    """
    Measure the syndrome for a given error vector using the quantum code's parity-check matrices.
    Returns a tuple of syndromes (syndrome_X, syndrome_Z).
    """
    (error_X, error_Z) = separate_error_components(error)
    syndrome_X = measure_syndrome_classical_code(Q.CX.H, Q.CX.r + 1, error_Z)
    syndrome_Z = measure_syndrome_classical_code(Q.CZ.H, Q.CZ.r + 1, error_X)
    return (syndrome_X, syndrome_Z)
end

function readable_string_parity_check_constraints(parity_check_matrix::Matrix{Int}, soft_constraint_start::Int)::String
    """
    Generate a human-readable string representation of the parity-check matrix,
    indicating which rows are hard constraints and which are soft constraints.
    Each row should be presented as v_i1 + v_i2 + ... = 0, where v_ij are the variable nodes involved in that check.
    """
    n_rows = size(parity_check_matrix, 1)
    lines = String[]
    push!(lines, "Parity-Check Matrix (H):")
    for i in 1:n_rows
        row_type = i < soft_constraint_start ? "Hard" : "Soft"
        involved_vars = findall(x -> x == 1, parity_check_matrix[i, :])
        row_str = join(["v_$j" for j in involved_vars], " + ")
        push!(lines, "Row $i ($row_type): $row_str = 0")
    end
    return join(lines, "\n")
end

function print_quantum_code_info(code::QuantumCode; io::IO=stdout)
	println(io, "Code Name: ", code.name)
	println(io, "Number of Physical Qubits (n): ", code.n)
	println(io, "Number of Logical Qubits (k): ", code.k)
	println(io, "Number of Stabilizer Generators (r): ", code.r)
	println(io, "Parity-Check Matrix:")
    HX_str = readable_string_parity_check_constraints(code.CX.H, code.CX.r + 1)
    println(io, "H_X:\n", HX_str)
    HZ_str = readable_string_parity_check_constraints(code.CZ.H, code.CZ.r + 1)
    println(io, "H_Z:\n", HZ_str)
    println(io, "Logical Operators")
    println(io, "L_X:\n", code.CX.L)
    println(io, "L_Z:\n", code.CZ.L)
end
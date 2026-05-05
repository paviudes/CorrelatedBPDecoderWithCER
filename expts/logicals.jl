using LinearAlgebra

function row_reduce_mod2(A::Matrix{Int})::Matrix{Int}
    """
    Perform row reduction of the binary matrix A modulo 2.
    This is a standard Gaussian elimination process, but all operations are done modulo 2.
    The output is the row-reduced echelon form of A.
    """
    (nrows, ncols) = size(A)
    A_reduced = copy(A)
    row = 1
    for col in 1:ncols
        if row > nrows
            break
        end
        # Find the pivot in the current column
        pivot_row = findfirst(r -> A_reduced[r, col] == 1, row:nrows)
        if pivot_row === nothing
            continue
        end
        # Swap the current row with the pivot row
        A_reduced[[row, pivot_row], :] = A_reduced[[pivot_row, row], :]
        # Eliminate below
        for r in (row + 1):nrows
            if A_reduced[r, col] == 1
                A_reduced[r, :] .= (A_reduced[r, :] .+ A_reduced[row, :]) .% 2
            end
        end
        row += 1
    end

    # Remove zero rows at the bottom
    nonzero_rows = findall(r -> any(A_reduced[r, :] .== 1), 1:nrows)
    A_reduced = A_reduced[nonzero_rows, :]
    
    return A_reduced
end

function nullspace_mod2(A::Matrix{Int})::Matrix{Int}
    """
    Compute the nullspace of the binary matrix A modulo 2.
    This can be done by performing row reduction on A and then finding the basis for the nullspace.
    The output is a matrix whose rows form a basis for the nullspace of A.
    """
    ncols = size(A, 2)
    A_reduced = row_reduce_mod2(A)
    
    # Identify pivot columns
    pivot_cols = findall(c -> any(A_reduced[:, c] .== 1), 1:ncols)
    free_cols = setdiff(1:ncols, pivot_cols)

    # Construct the nullspace basis
    nullspace_basis = []
    for free_col in free_cols
        basis_vector = zeros(Int, ncols)
        basis_vector[free_col] = 1
        for (i, pivot_col) in enumerate(pivot_cols)
            if A_reduced[i, free_col] == 1
                basis_vector[pivot_col] = 1
            end
        end
        push!(nullspace_basis, basis_vector)
    end
    
    return hcat(nullspace_basis...)
end

function compute_logical_operators(stabilizers_X::Matrix{Int}, stabilizers_Z::Matrix{Int})
    """
    Given the parity-check matrices HX and HZ of a CSS code, compute the logical operators of the code.
    The logical operators can be found by computing the nullspace of the combined parity-check matrix [HX; HZ] and then identifying which of those operators commute with all stabilizers but are not themselves in the stabilizer group.
    
    Given the Null spaces:
    NX := nullspace_mod2(HX)
    NZ := nullspace_mod2(HZ)

    we know that
    NX = span(HZ | LZ) where LZ are the logical Z operators,
    NZ = span(HX | LX) where LX are the logical X operators.

    To isolate the logical operators LZ from NX, we should find the vectors in NX that are not in the span of HZ.
    This can be done by adding each row of NX to HZ and checking if its rank increases. If it does, then that row corresponds to a logical operator.

    Similarly, to isolate the logical operators LX from NZ, we should find the vectors in NZ that are not in the span of HX.
    """

    # Compute the nullspace of HX to get all operators that commute with the X-stabilizers, i.e. NZ = span(HZ | LZ)
    normalizers_Z = nullspace_mod2(stabilizers_X)
    nb_normalizers_Z = size(normalizers_Z, 1)

    # Isolate the logical Z operators from the normalizers by checking which ones are not in the span of HZ.
    logical_Z_operators = []
    for i in 1:nb_normalizers_Z
        candidate = normalizers_Z[i, :]
        # Check if candidate is in the span of HZ by checking if adding it to HZ increases the rank.
        augmented_matrix = vcat(stabilizers_Z, candidate')
        if rank(augmented_matrix) > rank(stabilizers_Z)
            push!(logical_Z_operators, candidate)
        end
    end
    logical_Z_operators = hcat(logical_Z_operators...)
    

    # Compute the nullspace of HZ to get all operators that commute with the Z-stabilizers, i.e. NX = span(HX | LX)
    normalizers_X = nullspace_mod2(stabilizers_Z)
    nb_normalizers_X = size(normalizers_X, 1)
    # Isolate the logical X operators from the normalizers by checking which ones are not in the span of HX.
    logical_X_operators = []
    for i in 1:nb_normalizers_X
        candidate = normalizers_X[i, :]
        # Check if candidate is in the span of HX by checking if adding it to HX increases the rank.
        augmented_matrix = vcat(stabilizers_X, candidate')
        if rank(augmented_matrix) > rank(stabilizers_X)
            push!(logical_X_operators, candidate)
        end
    end
    logical_X_operators = hcat(logical_X_operators...)

    return (logical_X_operators, logical_Z_operators)
end
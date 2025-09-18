using LinearAlgebra

function generate_Hamming_Parity_Check_Matrix(r::Int)::Matrix{Int}
    """
    Generate the parity-check matrix for a Hamming code with parameter r.
    The resulting code will have length n = 2^r - 1 and dimension k = n - r.
    The columns of the parity-check matrix are all non-zero binary vectors of length r.
    """
    n = 2^r - 1
    parity_check_matrix = zeros(Int, r, n)
    
    # Fill the parity-check matrix with binary representations of numbers from 1 to n
    for j in 1:n
        binary_repr = reverse(digits(j, base=2, pad=r))
        parity_check_matrix[:, j] = binary_repr
    end
    return parity_check_matrix
end
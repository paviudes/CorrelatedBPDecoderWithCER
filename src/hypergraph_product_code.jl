using LinearAlgebra

function get_hypergraph_product_code_H(H1::Matrix{Int}, H2::Matrix{Int})
    """
    Given two parity-check matrices H1 and H2, construct the hypergraph product code's parity-check matrix using Section 6 of https://arxiv.org/pdf/0903.0566.
    H_X = (H1 ⊗ I_n2, I_r1 ⊗ H2^T)
    H_Z = (I_n1 ⊗ H2, H1^T ⊗ I_r2)
    where
        - ⊗ is the Kronecker product
        - H1 is of size r1 x n1
        - H2 is of size r2 x n2
        - I_n is the identity matrix of size n
    """
    (r1, n1) = size(H1)
    (r2, n2) = size(H2)
    I_n1 = Matrix{Int}(I, n1, n1)
    I_r1 = Matrix{Int}(I, r1, r1)
    I_n2 = Matrix{Int}(I, n2, n2)
    I_r2 = Matrix{Int}(I, r2, r2)

    # Construct H_X and H_Z using Section 6 of the paper
    H_X = hcat(kron(H1, I_n2), kron(I_r1, H2'))
    H_Z = hcat(kron(I_n1, H2), kron(H1', I_r2))

    return (H_X, H_Z)
end
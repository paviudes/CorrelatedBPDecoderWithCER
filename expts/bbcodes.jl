using LinearAlgebra
using DelimitedFiles

# ----------------------------------------------------------------------------
# Bivariate Bicycle (BB) quantum LDPC codes from
#   Bravyi, Cross, Gambetta, Maslov, Rall, Yoder (2024),
#   "High-threshold and low-overhead fault-tolerant quantum memory."
#   https://arxiv.org/abs/2308.07915
#
# A BB code is parameterized by integers (l, m) and two polynomials
#   A(x, y), B(x, y) ∈ F_2[x, y] / (x^l - 1, y^m - 1)
# yielding a CSS code with parameters [[2lm, k, d]].
#
# Stabilizer parity-check matrices:
#   H_X = [A | B]      (lm rows × 2lm cols)
#   H_Z = [B^T | A^T]  (lm rows × 2lm cols)
# X-stabilizers act on qubits indexed 1..lm (left block) and lm+1..2lm (right
# block) according to the 1-entries of each row of H_X; similarly for Z.
# ----------------------------------------------------------------------------

struct BBCode
    HX::Matrix{Int}
    HZ::Matrix{Int}
    n::Int
    k::Int
    d::Int
    l::Int
    m::Int
    A_monomials::Vector{Tuple{Int,Int}}
    B_monomials::Vector{Tuple{Int,Int}}
    function BBCode(HX, HZ, n, k, d, l, m, A_monomials, B_monomials)
        new(HX, HZ, n, k, d, l, m, A_monomials, B_monomials)
    end
end

# ============================================================================
# BB code construction.
# ============================================================================

function _cyclic_shift(n::Int)::Matrix{Int}
    """
    Build the n × n cyclic shift matrix S, given by
    S[i, j] = 1 if j ≡ i + 1 (mod n)
            = 0 otherwise.
    """
    cyclic = zeros(Int, n, n)
    @inbounds for i in 1:n
        cyclic[i, mod1(i + 1, n)] = 1
    end
    return cyclic
end

function _polynomial_matrix(l::Int, m::Int, monomials::Vector{Tuple{Int,Int}})::Matrix{Int}
    """
    Build the l m × l m matrix corresponding to a bivariate polynomial A(x, y) = ∑_(a,b) x^a y^b.
    x and y are defined as
    x = S_l ⊗ I_m
    y = I_l ⊗ S_m
    where S_n is the n × n cyclic shift matrix given by `_cyclic_shift(n)`.
    Hence:
    x^a y^b = (S_l^a ⊗ I_m) * (I_l ⊗ S_m^b).
            = S_l^a ⊗ S_m^b
    """
    Sl = _cyclic_shift(l)
    Sm = _cyclic_shift(m)
    Il = Matrix{Int}(I, l, l)
    Im = Matrix{Int}(I, m, m)

    # Compute the matrix corresponding to the polynomial A(x, y) = ∑_(a,b) x^a y^b.
    n_qubits = l * m
    poly_matrix = zeros(Int, n_qubits, n_qubits)
    for (a, b) in monomials
        Sla = a == 0 ? Il : Sl^a # matrix power.
        Smb = b == 0 ? Im : Sm^b # matrix power.
        monomial_term = kron(Sla, Smb) # Kronecker product.
        poly_matrix = poly_matrix .+ monomial_term
    end
    
    # Since we are working over F_2, we take the result modulo 2.
    binary_mat = mod.(poly_matrix, 2)
    
    return binary_mat
end

function bb_code(
    l::Int, m::Int,
    A_monomials::Vector{Tuple{Int,Int}},
    B_monomials::Vector{Tuple{Int,Int}};
    distance::Int = -1 # optional distance parameter, since computing the distance is expensive. If not provided, it will be set to -1.
)
    """
    Construct a Bicycle Bivariate (BB) code given by parameters (l, m) and two bivariate polynomials A and B, following the prescription in Section 4 of https://arxiv.org/abs/2308.07915.

    The code is constructed as follows:
    1. Construct a cycling shift matrix S_l of size l × l and S_m of size m × m.
    2. Construct the matrices: x = S_l ⊗ I_m and y = I_l ⊗ S_m.
    3. Construct the matrices A and B corresponding to the input polynomials, as A = ∑_(a,b) x^a y^b and B = ∑_(a,b) x^a y^b.
    4. Construct the parity-check matrices H_X = [A | B] and H_Z = [B^T | A^T].

    The resulting [[n,k,d]] BB code has
    - n = 2 l m physical qubits
    - k = 2 l m - rank(H_X) - rank(H_Z) logical qubits
    - d inferred from Table 3 of https://arxiv.org/abs/2308.07915.

    We will return the parity-check matrices H_X and H_Z, and the [[n, k, d]] parameters of the code
    """
    A = _polynomial_matrix(l, m, A_monomials)
    B = _polynomial_matrix(l, m, B_monomials)

    HX = mod.(hcat(A,  B),  2)
    HZ = mod.(hcat(B', A'), 2)

    # Check commutation condition for CSS codes: H_X * H_Z^T = 0 (mod 2)
    css_condition_ok = mod.(HX * HZ', 2) .== 0
    # Identify the non-zero elements and determine which stabilizer generators anti-commute, if any.
    if all(css_condition_ok)
        println("CSS condition satisfied: H_X * H_Z^T = 0 (mod 2)")
    else
        anticommuting_pairs = findall(!, css_condition_ok)
        for (i, j) in anticommuting_pairs
            println("Stabilizer generator $(i) of H_X anti-commutes with stabilizer generator $(j) of H_Z.")
        end
        error("CSS condition not satisfied: H_X * H_Z^T ≠ 0 (mod 2).")
    end

    n_qubits = 2 * l * m
    k_logical = n_qubits - size(HX, 1) - size(HZ, 1)

    # Dictionary with the code details.
    code = BBCode(HX, HZ, n_qubits, k_logical, distance, l, m, A_monomials, B_monomials)
    return code
end

function save_bb_code(bbc::BBCode; prefix::String="./../code")
    """
    Save a BB code's two matrices to plain-text files:
    <prefix>_HX.txt
    <prefix>_HZ.txt
    We will save these files in <prefix>/bb_code/, so the full paths will be `<prefix>/bb_code/<prefix>_HX.txt` and `<prefix>/bb_code/<prefix>_HZ.txt`.
    We will also write the
    - code parameters (n, k, d) in a file `<prefix>/bb_code/parameters.txt` for reference.
    - code hyperparameters (l, m, A_monomials, B_monomials) in a file `<prefix>/bb_code/hyperparameters.txt` for reference.
    """
    if !isdir("$(prefix)")
        mkdir("$(prefix)")
    end
    writedlm("$(prefix)/HX.txt", bbc.HX, ' ')
    writedlm("$(prefix)/HZ.txt", bbc.HZ, ' ')
    open("$(prefix)/parameters.txt", "w") do io
        println(io, "n: ", bbc.n)
        println(io, "k: ", bbc.k)
        println(io, "d: ", bbc.d)
    end
    open("$(prefix)/hyperparameters.txt", "w") do io
        println(io, "l: ", bbc.l)
        println(io, "m: ", bbc.m)
        # Store the monomials in a human-readable format, e.g. "x^3 y^0" for the monomial (3, 0).
        println(io, "A(x, y): ", join(["x^$(a) y^$(b)" for (a, b) in bbc.A_monomials], ", "))
        println(io, "B(x, y): ", join(["x^$(a) y^$(b)" for (a, b) in bbc.B_monomials], ", "))
    end
    println("BB code saved to $(prefix)/")
    return nothing
end

#============================================================================
Summary of code parameters and constructions from Table 3 of https://arxiv.org/abs/2308.07915.
Columns:
- [n, k, d] : Code parameters
- r         : Net encoding rate
- (l, m)    : Construction parameters
- A, B      : Polynomial definitions

| [n, k, d]        | r     | (l, m) | A                  | B                      |
|------------------|-------|--------|--------------------|------------------------|
| [72, 12, 6]      | 1/12  | (6, 6) | x^3 + y + y^2      | y^3 + x + x^2          |
| [90, 8, 10]      | 1/23  | (15, 3)| x^9 + y + y^2      | 1 + x^2 + x^7          |
| [108, 8, 10]     | 1/27  | (9, 6) | x^3 + y + y^2      | y^3 + x + x^2          |
| [144, 12, 12]    | 1/24  | (12, 6)| x^3 + y + y^2      | y^3 + x + x^2          |
| [288, 12, 18]    | 1/48  | (12,12)| x^3 + y^2 + y^7    | y^3 + x + x^2          |
| [360, 12, ≤24]   | 1/60  | (30, 6)| x^9 + y + y^2      | y^3 + x^25 + x^26      |
| [756, 16, ≤34]   | 1/95  | (21,18)| x^3 + y^10 + y^17  | y^5 + x^3 + x^19       |

Notes:
- Distances with ≤ indicate lower bounds.
- Polynomials A and B define the structure used in the construction.
=# # ============================================================================

function bb_code_72(;prefix::String="./../data/codes/72q_BB_code")
    """
    Construct the [[72, 12, 6]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]; distance=6)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_90(;prefix::String="./../data/codes/90q_BB_code")
    """
    Construct the [[90, 8, 10]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)]; distance=10)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_108(;prefix::String="./../data/codes/108q_BB_code")
    """
    Construct the [[108, 8, 10]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]; distance=10)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_144(;prefix::String="./../data/codes/144q_BB_code")
    """
    Construct the [[144, 12, 12]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]; distance=12)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_288(;prefix::String="./../data/codes/288q_BB_code")
    """
    Construct the [[288, 12, 18]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]; distance=18)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_360(;prefix::String="./../data/codes/360q_BB_code")
    """
    Construct the [[360, 12, ≤24]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(30, 6, [(9, 0), (0, 1), (0, 2)], [(0, 3), (25, 0), (26, 0)]; distance=24)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end

function bb_code_756(;prefix::String="./../data/codes/756q_BB_code")
    """
    Construct the [[756, 16, ≤34]] BB code from Table 3 of https://arxiv.org/abs/2308.07915.
    """
    bbc = bb_code(21, 18, [(3, 0), (0, 10), (0, 17)], [(0, 5), (3, 0), (19, 0)]; distance=34)
    save_bb_code(bbc; prefix=prefix)
    return bbc
end
module BBCodes

# ============================================================================
# BBCodes — Bivariate Bicycle (BB) quantum LDPC code construction
# ============================================================================
#
# Standalone sibling package to CorrelatedBPDecoderWithCER. Builds the
# stabilizer parity-check matrices (H_X, H_Z) and logical operators (L_X, L_Z)
# of BB codes from Bravyi, Cross, Gambetta, Maslov, Rall, Yoder (2024),
# "High-threshold and low-overhead fault-tolerant quantum memory."
# (arXiv:2308.07915), and writes them to disk.
#
# The module itself uses only LinearAlgebra + DelimitedFiles (both standard
# libraries) and has no dependency on the main decoder package. The project
# also lists ArgParse — used only by the expts/build_bb_code.jl CLI driver that
# runs under this project, not by the module below.
#
# Usage (REPL):
#
#     julia --project=BBCodes
#     julia> using BBCodes
#     julia> bbc = bb_code_72_12_6(; prefix="./my_codes/72q_BB_code")   # a Table-3 preset
#     julia> # …or a custom code from human-readable polynomials:
#     julia> A = parse_bb_polynomial("x^3 + y + y^2")
#     julia> B = parse_bb_polynomial("y^3 + x + x^2")
#     julia> save_bb_code(bb_code(6, 6, A, B; distance=6); prefix="./my_codes/custom")
#
# Standalone / shell driver: see expts/build_bb_code.sh, which writes a
# settings TOML (with Table 3 shown as comments), opens it in your editor, and
# runs expts/build_bb_code.jl against this package.
#
# First-time install:
#
#     cd CorrelatedBPDecoderWithCER
#     julia --project=BBCodes -e 'using Pkg; Pkg.instantiate()'
# ============================================================================

using LinearAlgebra
using DelimitedFiles

# Order matters: construction.jl calls compute_logical_operators (from
# logicals.jl) at runtime, so logicals.jl must be included first.
#
# NB: the construction file is named construction.jl rather than bbcodes.jl on
# purpose — the module file is BBCodes.jl, and on a case-insensitive filesystem
# (macOS APFS default) "BBCodes.jl" and "bbcodes.jl" collide as one file.
include("logicals.jl")
include("construction.jl")

# --- code construction & IO ---
export BBCode
export bb_code, save_bb_code, parse_bb_polynomial

# --- logical-operator utilities (also useful on their own) ---
export compute_logical_operators, nullspace_mod2, row_reduce_mod2

# --- Table-3 preset constructors + lookup helpers ---
export bb_code_72_12_6, bb_code_90_8_10, bb_code_108_8_10, bb_code_144_12_12,
       bb_code_288_12_18, bb_code_360_12_24, bb_code_756_16_34
export BB_CODE_TABLE, BB_CODE_CONSTRUCTORS, bb_code_by_label, print_bb_code_table

end # module

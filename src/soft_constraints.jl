# soft_constraints.jl
# ----------------------------------------------------------------------------
# Correlation-adapted check-to-variable messages ("enriched check node").
#
# The standard check node treats its variables as independent given the parity
# constraint. Here each check factor carries, beyond the parity, one log-weight
# per configuration x of its support, the TABLE T_c:
#
#     ψ_c(x) = 1[⊕ x = s_c] · exp( α · T_c(x) )
#
# and the message to v is the exact marginalisation of that factor against the
# incoming messages:
#
#     m_{c→v} = ln [ Z_v(0) / Z_v(1) ],
#     Z_v(e_v) = Σ_{ξ: e_v ⊕ ξ = s_c} Π_{u ≠ v} q_u(ξ_u) · exp(α T_c(e_v, ξ))
#
# with q_u(·) the probabilities encoded by the incoming LLR m_{u→c}. See
# refs/soft_check_nbp.tex for the derivation. Two tables are available
# (`enriched_table` in the hyperparameters TOML, `SoftCheckTables.table_kind`):
#
#   * "pairwise": T_c(x) = Σ_{(i,k) ⊂ N(c)} J_ik x_i x_k, the CER couplings as
#     Ising pair terms. At α = 0 (or J = 0) the rule reduces exactly to the tanh
#     rule 2·atanh[(-1)^{s_c} Π tanh(m/2)]. Measured on 2026-10-15 to misstate
#     the check's joint badly (CLAUDE.md, "The pairwise factor is mis-specified"):
#     the marginal rates already contain the pair events that e^J adds again.
#   * "bethe": T_c(x) = ln P_c(x) − Σ_u ln p_u(x_u), the check's true joint over
#     the product of its marginals, read from a table file built from data
#     (expts/misc/build_bethe_tables.py). With the marginals as incoming messages
#     the check's local joint is then P_c itself, and Π_c ψ_c over the checks is
#     the Bethe approximation of the error model over check regions. α = 1 is
#     exact at the first layer.
#
# Only the check-to-variable rule changes. The variable-to-check rule and the
# readout, which carry every learnable message weight, are untouched. The one
# new learnable quantity is the scalar α (`coupling_scale`), shared by every
# entry of the table; α = 1 is the Bayesian value of either table.
#
# Implementation (`message_check_to_vertex_enriched!`, one check and one
# sample at a time): the softplus parts of ln q_u are common to every
# configuration and cancel in the message, so with b(k) the bit vector of
# configuration k and m the d incoming raw LLRs,
#
#     LW(k)   = α T_c(k) − b(k)·m
#     S_v^e   = log-sum-exp over {k : parity(k) = s_c, b_v(k) = e} of LW(k)
#     m_{c→v} = S_v^0 − S_v^1 − m_v,  clamped to ±ENRICHED_MESSAGE_CAP.
#
# ONE enumeration of the 2^d configurations serves all d targets: each target's
# own factor is left in LW and divided out by the final subtraction. Each class
# is stabilised by its own maximum, because Float32 products of six
# probabilities times exp(α T) under/overflow long before the LLRs saturate.
#
# Its adjoint (`backward_message_check_to_vertex_enriched!`) is the derivative
# of a log-sum-exp, a softmax: with π_v^e the softmax of LW inside class (v, e)
# and a_v = ∂L/∂m_{c→v} (0 where the cap clamped),
#
#     ω(k)        = Σ_v a_v [π_v^0(k) − π_v^1(k)]
#     ∂L/∂m_u     = −Σ_k ω(k) b_u(k) − a_u          (exactly 0 for u = v: extrinsic)
#     ∂L/∂(αT)(k) = ω(k),   ∂L/∂α = Σ_k ω(k) T_c(k)
#
# Every factor is a convex weight or a bit, so |∂m_{c→v}/∂m_u| ≤ 1 and nothing
# overflows; the cost equals the forward's; nothing but the layer's incoming
# messages is needed from the tape. Prototyped against central finite
# differences (outputs/proto/enriched_table_node.py, 2026-10-19).
#
# Two implementations of the same arithmetic live here:
#   * `apply_enriched_checks!` / `backward_enriched_checks!`
#                                  CPU, allocation-free loops; the forward is
#                                  Enzyme-friendly (Enzyme is the oracle in the
#                                  tests), the adjoint is the training path.
#   * `apply_enriched_checks_gpu` / `backward_enriched_checks_gpu`
#                                  dense broadcasts, reductions and matmuls only,
#                                  so they run unchanged on Array, MtlArray and
#                                  CuArray (testing, and training under USE_GPU=1).
# All consume the same `SoftCheckTables`, built once from (H, connectivity, J)
# and, for the Bethe table, the table file.
# ----------------------------------------------------------------------------

const CHECK_NODE_TANH::Int = 0
const CHECK_NODE_ENRICHED::Int = 1

"Which log-weight table the enriched check node carries (`enriched_table` in the
hyperparameters TOML): the CER pair couplings as Ising terms, or the check's
true joint over the product of its marginals (the Bethe factor), from a file."
const ENRICHED_TABLE_PAIRWISE::String = "pairwise"
const ENRICHED_TABLE_BETHE::String = "bethe"

"Additive log-domain penalty that excludes a configuration from a log-sum-exp
class on the GPU path. Finite, so no ±Inf arithmetic anywhere: exp(-1e30 - m)
is exactly 0 in Float32 for any finite m, and a class can never be empty."
const EXCLUDED_CONFIGURATION_LOG_WEIGHT::Float32 = -1.0f30

"Cap on |m_{c→v}| from the enriched rule. Chosen to equal the saturation the
tanh path already has: `safe_atanh_exp_signed!` clips exp(magnitude) at
1 - eps(Float32), so |2 atanh(·)| never exceeds this value there either."
const ENRICHED_MESSAGE_CAP::Float32 = 2.0f0 * atanh(1.0f0 - eps(Float32))

# =============================================================================
#   α and its logit: keeping the coupling scale inside [0, 1] without a wall
# =============================================================================
# α is the ratio between the coupling strength the decoder USES and the one CER
# MEASURED, so both ends of [0, 1] mean something and neither may be crossed:
#
#   α < 0  INVERTS every coupling. J already carries the sign (128 of the 540
#          couplings here are negative), so a negative α turns each
#          measured-correlated pair into an anti-correlated one and vice versa.
#          That is not a temperature, it is a claim the measurement had the
#          wrong sign. Measured: one run learned α = -0.541 and returned 4952
#          failures against a 330 baseline.
#   α > 1  asserts the pairs are MORE strongly coupled than the experiment
#          found. The loopy-BP correction runs the other way — cycles overcount
#          evidence, so the couplings want DISCOUNTING, exactly as normalised
#          min-sum scales its messages down.
#
# So training learns an UNCONSTRAINED logit θ and the model uses
#
#     α = 1 / (1 + exp(-θ))
#
# rather than clamping α after each step. A projection would leave the gradient
# pushing into the wall and park θ exactly on it; the logistic has no wall at
# all — it just takes ever larger θ to move α further, so the boundary is
# approached and never reached.
#
# Adam absorbs most of the link's Jacobian: dα/dθ = α(1-α) ≈ 0.24 over the
# useful range, a near-constant rescaling, and Adam normalises by the gradient's
# running magnitude. What survives is the intended part — the shrinking step in
# α as either boundary is approached.
#
# The link lives HERE, above the kernel: `apply_enriched_checks!` still takes α
# itself, so `α = 0` remains expressible and the "α = 0 reproduces the tanh
# rule" test stays exact. Only the learnable parameter is reparameterised.

"How far inside (0, 1) a requested α is placed before taking its logit. α = 0
and α = 1 are logit ∓Inf, and an infinite parameter makes Enzyme's gradient NaN,
which would NaN-skip every batch. 1e-6 is far below any decodable difference —
the classical scan separates α = 0.42 from α = 0.6 — so the nudge is invisible
except that it keeps the parameter finite, and `logit_from_coupling_scale` makes
it silently. For an exact α = 0, use `check_node = \"tanh\"`, which is the same
rule and needs no couplings at all."
const COUPLING_SCALE_LINK_MARGIN::Float32 = 1.0f-6

function coupling_scale_from_logit(coupling_logit::Real)::Float32
    """
    α = 1 / (1 + exp(-θ)), the standard increasing logistic, mapping the whole
    real line onto (0, 1).

    Written out rather than calling `sigmoid` from utils.jl: that one is
    `1/(1 + exp(x))`, the DECREASING convention used for σ(μ) = P(error), and
    reusing it here would silently flip the sign of θ.

    In exact arithmetic α never reaches 1; in Float32 it rounds to exactly 1 at
    θ ≥ 16.64. That corner is benign, not a failure: α = 1 is the Bayesian value
    the pairwise prior asks for, and dα/dθ = α(1-α) rounds to 0 there too, so a
    runaway θ parks α at 1 and stops moving it rather than producing anything
    infinite. If a run ends with α = 1 exactly, read it as "training pushed α to
    the top of the range", not as a converged interior optimum.
    """
    alpha::Float32 = 1.0f0 / (1.0f0 + exp(-Float32(coupling_logit)))
    return alpha
end

function logit_from_coupling_scale(coupling_scale::Real)::Float32
    """
    θ = log(α / (1 - α)), the inverse link, for turning a human-specified α
    (`coupling_scale_init`, or the fixed α of a classical run) into the
    parameter that is actually stored and trained.

    TWO DIFFERENT THINGS are clamped here, and only one of them is a mistake:

      α ∈ {0, 1}         IN range — α = 1 is the Bayesian value and the default —
                         but not representable, since the logit is ∓Inf. Nudged
                         inside by `COUPLING_SCALE_LINK_MARGIN` and left SILENT:
                         1e-6 is orders of magnitude below any decodable
                         difference, and warning here would fire on every load of
                         a weights file that predates the coupling field.
      α < 0 or α > 1     OUT of range, and a statement about the physics that the
                         run cannot honour. Warned about every time.
    """
    requested::Float32 = Float32(coupling_scale)
    if requested < 0.0f0 || requested > 1.0f0
        @warn "logit_from_coupling_scale: α = $(requested) lies outside [0, 1] and was " *
              "clamped. α is the fraction of the MEASURED coupling strength the decoder " *
              "uses: below 0 it inverts every coupling, which negates the physics (J " *
              "already carries the sign of each correlation), and above 1 it claims more " *
              "coupling than CER found. For an exact α = 0 use check_node = \"tanh\"."
    end
    clamped::Float32 = clamp(requested, COUPLING_SCALE_LINK_MARGIN, 1.0f0 - COUPLING_SCALE_LINK_MARGIN)
    logit::Float32 = log(clamped / (1.0f0 - clamped))
    return logit
end

function check_node_code(check_node_name::AbstractString)::Int
    """
    Resolve the `check_node` hyperparameter string to its integer code, so an
    unknown name throws at configuration time rather than inside a forward pass.
    """
    normalised_name::String = lowercase(strip(String(check_node_name)))
    check_node_kind::Int = CHECK_NODE_TANH
    if normalised_name == "tanh"
        check_node_kind = CHECK_NODE_TANH
    elseif normalised_name == "enriched"
        check_node_kind = CHECK_NODE_ENRICHED
    else
        throw(ArgumentError(
            "Unknown check_node \"$(check_node_name)\". Supported: \"tanh\" " *
            "(standard rule) and \"enriched\" (couplings inside the check factor)."))
    end
    return check_node_kind
end

function check_node_name(check_node_kind::Int)::String
    """
    Inverse of `check_node_code`: the hyperparameter string for a code, used
    when a model file records which rule its weights were trained under.
    """
    check_node_label::String = "tanh"
    if check_node_kind == CHECK_NODE_ENRICHED
        check_node_label = "enriched"
    elseif check_node_kind != CHECK_NODE_TANH
        throw(ArgumentError("Unknown check node code $(check_node_kind)."))
    end
    return check_node_label
end

function enriched_table_kind(enriched_table_name::AbstractString)::String
    """
    Validate the `enriched_table` hyperparameter string and return it
    normalised, so an unknown name throws at configuration time.
    """
    normalised_name::String = lowercase(strip(String(enriched_table_name)))
    if normalised_name != ENRICHED_TABLE_PAIRWISE && normalised_name != ENRICHED_TABLE_BETHE
        throw(ArgumentError(
            "Unknown enriched_table \"$(enriched_table_name)\". Supported: " *
            "\"$(ENRICHED_TABLE_PAIRWISE)\" (the CER couplings as Ising pair terms) and " *
            "\"$(ENRICHED_TABLE_BETHE)\" (the check's joint over its marginals, from a table file)."))
    end
    return normalised_name
end

function bethe_table_path(cer_data_file::AbstractString)::String
    """
    The Bethe table file that belongs to a CER file: `bethe_table_<key>.txt`
    next to `correlated_weights_<key>.txt`, as expts/misc/build_bethe_tables.py
    writes it. The key is whatever follows `correlated_weights_` in the CER file's
    name, so the table of a test distribution sits next to its priors.
    """
    cer_file_name::String = basename(String(cer_data_file))
    cer_prefix::String = "correlated_weights_"
    if !startswith(cer_file_name, cer_prefix)
        throw(ArgumentError(
            "bethe_table_path: the CER file \"$(cer_data_file)\" is not named " *
            "correlated_weights_<key>.txt, so no Bethe table file name follows from it."))
    end
    table_file_name::String = "bethe_table_" * cer_file_name[length(cer_prefix) + 1:end]
    table_path::String = joinpath(dirname(String(cer_data_file)), table_file_name)
    return table_path
end

function read_bethe_log_weights(table_file::AbstractString)::Dict{Int, Vector{Float32}}
    """
    Read a Bethe table file: one line per check, `<check> : v_1 v_2 ... v_{2^d}`,
    with `<check>` the 1-based row of H and v_k the log-weight
    ln P_c(k) − Σ_u ln p_u(b_u(k)) of configuration k, in the order of
    `configuration_table` (configuration k has bit (k−1) >> (slot−1) & 1 in
    slot `slot`, slots in increasing qubit order). Blank lines and `#` comments
    are skipped. Every value must be finite.
    """
    if !isfile(table_file)
        throw(ArgumentError(
            "read_bethe_log_weights: no Bethe table file at \"$(table_file)\". Build it with " *
            "python3 misc/build_bethe_tables.py --workdir <codename> --key <key>."))
    end
    log_weights::Dict{Int, Vector{Float32}} = Dict{Int, Vector{Float32}}()
    for (line_number, raw_line) in enumerate(eachline(table_file))
        line::String = strip(raw_line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        separator::Union{UnitRange{Int}, Nothing} = findfirst(":", line)
        if separator === nothing
            throw(ArgumentError(
                "read_bethe_log_weights: line $(line_number) of \"$(table_file)\" has no ':'."))
        end
        check::Int = parse(Int, strip(line[1:first(separator) - 1]))
        values::Vector{Float32} = [parse(Float32, token) for token in split(strip(line[last(separator) + 1:end]))]
        if !all(isfinite, values)
            throw(ArgumentError(
                "read_bethe_log_weights: check $(check) in \"$(table_file)\" has a non-finite log-weight."))
        end
        if haskey(log_weights, check)
            throw(ArgumentError(
                "read_bethe_log_weights: check $(check) appears twice in \"$(table_file)\"."))
        end
        log_weights[check] = values
    end
    return log_weights
end

struct SoftCheckTables
    """
    Everything the enriched check-node kernels need, precomputed once.

    A check is ENRICHED when at least one CER pair lies inside its support.
    Every CER pair is assigned to exactly one check (the first, in row order,
    whose support contains both endpoints), so no coupling is counted twice.
    Pairs whose endpoints share no check cannot be represented by a check
    factor at all; they are counted in `n_pairs_unassigned` and dropped, with a
    warning at build time.

    Fields:
    - `n_enriched_checks`, `check_degree` (d), `n_configurations` (2^d).
      All enriched checks must have the same degree; the dense GPU formulation
      relies on it.
    - `enriched_check_ids`: row of H for each enriched check.
    - `edge_neurons` (d × n_enriched): neuron index of each slot of each
      enriched check, slots in increasing variable order.
    - `configuration_bits` (2^d × d): bit of slot j in configuration k, as
      0/1 Float32 so the kernels can use it arithmetically.
    - `configuration_parity` (2^d): parity of configuration k, 0/1 Float32.
    - `table_kind`: ENRICHED_TABLE_PAIRWISE or ENRICHED_TABLE_BETHE, which of
      the two log-weight tables below is loaded.
    - `log_weight_table` (2^d × n_enriched): T_c(k), the UNSCALED log-weight of
      configuration k; α multiplies it at run time. For the pairwise table it is
      the coupling energy E_c(k) = Σ_{pairs ⊂ N(c)} J e_i e_k; for the Bethe
      table it is ln P_c(k) − Σ_u ln p_u(b_u(k)) as read from the table file.
    - `n_pairs_assigned`, `n_pairs_unassigned`: bookkeeping for the console
      summary and the tests.
    """
    n_enriched_checks::Int
    check_degree::Int
    n_configurations::Int
    enriched_check_ids::Vector{Int}
    edge_neurons::Matrix{Int}
    configuration_bits::Matrix{Float32}
    configuration_parity::Vector{Float32}
    table_kind::String
    log_weight_table::Matrix{Float32}
    n_pairs_assigned::Int
    n_pairs_unassigned::Int
end

function empty_soft_check_tables()::SoftCheckTables
    """
    The tables of a model with the standard check node: nothing enriched.
    """
    empty_tables::SoftCheckTables = SoftCheckTables(
        0, 0, 0,
        Int[],
        Matrix{Int}(undef, 0, 0),
        Matrix{Float32}(undef, 0, 0),
        Float32[],
        ENRICHED_TABLE_PAIRWISE,
        Matrix{Float32}(undef, 0, 0),
        0, 0
    )
    return empty_tables
end

function configuration_table(check_degree::Int)::Tuple{Matrix{Float32}, Vector{Float32}}
    """
    All 2^d binary configurations of a degree-d check, as a (2^d × d) 0/1
    matrix, together with their parities. Configuration k (1-based) has bit j
    equal to bit (j-1) of the integer k-1, so k = 1 is all zeros.
    """
    n_configurations::Int = 2^check_degree
    bits::Matrix{Float32} = zeros(Float32, n_configurations, check_degree)
    parities::Vector{Float32} = zeros(Float32, n_configurations)
    for configuration_index in 1:n_configurations
        integer_code::Int = configuration_index - 1
        parity::Int = 0
        for slot in 1:check_degree
            bit::Int = (integer_code >> (slot - 1)) & 1
            bits[configuration_index, slot] = Float32(bit)
            parity = parity ⊻ bit
        end
        parities[configuration_index] = Float32(parity)
    end
    return (bits, parities)
end

function build_soft_check_tables(
    parity_check_matrix::AbstractMatrix{Bool},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32},
    neuron_to_check_variable::Dict{Int, Tuple{Int, Int}};
    table_kind::String = ENRICHED_TABLE_PAIRWISE,
    bethe_log_weights::Dict{Int, Vector{Float32}} = Dict{Int, Vector{Float32}}(),
    verbose::Bool = true
)::SoftCheckTables
    """
    Build the `SoftCheckTables` for a parity-check matrix and CER couplings.
    `connectivity` is the (n_pairs × 2) matrix of 1-based qubit pairs and
    `correlation_strengths` the matching J_ik, exactly as `parse_cer_data`
    returns them. `neuron_to_check_variable` is the base's neuron map, used to
    locate the neuron of each (check, variable) edge.

    The enriched checks are those whose support holds at least one pair,
    whatever the table: the pairs say where the correlations live. With
    `table_kind = ENRICHED_TABLE_PAIRWISE` the log-weight table is the coupling
    energy of every configuration. With `ENRICHED_TABLE_BETHE` it is read from
    `bethe_log_weights` (`read_bethe_log_weights`), which must hold a row of
    2^d finite values for every enriched check; rows of other checks are ignored.
    """
    validated_table_kind::String = enriched_table_kind(table_kind)
    n_checks::Int = size(parity_check_matrix, 1)
    n_pairs::Int = size(connectivity, 1)
    if length(correlation_strengths) != n_pairs
        throw(ArgumentError(
            "build_soft_check_tables: $(n_pairs) pairs but " *
            "$(length(correlation_strengths)) couplings."))
    end

    # Which checks contain each qubit, so pair assignment is a small intersection.
    checks_of_qubit::Vector{Vector{Int}} = [
        findall(parity_check_matrix[:, qubit]) for qubit in 1:size(parity_check_matrix, 2)
    ]

    # Assign every pair to the first check whose support holds both endpoints.
    pairs_of_check::Vector{Vector{Int}} = [Int[] for _ in 1:n_checks]
    n_pairs_unassigned::Int = 0
    for pair_index in 1:n_pairs
        qubit_a::Int = connectivity[pair_index, 1]
        qubit_b::Int = connectivity[pair_index, 2]
        shared_checks::Vector{Int} = intersect(checks_of_qubit[qubit_a], checks_of_qubit[qubit_b])
        if isempty(shared_checks)
            n_pairs_unassigned += 1
        else
            push!(pairs_of_check[shared_checks[1]], pair_index)
        end
    end
    n_pairs_assigned::Int = n_pairs - n_pairs_unassigned
    if n_pairs_unassigned > 0 && verbose
        @warn "build_soft_check_tables: $(n_pairs_unassigned) of $(n_pairs) CER pairs " *
              "share no check and cannot enter any check factor; they are dropped."
    end

    enriched_check_ids::Vector{Int} = findall(!isempty, pairs_of_check)
    n_enriched_checks::Int = length(enriched_check_ids)
    if n_enriched_checks == 0
        throw(ArgumentError(
            "build_soft_check_tables: no check contains a CER pair, so the " *
            "enriched check node would be identical to the standard one."))
    end

    # All enriched checks must share one degree (the dense GPU path reshapes on it).
    degrees::Vector{Int} = [count(parity_check_matrix[check, :]) for check in enriched_check_ids]
    check_degree::Int = degrees[1]
    if any(degree -> degree != check_degree, degrees)
        throw(ArgumentError(
            "build_soft_check_tables: enriched checks have unequal degrees " *
            "$(sort(unique(degrees))); the enriched kernel needs one common degree."))
    end
    n_configurations::Int = 2^check_degree

    # Neuron of each (check, variable) edge.
    neuron_of_edge::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}()
    for (neuron_index, check_and_variable) in neuron_to_check_variable
        neuron_of_edge[check_and_variable] = neuron_index
    end

    configuration_tables::Tuple{Matrix{Float32}, Vector{Float32}} = configuration_table(check_degree)
    configuration_bits::Matrix{Float32} = configuration_tables[1]
    configuration_parity::Vector{Float32} = configuration_tables[2]

    edge_neurons::Matrix{Int} = zeros(Int, check_degree, n_enriched_checks)
    log_weight_table::Matrix{Float32} = zeros(Float32, n_configurations, n_enriched_checks)
    for (enriched_index, check) in enumerate(enriched_check_ids)
        support::Vector{Int} = findall(parity_check_matrix[check, :])
        slot_of_qubit::Dict{Int, Int} = Dict{Int, Int}(qubit => slot for (slot, qubit) in enumerate(support))
        for (slot, qubit) in enumerate(support)
            edge_neurons[slot, enriched_index] = neuron_of_edge[(check, qubit)]
        end
        if validated_table_kind == ENRICHED_TABLE_PAIRWISE
            for pair_index in pairs_of_check[check]
                slot_a::Int = slot_of_qubit[connectivity[pair_index, 1]]
                slot_b::Int = slot_of_qubit[connectivity[pair_index, 2]]
                coupling::Float32 = correlation_strengths[pair_index]
                for configuration_index in 1:n_configurations
                    log_weight_table[configuration_index, enriched_index] +=
                        coupling *
                        configuration_bits[configuration_index, slot_a] *
                        configuration_bits[configuration_index, slot_b]
                end
            end
        else
            if !haskey(bethe_log_weights, check)
                throw(ArgumentError(
                    "build_soft_check_tables: enriched check $(check) has no row in the Bethe " *
                    "table (rows for $(sort(collect(keys(bethe_log_weights))))). Rebuild the " *
                    "table file with every check of this parity-check matrix."))
            end
            bethe_row::Vector{Float32} = bethe_log_weights[check]
            if length(bethe_row) != n_configurations
                throw(ArgumentError(
                    "build_soft_check_tables: the Bethe table row of check $(check) has " *
                    "$(length(bethe_row)) entries; a degree-$(check_degree) check needs " *
                    "$(n_configurations)."))
            end
            log_weight_table[:, enriched_index] .= bethe_row
        end
    end

    tables::SoftCheckTables = SoftCheckTables(
        n_enriched_checks,
        check_degree,
        n_configurations,
        enriched_check_ids,
        edge_neurons,
        configuration_bits,
        configuration_parity,
        validated_table_kind,
        log_weight_table,
        n_pairs_assigned,
        n_pairs_unassigned
    )
    return tables
end

"How many (2^d × n_enriched × batch) Float32 arrays the dense kernel holds. Two
are the real ones — the log-weights and their exponentials — and the rest covers
the (d × n_enriched × batch) buffers and headroom. These are allocated ONCE per
chunk in `GPUSoftCheckState`, not per layer, so this is the whole cost."
const ENRICHED_KERNEL_LIVE_ARRAYS::Int = 4

"Device memory the enriched kernel's own tensors may occupy at the batch size
chosen when the caller gave no explicit budget.

1 GB makes this a no-op at the 16384 fallback for a degree-6 check — the state
is released after every chunk now, so peak memory is one chunk's working set
however many chunks there are, and there is no reason to penalise the enriched
node relative to the standard one. The cap still bites for a higher-degree code,
where 2^d grows the per-sample cost fast (degree 8 is 4x degree 6)."
const ENRICHED_KERNEL_MEMORY_BUDGET_BYTES::Int = 1024 * 1024 * 1024

function enriched_kernel_bytes_per_sample(tables::SoftCheckTables)::Int
    """
    Device memory the dense GPU kernel needs PER SAMPLE beyond the standard
    forward pass, so batch sizing can account for it. Zero for empty tables.
    """
    bytes_per_sample::Int = tables.n_configurations * tables.n_enriched_checks * 4 * ENRICHED_KERNEL_LIVE_ARRAYS
    return bytes_per_sample
end

function cap_batch_size_for_enriched_kernel(batch_size::Int, tables::SoftCheckTables)::Int
    """
    Lower `batch_size` until the enriched kernel's tensors fit in
    `ENRICHED_KERNEL_MEMORY_BUDGET_BYTES`, rounding down to a power of two.
    Returns `batch_size` unchanged for the standard check node (empty tables),
    so nothing about the tanh path moves.

    This exists because the batch size is NOT always derived: with no
    `gpu_memory` and no `prediction_batch_size`, `resolve_prediction_batch_size`
    returns its fallback outright, and the fallback knows nothing about the
    check node. An explicit `prediction_batch_size` is still honoured — the
    caller has then said what they want.
    """
    bytes_per_sample::Int = enriched_kernel_bytes_per_sample(tables)
    if bytes_per_sample <= 0
        return batch_size
    end
    affordable_batch_size::Int = ENRICHED_KERNEL_MEMORY_BUDGET_BYTES ÷ bytes_per_sample
    if affordable_batch_size >= batch_size
        return batch_size
    end
    capped_batch_size::Int = max(1, 1 << floor(Int, log2(max(1, affordable_batch_size))))
    return capped_batch_size
end

function describe_soft_check_tables(tables::SoftCheckTables)::String
    """
    One-line summary for the console.
    """
    description::String =
        "$(tables.n_enriched_checks) enriched checks of degree $(tables.check_degree) " *
        "($(tables.n_configurations) configurations each), " *
        "$(tables.n_pairs_assigned) couplings assigned, $(tables.n_pairs_unassigned) dropped, " *
        "table: $(tables.table_kind)"
    return description
end

# ----------------------------------------------------------------------------
# CPU kernels (training path): one check and one sample at a time
# ----------------------------------------------------------------------------

struct EnrichedKernelWorkspace
    """
    Scratch vectors for `message_check_to_vertex_enriched!` and its adjoint,
    allocated once per call of the batch-level functions so the per-(check,
    sample) kernels allocate nothing. Sizes: d for the per-slot vectors, 2^d for
    the per-configuration ones.
    """
    incoming_messages::Vector{Float32}            # m_{u→c}, one per slot
    outgoing_messages::Vector{Float32}            # m_{c→v}, one per slot
    configuration_log_weights::Vector{Float32}    # LW(k)
    class_maximum_zero::Vector{Float32}           # per slot: max of LW over class (v, 0)
    class_maximum_one::Vector{Float32}            # per slot: max of LW over class (v, 1)
    class_sum_zero::Vector{Float32}               # per slot: Σ exp(LW − max) over class (v, 0)
    class_sum_one::Vector{Float32}                # per slot: Σ exp(LW − max) over class (v, 1)
    adjoint_outgoing_messages::Vector{Float32}    # a_v = ∂L/∂m_{c→v}, one per slot
    adjoint_incoming_messages::Vector{Float32}    # ∂L/∂m_{u→c}, one per slot
    adjoint_configuration_weights::Vector{Float32}  # ω(k)
end

function enriched_kernel_workspace(tables::SoftCheckTables)::EnrichedKernelWorkspace
    """
    A workspace sized for `tables`.
    """
    check_degree::Int = tables.check_degree
    n_configurations::Int = tables.n_configurations
    workspace::EnrichedKernelWorkspace = EnrichedKernelWorkspace(
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, n_configurations),
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, check_degree),
        zeros(Float32, n_configurations)
    )
    return workspace
end

function enriched_class_sums!(
    workspace::EnrichedKernelWorkspace,
    syndrome_bit::Float32,
    log_weight_table::AbstractMatrix{Float32},
    enriched_index::Int,
    coupling_scale::Float32,
    tables::SoftCheckTables
)::Nothing
    """
    The part of the enriched rule the forward and its adjoint share, for one
    check and one sample, from `workspace.incoming_messages`:

      1. LW(k) = α T_c(k) − Σ_slots b_slot(k) m_slot, every configuration k.
      2. For each slot v and bit e, the maximum of LW over the class
         {k : parity(k) = s_c, b_v(k) = e} and the sum of exp(LW − maximum)
         over it — a log-sum-exp stabilised by the class's own maximum, so no
         class can underflow to zero: it always holds its maximum, whose term
         is exactly 1.

    Leaves LW in `configuration_log_weights` and the maxima and sums in the
    `class_*` vectors. Every class is non-empty for a check of degree ≥ 2
    (2^(d−2) members), so every maximum is finite.
    """
    check_degree::Int = tables.check_degree
    n_configurations::Int = tables.n_configurations

    @inbounds for configuration_index in 1:n_configurations
        log_weight::Float32 = coupling_scale * log_weight_table[configuration_index, enriched_index]
        for slot in 1:check_degree
            log_weight -= tables.configuration_bits[configuration_index, slot] * workspace.incoming_messages[slot]
        end
        workspace.configuration_log_weights[configuration_index] = log_weight
    end

    @inbounds for slot in 1:check_degree
        workspace.class_maximum_zero[slot] = -Inf32
        workspace.class_maximum_one[slot] = -Inf32
        workspace.class_sum_zero[slot] = 0.0f0
        workspace.class_sum_one[slot] = 0.0f0
    end
    @inbounds for configuration_index in 1:n_configurations
        if tables.configuration_parity[configuration_index] == syndrome_bit
            log_weight_of_configuration::Float32 = workspace.configuration_log_weights[configuration_index]
            for slot in 1:check_degree
                if tables.configuration_bits[configuration_index, slot] == 0.0f0
                    workspace.class_maximum_zero[slot] = max(workspace.class_maximum_zero[slot], log_weight_of_configuration)
                else
                    workspace.class_maximum_one[slot] = max(workspace.class_maximum_one[slot], log_weight_of_configuration)
                end
            end
        end
    end
    @inbounds for configuration_index in 1:n_configurations
        if tables.configuration_parity[configuration_index] == syndrome_bit
            log_weight_to_sum::Float32 = workspace.configuration_log_weights[configuration_index]
            for slot in 1:check_degree
                if tables.configuration_bits[configuration_index, slot] == 0.0f0
                    workspace.class_sum_zero[slot] += exp(log_weight_to_sum - workspace.class_maximum_zero[slot])
                else
                    workspace.class_sum_one[slot] += exp(log_weight_to_sum - workspace.class_maximum_one[slot])
                end
            end
        end
    end
    return nothing
end

function message_check_to_vertex_enriched!(
    workspace::EnrichedKernelWorkspace,
    syndrome_bit::Float32,
    log_weight_table::AbstractMatrix{Float32},
    enriched_index::Int,
    coupling_scale::Float32,
    tables::SoftCheckTables
)::Nothing
    """
    The enriched check-to-variable rule for ONE check and ONE sample: reads the
    d incoming raw LLRs m_{u→c} from `workspace.incoming_messages` and writes
    the d outgoing messages m_{c→v} to `workspace.outgoing_messages`.

    With b(k) the bit vector of configuration k (`tables.configuration_bits`),
    T_c the check's column of `log_weight_table` and α = `coupling_scale`:

        LW(k)   = α T_c(k) − b(k)·m
        S_v^e   = log-sum-exp over {k : parity(k) = s_c, b_v(k) = e} of LW(k)
        m_{c→v} = S_v^0 − S_v^1 − m_v,   clamped to ±ENRICHED_MESSAGE_CAP

    The subtraction of m_v divides out the target's own factor, which LW
    includes so that one enumeration serves every target; the message does not
    depend on m_v at all. The softplus terms of ln q_u(·) are omitted from LW
    because they are the same for every configuration and cancel in the
    difference. At T_c = 0 (or α = 0) this is the tanh rule.
    """
    enriched_class_sums!(workspace, syndrome_bit, log_weight_table, enriched_index, coupling_scale, tables)
    @inbounds for slot in 1:tables.check_degree
        log_sum_exp_zero::Float32 = workspace.class_maximum_zero[slot] + log(workspace.class_sum_zero[slot])
        log_sum_exp_one::Float32 = workspace.class_maximum_one[slot] + log(workspace.class_sum_one[slot])
        message::Float32 = log_sum_exp_zero - log_sum_exp_one - workspace.incoming_messages[slot]
        workspace.outgoing_messages[slot] = clamp(message, -ENRICHED_MESSAGE_CAP, ENRICHED_MESSAGE_CAP)
    end
    return nothing
end

function backward_message_check_to_vertex_enriched!(
    workspace::EnrichedKernelWorkspace,
    syndrome_bit::Float32,
    log_weight_table::AbstractMatrix{Float32},
    enriched_index::Int,
    coupling_scale::Float32,
    tables::SoftCheckTables
)::Float32
    """
    The adjoint of `message_check_to_vertex_enriched!` for one check and one
    sample. Reads the incoming messages m_{u→c} (`workspace.incoming_messages`),
    the forward's outgoing messages (`workspace.outgoing_messages`, to find the
    clamped ones) and the adjoint of the outgoing messages a_v = ∂L/∂m_{c→v}
    (`workspace.adjoint_outgoing_messages`); writes ∂L/∂m_{u→c} to
    `workspace.adjoint_incoming_messages` and ω(k) = ∂L/∂(α T_c)(k) to
    `workspace.adjoint_configuration_weights`, and returns Σ_k ω(k) T_c(k), the
    check's contribution to ∂L/∂α.

    The forward's class log-sum-exps are recomputed here rather than taped
    (`enriched_class_sums!`, the same cost as the forward). With
    π_v^e(k) = exp(LW(k) − S_v^e) the softmax inside class (v, e):

        ω(k)      = Σ_v a_v [π_v^0(k) − π_v^1(k)]
        ∂L/∂m_u   = −Σ_k ω(k) b_u(k) − a_u

    The −a_u removes the u = v term of the sum, which is a_u exactly: the
    message is extrinsic, so ∂m_{c→v}/∂m_v = 0. A slot whose forward message
    sat on the cap has a_v = 0: the clamp's derivative is zero there.
    """
    check_degree::Int = tables.check_degree
    n_configurations::Int = tables.n_configurations
    enriched_class_sums!(workspace, syndrome_bit, log_weight_table, enriched_index, coupling_scale, tables)

    @inbounds for slot in 1:check_degree
        if abs(workspace.outgoing_messages[slot]) >= ENRICHED_MESSAGE_CAP
            workspace.adjoint_outgoing_messages[slot] = 0.0f0
        end
    end
    @inbounds for configuration_index in 1:n_configurations
        workspace.adjoint_configuration_weights[configuration_index] = 0.0f0
    end
    @inbounds for configuration_index in 1:n_configurations
        if tables.configuration_parity[configuration_index] == syndrome_bit
            log_weight_of_configuration::Float32 = workspace.configuration_log_weights[configuration_index]
            softmax_difference::Float32 = 0.0f0
            for slot in 1:check_degree
                if tables.configuration_bits[configuration_index, slot] == 0.0f0
                    softmax_difference += workspace.adjoint_outgoing_messages[slot] *
                        exp(log_weight_of_configuration - workspace.class_maximum_zero[slot]) / workspace.class_sum_zero[slot]
                else
                    softmax_difference -= workspace.adjoint_outgoing_messages[slot] *
                        exp(log_weight_of_configuration - workspace.class_maximum_one[slot]) / workspace.class_sum_one[slot]
                end
            end
            workspace.adjoint_configuration_weights[configuration_index] = softmax_difference
        end
    end

    adjoint_coupling_scale::Float32 = 0.0f0
    @inbounds for slot in 1:check_degree
        workspace.adjoint_incoming_messages[slot] = -workspace.adjoint_outgoing_messages[slot]
    end
    @inbounds for configuration_index in 1:n_configurations
        configuration_weight::Float32 = workspace.adjoint_configuration_weights[configuration_index]
        for slot in 1:check_degree
            workspace.adjoint_incoming_messages[slot] -=
                configuration_weight * tables.configuration_bits[configuration_index, slot]
        end
        adjoint_coupling_scale += configuration_weight * log_weight_table[configuration_index, enriched_index]
    end
    return adjoint_coupling_scale
end

# ----------------------------------------------------------------------------
# CPU kernels, batch level
# ----------------------------------------------------------------------------

function apply_enriched_checks!(
    messages_c2v::AbstractMatrix{Float32},
    messages_v2c::AbstractMatrix{Float32},
    syndromes_batch::AbstractMatrix{Bool},
    coupling_scale::AbstractVector{Float32},
    tables::SoftCheckTables
)::Nothing
    """
    Overwrite the rows of `messages_c2v` belonging to enriched checks with the
    correlation-adapted messages, for every sample in the batch:
    `message_check_to_vertex_enriched!` per (check, sample), gathering the
    check's incoming messages by `tables.edge_neurons` and scattering its
    outgoing ones back.

    Inputs:
    - `messages_v2c` (nb_neurons × n_samples): the RAW variable-to-check LLRs
      m_{v→c} of this layer (not the log-tanh activations).
    - `syndromes_batch` (n_checks × n_samples).
    - `coupling_scale`: length-1 vector holding α. A vector rather than a
      scalar so Enzyme can carry its gradient as a `Duplicated` argument like
      the other weights.

    The standard rule is expected to have filled `messages_c2v` already; rows
    of non-enriched checks are left exactly as they were.

    Plain loops, no allocation inside the sample loop, branches only on
    constant table data: this is what keeps the function differentiable by
    Enzyme in reverse mode alongside the rest of the forward pass.
    """
    n_samples::Int = size(messages_v2c, 2)
    check_degree::Int = tables.check_degree
    alpha::Float32 = coupling_scale[1]
    workspace::EnrichedKernelWorkspace = enriched_kernel_workspace(tables)

    @inbounds for sample in 1:n_samples
        for enriched_index in 1:tables.n_enriched_checks
            check::Int = tables.enriched_check_ids[enriched_index]
            syndrome_bit::Float32 = Float32(syndromes_batch[check, sample])
            for slot in 1:check_degree
                workspace.incoming_messages[slot] = messages_v2c[tables.edge_neurons[slot, enriched_index], sample]
            end
            message_check_to_vertex_enriched!(
                workspace, syndrome_bit, tables.log_weight_table, enriched_index, alpha, tables)
            for slot in 1:check_degree
                messages_c2v[tables.edge_neurons[slot, enriched_index], sample] = workspace.outgoing_messages[slot]
            end
        end
    end
    return nothing
end

function backward_enriched_checks!(
    adjoint_messages_v2c::AbstractMatrix{Float32},
    adjoint_log_weight_table::AbstractMatrix{Float32},
    adjoint_messages_c2v::AbstractMatrix{Float32},
    messages_c2v::AbstractMatrix{Float32},
    messages_v2c::AbstractMatrix{Float32},
    syndromes_batch::AbstractMatrix{Bool},
    coupling_scale::Float32,
    tables::SoftCheckTables
)::Float32
    """
    The adjoint of `apply_enriched_checks!` for a batch:
    `backward_message_check_to_vertex_enriched!` per (check, sample).

    Inputs: `adjoint_messages_c2v` = ∂L/∂m_{c→v} (nb_neurons × n_samples), read
    on the enriched checks' rows; `messages_c2v`, the forward's OUTGOING
    messages of this layer (the clamped ones get a zero adjoint);
    `messages_v2c`, the forward's raw incoming messages; `syndromes_batch`; α.

    Outputs:
    - `adjoint_messages_v2c`: ∂L/∂m_{u→c}, OVERWRITTEN on the rows of the
      enriched checks' edges and untouched elsewhere. Those rows receive their
      whole adjoint here — the tanh chain's contribution to them is wrong (the
      forward overwrote those checks' messages) and is discarded by the
      overwrite, which the caller relies on.
    - `adjoint_log_weight_table` (2^d × n_enriched): ACCUMULATES
      ∂L/∂(α T_c)(k) summed over the samples.
    - returns the batch's contribution to ∂L/∂α, Σ_k ω(k) T_c(k) summed over
      checks and samples.
    """
    n_samples::Int = size(messages_v2c, 2)
    check_degree::Int = tables.check_degree
    n_configurations::Int = tables.n_configurations
    workspace::EnrichedKernelWorkspace = enriched_kernel_workspace(tables)
    adjoint_coupling_scale::Float32 = 0.0f0

    @inbounds for sample in 1:n_samples
        for enriched_index in 1:tables.n_enriched_checks
            check::Int = tables.enriched_check_ids[enriched_index]
            syndrome_bit::Float32 = Float32(syndromes_batch[check, sample])
            for slot in 1:check_degree
                neuron::Int = tables.edge_neurons[slot, enriched_index]
                workspace.incoming_messages[slot] = messages_v2c[neuron, sample]
                workspace.outgoing_messages[slot] = messages_c2v[neuron, sample]
                workspace.adjoint_outgoing_messages[slot] = adjoint_messages_c2v[neuron, sample]
            end
            adjoint_coupling_scale += backward_message_check_to_vertex_enriched!(
                workspace, syndrome_bit, tables.log_weight_table, enriched_index, coupling_scale, tables)
            for slot in 1:check_degree
                adjoint_messages_v2c[tables.edge_neurons[slot, enriched_index], sample] =
                    workspace.adjoint_incoming_messages[slot]
            end
            for configuration_index in 1:n_configurations
                adjoint_log_weight_table[configuration_index, enriched_index] +=
                    workspace.adjoint_configuration_weights[configuration_index]
            end
        end
    end
    return adjoint_coupling_scale
end

# ----------------------------------------------------------------------------
# GPU formulation (testing path). Dense algebra, and ALLOCATION-FREE per layer.
#
# Two things make this kernel's memory behaviour different from the standard
# rule's, and only the first is inherent:
#
#   1. The working set really is ~10x larger. The tanh rule holds one number per
#      EDGE (nb_neurons = 216 for the BB code); the enriched rule holds one per
#      CONFIGURATION of every check (2^d * n_enriched = 64 * 36 = 2304). That is
#      the price of marginalising the check factor exactly instead of factorising
#      it, and it cannot be avoided.
#   2. Allocating that per layer is fatal. Julia's GC decides when to run from
#      the size of the HOST objects it tracks, and an MtlArray/CuArray is a small
#      wrapper around a large device buffer — so `n_layers` of them accumulate
#      without the GC feeling any pressure. On unified memory (Apple silicon)
#      they come out of system RAM and there is no allocation failure to trigger
#      a retry, so the process is simply OOM-killed. At 8192 samples x 90 layers
#      that was ~12 GB of dead tensors.
#
# So every array this kernel needs is allocated ONCE, in `GPUSoftCheckState`, and
# written in place. The layer loop allocates nothing: peak device memory is one
# layer's working set no matter how many layers there are. Buffers are sized from
# the chunk's sample count, and `build_gpu_state` — hence this state — is rebuilt
# per chunk, so a smaller final chunk gets correctly sized buffers.
#
# The slot's own factor need NOT be excluded from the log-weights. Within the
# class e_v = 0 that factor is the constant log q_v(0), and within e_v = 1 it is
# the constant log q_v(1), so it contributes exactly
#     log q_v(0) - log q_v(1) = m_{v->c}
# to the message — which is the `- m_{v->c}` the CPU kernel subtracts at the end.
# So ONE log-weight tensor serves every target, and the per-target work is a
# (2d x 2^d) mask matmul.
#
# One global stabiliser (the max over parity-selected configurations) never
# gives NaN: the class holding the maximum sums to at least 1. It is exact while
# every incoming |m_{v->c}| stays below ~70. Each target's two class sums still
# contain the target's OWN factor q_v, so a huge incoming LLR on the target
# (|m_{v->c}| > ~87) can push one sum below Float32's range: log(0) = -Inf and
# the message saturates at ±ENRICHED_MESSAGE_CAP even when the true message is
# small. The CPU kernel stabilises each class separately and has no such limit.
# At unit weights a BB-code v2c message is at most 2 x 16.6 + |l|, far below it.
# ----------------------------------------------------------------------------

struct GPUSoftCheckState{A2 <: AbstractMatrix{Float32}, A3 <: AbstractArray{Float32, 3}}
    """
    The `SoftCheckTables` rearranged for the dense formulation, plus every
    working buffer the kernel needs, on whichever array type the GPU backend
    uses (or plain `Array` with no backend).

    Constant tables:
    - `gather` (d*n_enriched x nb_neurons): 0/1, pulls the enriched edges'
      messages out of the full message matrix; row (slot, check) is flattened
      so that a free reshape gives (d, n_enriched*batch).
    - `scatter` (nb_neurons x d*n_enriched): its transpose, puts them back.
    - `keep_mask` (nb_neurons x 1): 1 on rows of non-enriched checks, 0 on rows
      the kernel overwrites.
    - `configuration_bits` (2^d x d): the bit table, for the log-weight matmul.
    - `class_mask_zero`, `class_mask_one` (d x 2^d): row v indicates the
      configurations with bit v = 0, resp. 1. One matmul each gives every
      target's class sum.
    - `slot_sum_row` (1 x d): a row of ones, so a column sum is a matmul.
    - `configuration_parity` (2^d x 1 x 1), `log_weight_table` (2^d x n_enriched x 1).
    - `syndromes_enriched` (1 x n_enriched x batch).

    Working buffers, all written in place, never reallocated:
    - `gathered_messages` (d*n_enriched x batch), `log_q_zero` (d x n_columns),
      `all_zero_log_weight` (1 x n_columns), `log_weights` (2^d x n_columns),
      `stabiliser` (1 x n_columns), `stabilised_weights` (2^d x n_columns),
      `class_sum_zero`, `class_sum_one` (d x n_columns),
      `enriched_messages` (d*n_enriched x batch),
      `updated_messages` (nb_neurons x batch)
      where n_columns = n_enriched * batch.
    """
    gather::A2
    scatter::A2
    keep_mask::A2
    configuration_bits::A2
    class_mask_zero::A2
    class_mask_one::A2
    slot_sum_row::A2
    configuration_parity::A3
    log_weight_table::A3
    syndromes_enriched::A3
    gathered_messages::A2
    log_q_zero::A2
    all_zero_log_weight::A2
    log_weights::A2
    stabiliser::A2
    stabilised_weights::A2
    class_sum_zero::A2
    class_sum_one::A2
    enriched_messages::A2
    updated_messages::A2
    check_degree::Int
    n_enriched_checks::Int
    n_configurations::Int
    nb_neurons::Int
    n_samples::Int
end

function build_gpu_soft_check_state(
    tables::SoftCheckTables,
    syndromes_batch::AbstractMatrix{Bool},
    nb_neurons::Int,
    to_device::Function
)::GPUSoftCheckState
    """
    Arrange `tables` for the dense kernel, allocate its working buffers for this
    chunk's sample count, and move everything to the device with `to_device`
    (a function `Array{Float32} -> device array`, e.g. `_to_device` in
    forward_gpu.jl, or `identity` for a CPU run).
    """
    check_degree::Int = tables.check_degree
    n_enriched_checks::Int = tables.n_enriched_checks
    n_configurations::Int = tables.n_configurations
    n_enriched_edges::Int = check_degree * n_enriched_checks
    n_samples::Int = size(syndromes_batch, 2)
    n_columns::Int = n_enriched_checks * n_samples

    gather_cpu::Matrix{Float32} = zeros(Float32, n_enriched_edges, nb_neurons)
    keep_mask_cpu::Matrix{Float32} = ones(Float32, nb_neurons, 1)
    for enriched_index in 1:n_enriched_checks
        for slot in 1:check_degree
            flat_row::Int = slot + check_degree * (enriched_index - 1)
            neuron::Int = tables.edge_neurons[slot, enriched_index]
            gather_cpu[flat_row, neuron] = 1.0f0
            keep_mask_cpu[neuron, 1] = 0.0f0
        end
    end

    configuration_bits_complement_cpu::Matrix{Float32} = 1.0f0 .- tables.configuration_bits
    syndromes_enriched_cpu::Array{Float32, 3} = reshape(
        Float32.(Matrix(syndromes_batch[tables.enriched_check_ids, :])),
        1, n_enriched_checks, n_samples
    )

    state::GPUSoftCheckState = GPUSoftCheckState(
        to_device(gather_cpu),
        to_device(Matrix{Float32}(transpose(gather_cpu))),
        to_device(keep_mask_cpu),
        to_device(tables.configuration_bits),
        to_device(Matrix{Float32}(transpose(configuration_bits_complement_cpu))),
        to_device(Matrix{Float32}(transpose(tables.configuration_bits))),
        to_device(ones(Float32, 1, check_degree)),
        to_device(reshape(tables.configuration_parity, n_configurations, 1, 1)),
        to_device(reshape(tables.log_weight_table, n_configurations, n_enriched_checks, 1)),
        to_device(syndromes_enriched_cpu),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, check_degree, n_columns)),
        to_device(zeros(Float32, 1, n_columns)),
        to_device(zeros(Float32, n_configurations, n_columns)),
        to_device(zeros(Float32, 1, n_columns)),
        to_device(zeros(Float32, n_configurations, n_columns)),
        to_device(zeros(Float32, check_degree, n_columns)),
        to_device(zeros(Float32, check_degree, n_columns)),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, nb_neurons, n_samples)),
        check_degree,
        n_enriched_checks,
        n_configurations,
        nb_neurons,
        n_samples
    )
    return state
end

function release_device_array!(array::AbstractArray)::Nothing
    """
    Return a device array's memory to the backend allocator immediately, rather
    than waiting for Julia's garbage collector.

    Julia's GC decides when to run from the size of the HOST objects it tracks,
    and an `MtlArray`/`CuArray` is a small wrapper around a large device buffer.
    Per-chunk state therefore accumulates unnoticed: measured, the enriched
    check node's ~187 MB of buffers per chunk reached ~12 GB by chunk 65 of 123,
    macOS paged the process out, the swap files filled the volume, and the run
    was killed with "the volume is out of space".

    UNSAFE, as the backend's own name says: only for arrays the caller knows are
    dead. Safe on a plain `Array` (a no-op) and safe to call twice.
    """
    @static if METAL_LOADED
        if array isa Metal.MtlArray
            Metal.unsafe_free!(array)
        end
    elseif CUDA_LOADED
        if array isa CUDA.CuArray
            CUDA.unsafe_free!(array)
        end
    end
    return nothing
end

function release_gpu_soft_check_state!(state::GPUSoftCheckState)::Nothing
    """
    Free every device array in a `GPUSoftCheckState`. Call once the chunk that
    built it is finished; the state must not be used afterwards.
    """
    release_device_array!(state.gather)
    release_device_array!(state.scatter)
    release_device_array!(state.keep_mask)
    release_device_array!(state.configuration_bits)
    release_device_array!(state.class_mask_zero)
    release_device_array!(state.class_mask_one)
    release_device_array!(state.slot_sum_row)
    release_device_array!(state.configuration_parity)
    release_device_array!(state.log_weight_table)
    release_device_array!(state.syndromes_enriched)
    release_device_array!(state.gathered_messages)
    release_device_array!(state.log_q_zero)
    release_device_array!(state.all_zero_log_weight)
    release_device_array!(state.log_weights)
    release_device_array!(state.stabiliser)
    release_device_array!(state.stabilised_weights)
    release_device_array!(state.class_sum_zero)
    release_device_array!(state.class_sum_one)
    release_device_array!(state.enriched_messages)
    release_device_array!(state.updated_messages)
    return nothing
end

function apply_enriched_checks_gpu(
    messages_c2v::AbstractMatrix{Float32},
    messages_v2c::AbstractMatrix{Float32},
    coupling_scale::Float32,
    state::GPUSoftCheckState
)::AbstractMatrix{Float32}
    """
    Dense-algebra twin of `apply_enriched_checks!`. Same arithmetic and the same
    ENRICHED_MESSAGE_CAP; the two agree to Float32 round-off for every message
    below the cap.

    OWNERSHIP. The returned matrix is `state.updated_messages`, a buffer this
    function owns and overwrites on the next call. That is safe in the forward
    pass because each layer consumes the previous layer's messages (into
    `m_v2c` and into the readout) BEFORE the next call rewrites the buffer. A
    caller that needs to keep the result across calls must copy it.
    """
    check_degree::Int = state.check_degree
    n_enriched_checks::Int = state.n_enriched_checks
    n_configurations::Int = state.n_configurations
    n_samples::Int = size(messages_v2c, 2)
    n_columns::Int = n_enriched_checks * n_samples
    if n_samples != state.n_samples
        throw(DimensionMismatch(
            "apply_enriched_checks_gpu: state was built for $(state.n_samples) samples " *
            "but was given $(n_samples). Rebuild the state for this chunk."))
    end

    # 1. Incoming LLRs of the enriched edges, as (d x n_enriched*batch), and the
    #    part of the log-weights that does not depend on the configuration.
    #
    #    softplus(-m) = max(-m, 0) + log(1 + exp(-|m|)); written with log(1 + x)
    #    rather than log1p because the latter is not among the math intrinsics
    #    every GPU backend provides, and exp(-|m|) <= 1 keeps the plain form
    #    exact to Float32 round-off.
    mul!(state.gathered_messages, state.gather, messages_v2c)
    incoming::AbstractMatrix{Float32} = reshape(state.gathered_messages, check_degree, n_columns)
    state.log_q_zero .= .-(max.(.-incoming, 0.0f0) .+ log.(1.0f0 .+ exp.(.-abs.(incoming))))
    # Sum over slots of log q(0): a column sum, done as a matmul so the kernel
    # needs no reduction primitive beyond `maximum!` below.
    mul!(state.all_zero_log_weight, state.slot_sum_row, state.log_q_zero)

    # 2. Log-weight of every configuration, ONCE for all targets. Since the bit
    #    table's complement is 1 - bits, the two matmuls
    #        bits * log q(1) + (1 - bits) * log q(0)
    #    collapse to  sum_slots log q(0) + bits * (log q(1) - log q(0)),  and
    #    log q(1) - log q(0) is just -m. So this is ONE (2^d x d)*(d x N) matmul
    #    plus a row broadcast, into a buffer that already exists.
    mul!(state.log_weights, state.configuration_bits, incoming)
    log_weights_3d::AbstractArray{Float32, 3} =
        reshape(state.log_weights, n_configurations, n_enriched_checks, n_samples)
    log_weights_3d .= reshape(state.all_zero_log_weight, 1, n_enriched_checks, n_samples) .-
        log_weights_3d .+
        coupling_scale .* state.log_weight_table .+
        abs.(state.configuration_parity .- state.syndromes_enriched) .* EXCLUDED_CONFIGURATION_LOG_WEIGHT

    # 3. One stabiliser per (check, sample), then both class sums for every
    #    target in a single matmul each.
    stabiliser_3d::AbstractArray{Float32, 3} =
        reshape(state.stabiliser, 1, n_enriched_checks, n_samples)
    maximum!(stabiliser_3d, log_weights_3d)
    state.stabilised_weights .= exp.(state.log_weights .- state.stabiliser)
    mul!(state.class_sum_zero, state.class_mask_zero, state.stabilised_weights)
    mul!(state.class_sum_one, state.class_mask_one, state.stabilised_weights)

    # 4. The message. Subtracting `incoming` removes the target's own factor,
    #    which step 2 included for every target alike.
    enriched_messages::AbstractMatrix{Float32} =
        reshape(state.enriched_messages, check_degree, n_columns)
    enriched_messages .= clamp.(
        log.(state.class_sum_zero) .- log.(state.class_sum_one) .- incoming,
        -ENRICHED_MESSAGE_CAP, ENRICHED_MESSAGE_CAP
    )

    # 5. Put the enriched rows back into the full message matrix.
    mul!(state.updated_messages, state.scatter, state.enriched_messages)
    state.updated_messages .= state.updated_messages .+ messages_c2v .* state.keep_mask
    return state.updated_messages
end
# ----------------------------------------------------------------------------
# GPU adjoint (training under USE_GPU=1). The same dense style as the forward:
# broadcasts, `maximum!`/`sum!` reductions and matmuls, on whatever array type
# the state holds. Buffers are sized for the TRAINING batch and allocated once
# per gradient call in `GPUEnrichedBackwardState`, never per layer.
#
# Unlike the forward kernel, every class is stabilised by its OWN maximum (the
# (2^d × d × n_columns) scratch tensor makes that one `maximum!` per bit value),
# so a class far below the global maximum keeps its softmax instead of
# underflowing to 0/0 — the forward's "exact while |m| < ~70" limit does not
# carry over to the gradient.
# ----------------------------------------------------------------------------

struct GPUEnrichedBackwardState{A2 <: AbstractMatrix{Float32}, A3 <: AbstractArray{Float32, 3}}
    """
    Constant tables and working buffers of `backward_enriched_checks_gpu`, on
    the device. n_columns = n_enriched × batch.

    Constant tables:
    - `membership_zero`, `membership_one` (2^d × d × 1): 1 where slot v's bit in
      configuration k is 0, resp. 1 — the class masks, broadcast over columns.
    - `configuration_bits_transposed` (d × 2^d): for −Bᵀ ω.

    Working buffers:
    - `gathered_adjoint`, `gathered_outgoing`, `gathered_incoming`
      (d·n_enriched × batch): a_v, the forward's m_{c→v}, m_{u→c}.
    - `log_weights` (2^d × n_columns): LW(k), parity exclusion applied.
    - `class_scores` (2^d × d × n_columns): the class-masked log-weights, then
      their class softmaxes times a_v.
    - `class_maximum`, `class_sum` (1 × d × n_columns).
    - `configuration_weights_part` (2^d × 1 × n_columns): one bit value's share
      of ω; `configuration_weights` (2^d × n_columns): ω.
    - `configuration_weights_batch_sum` (2^d × n_enriched × 1): ω summed over
      the batch, for the table's and α's gradients.
    - `adjoint_incoming` (d·n_enriched × batch): ∂L/∂m_{u→c} on the enriched edges.
    - `adjoint_messages_v2c` (nb_neurons × batch): the same scattered into the
      full message matrix, zero on every other row.
    - `adjoint_log_weight_table` (2^d × n_enriched): ACCUMULATES ∂L/∂(αT) over
      the calls (layers) made with this state.
    """
    membership_zero::A3
    membership_one::A3
    configuration_bits_transposed::A2
    gathered_adjoint::A2
    gathered_outgoing::A2
    gathered_incoming::A2
    log_weights::A2
    class_scores::A3
    class_maximum::A3
    class_sum::A3
    configuration_weights_part::A3
    configuration_weights::A2
    configuration_weights_batch_sum::A3
    adjoint_incoming::A2
    adjoint_messages_v2c::A2
    adjoint_log_weight_table::A2
    n_columns::Int
end

function build_gpu_enriched_backward_state(
    tables::SoftCheckTables,
    nb_neurons::Int,
    n_samples::Int,
    to_device::Function
)::GPUEnrichedBackwardState
    """
    Allocate the adjoint's buffers for a batch of `n_samples` and move the
    constant tables to the device with `to_device` (as `build_gpu_soft_check_state`).
    """
    check_degree::Int = tables.check_degree
    n_enriched_checks::Int = tables.n_enriched_checks
    n_configurations::Int = tables.n_configurations
    n_enriched_edges::Int = check_degree * n_enriched_checks
    n_columns::Int = n_enriched_checks * n_samples
    membership_one_cpu::Array{Float32, 3} = reshape(copy(tables.configuration_bits), n_configurations, check_degree, 1)
    membership_zero_cpu::Array{Float32, 3} = 1.0f0 .- membership_one_cpu
    state::GPUEnrichedBackwardState = GPUEnrichedBackwardState(
        to_device(membership_zero_cpu),
        to_device(membership_one_cpu),
        to_device(Matrix{Float32}(transpose(tables.configuration_bits))),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, n_configurations, n_columns)),
        to_device(zeros(Float32, n_configurations, check_degree, n_columns)),
        to_device(zeros(Float32, 1, check_degree, n_columns)),
        to_device(zeros(Float32, 1, check_degree, n_columns)),
        to_device(zeros(Float32, n_configurations, 1, n_columns)),
        to_device(zeros(Float32, n_configurations, n_columns)),
        to_device(zeros(Float32, n_configurations, n_enriched_checks, 1)),
        to_device(zeros(Float32, n_enriched_edges, n_samples)),
        to_device(zeros(Float32, nb_neurons, n_samples)),
        to_device(zeros(Float32, n_configurations, n_enriched_checks)),
        n_columns
    )
    return state
end

function release_gpu_enriched_backward_state!(state::GPUEnrichedBackwardState)::Nothing
    """
    Free every device array of a `GPUEnrichedBackwardState`; the state must not
    be used afterwards.
    """
    release_device_array!(state.membership_zero)
    release_device_array!(state.membership_one)
    release_device_array!(state.configuration_bits_transposed)
    release_device_array!(state.gathered_adjoint)
    release_device_array!(state.gathered_outgoing)
    release_device_array!(state.gathered_incoming)
    release_device_array!(state.log_weights)
    release_device_array!(state.class_scores)
    release_device_array!(state.class_maximum)
    release_device_array!(state.class_sum)
    release_device_array!(state.configuration_weights_part)
    release_device_array!(state.configuration_weights)
    release_device_array!(state.configuration_weights_batch_sum)
    release_device_array!(state.adjoint_incoming)
    release_device_array!(state.adjoint_messages_v2c)
    release_device_array!(state.adjoint_log_weight_table)
    return nothing
end

function backward_enriched_checks_gpu(
    adjoint_messages_c2v::AbstractMatrix{Float32},
    messages_c2v::AbstractMatrix{Float32},
    messages_v2c::AbstractMatrix{Float32},
    coupling_scale::Float32,
    forward_state::GPUSoftCheckState,
    backward_state::GPUEnrichedBackwardState
)::Tuple{AbstractMatrix{Float32}, Float32}
    """
    `backward_enriched_checks!` on the device, for one layer. Arguments as the
    CPU function: ∂L/∂m_{c→v} (read on the enriched rows), the forward's
    outgoing and incoming messages of the layer, α, and the two states.

    Returns (∂L/∂m_{u→c} on the enriched edges scattered into a
    (nb_neurons × batch) matrix that is zero on every other row, the layer's
    contribution to ∂L/∂α). The matrix is `backward_state.adjoint_messages_v2c`,
    overwritten on the next call: consume it before calling again. The table's
    gradient accumulates in `backward_state.adjoint_log_weight_table`.
    """
    check_degree::Int = forward_state.check_degree
    n_enriched_checks::Int = forward_state.n_enriched_checks
    n_configurations::Int = forward_state.n_configurations
    n_samples::Int = size(messages_v2c, 2)
    n_columns::Int = backward_state.n_columns
    if n_samples != forward_state.n_samples || n_columns != n_enriched_checks * n_samples
        throw(DimensionMismatch(
            "backward_enriched_checks_gpu: the states were built for " *
            "$(forward_state.n_samples) samples but were given $(n_samples)."))
    end

    # 1. The enriched edges' values, as (d × n_columns): a_v, with the clamped
    #    slots' adjoint zeroed; the forward's outgoing messages say which.
    mul!(backward_state.gathered_adjoint, forward_state.gather, adjoint_messages_c2v)
    mul!(backward_state.gathered_outgoing, forward_state.gather, messages_c2v)
    mul!(backward_state.gathered_incoming, forward_state.gather, messages_v2c)
    adjoint_outgoing::AbstractMatrix{Float32} = reshape(backward_state.gathered_adjoint, check_degree, n_columns)
    outgoing::AbstractMatrix{Float32} = reshape(backward_state.gathered_outgoing, check_degree, n_columns)
    incoming::AbstractMatrix{Float32} = reshape(backward_state.gathered_incoming, check_degree, n_columns)
    adjoint_outgoing .= adjoint_outgoing .* (abs.(outgoing) .< ENRICHED_MESSAGE_CAP)

    # 2. LW(k) = α T_c(k) − b(k)·m, parity-inconsistent configurations excluded.
    mul!(backward_state.log_weights, forward_state.configuration_bits, incoming)
    log_weights_3d::AbstractArray{Float32, 3} =
        reshape(backward_state.log_weights, n_configurations, n_enriched_checks, n_samples)
    log_weights_3d .= coupling_scale .* forward_state.log_weight_table .- log_weights_3d .+
        abs.(forward_state.configuration_parity .- forward_state.syndromes_enriched) .* EXCLUDED_CONFIGURATION_LOG_WEIGHT

    # 3. ω(k) = Σ_v a_v [π_v^0(k) − π_v^1(k)], one bit value at a time: mask the
    #    other class out, stabilise by the class maximum, normalise, weight by a_v.
    log_weights_columns::AbstractArray{Float32, 3} = reshape(backward_state.log_weights, n_configurations, 1, n_columns)
    adjoint_columns::AbstractArray{Float32, 3} = reshape(backward_state.gathered_adjoint, 1, check_degree, n_columns)
    fill!(backward_state.configuration_weights, 0.0f0)
    for (membership, sign) in ((backward_state.membership_zero, 1.0f0), (backward_state.membership_one, -1.0f0))
        backward_state.class_scores .= log_weights_columns .+ EXCLUDED_CONFIGURATION_LOG_WEIGHT .* (1.0f0 .- membership)
        maximum!(backward_state.class_maximum, backward_state.class_scores)
        backward_state.class_scores .= exp.(backward_state.class_scores .- backward_state.class_maximum)
        sum!(backward_state.class_sum, backward_state.class_scores)
        backward_state.class_scores .= backward_state.class_scores ./ backward_state.class_sum .* adjoint_columns
        sum!(backward_state.configuration_weights_part, backward_state.class_scores)
        backward_state.configuration_weights .= backward_state.configuration_weights .+
            sign .* reshape(backward_state.configuration_weights_part, n_configurations, n_columns)
    end

    # 4. ∂L/∂m_u = −Bᵀ ω − a_u, scattered back into the full message matrix.
    adjoint_incoming::AbstractMatrix{Float32} = reshape(backward_state.adjoint_incoming, check_degree, n_columns)
    mul!(adjoint_incoming, backward_state.configuration_bits_transposed, backward_state.configuration_weights)
    adjoint_incoming .= .-adjoint_incoming .- adjoint_outgoing
    mul!(backward_state.adjoint_messages_v2c, forward_state.scatter, backward_state.adjoint_incoming)

    # 5. The table's gradient, summed over the batch, and α's.
    configuration_weights_3d::AbstractArray{Float32, 3} =
        reshape(backward_state.configuration_weights, n_configurations, n_enriched_checks, n_samples)
    sum!(backward_state.configuration_weights_batch_sum, configuration_weights_3d)
    backward_state.adjoint_log_weight_table .= backward_state.adjoint_log_weight_table .+
        reshape(backward_state.configuration_weights_batch_sum, n_configurations, n_enriched_checks)
    adjoint_coupling_scale::Float32 = Float32(sum(
        backward_state.configuration_weights_batch_sum .* forward_state.log_weight_table))
    return (backward_state.adjoint_messages_v2c, adjoint_coupling_scale)
end

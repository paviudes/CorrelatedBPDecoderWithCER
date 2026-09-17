# soft_constraints.jl
# ----------------------------------------------------------------------------
# Correlation-adapted check-to-variable messages ("enriched check node").
#
# The standard check node treats its variables as independent given the parity
# constraint. Here each check factor additionally carries the pairwise Ising
# couplings J_ik (from the CER data) between variables in its support:
#
#     ψ_c(e) = 1[⊕ e = s_c] · exp( α · Σ_{(i,k) ⊂ N(c)} J_ik e_i e_k )
#
# and the message to v is the exact marginalisation of that factor:
#
#     m_{c→v} = ln [ Z_v(0) / Z_v(1) ],
#     Z_v(e_v) = Σ_{ξ: e_v ⊕ ξ = s_c} Π_{u ≠ v} q_u(ξ_u) · exp(α Σ J e e)   (e_v substituted)
#
# with q_u(·) the probabilities encoded by the incoming LLR m_{u→c}. See
# refs/soft_check_nbp.tex for the derivation. At α = 0 (or J = 0) this reduces
# exactly to the tanh rule 2·atanh[(-1)^{s_c} Π tanh(m/2)].
#
# Only the check-to-variable rule changes. The variable-to-check rule and the
# readout, which carry every learnable message weight, are untouched. The one
# new learnable quantity is the scalar α (`coupling_scale`), shared by all
# couplings; α = 1 is the Bayesian value for the pairwise prior.
#
# Implementation: for a check of degree d, enumerate all 2^d configurations
# ONCE per (check, sample), weight each by Π q_u × exp(α E_c), and bin the
# weights by (target slot, target bit) with the parity selector applied. Each
# target's own factor q_v(e_v) is divided out afterwards, which is exactly
# the subtraction of m_{v→c} in the log domain — so ONE enumeration serves all
# d targets. Everything is done in log space with a maximum subtracted, because
# Float32 products of six probabilities times exp(Σ J) under/overflow long
# before the LLRs saturate.
#
# Two implementations of the same arithmetic live here:
#   * `apply_enriched_checks!`     CPU, allocation-free loops, Enzyme-friendly
#                                  (the training path).
#   * `apply_enriched_checks_gpu`  dense broadcasts and matmuls only, so it
#                                  runs unchanged on Array, MtlArray and CuArray
#                                  (the testing path, see forward_gpu.jl).
# Both consume the same `SoftCheckTables`, built once from (H, connectivity, J).
# ----------------------------------------------------------------------------

const CHECK_NODE_TANH::Int = 0
const CHECK_NODE_ENRICHED::Int = 1

"Additive log-domain penalty that excludes a configuration from a log-sum-exp
class on the GPU path. Finite, so no ±Inf arithmetic anywhere: exp(-1e30 - m)
is exactly 0 in Float32 for any finite m, and a class can never be empty."
const EXCLUDED_CONFIGURATION_LOG_WEIGHT::Float32 = -1.0f30

"Cap on |m_{c→v}| from the enriched rule. Chosen to equal the saturation the
tanh path already has: `safe_atanh_exp_signed!` clips exp(magnitude) at
1 - eps(Float32), so |2 atanh(·)| never exceeds this value there either."
const ENRICHED_MESSAGE_CAP::Float32 = 2.0f0 * atanh(1.0f0 - eps(Float32))

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
    - `pair_energy` (2^d × n_enriched): E_c(k) = Σ_{pairs ⊂ N(c)} J e_i e_k,
      the UNSCALED coupling energy of configuration k; α multiplies it at
      run time.
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
    pair_energy::Matrix{Float32}
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
    verbose::Bool = true
)::SoftCheckTables
    """
    Build the `SoftCheckTables` for a parity-check matrix and CER couplings.
    `connectivity` is the (n_pairs × 2) matrix of 1-based qubit pairs and
    `correlation_strengths` the matching J_ik, exactly as `parse_cer_data`
    returns them. `neuron_to_check_variable` is the base's neuron map, used to
    locate the neuron of each (check, variable) edge.
    """
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
    pair_energy::Matrix{Float32} = zeros(Float32, n_configurations, n_enriched_checks)
    for (enriched_index, check) in enumerate(enriched_check_ids)
        support::Vector{Int} = findall(parity_check_matrix[check, :])
        slot_of_qubit::Dict{Int, Int} = Dict{Int, Int}(qubit => slot for (slot, qubit) in enumerate(support))
        for (slot, qubit) in enumerate(support)
            edge_neurons[slot, enriched_index] = neuron_of_edge[(check, qubit)]
        end
        for pair_index in pairs_of_check[check]
            slot_a::Int = slot_of_qubit[connectivity[pair_index, 1]]
            slot_b::Int = slot_of_qubit[connectivity[pair_index, 2]]
            coupling::Float32 = correlation_strengths[pair_index]
            for configuration_index in 1:n_configurations
                pair_energy[configuration_index, enriched_index] +=
                    coupling *
                    configuration_bits[configuration_index, slot_a] *
                    configuration_bits[configuration_index, slot_b]
            end
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
        pair_energy,
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
        "$(tables.n_pairs_assigned) couplings assigned, $(tables.n_pairs_unassigned) dropped"
    return description
end

# ----------------------------------------------------------------------------
# CPU kernel (training path)
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
    correlation-adapted messages, for every sample in the batch.

    Inputs:
    - `messages_v2c` (nb_neurons × n_samples): the RAW variable-to-check LLRs
      m_{v→c} of this layer (not the log-tanh activations).
    - `syndromes_batch` (n_checks × n_samples).
    - `coupling_scale`: length-1 vector holding α. A vector rather than a
      scalar so Enzyme can carry its gradient as a `Duplicated` argument like
      the other weights.

    The standard rule is expected to have filled `messages_c2v` already; rows
    of non-enriched checks are left exactly as they were.

    Per (check, sample):
      1. log q_u(0) = -softplus(-m_u),  log q_u(1) = -m_u - softplus(-m_u).
      2. LW(k) = α E_c(k) + Σ_slots [log q(0) + bit(k, slot) (log q(1) - log q(0))].
      3. For each slot v and bit e_v, the log-sum-exp of LW over configurations
         with bit(k, v) = e_v and parity(k) = s_c, stabilised by that class's
         own maximum (two passes over the configurations).
      4. m_{c→v} = LSE_v(0) - LSE_v(1) - m_{v→c}, clamped to ±ENRICHED_MESSAGE_CAP.
    Step 4's subtraction divides out the target's own factor q_v(e_v), which
    LW included so that one enumeration serves all d targets.

    Plain loops, no allocation inside the sample loop, branches only on
    constant table data: this is what keeps the function differentiable by
    Enzyme in reverse mode alongside the rest of the forward pass.
    """
    n_samples::Int = size(messages_v2c, 2)
    check_degree::Int = tables.check_degree
    n_configurations::Int = tables.n_configurations
    alpha::Float32 = coupling_scale[1]

    incoming_messages::Vector{Float32} = zeros(Float32, check_degree)
    log_q_zero::Vector{Float32} = zeros(Float32, check_degree)
    log_q_one::Vector{Float32} = zeros(Float32, check_degree)
    configuration_log_weights::Vector{Float32} = zeros(Float32, n_configurations)
    class_maximum_zero::Vector{Float32} = zeros(Float32, check_degree)
    class_maximum_one::Vector{Float32} = zeros(Float32, check_degree)
    class_sum_zero::Vector{Float32} = zeros(Float32, check_degree)
    class_sum_one::Vector{Float32} = zeros(Float32, check_degree)

    @inbounds for sample in 1:n_samples
        for enriched_index in 1:tables.n_enriched_checks
            check::Int = tables.enriched_check_ids[enriched_index]
            syndrome_bit::Float32 = Float32(syndromes_batch[check, sample])

            # 1. Incoming LLRs and their log-probabilities.
            for slot in 1:check_degree
                neuron::Int = tables.edge_neurons[slot, enriched_index]
                incoming::Float32 = messages_v2c[neuron, sample]
                softplus_of_negative::Float32 = softplus(-incoming)
                incoming_messages[slot] = incoming
                log_q_zero[slot] = -softplus_of_negative
                log_q_one[slot] = -incoming - softplus_of_negative
            end

            # 2. Log-weight of every configuration.
            for configuration_index in 1:n_configurations
                log_weight::Float32 = alpha * tables.pair_energy[configuration_index, enriched_index]
                for slot in 1:check_degree
                    log_weight += log_q_zero[slot] +
                        tables.configuration_bits[configuration_index, slot] *
                        (log_q_one[slot] - log_q_zero[slot])
                end
                configuration_log_weights[configuration_index] = log_weight
            end

            # 3a. Class maxima over parity-consistent configurations.
            for slot in 1:check_degree
                class_maximum_zero[slot] = -Inf32
                class_maximum_one[slot] = -Inf32
                class_sum_zero[slot] = 0.0f0
                class_sum_one[slot] = 0.0f0
            end
            for configuration_index in 1:n_configurations
                if tables.configuration_parity[configuration_index] == syndrome_bit
                    log_weight_of_configuration::Float32 = configuration_log_weights[configuration_index]
                    for slot in 1:check_degree
                        if tables.configuration_bits[configuration_index, slot] == 0.0f0
                            class_maximum_zero[slot] = max(class_maximum_zero[slot], log_weight_of_configuration)
                        else
                            class_maximum_one[slot] = max(class_maximum_one[slot], log_weight_of_configuration)
                        end
                    end
                end
            end

            # 3b. Class sums, each stabilised by its own maximum.
            for configuration_index in 1:n_configurations
                if tables.configuration_parity[configuration_index] == syndrome_bit
                    log_weight_to_sum::Float32 = configuration_log_weights[configuration_index]
                    for slot in 1:check_degree
                        if tables.configuration_bits[configuration_index, slot] == 0.0f0
                            class_sum_zero[slot] += exp(log_weight_to_sum - class_maximum_zero[slot])
                        else
                            class_sum_one[slot] += exp(log_weight_to_sum - class_maximum_one[slot])
                        end
                    end
                end
            end

            # 4. Messages: divide out the target's own factor, then cap.
            for slot in 1:check_degree
                log_sum_exp_zero::Float32 = class_maximum_zero[slot] + log(class_sum_zero[slot])
                log_sum_exp_one::Float32 = class_maximum_one[slot] + log(class_sum_one[slot])
                message::Float32 = log_sum_exp_zero - log_sum_exp_one - incoming_messages[slot]
                neuron_out::Int = tables.edge_neurons[slot, enriched_index]
                messages_c2v[neuron_out, sample] = clamp(message, -ENRICHED_MESSAGE_CAP, ENRICHED_MESSAGE_CAP)
            end
        end
    end
    return nothing
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
# One global stabiliser (the max over parity-selected configurations) is safe.
# The class holding the maximum sums to at least 1, so it never underflows; the
# other can only underflow to 0 when the two classes are more than ~87 nats
# apart, which makes log(0) = -Inf, the message +/-Inf, and the clamp to
# ENRICHED_MESSAGE_CAP the right answer. Both classes underflowing — the only
# way to reach NaN — is impossible.
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
    - `configuration_parity` (2^d x 1 x 1), `pair_energy` (2^d x n_enriched x 1).
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
    pair_energy::A3
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
        to_device(reshape(tables.pair_energy, n_configurations, n_enriched_checks, 1)),
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
    release_device_array!(state.pair_energy)
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
        coupling_scale .* state.pair_energy .+
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
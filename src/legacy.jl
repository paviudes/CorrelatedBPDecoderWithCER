# Legacy functions that are no longer used in the current implementation
# These functions were moved here to keep the main codebase clean

function v2c_to_c2v(
    activated_m_v2c_magnitudes,
    activated_m_v2c_signs,
    syndromes_batch,
    base
)
    """
    Functional (Zygote-friendly) V2C → C2V.
    """

    # Magnitudes: standard sparse matmul
    activated_m_c2v_magnitudes = base.adj_V2C_C2V * activated_m_v2c_magnitudes

    # Signs: accumulate parity via adjacency
    # Convert Bool → Int for summation
    signs_int = Int.(activated_m_v2c_signs)

    parity = base.adj_V2C_C2V * signs_int

    # mod 2 for XOR
    activated_m_c2v_signs = mod.(parity, 2) .== 1

    # add syndrome
    activated_m_c2v_signs = xor.(activated_m_c2v_signs, syndromes_batch[base.neuron_to_checks, :])

    # Compute the messages from check to vertex by applying the inverse activation function safely
    m_c2v = safe_atanh_exp_signed(activated_m_c2v_magnitudes, activated_m_c2v_signs)

    return m_c2v
end

function c2v_to_v2c!(
    activated_m_v2c_magnitudes,        # output: magnitudes (log-domain)
    activated_m_v2c_signs,             # output: sign bits
    messages_v2c,                   # buffer (pre-activation real values)
    weighted_channel_llrs,   # buffer (same size as channel_llrs)
    messages_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base,
    weight_matrix
)
    """
    Compute the messages from variable nodes to check nodes (V2C) given the messages from check nodes to variable nodes (C2V) and the initial LLRs, for a single layer.

    m^(t)_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)

    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^(t-1)_(v,c;c',v) is a weight.

    Instead of returning complex values, we represent:
        log(tanh(x/2)) = magnitude + i π * sign

    as:
    - magnitude: log(tanh(|x|/2))
    - sign:      1_{x < 0}
    """

    # update sparse values instead of rebuilding structure
    weight_matrix.nzval .= weights_messages

    # compute: ∑_(c' ∈ N(v) - c) ...
    mul!(messages_v2c, weight_matrix .* base.adj_C2V_V2C, messages_c2v_previous)

    # ---- channel contribution ----
    # weighted_channel_llrs = weights_llrs .* channel_llrs
    @. weighted_channel_llrs = weights_llrs * channel_llrs

    # m_v2c += base_projection * weighted_channel_llrs
    mul!(messages_v2c, base.adj_initialize_V2C, weighted_channel_llrs, 1f0, 1f0)

    # apply the activation function a(x) = log(tanh(x/2)) split into magnitude + sign
    safe_log_tanh_split!(activated_m_v2c_magnitudes, activated_m_v2c_signs, messages_v2c)

    return nothing
end

function c2v_to_v2c(
    m_c2v_previous,
    weights_llrs,
    weights_messages,
    channel_llrs,
    base
)
    """
    Functional (Zygote-friendly) version of C2V → V2C.
    """

    weight_matrix = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        weights_messages,
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )

    # Compute \sum_(c' ∈ N(v) - c) W^(t-1)_(v,c;c',v) m^(t-1)_(c'->v)
    m_v2c = (weight_matrix .* base.adj_C2V_V2C) * m_c2v_previous
    # Compute b^(t)_v l_v contribution
    scaled_llrs = base.adj_initialize_V2C * (weights_llrs .* channel_llrs)
    m_v2c = m_v2c .+ scaled_llrs
    
    # apply activation
    activated_m_v2c_magnitudes, activated_m_v2c_signs =
        safe_log_tanh_split(m_v2c)

    return (activated_m_v2c_magnitudes, activated_m_v2c_signs)
end

function readout!(
    posterior_llrs,
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base,
    weight_matrix
)
    """
    Compute the LLRs at the readout layer given the activated neurons at the C2V layer and the initial LLRs, for a single layer.
    μ^t_v = b^(t)_v l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    where
    - l_v is the initial LLR corresponding to variable node v
    - N(v) is the set of check nodes connected to variable node v
    - W^t_(v; c,v) is a weight.
    """

    weight_matrix.nzval .= weights_readout

    mul!(posterior_llrs, weight_matrix .* base.adj_C2V_readout, m_c2v)

    @. posterior_llrs += weights_llrs * channel_llrs

    return nothing
end

function readout(
    m_c2v,
    weights_readout,
    weights_llrs,
    channel_llrs,
    base
)
    """
    Functional (Zygote-friendly) version of readout.
    """

    # build sparse weight matrix (no nzval mutation)
    weight_matrix = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        weights_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    # message contribution
    posterior_llrs =
        (weight_matrix .* base.adj_C2V_readout) * m_c2v

    # channel contribution
    posterior_llrs =
        posterior_llrs .+ (weights_llrs .* channel_llrs)

    return posterior_llrs
end

function compute_layer!(
    messages_c2v,
    activated_m_c2v_magnitudes,         # buffer: C2V magnitudes
    activated_m_c2v_signs,              # buffer: C2V signs
    messages_v2c,
    activated_m_v2c_magnitudes,         # buffer: V2C magnitudes
    activated_m_v2c_signs,              # buffer: V2C signs
    weighted_channel_llrs,    # buffer: weights_llrs .* channel_llrs
    posterior_llrs,
    syndromes_batch,
    initial_llrs_batch,
    bpnn,
    layer,
    weight_matrix_v2c,
    weight_matrix_readout
)
    """
    Compute the LLRs at layer `t` given the activated neurons at the V2C layer and the syndromes, at layer `t-1`.

    We have to compute two forward passes:
    1. m^t_(v->c) = b^(t)_v l_v + ∑_(c' ∈ N(v) - c) m^(t-1)_(c'->v) W^(t-1)_(v,c;c',v)
    2. a(m^t_(c->v)) = i π s_c + ∑_(v' ∈ N(c) - v) a(m^t_(v'->c))

    Instead of complex numbers:
    - we store magnitudes and signs separately
    - signs are propagated via XOR (parity)

    We compute the LLRs at layer `t` from the messages m^t_(c->v) and the initial LLRs using the readout weights:
    μ^t_v = l_v + ∑_(c ∈ N(v)) m^t_(c->v) W^t_(v; c,v)
    """

    base = bpnn.base
    nsamples = size(initial_llrs_batch, 2)

    # ---- Slice the weights relevant for the current layer ----
    weights_messages, weights_llr = get_layer_weights(bpnn.weights_c2v_v2c, bpnn.weights_llrs, base, layer, nsamples)

    # -------------------------
    # 1. C2V → V2C
    # -------------------------
    c2v_to_v2c!(
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        messages_v2c,
        weighted_channel_llrs,
        messages_c2v,
        weights_llr,
        weights_messages,
        initial_llrs_batch,
        base,
        weight_matrix_v2c
    )

    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    v2c_to_c2v!(
        messages_c2v,
        activated_m_c2v_magnitudes,
        activated_m_c2v_signs,
        activated_m_v2c_magnitudes,
        activated_m_v2c_signs,
        syndromes_batch,
        base
    )

    # -------------------------
    # 3. Readout
    # -------------------------
    readout!(
        posterior_llrs,
        messages_c2v,
        bpnn.weights_c2v_readout,
        weights_llr,
        initial_llrs_batch,
        base,
        weight_matrix_readout
    )

    return nothing
end

function compute_layer(
    messages_c2v,
    syndromes_batch,
    initial_llrs_batch,
    bpnn,
    layer
)
    """
    Functional (Zygote-friendly) version of one layer.
    """

    base = bpnn.base
    nsamples = size(initial_llrs_batch, 2)

    # ---- weights ----
    weights_start = (layer - 1) * base.nb_weights_c2v_v2c + 1
    weights_end   = layer * base.nb_weights_c2v_v2c

    weights_messages = @view bpnn.weights_c2v_v2c[weights_start:weights_end]

    weights_llr = bpnn.weights_llrs[
        (layer - 1)*base.code_n_bits + 1 : layer*base.code_n_bits
    ] .* ones(Float32, 1, nsamples)

    # -------------------------
    # 1. C2V → V2C
    # -------------------------
    m_v2c_magnitudes, m_v2c_signs =
        c2v_to_v2c(
            messages_c2v,
            weights_llr,
            weights_messages,
            initial_llrs_batch,
            base
        )

    # -------------------------
    # 2. V2C → C2V
    # -------------------------
    messages_c2v_new =
        v2c_to_c2v(
            m_v2c_magnitudes,
            m_v2c_signs,
            syndromes_batch,
            base
        )

    # -------------------------
    # 3. Readout
    # -------------------------
    weight_matrix_readout = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        bpnn.weights_c2v_readout,
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    posterior_llrs =
        (weights_llr .* initial_llrs_batch) .+
        (weight_matrix_readout .* base.adj_C2V_readout) * messages_c2v_new

    return (messages_c2v_new, posterior_llrs)
end

function forward_pass(bpnn, initial_llrs_batch, syndromes_batch)
    """
    Functional version of the forward pass: bpnn(initial_llrs_batch, syndromes_batch), returning the LLRs at each layer as a vector of matrices.
    """

    base = bpnn.base
    n_batches = size(initial_llrs_batch, 2)

    messages_c2v = zeros(Float32, base.nb_neurons_per_layer, n_batches)

    posterior_list = Vector{Matrix{Float32}}(undef, base.n_layers)

    for layer in 1:base.n_layers
        messages_c2v, posterior_llrs =
            compute_layer(
                messages_c2v,
                syndromes_batch,
                initial_llrs_batch,
                bpnn,
                layer
            )

        posterior_list[layer] = posterior_llrs
    end

    # Stack into 3D tensor: (n_bits × n_batches × n_layers)
    posterior_llrs_all = cat(posterior_list...; dims=3)

    return posterior_llrs_all
end

function (bpnn::NachmaniNeuralBP)(
    initial_llrs_batch::AbstractMatrix{<:Real},
    syndromes_batch::BitMatrix
)
    """
    Forward pass through the Neural Network for Nachmani et al. architecture, returning the LLRs at each layer.
    """

    base = bpnn.base
    n_batches = size(initial_llrs_batch, 2)
    neurons_per_layer = base.nb_neurons_per_layer

    # Buffers
    messages_c2v      = zeros(Float32, neurons_per_layer, n_batches)

    # C2V buffers (magnitude + sign)
    m_c2v_magnitudes  = similar(messages_c2v)
    m_c2v_signs       = falses(size(messages_c2v))

    # V2C buffers
    messages_v2c      = similar(messages_c2v)

    # V2C activated (magnitude + sign)
    m_v2c_magnitudes  = similar(messages_v2c)
    m_v2c_signs       = falses(size(messages_v2c))

    # Channel contribution buffer
    weighted_channel_llrs = similar(initial_llrs_batch)
    # this will carry weights_llrs .* channel_llrs to avoid recomputing

    posterior_llrs_layer = zeros(Float32, base.code_n_bits, n_batches)

    # Sparse templates
    weight_matrix_v2c = sparse(
        base.non_zero_rows_C2V_V2C,
        base.non_zero_cols_C2V_V2C,
        zeros(Float32, length(base.non_zero_rows_C2V_V2C)),
        base.nb_neurons_per_layer,
        base.nb_neurons_per_layer
    )

    weight_matrix_readout = sparse(
        base.non_zero_rows_C2V_readout,
        base.non_zero_cols_C2V_readout,
        zeros(Float32, length(base.non_zero_rows_C2V_readout)),
        base.code_n_bits,
        base.nb_neurons_per_layer
    )

    # NOTE: previously this was `Zygote.Buffer(zeros(...))` so Zygote could track
    # mutations through this array during reverse-mode AD. Since this legacy
    # forward_pass is a plain forward evaluator (never used inside an AD path
    # — Enzyme is the AD engine, and it operates on the newer *_with_weights!
    # kernels in forward_pass_weights.jl), a regular Array works.
    posterior_llrs = zeros(Float32, base.code_n_bits, n_batches, base.n_layers)

    for layer in 1:base.n_layers
        compute_layer!(
            messages_c2v,
            m_c2v_magnitudes,
            m_c2v_signs,
            messages_v2c,
            m_v2c_magnitudes,
            m_v2c_signs,
            weighted_channel_llrs,
            posterior_llrs_layer,
            syndromes_batch,
            initial_llrs_batch,
            bpnn,
            layer,
            weight_matrix_v2c,
            weight_matrix_readout
        )

        posterior_llrs[:, :, layer] = posterior_llrs_layer
    end

    return copy(posterior_llrs)
end

# ============================================================================
# Legacy result collectors (moved out of postprocessing.jl).
# `postprocessing.jl` now keeps only the general `collect_decoder_statistics`
# (multi-file) and the combined `collect_standard_decoder_statistics`. These
# grid/Ising-specific collectors remain here (still exported) so existing
# callers such as expts/neural_bp_experiments.jl keep working. `fmt_probs` and
# `compute_std_assuming_bernoulli` are resolved at call time, so their being
# defined in other files included later/earlier is fine.
# ============================================================================

function collect_decoder_statistics_correlated(per_qubit_error_probs::AbstractVector{<:Real}, neighbour_error_probs::AbstractVector{<:Real}, num_samples_per_error_rate::Int, n_layers::Int, n_epochs::Int; prefix::String="./../data")::DataFrame
    """
    Collect Neural BP decoder statistics for the correlated (Ballistic) error
    model. There's one file per simulation run, produced by
    `neural_bp_experiments.jl` and named by the convention it builds:

        simulation_results_test_<pq_tag>_s_<s>_nlayers_<n_layers>_epochs_<n_epochs>_trained_using_train_<pq_tag>_s_<s>.csv

    where `<pq_tag>` is `fmt_probs(p, q)`. Reconstructs the expected filenames
    from the p×q×samples grid, reads each, and stacks them into one DataFrame.
    (Legacy: prefer building the file list and calling `collect_decoder_statistics`.)
    """
    num_files = 0
    missing_files = String[]
    for p in per_qubit_error_probs, q in neighbour_error_probs, s in 1:num_samples_per_error_rate
        pq_tag = fmt_probs(Float64(p), Float64(q))
        training_file = "train_$(pq_tag)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_$(pq_tag)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
        if isfile(results_file)
            num_files += 1
        else
            push!(missing_files, results_file)
        end
    end
    if (size(missing_files, 1) > 0)
        @warn ("$(size(missing_files, 1)) files are missing:\n$(missing_files)")
    end

    all_stats = DataFrame(
        algo = Vector{String}(undef, num_files),
        error_model_name = Vector{String}(undef, num_files),
        error_model_parameters_description = Vector{String}(undef, num_files),
        num_samples_per_error_rate = Vector{Int}(undef, num_files),
        n_iterations_BP = Vector{Int}(undef, num_files),
        rounds_per_BP = Vector{Int}(undef, num_files),
        weight_soft_constraint = Vector{Float64}(undef, num_files),
        num_failures = Vector{Int}(undef, num_files),
        average_logical_error_rate = Vector{Float64}(undef, num_files),
        std_logical_error_rate = Vector{Float64}(undef, num_files),
        runtime = Vector{Float64}(undef, num_files)
    )
    file_index = 1
    for p in per_qubit_error_probs, q in neighbour_error_probs, s in 1:num_samples_per_error_rate
        pq_tag = fmt_probs(Float64(p), Float64(q))
        training_file = "train_$(pq_tag)_s_$(s)"
        results_file = "$(prefix)/results/simulation_results_test_$(pq_tag)_s_$(s)_nlayers_$(n_layers)_epochs_$(n_epochs)_trained_using_$(training_file).csv"
        if isfile(results_file)
            stats_dataframe = CSV.read(results_file, DataFrame)
            all_stats[file_index, :algo] = stats_dataframe[1, :algo]
            all_stats[file_index, :error_model_name] = stats_dataframe[1, :error_model_name]
            all_stats[file_index, :error_model_parameters_description] = stats_dataframe[1, :error_model_parameters_description]
            all_stats[file_index, :num_samples_per_error_rate] = stats_dataframe[1, :num_samples_per_error_rate]
            all_stats[file_index, :n_iterations_BP] = stats_dataframe[1, :n_iterations_BP]
            all_stats[file_index, :rounds_per_BP] = stats_dataframe[1, :rounds_per_BP]
            all_stats[file_index, :weight_soft_constraint] = stats_dataframe[1, :weight_soft_constraint]
            all_stats[file_index, :num_failures] = stats_dataframe[1, :num_failures]
            all_stats[file_index, :average_logical_error_rate] = stats_dataframe[1, :average_logical_error_rate]
            all_stats[file_index, :std_logical_error_rate] = compute_std_assuming_bernoulli(all_stats[file_index, :average_logical_error_rate], all_stats[file_index, :num_samples_per_error_rate])
            all_stats[file_index, :runtime] = stats_dataframe[1, :runtime]
            file_index += 1
        end
    end
    return all_stats
end

function collect_standard_decoder_statistics_correlated(prefix::String="./../data", ntrials::Int=100000; standard_BP_output_file::String="standard_bp_failure_rates.txt")::DataFrame
    """
    Legacy Ising-only entry point, kept for back-compat with existing callers
    (e.g. neural_bp_experiments.jl). Delegates to the combined
    `collect_standard_decoder_statistics(:Ising; ...)` in postprocessing.jl.
    `ntrials` is accepted for signature compatibility but ignored — the combined
    collector now reads the trial count from the file.
    """
    stats_df = collect_standard_decoder_statistics(:Ising; prefix=prefix, standard_BP_output_file=standard_BP_output_file)
    return stats_df
end

# ============================================================================
# StandardBPDecoderStatistics — statistics for the standard BP decoder.
# Renamed from `DecoderStatistics` and moved here: standard BP is not the current
# focus, and the Neural BP path now has its own `NeuralBPDecoderStatistics` (in
# postprocessing.jl) with fields named for layers/epochs rather than for BP
# iterations/rounds. `compute_std_assuming_bernoulli` is defined in
# postprocessing.jl (included earlier) and resolved at call time.
# ============================================================================

struct StandardBPDecoderStatistics
    """
    Statistics for the standard Belief Propagation decoder.

    Every field is a Vector, so a single struct can hold ONE record (all fields
    length 1) or MANY (e.g. after concatenating a parameter sweep). The two inner
    constructors are a scalar/per-simulation form (computes the logical error rate
    and its Bernoulli std, then stores each field as a 1-element Vector) and a raw
    form taking every field as an already-built Vector (used by `vcat` and the
    `DataFrame` builder).
    """
    algo::Vector{String} # "SumProduct" or "MinSum"
    error_model_name::Vector{String} # e.g. "ExplicitErrorModel" or "IsingModel" or "CircuitLevelModel"
    error_model_parameters_description::Vector{String} # e.g. "p=0.0011,q=0.0007" or "p=0.0011" or a filename
    num_samples_per_error_rate::Vector{Int} # number of trials (ntrials) for a given error rate
    n_iterations_BP::Vector{Int} # number of iterations of BP
    rounds_per_BP::Vector{Int} # number of rounds per iteration of BP
    weight_soft_constraint::Vector{Float64} # weight of the soft constraint in the BP decoder
    num_failures::Vector{Int} # number of logical failures observed in the trials
    average_logical_error_rate::Vector{Float64} # average logical error rate = num_failures / num_samples_per_error_rate
    std_logical_error_rate::Vector{Float64} # standard deviation of the logical error rate, computed assuming a Bernoulli distribution
    runtime::Vector{Float64} # total runtime of the decoder in seconds over all trials

    # --- Raw (array) constructor: every field already a Vector. ---------------
    function StandardBPDecoderStatistics(algo::Vector{String}, error_model_name::Vector{String},
            error_model_parameters_description::Vector{String},
            num_samples_per_error_rate::Vector{Int}, n_iterations_BP::Vector{Int},
            rounds_per_BP::Vector{Int}, weight_soft_constraint::Vector{Float64},
            num_failures::Vector{Int}, average_logical_error_rate::Vector{Float64},
            std_logical_error_rate::Vector{Float64}, runtime::Vector{Float64})
        n = length(algo)
        lengths_match = all(==(n), (length(error_model_name), length(error_model_parameters_description),
                    length(num_samples_per_error_rate), length(n_iterations_BP),
                    length(rounds_per_BP), length(weight_soft_constraint),
                    length(num_failures), length(average_logical_error_rate),
                    length(std_logical_error_rate), length(runtime)))
        if !lengths_match
            throw(ArgumentError("All StandardBPDecoderStatistics field vectors must have the same length."))
        end
        new(algo, error_model_name, error_model_parameters_description,
            num_samples_per_error_rate, n_iterations_BP, rounds_per_BP,
            weight_soft_constraint, num_failures, average_logical_error_rate,
            std_logical_error_rate, runtime)
    end

    # --- Scalar/per-simulation constructor. -----------------------------------
    function StandardBPDecoderStatistics(algo::String, error_model_name::String, error_model_parameters_description::String, num_samples_per_error_rate::Int, num_iterations_BP::Int, num_rounds_per_iteration_BP::Int, weight_soft_constraint::Float64; num_failures::Int=0, failures::Vector{Bool}=zeros(Bool, num_samples_per_error_rate), runtime::Float64=0.0)
        if !(algo in ("SumProduct", "MinSum"))
            throw(ArgumentError("Algorithm must be either 'SumProduct' or 'MinSum'."))
        end
        if num_samples_per_error_rate < 0
            throw(ArgumentError("Number of samples per error rate must be non-negative."))
        end
        if (num_failures == 0) && (length(failures) > 0)
            num_failures = count(failures)
        end
        if (num_samples_per_error_rate == 0) || (num_failures == 0)
            average_logical_error_rate = 0.0
            std_logical_error_rate = 0.0
        else
            average_logical_error_rate = num_failures / num_samples_per_error_rate
            # NOTE: preserved from the original shared constructor, which divided
            # by the iteration count. The trial count (num_samples_per_error_rate)
            # is the statistically correct denominator; left unchanged here to keep
            # the rename behaviour-preserving for the (test-only) standard path.
            std_logical_error_rate = compute_std_assuming_bernoulli(average_logical_error_rate, num_iterations_BP)
        end
        # Store every field as a 1-element Vector (delegates to the raw form).
        new([algo], [error_model_name], [error_model_parameters_description],
            [num_samples_per_error_rate], [num_iterations_BP], [num_rounds_per_iteration_BP],
            [weight_soft_constraint], [num_failures], [average_logical_error_rate],
            [std_logical_error_rate], [runtime])
    end
end

function Base.vcat(stats::StandardBPDecoderStatistics...)::StandardBPDecoderStatistics
    """
    Concatenate several `StandardBPDecoderStatistics` into one by stacking each
    field vector.
    """
    if isempty(stats)
        throw(ArgumentError("vcat needs at least one StandardBPDecoderStatistics."))
    end
    combined = StandardBPDecoderStatistics(
        (reduce(vcat, getfield(s, f) for s in stats) for f in fieldnames(StandardBPDecoderStatistics))...
    )
    return combined
end

"""
    StandardBPDecoderStatistics(df::DataFrame) -> StandardBPDecoderStatistics

Build a multi-record `StandardBPDecoderStatistics` from a DataFrame whose columns
are the struct's field names.
"""
function StandardBPDecoderStatistics(df::DataFrame)::StandardBPDecoderStatistics
    stats = StandardBPDecoderStatistics(
        (Vector(df[!, f]) for f in fieldnames(StandardBPDecoderStatistics))...
    )
    return stats
end

"""
Check that all provided symbols are valid fields of `StandardBPDecoderStatistics`.
Warns about any invalid key and returns only the valid ones.
"""
function check_valid_fields_StandardBPDecoderStatistics(keys::Vector{Symbol})::Vector{Symbol}
    allowed = fieldnames(StandardBPDecoderStatistics)
    valid = intersect(keys, allowed)
    invalid = setdiff(keys, allowed)
    if !isempty(invalid)
        @warn ("Invalid keys found: $(collect(invalid))")
    end
    return valid
end

# ============================================================================
# Legacy dataframe-filtering helpers (moved out of postprocessing.jl). Only used
# by the legacy expts drivers (ballistic_errors.jl, misc/explicit_errors.jl);
# kept exported so those callers keep working.
# ============================================================================

function check_approximate(col::AbstractVector, val; atol::Float64=1e-8, rtol::Float64=1e-5)::BitVector
    """
    Check if the values in the column are approximately equal to the given value `val` using `isapprox` for Real values,
    or == for String values. Returns a BitVector indicating which elements are approximately equal or equal.
    """
    if eltype(col) <: Real && isa(val, Real)
        matches = isapprox.(col, val; atol=atol, rtol=rtol)
    elseif eltype(col) <: AbstractString && isa(val, AbstractString)
        matches = col .== val
    else
        # Fallback to == for other types
        matches = col .== val
    end
    return matches
end

function extract_collected_data(stats_dataframe::DataFrame, select_parameters::Dict{Symbol, AbstractVector{<:Any}}, display_parameters::Vector{Symbol})::DataFrame
    """
    Extract data from a dataframe that has a specific set of columns and rows corresponding to the values for the columns.
    Returns the rows whose `select_parameters` columns match, projected onto `display_parameters`.
    """
    valid_parameter_names = check_valid_fields_StandardBPDecoderStatistics(collect(keys(select_parameters)))
    valid_parameter_values = Iterators.product(collect([select_parameters[param] for param in valid_parameter_names])...)

    # Each StandardBPDecoderStatistics field is a Vector, so the per-column element
    # type is `eltype(fieldtype(...))` (e.g. String for a Vector{String} field).
    focused_dataframe = DataFrame(
        [name => eltype(fieldtype(StandardBPDecoderStatistics, name))[] for name in display_parameters]...
    )
    for values in valid_parameter_values
        filter_condition = reduce((acc, (param, val)) -> acc .& check_approximate(stats_dataframe[!, param], val), zip(valid_parameter_names, values), init=trues(nrow(stats_dataframe)))
        matching_rows = stats_dataframe[filter_condition, :]
        if nrow(matching_rows) > 0
            append!(focused_dataframe, matching_rows[:, display_parameters])
        end
    end
    return focused_dataframe
end
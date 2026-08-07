function predict_neuralbp(bpnn::NeuralBP, syndromes::BitMatrix; batch_size::Int = 1024)::Array{Bool, 3}
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    The samples are processed in batches of `batch_size` to keep GPU memory
    manageable; per-batch posterior tensors are hard-thresholded to Bool
    immediately and written into the output, so only one batch's worth of
    Float32 LLRs lives on the GPU at a time.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `batch_size::Int=1024`: How many samples to push through `forward_pass_gpu` at once. Smaller = less peak memory, more launch overhead.

    Returns:
    - `predicted_recoveries::Array{Bool, 3}`: shape (n_bits × n_samples × n_layers).
    """
    n_total  = size(syndromes, 2)
    n_bits   = bpnn.base.code_n_bits
    n_layers = bpnn.base.n_layers

    predicted_recoveries = falses(n_bits, n_total, n_layers)

    for start in 1:batch_size:n_total
        stop = min(start + batch_size - 1, n_total)
        chunk_synd = syndromes[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)
        chunk_post = forward_pass_gpu(bpnn, chunk_llrs, chunk_synd)
        @views predicted_recoveries[:, start:stop, :] .= (chunk_post .< 0)
    end

    return predicted_recoveries
end

function check_bp_solutions(parity_check_matrix::Matrix{Int}, logicals::Matrix{Int}, errors::BitMatrix, proposed_recoveries::Array{Bool, 3})::BitVector
    """
    Score each sample the way a real decoder is scored, in two steps:

      1. COMMIT (by syndrome — decode-time information). Pick the FIRST layer
         whose recovery clears the syndrome, i.e. the residual `e ⊕ r` commutes
         with the stabilizers: `parity_check_matrix * (e ⊕ r) = 0`. This mimics
         standard BP, which stops at the first iteration whose residual has zero
         syndrome. Only the syndrome (known at decode time) picks the layer.

      2. SCORE (by logicals). The committed residual is a SUCCESS iff it is also
         logically trivial — it commutes with the logical operators:
         `logicals * (e ⊕ r) = 0`. The committed layer already has zero syndrome,
         so only the (few) logical rows are checked here.

    A sample with NO syndrome-clearing layer is a decode failure (there is no OSD
    backstop). We commit to ONE layer by syndrome and score only that layer — NOT
    "any layer that happens to be logically correct", which would hand the decoder
    oracle knowledge of the true error and massively over-count success on small
    codes (few logical cosets).

    Arguments:
    - `parity_check_matrix::Matrix{Int}`: stabilizer checks H — selects the layer.
    - `logicals::Matrix{Int}`: logical operators L — scores the committed layer.
    - `errors::BitMatrix`: one true error pattern per column.
    - `proposed_recoveries::Array{Bool, 3}`: `(n_bits, n_samples, n_layers)` hard
      decisions, one recovery per layer.

    Returns:
    - `is_correct::BitVector`: true for each correctly decoded sample.
    """
    n_samples = size(errors, 2)
    is_correct = falses(n_samples)
    for i in 1:n_samples
        # Residual e ⊕ r for every layer at once: (n_bits, n_layers).
        residuals = errors[:, i] .⊻ proposed_recoveries[:, i, :]
        # Per-layer syndrome weight; zero ⇔ that layer's residual clears the syndrome.
        layer_syndrome_weight = vec(sum(mod.(parity_check_matrix * residuals, 2), dims = 1))
        committed_layer = findfirst(==(0), layer_syndrome_weight)
        if committed_layer === nothing
            continue  # no layer produced a syndrome-valid correction -> failure
        end
        # Score ONLY the committed layer: success iff it is also logically trivial.
        committed_residual = residuals[:, committed_layer]
        logical_syndrome = mod.(logicals * committed_residual, 2)
        if all(logical_syndrome .== 0)
            is_correct[i] = true
        end
    end
    return is_correct
end

"""
    count_syndrome_satisfactions(parity_check_matrix, logicals, errors, proposed_recoveries) -> NamedTuple

Decompose decoder outcomes into three mutually exclusive buckets, using EXACTLY
the commit rule of `check_bp_solutions` (first layer whose residual has zero
syndrome weight):

  1. SUCCESS             — some layer cleared the syndrome, and the first such
                           layer is also logically trivial.
  2. COSET FAILURE       — some layer cleared the syndrome, but the first such
                           layer carries a logical.
  3. CONVERGENCE FAILURE — no layer ever cleared the syndrome.

`check_bp_solutions` already computes both facts but collapses (2) and (3) into
the same `false`. This surfaces the distinction, because they call for opposite
fixes: (2) rising implicates the correlation prior pointing along the code's
cycle space (which is spanned by stabilizers AND logicals), whereas (3) rising
implicates the correlation term stealing gradient from the syndrome term during
training.

Also returns, per sample, the layer committed to (0 when none cleared) and the
smallest residual syndrome weight reached across all layers — the latter says
whether a non-converging sample was close or wildly off.

This is ADDITIVE. `check_bp_solutions` is deliberately left untouched as an
independent reference implementation, so the equality of the two `is_correct`
vectors is a real test rather than a tautology; see
`expts/misc/test_syndrome_diagnosis.jl`.

Arguments and shapes are identical to `check_bp_solutions`.
"""
function count_syndrome_satisfactions(
    parity_check_matrix::Matrix{Int},
    logicals::Matrix{Int},
    errors::BitMatrix,
    proposed_recoveries::Array{Bool, 3}
)::NamedTuple
    n_samples::Int = size(errors, 2)
    n_layers::Int = size(proposed_recoveries, 3)

    syndrome_cleared::BitVector = falses(n_samples)
    is_correct::BitVector = falses(n_samples)
    committed_layers::Vector{Int} = zeros(Int, n_samples)      # 0 = no layer ever cleared
    min_syndrome_weights::Vector{Int} = zeros(Int, n_samples)
    # Weight of the TRUE error, carried per sample so the failure records can be
    # read against it: coset tipping should concentrate at weights where the
    # residual can reach a low-weight logical, convergence failure should not.
    error_weights::Vector{Int} = zeros(Int, n_samples)

    for i in 1:n_samples
        error_weights[i] = count(view(errors, :, i))
        # NOTE: `residuals` and `committed_residual` are deliberately left
        # unannotated. Broadcasting a BitVector against an Array{Bool,2} may yield
        # either container, and pinning the type would force a `convert` — i.e. a
        # fresh 72x100 copy — on every one of up to 10^6 iterations.
        residuals = errors[:, i] .⊻ proposed_recoveries[:, i, :]
        layer_syndrome_weight::Vector{Int} = vec(sum(mod.(parity_check_matrix * residuals, 2), dims = 1))
        min_syndrome_weights[i] = minimum(layer_syndrome_weight)

        committed_layer::Union{Int, Nothing} = findfirst(==(0), layer_syndrome_weight)
        if committed_layer === nothing
            continue  # convergence failure: no layer produced a syndrome-valid correction
        end

        syndrome_cleared[i] = true
        committed_layers[i] = committed_layer

        # Score ONLY the committed layer: success iff it is also logically trivial.
        committed_residual = residuals[:, committed_layer]
        logical_syndrome::Vector{Int} = mod.(logicals * committed_residual, 2)
        if all(logical_syndrome .== 0)
            is_correct[i] = true
        end
    end

    n_syndrome_cleared::Int = count(syndrome_cleared)
    n_correct::Int = count(is_correct)

    diagnosis::NamedTuple = (
        # per-sample — kept because the paired (McNemar) comparison needs them
        syndrome_cleared       = syndrome_cleared,
        is_correct             = is_correct,
        committed_layer        = committed_layers,
        min_syndrome_weight    = min_syndrome_weights,
        error_weight           = error_weights,
        # aggregate
        n_samples              = n_samples,
        n_layers               = n_layers,
        n_syndrome_cleared     = n_syndrome_cleared,
        n_correct              = n_correct,
        n_coset_failures       = n_syndrome_cleared - n_correct,
        n_convergence_failures = n_samples - n_syndrome_cleared,
    )
    return diagnosis
end

"""
    concatenate_diagnoses(chunk_diagnoses) -> NamedTuple

Stitch per-chunk `count_syndrome_satisfactions` results into one whole-run
result, preserving sample order. Aggregates are recomputed from the concatenated
per-sample vectors rather than summed from the chunks, so the two can never
disagree.
"""
function concatenate_diagnoses(chunk_diagnoses::Vector{<:NamedTuple})::NamedTuple
    syndrome_cleared::BitVector = reduce(vcat, [chunk.syndrome_cleared for chunk in chunk_diagnoses])
    is_correct::BitVector = reduce(vcat, [chunk.is_correct for chunk in chunk_diagnoses])
    committed_layer::Vector{Int} = reduce(vcat, [chunk.committed_layer for chunk in chunk_diagnoses])
    min_syndrome_weight::Vector{Int} = reduce(vcat, [chunk.min_syndrome_weight for chunk in chunk_diagnoses])
    error_weight::Vector{Int} = reduce(vcat, [chunk.error_weight for chunk in chunk_diagnoses])

    n_samples::Int = length(is_correct)
    n_layers::Int = maximum(chunk.n_layers for chunk in chunk_diagnoses)
    n_syndrome_cleared::Int = count(syndrome_cleared)
    n_correct::Int = count(is_correct)

    diagnosis::NamedTuple = (
        syndrome_cleared       = syndrome_cleared,
        is_correct             = is_correct,
        committed_layer        = committed_layer,
        min_syndrome_weight    = min_syndrome_weight,
        error_weight           = error_weight,
        n_samples              = n_samples,
        n_layers               = n_layers,
        n_syndrome_cleared     = n_syndrome_cleared,
        n_correct              = n_correct,
        n_coset_failures       = n_syndrome_cleared - n_correct,
        n_convergence_failures = n_samples - n_syndrome_cleared,
    )
    return diagnosis
end

"""
    mean_committed_layer(diagnosis) -> Float64

Average committed layer over the samples that ACTUALLY cleared the syndrome.

Samples that never cleared carry `committed_layer = 0`, and averaging those in
would conflate "converged early" with "never converged" — the two things this
diagnostic exists to separate. Returns `NaN` when nothing cleared.
"""
function mean_committed_layer(diagnosis::NamedTuple)::Float64
    cleared_layers::Vector{Int} = diagnosis.committed_layer[diagnosis.syndrome_cleared]
    if isempty(cleared_layers)
        return NaN
    end
    average_layer::Float64 = sum(cleared_layers) / length(cleared_layers)
    return average_layer
end

function predict_and_check_neuralbp(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    errors::BitMatrix;
    batch_size::Int = 1024,
)::BitVector
    """
    Predict the recoveries for the given syndromes using the trained NeuralBP model.
    Then check if the predicted recoveries correctly fix the errors according to the parity-check matrix.
    Arguments:
    - `bpnn::NeuralBP`: The trained NeuralBP model.
    - `syndromes::BitMatrix`: A matrix where each column represents a syndrome corresponding to an error pattern.
    - `errors::BitMatrix`: A matrix where each row represents an error pattern.
    - `batch_size::Int=1024`: How many samples to push through `forward_pass_gpu` at once. Smaller = less peak memory, more launch overhead.
    Returns:
    - `is_correct::BitVector`: A vector indicating whether each recovery correctly fixes the corresponding error.

    A sample is decoded correctly when the FIRST layer whose recovery clears the
    syndrome (the BP stopping condition) also lands in the correct logical coset;
    see `check_bp_solutions`.
    """
    n_samples = size(syndromes, 2)

    # Stabilizer checks H (used to commit to a layer by syndrome) and the logical
    # operators L (used to score that committed layer). `parity_check_matrix_dual`
    # is `vcat(H, logicals)`, so the logical rows are its tail past the H rows.
    parity_check_matrix = convert.(Int, bpnn.base.parity_check_matrix)
    n_checks = size(parity_check_matrix, 1)
    logicals = convert.(Int, bpnn.base.parity_check_matrix_dual[n_checks + 1:end, :])

    # When the GPU is not engaged, process all samples at once without batching.
    # `gpu_active()` is false when USE_GPU=0 at runtime OR no GPU backend is
    # compiled in for this platform — in which case we use
    # `forward_pass_with_weights` instead of `forward_pass_gpu`.
    if !gpu_active()
        posterior_llrs = forward_pass_with_weights(bpnn, repeat(bpnn.base.initial_llrs, 1, n_samples), syndromes)
        proposed_recoveries = Array(posterior_llrs .< 0) # shape (n_bits, n_samples, n_layers)
        is_correct = check_bp_solutions(parity_check_matrix, logicals, errors, proposed_recoveries)
        return is_correct
    end

    print_info("Using GPU for predictions with batch size = $(batch_size). Total samples = $(n_samples).")

    is_correct = falses(n_samples)
    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)

        # Determine the syndromes, errors, and initial LLRs for the current batch.
        chunk_syndromes  = syndromes[:, start:stop]
        chunk_errors  = errors[:, start:stop]
        chunk_llrs  = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        # Predict the recoveries for the chunk of syndromes using the trained NeuralBP model.
        chunk_posterior_llrs = forward_pass_gpu(bpnn, chunk_llrs, chunk_syndromes)

        # Hard threshold the posterior LLRs to get proposed recoveries, and check if they correctly fix the errors.
        chunk_recoveries  = Array(chunk_posterior_llrs .< 0) # shape (n_bits, batch_size, n_layers)

        # Commit to the first syndrome-clearing layer, then score its logical coset.
        @views is_correct[start:stop] .= check_bp_solutions(parity_check_matrix, logicals, chunk_errors, chunk_recoveries)
    end

    return is_correct
end

"""
    predict_and_diagnose_neuralbp(bpnn, syndromes, errors; batch_size) -> NamedTuple

The diagnostic twin of `predict_and_check_neuralbp`: identical forward pass,
identical `gpu_active()` branch, identical batching and identical
`parity_check_matrix_dual` tail-slice for the logicals — but it scores each chunk
with `count_syndrome_satisfactions` instead of `check_bp_solutions`, so failures
are split into coset failures and convergence failures.

This is an ALTERNATIVE to `predict_and_check_neuralbp`, not an addition to it:
`neuralbp_test_predictions` calls one or the other, so a diagnostic run performs
exactly ONE forward pass, the same as a normal run.

`predict_and_check_neuralbp` deliberately still calls `check_bp_solutions`. The
two scoring implementations are kept independent so that asserting
`diagnosis.is_correct == check_bp_solutions(...)` is a genuine test of agreement.
"""
function predict_and_diagnose_neuralbp(
    bpnn::NeuralBP,
    syndromes::BitMatrix,
    errors::BitMatrix;
    batch_size::Int = 1024,
)::NamedTuple
    n_samples::Int = size(syndromes, 2)

    # Stabilizer checks H (commit) and logical operators L (score); the logical
    # rows are the tail of `parity_check_matrix_dual` past the H rows.
    parity_check_matrix::Matrix{Int} = convert.(Int, bpnn.base.parity_check_matrix)
    n_checks::Int = size(parity_check_matrix, 1)
    logicals::Matrix{Int} = convert.(Int, bpnn.base.parity_check_matrix_dual[n_checks + 1:end, :])

    if !gpu_active()
        posterior_llrs = forward_pass_with_weights(bpnn, repeat(bpnn.base.initial_llrs, 1, n_samples), syndromes)
        proposed_recoveries::Array{Bool, 3} = Array(posterior_llrs .< 0)
        diagnosis::NamedTuple = count_syndrome_satisfactions(parity_check_matrix, logicals, errors, proposed_recoveries)
        return diagnosis
    end

    print_info("Using GPU for predictions with batch size = $(batch_size). Total samples = $(n_samples). [diagnostic mode]")

    chunk_diagnoses::Vector{NamedTuple} = NamedTuple[]
    for start in 1:batch_size:n_samples
        stop::Int = min(start + batch_size - 1, n_samples)

        chunk_syndromes = syndromes[:, start:stop]
        chunk_errors = errors[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        chunk_posterior_llrs = forward_pass_gpu(bpnn, chunk_llrs, chunk_syndromes)
        chunk_recoveries::Array{Bool, 3} = Array(chunk_posterior_llrs .< 0)

        push!(chunk_diagnoses,
              count_syndrome_satisfactions(parity_check_matrix, logicals, chunk_errors, chunk_recoveries))
    end

    whole_run_diagnosis::NamedTuple = concatenate_diagnoses(chunk_diagnoses)
    return whole_run_diagnosis
end

"Environment variables searched for a GPU memory specification, in priority order,
paired with the label used to report which one was used."
const GPU_MEMORY_ENVIRONMENT_VARIABLES = Tuple{String, String}[
    ("GPU_MEMORY",        "ENV[\"GPU_MEMORY\"]"),
    ("SLURM_MEM_PER_GPU", "SLURM --mem-per-gpu"),
]

"Prediction batch size used when no memory specification is available anywhere.
This is the value that was hard-coded in this file before it became derivable."
const FALLBACK_PREDICTION_BATCH_SIZE = 16384

"""
    resolve_prediction_batch_size(bpnn; batch_size=0, gpu_memory="", default_batch_size=16384) -> Int

Decide how many samples to push through the GPU at once, WITHOUT editing source
and re-triggering precompilation. Resolution order, first hit wins:

  1. `batch_size > 0`                  — explicit override, used verbatim
  2. `gpu_memory`                      — `gpu_memory = "16G"` in the hyperparameters TOML
  3. `ENV["GPU_MEMORY"]`               — manual escape hatch, same string format
  4. `ENV["SLURM_MEM_PER_GPU"]`        — set automatically by `--mem-per-gpu`
  5. `default_batch_size`              — `FALLBACK_PREDICTION_BATCH_SIZE`, 16384

Step 4 is the useful one on the cluster: SLURM exports `--mem-per-gpu=16G` to the
job as `SLURM_MEM_PER_GPU=16384`, so the batch size tracks the allocation with no
configuration at all. The geometry (`n_bits`, `n_layers`, `nb_neurons`) is read
off the model, so a different code or depth re-sizes automatically.

A memory string that fails to parse is reported and falls back to
`default_batch_size` rather than aborting a run that is otherwise fine.
"""
function resolve_prediction_batch_size(
    bpnn::NeuralBP;
    batch_size::Int = 0,
    gpu_memory::AbstractString = "",
    default_batch_size::Int = FALLBACK_PREDICTION_BATCH_SIZE,
)::Int
    if batch_size > 0
        return batch_size
    end

    memory_specification::String = String(gpu_memory)
    specification_source::String = "hyperparameters `gpu_memory`"

    if isempty(memory_specification)
        for (environment_variable, source_label) in GPU_MEMORY_ENVIRONMENT_VARIABLES
            candidate_specification::String = get(ENV, environment_variable, "")
            if !isempty(candidate_specification)
                memory_specification = candidate_specification
                specification_source = source_label
                break
            end
        end
    end

    if isempty(memory_specification)
        return default_batch_size
    end

    # A malformed specification must not abort a run that is otherwise fine, so
    # fall back to the previous hard-coded value and say so loudly.
    memory_in_mb::Int = 0
    try
        memory_in_mb = parse_memory(memory_specification)
    catch parse_error
        @warn "Could not interpret the GPU memory specification $(repr(memory_specification)) " *
              "from $(specification_source): $(parse_error). " *
              "Falling back to batch_size = $(default_batch_size)."
        return default_batch_size
    end

    resolved_batch_size::Int = compute_optimal_batch_size_for(
        memory_in_mb;
        n_bits     = bpnn.base.code_n_bits,
        n_layers   = bpnn.base.n_layers,
        nb_neurons = bpnn.base.nb_neurons_per_layer,
    )
    print_info("Prediction batch size $(resolved_batch_size) derived from $(memory_in_mb) MB " *
               "($(memory_specification), via $(specification_source)).")
    return resolved_batch_size
end

function neuralbp_test_predictions(
    bpnn::NeuralBP,
    test_errors_file::String;
    batch_size::Int = 0,
    gpu_memory::AbstractString = "",
    diagnose::Bool = false,
)::Union{BitVector, NamedTuple}
    """
    Predict the recoveries for the given test syndromes using the trained Neural BP model.
    Test these predictions to see if they match the expected recoveries.

    `batch_size` / `gpu_memory` are passed to `resolve_prediction_batch_size`; the
    defaults reproduce the previous hard-coded 16384 when neither is supplied and
    no SLURM GPU-memory variable is set.

    `diagnose = false` (the default) returns the `BitVector` it always returned,
    via `predict_and_check_neuralbp`, along exactly the code path it always took.
    `diagnose = true` instead returns the full `count_syndrome_satisfactions`
    NamedTuple, which splits failures into coset failures and convergence
    failures. The two are alternatives, so a diagnostic run costs exactly one
    forward pass — the same as a normal run.
    """
    test_errors::BitMatrix = convert.(Bool, readdlm(test_errors_file, Int))
    test_syndromes::BitMatrix = convert.(Bool, mod.(bpnn.base.parity_check_matrix * test_errors, 2))

    # Separate the training block from the testing block on the console.
    print_console_rule()

    resolved_batch_size::Int = resolve_prediction_batch_size(
        bpnn; batch_size = batch_size, gpu_memory = gpu_memory
    )

    if diagnose
        diagnosis::NamedTuple = predict_and_diagnose_neuralbp(
            bpnn, test_syndromes, test_errors; batch_size = resolved_batch_size
        )
        return diagnosis
    end

    is_correct::BitVector = predict_and_check_neuralbp(
        bpnn, test_syndromes, test_errors; batch_size = resolved_batch_size
    )
    return is_correct
end
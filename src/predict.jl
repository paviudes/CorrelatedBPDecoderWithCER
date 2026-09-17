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

    gpu_state::Union{GPUState, Nothing} = nothing
    for start in 1:batch_size:n_total
        stop = min(start + batch_size - 1, n_total)
        chunk_synd = syndromes[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)
        gpu_state = reusable_gpu_state(gpu_state, bpnn, chunk_synd)
        @views predicted_recoveries[:, start:stop, :] .= predict_recoveries_gpu(gpu_state, chunk_llrs)
    end
    if gpu_state !== nothing
        release_gpu_state!(gpu_state)
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
`expts/misc/test_syndrome_diagnosis.jl`. That independence is worth more than
the speed here: this function is vectorised over samples and `check_bp_solutions`
is not, so the two now differ in implementation as well as in what they report,
which makes the agreement test stronger. It also means the NON-diagnostic path
is now the slow one on large test sets — use `--diagnose true`, or ask for the
same treatment there.

Arguments and shapes are identical to `check_bp_solutions`.
"""
function count_syndrome_satisfactions(
    parity_check_matrix::Matrix{Int},
    logicals::Matrix{Int},
    errors::BitMatrix,
    proposed_recoveries::Array{Bool, 3}
)::NamedTuple
    n_bits::Int = size(errors, 1)
    n_samples::Int = size(errors, 2)
    n_layers::Int = size(proposed_recoveries, 3)
    n_checks::Int = size(parity_check_matrix, 1)
    n_logicals::Int = size(logicals, 1)

    syndrome_cleared::BitVector = falses(n_samples)
    is_correct::BitVector = falses(n_samples)
    committed_layers::Vector{Int} = zeros(Int, n_samples)      # 0 = no layer ever cleared
    min_syndrome_weights::Vector{Int} = fill(typemax(Int), n_samples)
    # Weight of the TRUE error, carried per sample so the failure records can be
    # read against it: coset tipping should concentrate at weights where the
    # residual can reach a low-weight logical, convergence failure should not.
    error_weights::Vector{Int} = vec(sum(errors, dims = 1))

    # ONE LAYER AT A TIME, ALL SAMPLES AT ONCE. The per-sample version this
    # replaced did a (n_checks x n_bits) * (n_bits x n_layers) integer matmul
    # INSIDE a loop over samples: 2.35e11 scalar Int64 MACs and 62 GB of
    # allocation churn for a 10^6-sample run, single-threaded, which dominated
    # the whole decode (the GPU forward pass is ~30 s by comparison). Here the
    # same arithmetic is n_layers BLAS calls per chunk into reused Float32
    # buffers, with no allocation in the loop.
    #
    # Float32 is exact here and the result is identical, not approximate: H and
    # the residual are 0/1, so every dot product is an integer no larger than
    # n_bits, far inside Float32's exact-integer range, and `mod(x, 2)` of an
    # exactly represented small integer is exact.
    parity_check_float::Matrix{Float32} = Float32.(parity_check_matrix)
    logicals_float::Matrix{Float32} = Float32.(logicals)
    residual_layer::Matrix{Float32} = zeros(Float32, n_bits, n_samples)
    check_values::Matrix{Float32} = zeros(Float32, n_checks, n_samples)

    for layer in 1:n_layers
        @inbounds for sample in 1:n_samples
            for bit in 1:n_bits
                residual_layer[bit, sample] =
                    Float32(errors[bit, sample] ⊻ proposed_recoveries[bit, sample, layer])
            end
        end
        mul!(check_values, parity_check_float, residual_layer)
        @inbounds for sample in 1:n_samples
            layer_syndrome_weight::Int = 0
            for check in 1:n_checks
                layer_syndrome_weight += Int(mod(check_values[check, sample], 2.0f0))
            end
            if layer_syndrome_weight < min_syndrome_weights[sample]
                min_syndrome_weights[sample] = layer_syndrome_weight
            end
            # FIRST clearing layer only, exactly as `findfirst(==(0), ...)` did.
            if layer_syndrome_weight == 0 && committed_layers[sample] == 0
                committed_layers[sample] = layer
                syndrome_cleared[sample] = true
            end
        end
    end

    # Score ONLY the committed layer: success iff it is also logically trivial.
    # Gathered once for every sample, then a single matmul against the logicals.
    committed_residual::Matrix{Float32} = zeros(Float32, n_bits, n_samples)
    @inbounds for sample in 1:n_samples
        committed_layer::Int = committed_layers[sample]
        if committed_layer > 0
            for bit in 1:n_bits
                committed_residual[bit, sample] =
                    Float32(errors[bit, sample] ⊻ proposed_recoveries[bit, sample, committed_layer])
            end
        end
    end
    logical_values::Matrix{Float32} = logicals_float * committed_residual
    @inbounds for sample in 1:n_samples
        if committed_layers[sample] > 0
            logical_syndrome_weight::Int = 0
            for logical_row in 1:n_logicals
                logical_syndrome_weight += Int(mod(logical_values[logical_row, sample], 2.0f0))
            end
            is_correct[sample] = logical_syndrome_weight == 0
        end
    end

    # A run with no layers leaves the running minimum at its sentinel; report 0
    # rather than typemax so the column stays meaningful.
    @inbounds for sample in 1:n_samples
        if min_syndrome_weights[sample] == typemax(Int)
            min_syndrome_weights[sample] = 0
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

    # BOTH paths chunk. The CPU branch used to take `batch_size` and ignore it,
    # running the whole test set in one call and keeping every layer:
    #     Float32 n_bits x n_samples x n_layers  +  the Bool array of the same shape
    # which for 72 qubits x 10^6 samples x 90 layers is 24.1 GB + 6.0 GB = 30 GB
    # PER PROCESS. Three parallel jobs asked for ~90 GB and took the machine down
    # (2026-09-14). Chunking is exact here because `check_bp_solutions` is
    # per-sample, so the split cannot change any answer.
    use_gpu_forward::Bool = gpu_active()
    device_name::String = "CPU"
    if use_gpu_forward
        device_name = "GPU"
    end
    print_info("Using $(device_name) for predictions with batch size = $(batch_size). Total samples = $(n_samples).")

    is_correct = falses(n_samples)
    # Reused across chunks; see `reusable_gpu_state`.
    gpu_state::Union{GPUState, Nothing} = nothing
    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)

        # Determine the syndromes, errors, and initial LLRs for the current batch.
        chunk_syndromes  = syndromes[:, start:stop]
        chunk_errors  = errors[:, start:stop]
        chunk_llrs  = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        # Predict the recoveries for the chunk of syndromes using the trained NeuralBP model.
        # `gpu_active()` is false when USE_GPU=0 at runtime OR no GPU backend is
        # compiled in for this platform; the CPU forward pass is then used.
        chunk_recoveries = nothing
        if use_gpu_forward
            gpu_state = reusable_gpu_state(gpu_state, bpnn, chunk_syndromes)
            chunk_recoveries = predict_recoveries_gpu(gpu_state, chunk_llrs)
        else
            chunk_posterior_llrs = forward_pass_with_weights(bpnn, chunk_llrs, chunk_syndromes)
            chunk_recoveries = Array(chunk_posterior_llrs .< 0) # (n_bits, batch, n_layers)
        end

        # Commit to the first syndrome-clearing layer, then score its logical coset.
        @views is_correct[start:stop] .= check_bp_solutions(parity_check_matrix, logicals, chunk_errors, chunk_recoveries)
    end
    if gpu_state !== nothing
        release_gpu_state!(gpu_state)
    end

    return is_correct
end

"""
    reusable_gpu_state(gpu_state, bpnn, chunk_syndromes) -> GPUState

The `GPUState` to use for this chunk: the one passed in, repointed at the
chunk's syndromes, or a freshly built one when there is none yet or the chunk
size has changed (which happens for the final, short chunk).

Everything in the state except the syndromes is a function of the model, so
rebuilding it per chunk re-derives and re-uploads the per-layer weight tensor
every time — ~16 MB and a CPU-side scatter per chunk, 123 times in a 10^6-sample
run. The old state is released before a rebuild, so nothing accumulates.
"""
function reusable_gpu_state(
    gpu_state::Union{GPUState, Nothing},
    bpnn::NeuralBP,
    chunk_syndromes::BitMatrix
)::GPUState
    if gpu_state !== nothing && gpu_state.n_samples == size(chunk_syndromes, 2)
        update_gpu_state_syndromes!(gpu_state, bpnn.base, chunk_syndromes)
        return gpu_state
    end
    if gpu_state !== nothing
        release_gpu_state!(gpu_state)
    end
    rebuilt_state::GPUState = build_gpu_state(bpnn, chunk_syndromes)
    return rebuilt_state
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

    # Chunked on BOTH devices, for the memory reason documented on
    # `predict_and_check_neuralbp`. `concatenate_diagnoses` makes the split exact.
    use_gpu_forward::Bool = gpu_active()
    device_name::String = "CPU"
    if use_gpu_forward
        device_name = "GPU"
    end
    print_info("Using $(device_name) for predictions with batch size = $(batch_size). Total samples = $(n_samples). [diagnostic mode]")

    n_chunks::Int = cld(n_samples, batch_size)
    # Report every ~10% of the chunks. Without this a run that dies mid-way is
    # indistinguishable from one that dies on the first chunk — which is exactly
    # the difference between "the batch is too big" and "something accumulates
    # across chunks", and the enriched check node has been mistaken for both.
    chunk_report_interval::Int = max(1, cld(n_chunks, 10))
    chunk_diagnoses::Vector{NamedTuple} = NamedTuple[]
    chunk_index::Int = 0
    # Reused across chunks; see `reusable_gpu_state`.
    gpu_state::Union{GPUState, Nothing} = nothing
    for start in 1:batch_size:n_samples
        stop::Int = min(start + batch_size - 1, n_samples)
        chunk_index += 1
        if chunk_index % chunk_report_interval == 0 || chunk_index == 1
            print_info("  chunk $(chunk_index)/$(n_chunks) (samples $(start)-$(stop))")
        end

        chunk_syndromes = syndromes[:, start:stop]
        chunk_errors = errors[:, start:stop]
        chunk_llrs = repeat(bpnn.base.initial_llrs, 1, stop - start + 1)

        chunk_recoveries = nothing
        if use_gpu_forward
            gpu_state = reusable_gpu_state(gpu_state, bpnn, chunk_syndromes)
            chunk_recoveries = predict_recoveries_gpu(gpu_state, chunk_llrs)
        else
            chunk_posterior_llrs = forward_pass_with_weights(bpnn, chunk_llrs, chunk_syndromes)
            chunk_recoveries = Array(chunk_posterior_llrs .< 0)
        end

        push!(chunk_diagnoses,
              count_syndrome_satisfactions(parity_check_matrix, logicals, chunk_errors, chunk_recoveries))
    end
    if gpu_state !== nothing
        release_gpu_state!(gpu_state)
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

Steps 2-4 size against a real memory budget and account for the enriched check
node through `extra_bytes_per_sample`. Step 5 has no budget to work from, so for
the enriched check node it is capped by `cap_batch_size_for_enriched_kernel`.
Step 1 is neither sized nor capped — an explicit request is used verbatim.

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
    # An explicit request is honoured as given: the caller has said what they want.
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

    # No budget was given anywhere, so the fallback applies — but the fallback
    # was chosen for the standard check node and knows nothing about the
    # enriched kernel's per-sample tensors. Cap it (a no-op for `tanh`).
    if isempty(memory_specification)
        capped_default_batch_size::Int = cap_batch_size_for_enriched_kernel(
            default_batch_size, bpnn.base.soft_check_tables
        )
        if capped_default_batch_size < default_batch_size
            print_info("Prediction batch size $(capped_default_batch_size) (capped from " *
                       "$(default_batch_size)): the enriched check node needs " *
                       "$(enriched_kernel_bytes_per_sample(bpnn.base.soft_check_tables)) bytes " *
                       "per sample. Set `prediction_batch_size` or `gpu_memory` to override.")
        end
        return capped_default_batch_size
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
        # Zero for the standard check node (empty tables).
        extra_bytes_per_sample = enriched_kernel_bytes_per_sample(bpnn.base.soft_check_tables),
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
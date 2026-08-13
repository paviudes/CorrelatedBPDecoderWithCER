# ============================================================================
# collect_gate_cer.jl — gather every diagnostic from the gate x CER sweep
# ============================================================================
# Invoked by `bash misc/sweep_gate_cer.sh --collect`; runnable directly as
#
#     julia --project="./../" misc/collect_gate_cer.jl [results_dir] [--outdir DIR]
#
# WRITES THREE FILES (nothing is aggregated away that cannot be recovered):
#
#   gate_cer_per_run.csv    one row per run — every column of the results CSV,
#                           plus failure-mode counts and rates, the error-weight
#                           and min-syndrome-weight profiles of each failure
#                           kind, the committed-layer profile, and whatever the
#                           debug log recorded (gate_open_fraction, realised
#                           per-epoch lambda/sparsity, correlation_penalty
#                           statistics).
#
#   gate_cer_per_arm.csv    one row per arm — mean, sd, min, max across seeds
#                           for each headline quantity. The sd IS the error bar
#                           every comparison in this project has to clear; a
#                           previously measured spread on provably identical
#                           configurations was 309 vs 620 failures (2.0x).
#
#   gate_cer_contrasts.csv  the four contrasts the sweep exists to resolve,
#                           each as a paired per-seed difference (mean, sd, t)
#                           AND a pooled two-proportion z:
#                             cer_ungated  vs nocer_ungated  (the original claim)
#                             cer_gated    vs nocer_gated    (does gating change it?)
#                             cer_ungated  vs cer_gated      (gate effect, CER arm)
#                             nocer_ungated vs nocer_gated   (gate effect, control)
#
# READING IT. The last two contrasts are the point. Convergence failures falling
# in `cer_ungated -> cer_gated` but NOT in `nocer_ungated -> nocer_gated` means
# the ungated correlation reward was obstructing syndrome clearing. Falling in
# both means gating helps generally — most plausibly through sparsity, which
# carries ~400x the across-layer weight of the correlation term.
# ============================================================================

using CSV
using DataFrames
using Printf
using Statistics

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "..", "..", "data",
                                     "72q_BB_cycles_1_debug", "results")

# `..._trained_using_<source>_s_<n>[_no_cer]_gc_<arm>_<gate>_seed_<k>.csv`
const RUN_PATTERN = r"_gc_(cer|nocer)_(ungated|gated)_seed_(\d+)\.csv$"

"""
    parse_run(filename) -> Union{NamedTuple, Nothing}

Pull (arm, gate, seed) out of a result filename, or `nothing` if it is not one
of this sweep's files.
"""
function parse_run(filename::String)::Union{NamedTuple, Nothing}
    filename_match::Union{RegexMatch, Nothing} = match(RUN_PATTERN, filename)
    if filename_match === nothing
        return nothing
    end
    run_key::NamedTuple = (
        arm = String(filename_match.captures[1]),
        gate = String(filename_match.captures[2]),
        seed = parse(Int, filename_match.captures[3]),
    )
    return run_key
end

"""
    rows_to_dataframe(rows) -> DataFrame

Build a DataFrame from a vector of `Dict{Symbol, Any}` rows whose key sets may
differ, filling absent entries with `missing`.

Done explicitly rather than via `push!(df, dict; cols = :union)` because that
path's behaviour on an initially-empty DataFrame varies between DataFrames.jl
versions, and this script has to run unattended on the cluster.
"""
function rows_to_dataframe(rows::Vector{Dict{Symbol, Any}})::DataFrame
    if isempty(rows)
        return DataFrame()
    end
    column_names::Vector{Symbol} = Symbol[]
    for row in rows
        for key in keys(row)
            if !(key in column_names)
                push!(column_names, key)
            end
        end
    end
    frame::DataFrame = DataFrame()
    for name in column_names
        frame[!, name] = Any[get(row, name, missing) for row in rows]
    end
    return frame
end

"Suffix identifying one run, used to pair the three result files together."
function run_suffix(arm::String, gate::String, seed::Int)::String
    suffix::String = "_gc_$(arm)_$(gate)_seed_$(seed).csv"
    return suffix
end

"""
    summarise_by_kind(failures, kind, column) -> NamedTuple

n / mean / median / max of `column` over the failure rows of a given kind.
Empty-safe: returns zeros and NaNs rather than throwing on an absent kind.
"""
function summarise_by_kind(failures::DataFrame, kind::String, column::Symbol)::NamedTuple
    selected::DataFrame = failures[failures.failure_kind .== kind, :]
    if nrow(selected) == 0
        return (n = 0, mean = NaN, median = NaN, max = NaN)
    end
    values::Vector{Float64} = Float64.(selected[!, column])
    stats::NamedTuple = (n = nrow(selected), mean = mean(values),
                         median = median(values), max = maximum(values))
    return stats
end

"""
    debug_log_summary(logs_dir, arm, gate, seed) -> Dict{Symbol, Any}

Whatever the `--isdebug true` TRAINING logs recorded for this run.

WHY THIS READS `logs/` AND NOT `results/`. The two directories hold different
halves of the diagnostics:

    results/  TEST-side  — failure counts, coset vs convergence, per-sample
                           failure rows, committed-layer profile
    logs/     TRAIN-side — gate_open_fraction, correlation_penalty, and the
                           REALISED per-epoch correlation_weight / sparsity

`gate_open_fraction` exists nowhere else, and it is the number that decides
whether the "gated" arms measured "aux terms applied conditionally" or merely
"aux terms off" — so it cannot be skipped. Written by
`save_training_debug_logs` (src/train.jl) to `<codename>/logs/`.

Entirely optional: every field defaults to NaN and `debug_log_found = false`, so
a sweep run without `--isdebug`, or one whose logs were not staged out, still
collects the full test-side picture. Any error reading a log is warned about and
swallowed rather than killing the collection.
"""
function debug_log_summary(logs_dir::String, arm::String, gate::String, seed::Int)::Dict{Symbol, Any}
    summary::Dict{Symbol, Any} = Dict(
        :gate_open_fraction_mean => NaN, :gate_open_fraction_min => NaN,
        :gate_open_fraction_max => NaN,
        :correlation_penalty_mean => NaN, :correlation_penalty_min => NaN,
        :correlation_penalty_frac_zero => NaN,
        :lambda_final_epoch => NaN, :sparsity_final_epoch => NaN,
        # `gate_threshold_logged` is the CONSISTENCY CHECK: a log matched to a
        # gated run must report tau > 0, an ungated one -1.0. A mismatch means the
        # log-to-run matching is wrong and no train-side column can be trusted.
        :gate_threshold_logged => NaN,
        :debug_log_written_rows => 0, :debug_log_total_rows => 0,
        :debug_log_found => false,
    )
    if !isdir(logs_dir)
        return summary
    end

    # Plain suffix matching, not a constructed Regex: the run tail is a literal
    # string and the two filenames are disjoint (one ends `<tail>.csv`, the other
    # `<tail>_individual_losses.csv`).
    run_tail::String = "_gc_$(arm)_$(gate)_seed_$(seed)"
    log_files::Vector{String} = readdir(logs_dir)
    hyperparameter_files::Vector{String} = filter(
        f -> startswith(f, "debugging_") && endswith(f, run_tail * ".csv"), log_files)
    loss_files::Vector{String} = filter(
        f -> startswith(f, "debugging_") && endswith(f, run_tail * "_individual_losses.csv"), log_files)

    try
        if length(hyperparameter_files) > 1
            @warn "more than one hyperparameter log matches $(run_tail): " *
                  "$(hyperparameter_files). Using the first; log-to-run matching is ambiguous."
        end
        if !isempty(hyperparameter_files)
            summary[:debug_log_found] = true
            hyperparameters::DataFrame = CSV.read(joinpath(logs_dir, hyperparameter_files[1]), DataFrame)

            # THE ROWS ARE PRE-ALLOCATED. `init_training_debug_logs` (src/train.jl)
            # zero-fills n_samples_to_log rows and only the batches actually logged
            # are written, so the TRAILING rows are all zeros. Taking `last()` of a
            # column therefore returns 0.0 for any run that did not happen to fill
            # every row — which is exactly what made seeds within one arm appear to
            # have trained under different schedules. Read the row with the highest
            # WRITTEN epoch instead.
            written_rows::Int = 0
            last_written_index::Int = 0
            if hasproperty(hyperparameters, :epoch)
                highest_epoch::Float64 = 0.0
                for (index, value) in enumerate(hyperparameters[!, :epoch])
                    epoch::Union{Float64, Nothing} = value === missing ? nothing :
                                                     tryparse(Float64, string(value))
                    if epoch === nothing || epoch <= 0
                        continue
                    end
                    written_rows += 1
                    if epoch >= highest_epoch
                        highest_epoch = epoch
                        last_written_index = index
                    end
                end
            end
            summary[:debug_log_written_rows] = written_rows
            summary[:debug_log_total_rows] = nrow(hyperparameters)

            if last_written_index > 0
                for (column, key) in ((:correlation_weight, :lambda_final_epoch),
                                      (:sparsity_importance, :sparsity_final_epoch),
                                      (:syndrome_gate_threshold, :gate_threshold_logged))
                    if !hasproperty(hyperparameters, column)
                        continue
                    end
                    value = hyperparameters[last_written_index, column]
                    parsed::Union{Float64, Nothing} = value === missing ? nothing :
                                                      tryparse(Float64, string(value))
                    if parsed !== nothing
                        summary[key] = parsed
                    end
                end
            end
        end

        if !isempty(loss_files)
            losses::DataFrame = CSV.read(joinpath(logs_dir, loss_files[1]), DataFrame)
            if hasproperty(losses, :gate_open_fraction)
                gate_values::Vector{Float64} = parse_packed_floats(losses[!, :gate_open_fraction])
                if !isempty(gate_values)
                    summary[:gate_open_fraction_mean] = mean(gate_values)
                    summary[:gate_open_fraction_min] = minimum(gate_values)
                    summary[:gate_open_fraction_max] = maximum(gate_values)
                end
            end
            if hasproperty(losses, :correlation_penalty)
                penalty_values::Vector{Float64} = parse_packed_floats(losses[!, :correlation_penalty])
                if !isempty(penalty_values)
                    summary[:correlation_penalty_mean] = mean(penalty_values)
                    summary[:correlation_penalty_min] = minimum(penalty_values)
                    summary[:correlation_penalty_frac_zero] =
                        count(iszero, penalty_values) / length(penalty_values)
                end
            end
        end
    catch log_error
        @warn "could not read the debug log for $(arm)/$(gate)/seed $(seed): $(log_error). " *
              "Test-side diagnostics are unaffected; gate_open_fraction will be missing."
    end
    return summary
end

"""
    parse_packed_floats(column) -> Vector{Float64}

The debug log stores one per-LAYER array per row, serialised as a string. Flatten
every row's array into one vector, skipping anything unparseable.
"""
function parse_packed_floats(column)::Vector{Float64}
    values::Vector{Float64} = Float64[]
    for entry in column
        if entry === missing
            continue
        end
        text::String = strip(string(entry))
        if isempty(text)
            continue
        end
        for token in split(replace(text, "[" => "", "]" => ""), ",")
            trimmed::String = strip(token)
            if isempty(trimmed)
                continue
            end
            parsed::Union{Float64, Nothing} = tryparse(Float64, trimmed)
            if parsed !== nothing
                push!(values, parsed)
            end
        end
    end
    return values
end

function collect_per_run(results_dir::String, logs_dir::String)::DataFrame
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for filename in sort(readdir(results_dir))
        if !startswith(filename, "simulation_results_")
            continue
        end
        run_key::Union{NamedTuple, Nothing} = parse_run(filename)
        if run_key === nothing
            continue
        end
        (arm, gate, seed) = (run_key.arm, run_key.gate, run_key.seed)
        suffix::String = run_suffix(arm, gate, seed)

        results_row = first(CSV.File(joinpath(results_dir, filename)))
        row::Dict{Symbol, Any} = Dict(
            :arm => arm, :gate => gate, :seed => seed,
            :label => "$(arm)_$(gate)",
        )
        # Every column of the results CSV, unchanged.
        for name in propertynames(results_row)
            row[name] = getproperty(results_row, name)
        end

        n_samples::Int = Int(results_row.num_samples_per_error_rate)
        n_failures::Int = Int(results_row.num_failures)
        n_coset::Int = Int(results_row.num_coset_failures)
        n_convergence::Int = Int(results_row.num_convergence_failures)
        n_cleared::Int = Int(results_row.num_syndrome_cleared)
        row[:logical_error_rate] = n_failures / n_samples
        row[:coset_rate] = n_coset / n_samples
        row[:convergence_rate] = n_convergence / n_samples
        row[:coset_rate_given_cleared] = n_cleared > 0 ? n_coset / n_cleared : NaN
        row[:convergence_share_of_failures] = n_failures > 0 ? n_convergence / n_failures : NaN

        # ---- per-sample failure profile -----------------------------------
        failures_path::String = joinpath(results_dir, "per_sample_failures_" *
                                         split(filename, "simulation_results_")[2])
        if isfile(failures_path)
            failures::DataFrame = CSV.read(failures_path, DataFrame)
            row[:n_failure_rows] = nrow(failures)
            for kind in ("coset", "convergence")
                for (column, short) in ((:error_weight, "errw"),
                                        (:min_syndrome_weight, "minsyn"),
                                        (:committed_layer, "layer"))
                    s = summarise_by_kind(failures, kind, column)
                    row[Symbol("$(kind)_$(short)_mean")] = s.mean
                    row[Symbol("$(kind)_$(short)_median")] = s.median
                    row[Symbol("$(kind)_$(short)_max")] = s.max
                end
            end
            # Convergence failures at min_syndrome_weight == 3 are one uncorrected
            # qubit away from clearing (HZ has column weight 3) — near-misses an
            # OSD step would mop up. Worth tracking separately.
            convergence_rows::DataFrame = failures[failures.failure_kind .== "convergence", :]
            near_miss::Int = nrow(convergence_rows) == 0 ? 0 :
                             count(==(3), convergence_rows.min_syndrome_weight)
            row[:convergence_near_miss_w3] = near_miss
            row[:convergence_near_miss_frac] = nrow(convergence_rows) > 0 ?
                                               near_miss / nrow(convergence_rows) : NaN
        end

        # ---- committed-layer profile ---------------------------------------
        profile_path::String = joinpath(results_dir, "layer_profile_" *
                                        split(filename, "simulation_results_")[2])
        if isfile(profile_path)
            profile::DataFrame = CSV.read(profile_path, DataFrame)
            committed_total::Vector{Int} = profile.n_correct .+ profile.n_coset_failures
            total_committed::Int = sum(committed_total)
            row[:n_layers_profiled] = nrow(profile)
            if total_committed > 0
                row[:layer1_clearing_fraction] = committed_total[1] / total_committed
                row[:beyond_layer1_fraction] = 1 - committed_total[1] / total_committed
                row[:mean_committed_layer_from_profile] =
                    sum(profile.committed_layer .* committed_total) / total_committed
            end
        end

        # ---- training-time debug log ---------------------------------------
        debug_summary::Dict{Symbol, Any} = debug_log_summary(logs_dir, arm, gate, seed)
        for (key, value) in debug_summary
            row[key] = value
        end
        # Assert the log belongs to this run. Expected tau is 0.5-ish for a gated
        # arm and -1.0 for an ungated one; anything else means the matching broke.
        expected_gate_sign::Float64 = gate == "gated" ? 1.0 : -1.0
        logged_gate = debug_summary[:gate_threshold_logged]
        row[:gate_threshold_consistent] = missing
        if !(logged_gate isa Number) || isnan(logged_gate)
            row[:gate_threshold_consistent] = missing
        else
            row[:gate_threshold_consistent] = sign(logged_gate) == expected_gate_sign
            if sign(logged_gate) != expected_gate_sign
                @warn "log/run MISMATCH for $(arm)/$(gate)/seed $(seed): the log reports " *
                      "syndrome_gate_threshold = $(logged_gate), which contradicts the arm. " *
                      "Train-side columns for this run are not trustworthy."
            end
        end

        push!(collected_rows, row)
    end
    per_run::DataFrame = rows_to_dataframe(collected_rows)
    if nrow(per_run) > 0
        sort!(per_run, [:arm, :gate, :seed])
    end
    return per_run
end

"Mean / sd / min / max across seeds, per arm."
function collect_per_arm(per_run::DataFrame)::DataFrame
    quantities::Vector{Symbol} = [
        :num_failures, :num_coset_failures, :num_convergence_failures,
        :logical_error_rate, :coset_rate, :convergence_rate,
        :coset_rate_given_cleared, :convergence_share_of_failures,
        :mean_committed_layer, :layer1_clearing_fraction,
        :convergence_near_miss_frac, :gate_open_fraction_mean, :runtime,
    ]
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for group in groupby(per_run, :label)
        row::Dict{Symbol, Any} = Dict(:label => group.label[1],
                                      :arm => group.arm[1],
                                      :gate => group.gate[1],
                                      :n_seeds => nrow(group))
        for quantity in quantities
            if !hasproperty(group, quantity)
                continue
            end
            values::Vector{Float64} = Float64.(collect(skipmissing(group[!, quantity])))
            values = filter(!isnan, values)
            if isempty(values)
                continue
            end
            row[Symbol("$(quantity)_mean")] = mean(values)
            row[Symbol("$(quantity)_sd")] = length(values) > 1 ? std(values) : 0.0
            row[Symbol("$(quantity)_min")] = minimum(values)
            row[Symbol("$(quantity)_max")] = maximum(values)
        end
        push!(collected_rows, row)
    end
    per_arm::DataFrame = rows_to_dataframe(collected_rows)
    if nrow(per_arm) > 0
        sort!(per_arm, :label)
    end
    return per_arm
end

"""
    contrast(per_run, label_a, label_b) -> Dict

Paired-by-seed difference (a - b) for the three failure counts, plus a pooled
two-proportion z on the summed counts.

The PAIRED statistic is the honest one: it uses the seed as a block, so it
measures the configuration effect against the seed-to-seed spread. The pooled z
uses only test-set uncertainty and will look far more significant than the
evidence warrants — it is reported alongside precisely so the two can be
compared.
"""
function contrast(per_run::DataFrame, label_a::String, label_b::String)::Dict{Symbol, Any}
    a::DataFrame = per_run[per_run.label .== label_a, :]
    b::DataFrame = per_run[per_run.label .== label_b, :]
    shared_seeds::Vector{Int} = sort(intersect(Set(a.seed), Set(b.seed)) |> collect)
    out::Dict{Symbol, Any} = Dict(:contrast => "$(label_a) - $(label_b)",
                                  :n_paired_seeds => length(shared_seeds))
    if isempty(shared_seeds)
        return out
    end
    for quantity in (:num_failures, :num_coset_failures, :num_convergence_failures)
        differences::Vector{Float64} = Float64[]
        for seed in shared_seeds
            va = a[a.seed .== seed, quantity][1]
            vb = b[b.seed .== seed, quantity][1]
            push!(differences, Float64(va) - Float64(vb))
        end
        out[Symbol("$(quantity)_paired_mean")] = mean(differences)
        out[Symbol("$(quantity)_paired_sd")] = length(differences) > 1 ? std(differences) : 0.0
        standard_error::Float64 = length(differences) > 1 ?
            std(differences) / sqrt(length(differences)) : 0.0
        out[Symbol("$(quantity)_paired_t")] = standard_error > 0 ?
            mean(differences) / standard_error : NaN

        total_a::Int = sum(a[in.(a.seed, Ref(shared_seeds)), quantity])
        total_b::Int = sum(b[in.(b.seed, Ref(shared_seeds)), quantity])
        trials::Int = sum(a[in.(a.seed, Ref(shared_seeds)), :num_samples_per_error_rate])
        pooled::Float64 = (total_a + total_b) / (2 * trials)
        se_pooled::Float64 = sqrt(pooled * (1 - pooled) * 2 / trials)
        out[Symbol("$(quantity)_pooled_a")] = total_a
        out[Symbol("$(quantity)_pooled_b")] = total_b
        out[Symbol("$(quantity)_pooled_z")] = se_pooled > 0 ?
            (total_a / trials - total_b / trials) / se_pooled : NaN
    end
    return out
end

function collect_contrasts(per_run::DataFrame)::DataFrame
    pairs = [("cer_ungated", "nocer_ungated"), ("cer_gated", "nocer_gated"),
             ("cer_ungated", "cer_gated"), ("nocer_ungated", "nocer_gated")]
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for (a, b) in pairs
        if !(a in per_run.label) || !(b in per_run.label)
            continue
        end
        push!(collected_rows, contrast(per_run, a, b))
    end
    contrasts::DataFrame = rows_to_dataframe(collected_rows)
    return contrasts
end

function print_report(per_arm::DataFrame, contrasts::DataFrame)::Nothing
    println(repeat("=", 96))
    println("PER ARM (mean +- sd across seeds)")
    println(repeat("-", 96))
    @printf("%-16s %5s %16s %16s %18s %10s\n",
            "arm", "seeds", "failures", "coset", "convergence", "gate open")
    for row in eachrow(per_arm)
        gate_open = NaN
        if hasproperty(per_arm, :gate_open_fraction_mean_mean) &&
           row.gate_open_fraction_mean_mean !== missing
            gate_open = row.gate_open_fraction_mean_mean
        end
        @printf("%-16s %5d %8.1f+-%-6.1f %8.1f+-%-6.1f %10.1f+-%-6.1f %10.3f\n",
                row.label, row.n_seeds,
                row.num_failures_mean, row.num_failures_sd,
                row.num_coset_failures_mean, row.num_coset_failures_sd,
                row.num_convergence_failures_mean, row.num_convergence_failures_sd,
                gate_open)
    end
    println(repeat("=", 96))
    println("\nCONTRASTS (paired by seed; t against the seed spread, z against test-set only)")
    println(repeat("-", 96))
    for row in eachrow(contrasts)
        @printf("  %-32s  n=%d\n", row.contrast, row.n_paired_seeds)
        for (quantity, name) in ((:num_convergence_failures, "convergence"),
                                 (:num_coset_failures, "coset"),
                                 (:num_failures, "total"))
            m = row[Symbol("$(quantity)_paired_mean")]
            s = row[Symbol("$(quantity)_paired_sd")]
            t = row[Symbol("$(quantity)_paired_t")]
            z = row[Symbol("$(quantity)_pooled_z")]
            @printf("      %-12s paired %+9.1f +- %-8.1f  t = %+6.2f      pooled z = %+7.2f\n",
                    name, m, s, t, z)
        end
        println()
    end
    println("  t is the statistic that matters: it measures the configuration effect")
    println("  against the seed-to-seed spread. z uses test-set uncertainty only and")
    println("  will overstate significance — two networks always differ.")
    return nothing
end

"""
    main(arguments) -> Nothing

Entry point. Deliberately a FUNCTION rather than a bare `if` block: at top-level
script scope a `while` loop opens a SOFT scope, so `argument_index += 1` inside
it is treated as a new local and the script dies with
`UndefVarError: argument_index not defined in local scope`. Inside a function
everything is hard local scope and the loop behaves as written.
"""
function main(arguments::Vector{String})::Nothing
    results_dir::String = DEFAULT_RESULTS_DIR
    output_dir::String = ""
    argument_index::Int = 1
    while argument_index <= length(arguments)
        if arguments[argument_index] == "--outdir"
            output_dir = arguments[argument_index + 1]
            argument_index += 2
        else
            results_dir = arguments[argument_index]
            argument_index += 1
        end
    end
    if !isdir(results_dir)
        error("results directory not found: $(results_dir)")
    end
    if isempty(output_dir)
        # Outputs land next to the inputs, i.e. data/<codename>/results/.
        output_dir = results_dir
    end
    logs_dir::String = normpath(joinpath(results_dir, "..", "logs"))

    # results/ holds the TEST-side diagnostics; logs/ holds the TRAIN-side ones
    # (gate_open_fraction above all). logs/ is optional — say so plainly rather
    # than letting a silent NaN column look like a measurement.
    n_debug_logs::Int = 0
    if isdir(logs_dir)
        n_debug_logs = count(f -> startswith(f, "debugging_") && endswith(f, ".csv"),
                             readdir(logs_dir))
    end
    println("[collect] results (test-side) : $(results_dir)")
    println("[collect] logs    (train-side): $(logs_dir)  — $(n_debug_logs) debug file(s)")
    if n_debug_logs == 0
        println("          none found: gate_open_fraction, correlation_penalty and the")
        println("          realised per-epoch lambda/sparsity will be blank. Everything")
        println("          else comes from results/ and is unaffected.")
    end
    println()

    per_run::DataFrame = collect_per_run(results_dir, logs_dir)
    if nrow(per_run) == 0
        error("no gate x CER runs found in $(results_dir) " *
              "(expecting names ending _gc_<arm>_<gate>_seed_<n>.csv)")
    end
    per_arm::DataFrame = collect_per_arm(per_run)
    contrasts::DataFrame = collect_contrasts(per_run)

    print_report(per_arm, contrasts)

    CSV.write(joinpath(output_dir, "gate_cer_per_run.csv"), per_run)
    CSV.write(joinpath(output_dir, "gate_cer_per_arm.csv"), per_arm)
    CSV.write(joinpath(output_dir, "gate_cer_contrasts.csv"), contrasts)
    println("\n  wrote $(joinpath(output_dir, "gate_cer_per_run.csv"))   ($(nrow(per_run)) rows x $(ncol(per_run)) cols)")
    println("  wrote $(joinpath(output_dir, "gate_cer_per_arm.csv"))   ($(nrow(per_arm)) rows)")
    println("  wrote $(joinpath(output_dir, "gate_cer_contrasts.csv"))   ($(nrow(contrasts)) rows)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

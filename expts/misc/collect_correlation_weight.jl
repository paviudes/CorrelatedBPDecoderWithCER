# ============================================================================
# collect_correlation_weight.jl — summarise the CER vs no-CER sweep across p
# ============================================================================
# Invoked by `bash misc/sweep_correlation_weight.sh --collect`; runnable directly:
#
#     julia --project="./../" misc/collect_correlation_weight.jl [results_dir] [--outdir DIR]
#
# WRITES THREE FILES:
#
#   correlation_weight_per_run.csv   one row per (p, arm, seed) — every column of
#                                      the results CSV, plus failure-mode counts
#                                      and rates, the error-weight / min-syndrome
#                                      -weight / committed-layer profile of each
#                                      failure kind, and the train-side debug log
#                                      (gate_open_fraction, realised per-epoch
#                                      lambda and sparsity, correlation_penalty).
#
#   correlation_weight_per_arm.csv   one row per (p, arm) — mean, sd, min, max
#                                      across seeds. The sd IS the error bar every
#                                      claim here has to clear: a previously
#                                      measured spread on provably identical
#                                      configurations was 309 vs 620 failures.
#
#   correlation_weight_contrasts.csv one row per p — the paired (cer - nocer)
#                                      difference, as a per-seed paired t AND a
#                                      pooled two-proportion z.
#
# HOW TO READ IT. The p axis is the discriminator, and the prediction is sharp.
# The correlation term rewards CO-FLIPPED pairs, whose expected count per sample
# is |C| * P11 ~ p^2, so the term is roughly (1.9e-3 / 5e-4)^2 ~ 14x stronger at
# the top of the p range than at the bottom.
#
#   gap grows with p          couplings are doing the work — the p^2 scaling
#                             predicts exactly this shape
#   gap flat across p         it is the single-qubit PRIORS, not the couplings;
#                             confirm by rerunning with --lambda 0
#   no gap anywhere           the revised J convention did not rescue it either;
#                             the 1/|C| normalisation is the remaining suspect
#   CER worse, worst at big p the reward is outrunning the syndrome term; check
#                             gate_open_fraction, then lower --lambda
#
# TWO PREMISE CHECKS, printed loudly rather than buried in a column, because each
# can turn a "null result" into a measurement artefact:
#
#   sparsity_final_epoch  must be 0. The sweep is predicated on it, and it is
#                         read back from the TRAINING LOG rather than trusted
#                         from the TOML. Non-zero invalidates the run.
#
#   gate_open_fraction    must be strictly between 0 and 1 on a gated arm. At 0
#                         the gate never opened, the aux terms never fired, and
#                         the arm silently measured "base loss only" — a null
#                         that says nothing about the couplings. At 1 the gate
#                         never closed and the arm is effectively ungated.
# ============================================================================

using CSV
using DataFrames
using Printf
using Statistics

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "..", "..", "data",
                                     "72q_BB_cycles_1_debug", "results")

# `..._trained_using_train_p_<p>_s_<n>[_no_cer]_cw<arm>_<gate>_sp<tag>_seed_<k>.csv`
const RUN_PATTERN = r"_train_p_([0-9.eE+-]+)_s_(\d+)(_no_cer)?_cw(cer|nocer)_(ungated|gated)_sp([0-9p]+)_seed_(\d+)\.csv$"

"""
    parse_run(filename) -> Union{NamedTuple, Nothing}

Pull (p, arm, gate, sparsity_tag, seed) out of a result filename, or `nothing`
if it is not one of this sweep's files.

The `_no_cer` group is captured but NOT used to decide the arm: the arm comes
from the explicit `cw<arm>` token in the run tag. Capturing it anyway lets the
cross-check below fire if the two ever disagree, which would mean `use_CER` and
the run tag had drifted apart in a generated TOML.
"""
function parse_run(filename::String)::Union{NamedTuple, Nothing}
    filename_match::Union{RegexMatch, Nothing} = match(RUN_PATTERN, filename)
    if filename_match === nothing
        return nothing
    end
    arm::String = String(filename_match.captures[4])
    has_no_cer_tag::Bool = filename_match.captures[3] !== nothing
    if has_no_cer_tag != (arm == "nocer")
        @warn "filename $(filename) has the `_no_cer` tag = $(has_no_cer_tag) but the " *
              "run tag says arm = $(arm). use_CER and run_tag have drifted apart; " *
              "this run's arm label cannot be trusted."
    end
    run_key::NamedTuple = (
        p = String(filename_match.captures[1]),
        data_seed = parse(Int, filename_match.captures[2]),
        arm = arm,
        gate = String(filename_match.captures[5]),
        sparsity_tag = String(filename_match.captures[6]),
        seed = parse(Int, filename_match.captures[7]),
    )
    return run_key
end

"""
    rows_to_dataframe(rows) -> DataFrame

Build a DataFrame from a vector of `Dict{Symbol, Any}` rows whose key sets may
differ, filling absent entries with `missing`. Done explicitly rather than via
`push!(df, dict; cols = :union)` because that path's behaviour on an initially
empty DataFrame varies between DataFrames.jl versions, and this has to run
unattended on the cluster.
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

"""
    debug_log_summary(logs_dir, run_tail) -> Dict{Symbol, Any}

Whatever the `--isdebug true` TRAINING logs recorded for this run.

WHY THIS READS `logs/` AND NOT `results/`. The two directories hold different
halves of the diagnostics:

    results/  TEST-side  — failure counts, coset vs convergence, per-sample
                           failure rows, committed-layer profile
    logs/     TRAIN-side — gate_open_fraction, correlation_penalty, and the
                           REALISED per-epoch correlation_weight / sparsity

`sparsity_final_epoch` is what verifies the premise of this entire sweep, so it
cannot be skipped. Everything defaults to NaN with `debug_log_found = false`, so
a run without `--isdebug` still collects the full test-side picture.
"""
function debug_log_summary(logs_dir::String, run_tail::String)::Dict{Symbol, Any}
    summary::Dict{Symbol, Any} = Dict(
        :gate_open_fraction_mean => NaN, :gate_open_fraction_min => NaN,
        :gate_open_fraction_max => NaN,
        :correlation_penalty_mean => NaN, :correlation_penalty_min => NaN,
        :correlation_penalty_frac_zero => NaN,
        :lambda_final_epoch => NaN, :sparsity_final_epoch => NaN,
        :gate_threshold_logged => NaN,
        :debug_log_written_rows => 0, :debug_log_total_rows => 0,
        :debug_log_found => false,
    )
    if !isdir(logs_dir)
        return summary
    end

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
            # column returns 0.0 for any run that did not fill every row — which is
            # exactly what once made seeds within one arm appear to have trained
            # under different schedules. Read the highest WRITTEN epoch instead.
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
        @warn "could not read the debug log for $(run_tail): $(log_error). " *
              "Test-side diagnostics are unaffected."
    end
    return summary
end

"""
    collect_per_run(results_dir, logs_dir) -> DataFrame

One row per result file, carrying every column of the results CSV plus the
per-sample, layer-profile and train-side diagnostics.
"""
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
        run_stem::String = split(filename, "simulation_results_")[2]

        row::Dict{Symbol, Any} = Dict(
            :p => run_key.p,
            :p_numeric => something(tryparse(Float64, run_key.p), NaN),
            :arm => run_key.arm,
            :gate => run_key.gate,
            :sparsity_tag => run_key.sparsity_tag,
            :seed => run_key.seed,
            :label => "$(run_key.arm)_p$(run_key.p)",
        )
        results_row = first(CSV.File(joinpath(results_dir, filename)))
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

        # ---- per-sample failure profile -------------------------------------
        failures_path::String = joinpath(results_dir, "per_sample_failures_" * run_stem)
        if isfile(failures_path)
            failures::DataFrame = CSV.read(failures_path, DataFrame)
            row[:n_failure_rows] = nrow(failures)
            for kind in ("coset", "convergence")
                for (column, short) in ((:error_weight, "errw"),
                                        (:min_syndrome_weight, "minsyn"),
                                        (:committed_layer, "layer"))
                    statistics::NamedTuple = summarise_by_kind(failures, kind, column)
                    row[Symbol("$(kind)_$(short)_mean")] = statistics.mean
                    row[Symbol("$(kind)_$(short)_median")] = statistics.median
                    row[Symbol("$(kind)_$(short)_max")] = statistics.max
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

        # ---- committed-layer profile ----------------------------------------
        profile_path::String = joinpath(results_dir, "layer_profile_" * run_stem)
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

        # ---- training-time debug log ----------------------------------------
        cer_tag::String = run_key.arm == "nocer" ? "_no_cer" : ""
        run_tail::String = "train_p_$(run_key.p)_s_$(run_key.data_seed)$(cer_tag)" *
                           "_cw$(run_key.arm)_$(run_key.gate)_sp$(run_key.sparsity_tag)" *
                           "_seed_$(run_key.seed)"
        debug_summary::Dict{Symbol, Any} = debug_log_summary(logs_dir, run_tail)
        for (key, value) in debug_summary
            row[key] = value
        end

        # THE PREMISE CHECK. This sweep only means anything if sparsity really was
        # pinned to zero; a non-zero realised value means the TOML override did not
        # take and the arm is not what it claims to be.
        logged_sparsity = debug_summary[:sparsity_final_epoch]
        row[:sparsity_is_zero] = missing
        if logged_sparsity isa Number && !isnan(logged_sparsity)
            row[:sparsity_is_zero] = isapprox(logged_sparsity, 0.0; atol = 1e-8)
            if !row[:sparsity_is_zero]
                @warn "PREMISE VIOLATED for $(run_tail): the training log reports " *
                      "sparsity_importance = $(logged_sparsity) at the final epoch, not 0. " *
                      "The sparsity counterweight was active and this run does not test " *
                      "what the sweep claims to test."
            end
        end

        expected_gate_sign::Float64 = run_key.gate == "gated" ? 1.0 : -1.0
        logged_gate = debug_summary[:gate_threshold_logged]
        row[:gate_threshold_consistent] = missing
        if logged_gate isa Number && !isnan(logged_gate)
            row[:gate_threshold_consistent] = sign(logged_gate) == expected_gate_sign
            if sign(logged_gate) != expected_gate_sign
                @warn "log/run MISMATCH for $(run_tail): the log reports " *
                      "syndrome_gate_threshold = $(logged_gate), which contradicts the arm. " *
                      "Train-side columns for this run are not trustworthy."
            end
        end

        push!(collected_rows, row)
    end
    per_run::DataFrame = rows_to_dataframe(collected_rows)
    if nrow(per_run) > 0
        sort!(per_run, [:p_numeric, :arm, :seed])
    end
    return per_run
end

"Mean / sd / min / max across seeds, per (p, arm)."
function collect_per_arm(per_run::DataFrame)::DataFrame
    quantities::Vector{Symbol} = [
        :num_failures, :num_coset_failures, :num_convergence_failures,
        :logical_error_rate, :coset_rate, :convergence_rate,
        :coset_rate_given_cleared, :convergence_share_of_failures,
        :mean_committed_layer, :layer1_clearing_fraction,
        :convergence_near_miss_frac, :gate_open_fraction_mean,
        :correlation_penalty_mean, :lambda_final_epoch, :sparsity_final_epoch,
        :runtime,
    ]
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for group in groupby(per_run, :label)
        row::Dict{Symbol, Any} = Dict(:label => group.label[1],
                                      :p => group.p[1],
                                      :p_numeric => group.p_numeric[1],
                                      :arm => group.arm[1],
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
        sort!(per_arm, [:p_numeric, :arm])
    end
    return per_arm
end

"""
    contrast(per_run, p_value) -> Dict

Paired-by-seed (cer - nocer) difference at one p, for the three failure counts,
plus a pooled two-proportion z on the summed counts.

The PAIRED statistic is the honest one: it uses the seed as a block, so it
measures the configuration effect against the seed-to-seed spread. The pooled z
uses only test-set uncertainty and will look far more significant than the
evidence warrants — it is reported alongside precisely so the two can be
compared. A previous +361 failure "effect" survived the z (z = +10.8) and died
on the t (t = +0.52).
"""
function contrast(per_run::DataFrame, p_value::String)::Dict{Symbol, Any}
    at_p::DataFrame = per_run[per_run.p .== p_value, :]
    cer_rows::DataFrame = at_p[at_p.arm .== "cer", :]
    nocer_rows::DataFrame = at_p[at_p.arm .== "nocer", :]
    shared_seeds::Vector{Int} = sort(collect(intersect(Set(cer_rows.seed), Set(nocer_rows.seed))))
    out::Dict{Symbol, Any} = Dict(
        :p => p_value,
        :p_numeric => something(tryparse(Float64, p_value), NaN),
        :contrast => "cer - nocer",
        :n_paired_seeds => length(shared_seeds),
    )
    if isempty(shared_seeds)
        return out
    end
    for quantity in (:num_failures, :num_coset_failures, :num_convergence_failures)
        differences::Vector{Float64} = Float64[]
        for seed in shared_seeds
            cer_value = cer_rows[cer_rows.seed .== seed, quantity][1]
            nocer_value = nocer_rows[nocer_rows.seed .== seed, quantity][1]
            push!(differences, Float64(cer_value) - Float64(nocer_value))
        end
        out[Symbol("$(quantity)_paired_mean")] = mean(differences)
        out[Symbol("$(quantity)_paired_sd")] = length(differences) > 1 ? std(differences) : 0.0
        standard_error::Float64 = length(differences) > 1 ?
            std(differences) / sqrt(length(differences)) : 0.0
        out[Symbol("$(quantity)_paired_t")] = standard_error > 0 ?
            mean(differences) / standard_error : NaN

        total_cer::Int = sum(cer_rows[in.(cer_rows.seed, Ref(shared_seeds)), quantity])
        total_nocer::Int = sum(nocer_rows[in.(nocer_rows.seed, Ref(shared_seeds)), quantity])
        trials::Int = sum(cer_rows[in.(cer_rows.seed, Ref(shared_seeds)), :num_samples_per_error_rate])
        pooled::Float64 = (total_cer + total_nocer) / (2 * trials)
        pooled_standard_error::Float64 = sqrt(pooled * (1 - pooled) * 2 / trials)
        out[Symbol("$(quantity)_pooled_cer")] = total_cer
        out[Symbol("$(quantity)_pooled_nocer")] = total_nocer
        out[Symbol("$(quantity)_pooled_z")] = pooled_standard_error > 0 ?
            (total_cer / trials - total_nocer / trials) / pooled_standard_error : NaN
    end
    # Relative effect, so the three p can be compared on one scale despite their
    # failure counts differing by orders of magnitude.
    baseline_failures::Float64 = mean(Float64.(nocer_rows[in.(nocer_rows.seed, Ref(shared_seeds)), :num_failures]))
    out[:num_failures_relative] = baseline_failures > 0 ?
        out[:num_failures_paired_mean] / baseline_failures : NaN
    return out
end

function collect_contrasts(per_run::DataFrame)::DataFrame
    p_values::Vector{String} = unique(per_run.p)
    sort!(p_values; by = value -> something(tryparse(Float64, value), Inf))
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for p_value in p_values
        push!(collected_rows, contrast(per_run, p_value))
    end
    contrasts::DataFrame = rows_to_dataframe(collected_rows)
    return contrasts
end

function print_report(per_run::DataFrame, per_arm::DataFrame, contrasts::DataFrame)::Nothing
    println(repeat("=", 100))
    println("PER ARM (mean +- sd across seeds)")
    println(repeat("-", 100))
    @printf("%-18s %5s %16s %16s %18s %10s\n",
            "p / arm", "seeds", "failures", "coset", "convergence", "gate open")
    for row in eachrow(per_arm)
        gate_open::Float64 = NaN
        if hasproperty(per_arm, :gate_open_fraction_mean_mean) &&
           row.gate_open_fraction_mean_mean !== missing
            gate_open = row.gate_open_fraction_mean_mean
        end
        @printf("%-18s %5d %8.1f+-%-6.1f %8.1f+-%-6.1f %10.1f+-%-6.1f %10.3f\n",
                row.label, row.n_seeds,
                row.num_failures_mean, row.num_failures_sd,
                row.num_coset_failures_mean, row.num_coset_failures_sd,
                row.num_convergence_failures_mean, row.num_convergence_failures_sd,
                gate_open)
    end

    println()
    println(repeat("=", 100))
    println("CONTRASTS: cer - nocer, paired by seed (t against the seed spread, z test-set only)")
    println(repeat("-", 100))
    for row in eachrow(contrasts)
        @printf("  p = %-10s  n = %d paired seed(s)\n", row.p, row.n_paired_seeds)
        for (quantity, name) in ((:num_failures, "total"),
                                 (:num_coset_failures, "coset"),
                                 (:num_convergence_failures, "convergence"))
            paired_mean_key::Symbol = Symbol("$(quantity)_paired_mean")
            if !hasproperty(contrasts, paired_mean_key) || row[paired_mean_key] === missing
                continue
            end
            @printf("      %-12s paired %+9.1f +- %-8.1f  t = %+6.2f      pooled z = %+7.2f\n",
                    name, row[paired_mean_key],
                    row[Symbol("$(quantity)_paired_sd")],
                    row[Symbol("$(quantity)_paired_t")],
                    row[Symbol("$(quantity)_pooled_z")])
        end
        if hasproperty(contrasts, :num_failures_relative) && row.num_failures_relative !== missing
            @printf("      %-12s %+.2f%% of the no-CER failure count\n",
                    "relative", 100 * row.num_failures_relative)
        end
        println()
    end
    println("  NEGATIVE means CER did BETTER. |t| >~ 3 at n = 3 seeds is the bar; z will")
    println("  overstate significance because two networks always differ.")

    println()
    println(repeat("=", 100))
    println("p-TREND (the discriminator: the correlation term scales as p^2)")
    println(repeat("-", 100))
    @printf("  %-12s %14s %14s %10s\n", "p", "cer - nocer", "relative", "t")
    for row in eachrow(contrasts)
        if !hasproperty(contrasts, :num_failures_paired_mean) ||
           row.num_failures_paired_mean === missing
            continue
        end
        relative::Float64 = row.num_failures_relative === missing ? NaN : row.num_failures_relative
        @printf("  %-12s %+14.1f %13.1f%% %+10.2f\n",
                row.p, row.num_failures_paired_mean, 100 * relative,
                row.num_failures_paired_t)
    end
    println()
    println("  more negative as p rises  -> the COUPLINGS are working (p^2 scaling)")
    println("  flat and negative         -> the single-qubit PRIORS; rerun with --lambda 0")
    println("  flat and ~zero            -> revised J changed nothing; suspect the 1/|C| norm")
    println("  positive, worst at big p  -> reward outrunning the syndrome; check gate_open_fraction")

    # ---- integrity checks, printed loudly rather than buried in a column ----
    println()
    println(repeat("=", 100))
    println("INTEGRITY")
    println(repeat("-", 100))
    n_logs::Int = count(value -> value === true, per_run.debug_log_found)
    @printf("  debug logs found              : %d / %d run(s)\n", n_logs, nrow(per_run))
    if hasproperty(per_run, :sparsity_is_zero)
        checked::Vector{Any} = collect(skipmissing(per_run.sparsity_is_zero))
        n_bad::Int = count(value -> value === false, checked)
        @printf("  sparsity == 0 at final epoch  : %d / %d checked%s\n",
                length(checked) - n_bad, length(checked),
                n_bad > 0 ? "   <-- $(n_bad) VIOLATION(S), see warnings above" : "")
    end
    if hasproperty(per_run, :gate_threshold_consistent)
        gate_checked::Vector{Any} = collect(skipmissing(per_run.gate_threshold_consistent))
        n_gate_bad::Int = count(value -> value === false, gate_checked)
        @printf("  gate threshold matches arm    : %d / %d checked%s\n",
                length(gate_checked) - n_gate_bad, length(gate_checked),
                n_gate_bad > 0 ? "   <-- $(n_gate_bad) MISMATCH(ES)" : "")
    end
    # A gated arm whose gate never opened measured "base loss only" and is not a
    # null result about the couplings; one whose gate never closed is ungated in
    # all but name. Both are silent failures unless printed.
    if hasproperty(per_run, :gate_open_fraction_mean)
        gated_runs::DataFrame = per_run[per_run.gate .== "gated", :]
        if nrow(gated_runs) > 0
            openings::Vector{Float64} = filter(!isnan,
                Float64.(collect(skipmissing(gated_runs.gate_open_fraction_mean))))
            if isempty(openings)
                println("  gate open fraction            : not logged (rerun with --isdebug true)")
            else
                n_never_open::Int = count(value -> value <= 1e-6, openings)
                n_always_open::Int = count(value -> value >= 1 - 1e-6, openings)
                @printf("  gate open fraction (gated)    : %.3f .. %.3f over %d run(s)\n",
                        minimum(openings), maximum(openings), length(openings))
                if n_never_open > 0
                    @printf("      <-- %d run(s) NEVER opened the gate: the aux terms never fired,\n",
                            n_never_open)
                    println("          so those arms measured base loss only. Not a null result.")
                end
                if n_always_open > 0
                    @printf("      <-- %d run(s) ALWAYS open: effectively ungated; lower tau.\n",
                            n_always_open)
                end
            end
        end
    end
    if hasproperty(per_run, :lambda_final_epoch)
        lambdas::Vector{Float64} = filter(!isnan,
            Float64.(collect(skipmissing(per_run.lambda_final_epoch))))
        if !isempty(lambdas)
            @printf("  realised lambda (final epoch) : %.4f .. %.4f\n",
                    minimum(lambdas), maximum(lambdas))
        end
    end
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

    n_debug_logs::Int = 0
    if isdir(logs_dir)
        n_debug_logs = count(f -> startswith(f, "debugging_") && endswith(f, ".csv"),
                             readdir(logs_dir))
    end
    println("[collect] results (test-side) : $(results_dir)")
    println("[collect] logs    (train-side): $(logs_dir)  — $(n_debug_logs) debug file(s)")
    if n_debug_logs == 0
        println("          none found: the sparsity == 0 PREMISE CHECK cannot run, and")
        println("          gate_open_fraction / correlation_penalty will be blank.")
        println("          Rerun the sweep with --isdebug true to get them.")
    end
    println()

    per_run::DataFrame = collect_per_run(results_dir, logs_dir)
    if nrow(per_run) == 0
        error("no correlation-weight sweep runs found in $(results_dir) " *
              "(expecting names ending _cw<cer|nocer>_<gate>_sp<tag>_seed_<n>.csv)")
    end
    per_arm::DataFrame = collect_per_arm(per_run)
    contrasts::DataFrame = collect_contrasts(per_run)

    print_report(per_run, per_arm, contrasts)

    per_run_path::String = joinpath(output_dir, "correlation_weight_per_run.csv")
    per_arm_path::String = joinpath(output_dir, "correlation_weight_per_arm.csv")
    contrasts_path::String = joinpath(output_dir, "correlation_weight_contrasts.csv")
    CSV.write(per_run_path, per_run)
    CSV.write(per_arm_path, per_arm)
    CSV.write(contrasts_path, contrasts)
    println("\n  wrote $(per_run_path)   ($(nrow(per_run)) rows x $(ncol(per_run)) cols)")
    println("  wrote $(per_arm_path)   ($(nrow(per_arm)) rows)")
    println("  wrote $(contrasts_path)   ($(nrow(contrasts)) rows)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

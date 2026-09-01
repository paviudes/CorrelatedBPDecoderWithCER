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

# THERE IS DELIBERATELY NO DEFAULT RESULTS DIRECTORY.
#
# This previously defaulted to 72q_BB_cycles_1_debug/results. That is a trap once
# more than one codename exists: running the collector with no argument would
# silently summarise a DIFFERENT experiment than the one just finished, and the
# output filenames are identical either way, so the mistake is invisible in the
# artefacts. `sweep_correlation_weight.sh --collect` always passes the directory
# explicitly (derived from the active data profile), so requiring it costs
# nothing and removes the failure mode.
const USAGE = """
    julia --project="./../" misc/collect_correlation_weight.jl <results_dir> [--outdir DIR]

  <results_dir> is required, e.g.
      ./../data/72q_BB_cycles_1_spread_comparison/results
      ./../data/72q_BB_cycles_1_debug/results

  Or let the sweep script pick it from the active data profile:
      bash misc/sweep_correlation_weight.sh --spread --collect
"""

# `..._trained_using_train_p_<p>_s_<n>[_no_cer]_cw<arm>_<gate>_sp<tag>[_lam<tag>]_seed_<k>.csv`
#
# The `_lam<tag>` group is OPTIONAL, which is what lets one collector read both
# vintages: the 2026-08-20 p-sweep wrote no lambda tag (it inherited the base
# TOML's anneal), the lambda sweep pins one. A run with no tag is recorded as
# `lambda_pinned = false` rather than as lambda = 0, because "annealed to 0.7623"
# and "pinned at 0" are opposite ends of the axis and must never be pooled.
# The dataset KEY is captured whole (`p_0.0005_s_1`, or `p_0.0005_sig_0.0005_s_2`
# for the per-gate-spread runs) and decomposed afterwards, so a new field in the
# filename layout does not require a new capture group here.
# Two run-tag vintages, both collected by one pattern:
#   _cw<arm>_<gate>_sp<tag>[_lam<tag>]_seed_<n>   sweep_correlation_weight.sh
#   _hp<arm>_sp<tag>[_lam<tag>]_seed_<n>          sweep_hyperparams.sh
# The gate token is optional because the second generator dropped it once the
# ungated path was deleted from loss.jl: every run is gated now.
const RUN_PATTERN = r"_trained_using_train_(p_[0-9.eE+-]+(?:_sig_[0-9.eE+-]+)?_s_\d+)(_no_cer)?_(?:cw|hp)(cer|nocer)(?:_(ungated|gated))?_sp([0-9p]+)(?:_lam([0-9p]+[du]?))?(?:_tau([0-9pe]+))?(?:_ct([0-9pe]+))?(?:_cp([a-z]+[0-9p]*))?(?:_cf([a-z_]+))?_seed_(\d+)\.csv$"

"""
    tag_to_number(tag) -> Float64

Undo the filename-safe encoding: "0p75" -> 0.75, "0" -> 0.0, "1p5" -> 1.5.
"""
function tag_to_number(tag::String)::Float64
    numeric::Union{Float64, Nothing} = tryparse(Float64, replace(tag, "p" => "."))
    if numeric === nothing
        return NaN
    end
    return numeric
end

"""
    parse_run(filename) -> Union{NamedTuple, Nothing}

Pull (p, arm, gate, sparsity_tag, lambda, seed) out of a result filename, or
`nothing` if it is not one of this sweep's files.

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
    dataset_key::String = String(filename_match.captures[1])
    arm::String = String(filename_match.captures[3])
    has_no_cer_tag::Bool = filename_match.captures[2] !== nothing
    if has_no_cer_tag != (arm == "nocer")
        @warn "filename $(filename) has the `_no_cer` tag = $(has_no_cer_tag) but the " *
              "run tag says arm = $(arm). use_CER and run_tag have drifted apart; " *
              "this run's arm label cannot be trusted."
    end
    lambda_tag::String = ""
    lambda_pinned::Bool = filename_match.captures[6] !== nothing
    if lambda_pinned
        lambda_tag = String(filename_match.captures[6])
    end
    # A trailing d/u marks an ANNEALED lambda (down/up). Strip it for the numeric
    # value and keep it as a separate field: an annealed run and a constant run at
    # the same peak lambda are different experiments and must not be pooled.
    lambda_schedule::String = "constant"
    lambda_numeric_tag::String = lambda_tag
    if endswith(lambda_tag, "d")
        lambda_schedule = "down"
        lambda_numeric_tag = chop(lambda_tag)
    elseif endswith(lambda_tag, "u")
        lambda_schedule = "up"
        lambda_numeric_tag = chop(lambda_tag)
    end
    lambda_value::Float64 = NaN
    if lambda_pinned
        lambda_value = tag_to_number(lambda_numeric_tag)
    end

    # Decompose the key. `sigma` is "" on the uniform-p datasets, which is what
    # distinguishes them from a per-gate-spread run that happens to use sigma = 0.
    key_match::Union{RegexMatch, Nothing} =
        match(r"^p_([0-9.eE+-]+)(?:_sig_([0-9.eE+-]+))?_s_(\d+)$", dataset_key)
    p_string::String = ""
    sigma_string::String = ""
    data_sample::Int = 0
    if key_match !== nothing
        p_string = String(key_match.captures[1])
        if key_match.captures[2] !== nothing
            sigma_string = String(key_match.captures[2])
        end
        data_sample = parse(Int, key_match.captures[3])
    end

    # Which generator wrote this, and whether it emitted a gate token.
    prefix::String = occursin("_hp$(arm)", filename) ? "hp" : "cw"
    gate_segment::String = filename_match.captures[4] === nothing ? "" :
                           "_" * String(filename_match.captures[4])
    run_key::NamedTuple = (
        dataset = dataset_key,
        prefix = prefix,
        gate_segment = gate_segment,
        p = p_string,
        sigma = sigma_string,
        data_seed = data_sample,
        arm = arm,
        gate = filename_match.captures[4] === nothing ? "gated" :
                                                 String(filename_match.captures[4]),
        sparsity_tag = String(filename_match.captures[5]),
        lambda_tag = lambda_tag,
        lambda_pinned = lambda_pinned,
        lambda = lambda_value,
        lambda_schedule = lambda_schedule,
        tau_tag = filename_match.captures[7] === nothing ? "" : String(filename_match.captures[7]),
        tau = filename_match.captures[7] === nothing ? NaN :
              tag_to_number(String(filename_match.captures[7])),
        certainty_gate_tag = filename_match.captures[8] === nothing ? "" :
                             String(filename_match.captures[8]),
        certainty_penalty = filename_match.captures[9] === nothing ? "entropy" :
                            String(filename_match.captures[10]),
        correlation_form = filename_match.captures[10] === nothing ? "bilinear" :
                           String(filename_match.captures[9]),
        seed = parse(Int, filename_match.captures[11]),
    )
    return run_key
end

"""
    label_for(run_key) -> String

The arm label used for grouping and contrasts. The no-CER baseline carries no
lambda (with `use_CER = false` the couplings do not exist, so the weight
multiplies nothing) and is always just "nocer".
"""
function label_for(run_key::NamedTuple)::String
    # The certainty penalty changes L2, which BOTH arms carry, so it has to be
    # part of the label: otherwise runs with different L2 terms would be pooled
    # into one arm mean and the contrast would compare mixtures.
    certainty_tag::String = ""
    if run_key.certainty_penalty != "entropy"
        certainty_tag = "_cp$(run_key.certainty_penalty)"
    end
    # The L3 form changes the CER arms only; the baseline is shared between forms
    # and stays untagged, so it can serve as the control for both.
    correlation_form_tag::String = ""
    if run_key.correlation_form != "bilinear"
        correlation_form_tag = "_cf$(run_key.correlation_form)"
    end
    # alpha3 is a swept axis, so it belongs in the label. Without it the three
    # sparsity arms pooled into one group and their COUNT was reported as the
    # seed count: nocer_tau0p5 showed "7 seeds" (5 real seeds of sp0p0, plus one
    # each of sp0p003 and sp0p01) and every hinge arm showed a spurious 3.
    certainty_gate_label::String = ""
    if !isempty(run_key.certainty_gate_tag)
        certainty_gate_label = "_ct$(run_key.certainty_gate_tag)"
    end
    sparsity_tag_segment::String = ""
    if run_key.sparsity_tag != "0p0"
        sparsity_tag_segment = "_sp$(run_key.sparsity_tag)"
    end
    if run_key.arm == "nocer"
        base_label::String = "nocer"
        if !isempty(run_key.tau_tag)
            base_label = "nocer_tau$(run_key.tau_tag)"
        end
        return base_label * certainty_gate_label * sparsity_tag_segment * certainty_tag
    end
    if !run_key.lambda_pinned
        return "cer_annealed" * certainty_gate_label * sparsity_tag_segment * certainty_tag
    end
    label::String = "lam$(run_key.lambda_tag)"
    if !isempty(run_key.tau_tag)
        label = label * "_tau$(run_key.tau_tag)"
    end
    label = label * certainty_gate_label * sparsity_tag_segment * certainty_tag * correlation_form_tag
    return label
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
            :dataset => run_key.dataset,
            :p => run_key.p,
            :p_numeric => something(tryparse(Float64, run_key.p), NaN),
            :sigma => run_key.sigma,
            :data_sample => run_key.data_seed,
            :arm => run_key.arm,
            :gate => run_key.gate,
            :sparsity_tag => run_key.sparsity_tag,
            :lambda_tag => run_key.lambda_tag,
            :lambda_pinned => run_key.lambda_pinned,
            :lambda => run_key.lambda,
            :seed => run_key.seed,
            :label => label_for(run_key),
            # Grouping key for the per-arm table: one row per (p, arm) so a sweep
            # that varies BOTH axes still tabulates correctly.
            # Grouped by DATASET, not by p: the per-gate-spread runs share one p
            # across three independent noise samples, and pooling them would hide
            # exactly the sample-to-sample variation the replicates exist to measure.
            :group => "$(label_for(run_key))__$(run_key.dataset)",
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
        lambda_segment::String = ""
        if run_key.lambda_pinned
            lambda_segment = "_lam$(run_key.lambda_tag)"
        end
        # Rebuild the tag exactly as the generator wrote it, or the training log
        # will not be found and every premise check comes back blank.
        # EVERY tag that emit_point puts in the run_tag must be reproduced here,
        # in the same order. The debug log is matched by `endswith(f, run_tail)`,
        # so a missing segment does not merely fail to match -- it silently
        # matches a DIFFERENT run whose name happens to end that way. Omitting
        # _cp/_cf once made all three certainty penalties resolve to the entropy
        # run's log: the same warning fired three times, the hinge arms inherited
        # entropy's diagnostics, and the 90 runs carrying those tags found no log
        # at all (156/246).
        certainty_gate_segment::String = ""
        if !isempty(run_key.certainty_gate_tag)
            certainty_gate_segment = "_ct$(run_key.certainty_gate_tag)"
        end
        certainty_segment::String = ""
        if run_key.certainty_penalty != "entropy"
            certainty_segment = "_cp$(run_key.certainty_penalty)"
        end
        correlation_segment::String = ""
        if run_key.correlation_form != "bilinear"
            correlation_segment = "_cf$(run_key.correlation_form)"
        end
        run_tail::String = "train_$(run_key.dataset)$(cer_tag)" *
                           "_$(run_key.prefix)$(run_key.arm)$(run_key.gate_segment)" *
                           "_sp$(run_key.sparsity_tag)$(lambda_segment)" *
                           (isempty(run_key.tau_tag) ? "" : "_tau$(run_key.tau_tag)") *
                           certainty_gate_segment * certainty_segment * correlation_segment *
                           "_seed_$(run_key.seed)"
        debug_summary::Dict{Symbol, Any} = debug_log_summary(logs_dir, run_tail)
        for (key, value) in debug_summary
            row[key] = value
        end

        # STALENESS GUARD. A result file OLDER than the training log that
        # supposedly produced its model was not regenerated after that training —
        # a failed or killed test phase quietly leaves the previous run's results
        # in place, and they collect as if fresh. This once paired a fixed
        # training run with a broken run's test files, identical runtimes and all.
        row[:result_mtime] = mtime(joinpath(results_dir, filename))
        row[:result_fresh] = missing
        log_candidates::Vector{String} = filter(
            f -> startswith(f, "debugging_") && endswith(f, run_tail * ".csv"),
            isdir(logs_dir) ? readdir(logs_dir) : String[])
        if !isempty(log_candidates)
            log_mtime::Float64 = mtime(joinpath(logs_dir, log_candidates[1]))
            row[:result_fresh] = row[:result_mtime] >= log_mtime
            if row[:result_fresh] === false
                @warn "STALE RESULT for $(run_tail): the result file predates its own " *
                      "training log by $(round((log_mtime - row[:result_mtime])/60; digits=1)) " *
                      "minutes. The test phase did not run after that training; these " *
                      "numbers belong to an earlier model."
            end
        end

        # THE PREMISE CHECK. sparsity_importance is a SWEPT AXIS, not a constant
        # pinned to zero, so the test is that the realised value matches the one
        # in this run's own filename tag (_sp0p003 -> 0.003) -- exactly the test
        # lambda already gets below. Asserting == 0 here flagged every deliberate
        # alpha3 arm as a violation.
        logged_sparsity = debug_summary[:sparsity_final_epoch]
        pinned_sparsity::Float64 = tag_to_number(run_key.sparsity_tag)
        row[:sparsity_matches_pin] = missing
        row[:sparsity_is_zero] = missing
        if logged_sparsity isa Number && !isnan(logged_sparsity)
            row[:sparsity_is_zero] = isapprox(logged_sparsity, 0.0; atol = 1e-8)
            if !isnan(pinned_sparsity)
                row[:sparsity_matches_pin] =
                    isapprox(logged_sparsity, pinned_sparsity; rtol = 1e-3, atol = 1e-8)
                if !row[:sparsity_matches_pin]
                    @warn "PREMISE VIOLATED for $(run_tail): pinned sparsity = " *
                          "$(pinned_sparsity) from the filename tag, but the training log " *
                          "reports sparsity_importance = $(logged_sparsity) at the final " *
                          "epoch. The TOML override did not take."
                end
            end
        end

        # THE SECOND PREMISE CHECK, and the one this sweep turns on. A lambda axis
        # is only an axis if the realised weight matches the pinned one; a TOML
        # override that failed to take would silently collapse several points onto
        # the base TOML's annealed 0.7623 and manufacture a flat trend.
        logged_lambda = debug_summary[:lambda_final_epoch]
        row[:lambda_matches_pin] = missing
        if run_key.lambda_pinned && logged_lambda isa Number && !isnan(logged_lambda)
            row[:lambda_matches_pin] = isapprox(logged_lambda, run_key.lambda; atol = 1e-6)
            if !row[:lambda_matches_pin]
                @warn "PREMISE VIOLATED for $(run_tail): pinned lambda = $(run_key.lambda) " *
                      "but the training log reports correlation_weight = $(logged_lambda) at " *
                      "the final epoch. This point is not at the lambda its filename claims."
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
        # `:label` rather than `:lambda`: the latter is NaN on the no-CER and
        # annealed arms, and NaN ordering is not something to rely on.
        sort!(per_run, [:dataset, :label, :seed])
    end
    return per_run
end

"""
    arm_order(label) -> Float64

Sort key that puts the baseline first, then the pinned lambdas in numeric order,
then the annealed arm. Without it "lam0p1" < "lam0p3" < "lam0p75" < "lam1p5"
holds only by luck of string ordering, and "lam1p5" would sort before "lam0p3"
the moment a two-digit lambda appears.
"""
function arm_order(label::String)::Float64
    if startswith(label, "nocer")
        return -2.0
    end
    if label == "cer_annealed"
        return Inf
    end
    if startswith(label, "lam")
        base_part::String = split(label, "_tau")[1]
        return tag_to_number(base_part[4:end])
    end
    return -1.0
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
    for group in groupby(per_run, :group)
        row::Dict{Symbol, Any} = Dict(:group => group.group[1],
                                      :label => group.label[1],
                                      :dataset => group.dataset[1],
                                      :sigma => group.sigma[1],
                                      :p => group.p[1],
                                      :p_numeric => group.p_numeric[1],
                                      :arm => group.arm[1],
                                      :lambda => group.lambda[1],
                                      :lambda_pinned => group.lambda_pinned[1],
                                      :arm_order => arm_order(group.label[1]),
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
        sort!(per_arm, [:dataset, :arm_order])
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
function contrast(per_run::DataFrame, dataset_key::String,
                  label_a::String, label_b::String)::Dict{Symbol, Any}
    at_p::DataFrame = per_run[per_run.dataset .== dataset_key, :]
    cer_rows::DataFrame = at_p[at_p.label .== label_a, :]
    nocer_rows::DataFrame = at_p[at_p.label .== label_b, :]
    shared_seeds::Vector{Int} = sort(collect(intersect(Set(cer_rows.seed), Set(nocer_rows.seed))))
    out::Dict{Symbol, Any} = Dict(
        :dataset => dataset_key,
        :p => nrow(at_p) > 0 ? at_p.p[1] : "",
        :sigma => nrow(at_p) > 0 ? at_p.sigma[1] : "",
        :contrast => "$(label_a) - $(label_b)",
        :label_a => label_a,
        :label_b => label_b,
        :arm_order_a => arm_order(label_a),
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
        out[Symbol("$(quantity)_pooled_a")] = total_cer
        out[Symbol("$(quantity)_pooled_b")] = total_nocer
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

"""
    collect_contrasts(per_run) -> DataFrame

THE DECOMPOSITION. Every CER-vs-no-CER number in this project so far has
confounded two changes, because `use_CER = false` swaps the single-qubit priors
AND removes the couplings at the same time. With a lambda axis they separate:

    nocer -> lam0     the PRIORS      (CER single-qubit rates vs flat p = 0.1)
    lam0  -> lam>0    the COUPLINGS   (same priors, coupling weight turned up)

So each pinned lambda is contrasted against `lam0`, and `lam0` against `nocer`.
Where there is no `lam0` (the 2026-08-20 p-sweep, which ran the annealed
schedule) the fallback is the old cer-vs-nocer contrast, so that vintage still
collects.
"""
function collect_contrasts(per_run::DataFrame)::DataFrame
    p_values::Vector{String} = sort(unique(per_run.dataset))
    collected_rows::Vector{Dict{Symbol, Any}} = Dict{Symbol, Any}[]
    for p_value in p_values
        labels_here::Vector{String} = unique(per_run[per_run.dataset .== p_value, :label])
        pinned::Vector{String} = sort(filter(l -> startswith(l, "lam") && l != "lam0", labels_here);
                                      by = arm_order)
        if "lam0" in labels_here
            # Couplings: each lambda against the lambda = 0 control.
            for label in pinned
                push!(collected_rows, contrast(per_run, p_value, label, "lam0"))
            end
            # Priors: the control against the flat-prior baseline.
            if "nocer" in labels_here
                push!(collected_rows, contrast(per_run, p_value, "lam0", "nocer"))
            end
        end
        # Older vintage, or a lambda sweep run with --no_nocer: keep the direct
        # comparison so nothing silently drops out of the report.
        if "cer_annealed" in labels_here && "nocer" in labels_here
            push!(collected_rows, contrast(per_run, p_value, "cer_annealed", "nocer"))
        end
        if !("lam0" in labels_here) && "nocer" in labels_here
            for label in pinned
                push!(collected_rows, contrast(per_run, p_value, label, "nocer"))
            end
        end
    end
    contrasts::DataFrame = rows_to_dataframe(collected_rows)
    return contrasts
end

function print_report(per_run::DataFrame, per_arm::DataFrame, contrasts::DataFrame)::Nothing
    println(repeat("=", 100))
    println("PER ARM (mean +- sd across seeds)")
    println(repeat("-", 100))
    @printf("%-26s %5s %16s %16s %18s %10s\n",
            "p / arm", "seeds", "failures", "coset", "convergence", "gate open")
    for row in eachrow(per_arm)
        gate_open::Float64 = NaN
        if hasproperty(per_arm, :gate_open_fraction_mean_mean) &&
           row.gate_open_fraction_mean_mean !== missing
            gate_open = row.gate_open_fraction_mean_mean
        end
        @printf("%-26s %5d %8.1f+-%-6.1f %8.1f+-%-6.1f %10.1f+-%-6.1f %10.3f\n",
                row.group, row.n_seeds,
                row.num_failures_mean, row.num_failures_sd,
                row.num_coset_failures_mean, row.num_coset_failures_sd,
                row.num_convergence_failures_mean, row.num_convergence_failures_sd,
                gate_open)
    end

    println()
    println(repeat("=", 100))
    println("CONTRASTS, paired by seed (t against the seed spread, z test-set only)")
    println(repeat("-", 100))
    for row in eachrow(contrasts)
        @printf("  %-26s  %-22s  n = %d paired seed(s)\n",
                row.dataset, row.contrast, row.n_paired_seeds)
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
            @printf("      %-12s %+.2f%% of the %s failure count\n",
                    "relative", 100 * row.num_failures_relative, row.label_b)
        end
        println()
    end
    println("  NEGATIVE means the FIRST arm did better. |t| >~ 3 at n = 3 seeds is the bar;")
    println("  z will overstate significance because two networks always differ.")

    # ---- the trend table: lambda if there is a lambda axis, else p -----------
    lambda_rows::DataFrame = contrasts
    if hasproperty(contrasts, :label_b)
        lambda_rows = contrasts[contrasts.label_b .== "lam0", :]
    end
    if nrow(lambda_rows) > 0
        println()
        println(repeat("=", 100))
        println("LAMBDA TREND, each arm against the lambda = 0 control (couplings OFF, CER priors ON)")
        println(repeat("-", 100))
        @printf("  %-26s %-10s %13s %13s %13s %10s\n",
                "dataset", "lambda", "coset", "convergence", "total", "t(total)")
        for row in eachrow(lambda_rows)
            if row.num_failures_paired_mean === missing
                continue
            end
            @printf("  %-26s %-10s %+13.1f %+13.1f %+13.1f %+10.2f\n",
                    row.dataset, replace(row.label_a, "lam" => ""),
                    row.num_coset_failures_paired_mean,
                    row.num_convergence_failures_paired_mean,
                    row.num_failures_paired_mean,
                    row.num_failures_paired_t)
        end
        println()
        println("  THE PREDICTION: coset selection is a discrete argmax flip, so its benefit")
        println("  should SATURATE in lambda, while the convergence damage is a continuous")
        println("  distortion and should grow ~LINEARLY. If so, total has an interior minimum.")
        println()
        println("  interior minimum in total      -> the trade is tunable; take that lambda to the p axis")
        println("  coset and convergence track    -> no lambda wins; the 1/|C| divisor is the problem")
        println("  coset flat from lambda = 0     -> the coset effect was never the couplings")
    end

    prior_rows::DataFrame = contrasts
    if hasproperty(contrasts, :label_a)
        prior_rows = contrasts[(contrasts.label_a .== "lam0") .& (contrasts.label_b .== "nocer"), :]
    end
    if nrow(prior_rows) > 0
        println()
        println(repeat("=", 100))
        println("PRIORS ALONE: lambda = 0 against no-CER (same couplings — none — different priors)")
        println(repeat("-", 100))
        for row in eachrow(prior_rows)
            if row.num_failures_paired_mean === missing
                continue
            end
            @printf("  %-26s total %+9.1f +- %-8.1f  t = %+6.2f   (%+.1f%% of no-CER)\n",
                    row.dataset, row.num_failures_paired_mean, row.num_failures_paired_sd,
                    row.num_failures_paired_t,
                    row.num_failures_relative === missing ? NaN : 100 * row.num_failures_relative)
        end
        println()
        println("  This is the contrast the project has never run. If it is large, every")
        println("  earlier CER-vs-no-CER result was measuring the single-qubit priors.")
    end

    # ---- integrity checks, printed loudly rather than buried in a column ----
    println()
    println(repeat("=", 100))
    println("INTEGRITY")
    println(repeat("-", 100))
    n_logs::Int = count(value -> value === true, per_run.debug_log_found)
    @printf("  debug logs found              : %d / %d run(s)\n", n_logs, nrow(per_run))
    if hasproperty(per_run, :debug_log_written_rows)
        empty_logs::Vector{String} = String[]
        for row in eachrow(per_run)
            if row.debug_log_found === true && row.debug_log_written_rows == 0
                push!(empty_logs, "$(row.dataset)/$(row.label)")
            end
        end
        if !isempty(empty_logs)
            @printf("  ZERO-WRITTEN debug logs       : %d run(s)   <-- TRAINING NEVER LOGGED A BATCH\n",
                    length(empty_logs))
            println("      A log that exists but has no written rows means every batch was")
            println("      NaN-skipped and every epoch rolled back: the model tested is the")
            println("      INITIAL weights, and its results are not a trained-decoder result.")
            for name in empty_logs
                println("        ", name)
            end
        end
    end
    if hasproperty(per_run, :sparsity_matches_pin)
        checked::Vector{Any} = collect(skipmissing(per_run.sparsity_matches_pin))
        n_bad::Int = count(value -> value === false, checked)
        @printf("  realised sparsity == pinned   : %d / %d checked%s\n",
                length(checked) - n_bad, length(checked),
                n_bad > 0 ? "   <-- $(n_bad) VIOLATION(S), see warnings above" : "")
    end
    if hasproperty(per_run, :result_fresh)
        fresh_checked::Vector{Any} = collect(skipmissing(per_run.result_fresh))
        n_stale::Int = count(value -> value === false, fresh_checked)
        @printf("  result newer than its log     : %d / %d checked%s\n",
                length(fresh_checked) - n_stale, length(fresh_checked),
                n_stale > 0 ? "   <-- $(n_stale) STALE: test phase did not rerun" : "")
    end
    if hasproperty(per_run, :lambda_matches_pin)
        lambda_checked::Vector{Any} = collect(skipmissing(per_run.lambda_matches_pin))
        n_lambda_bad::Int = count(value -> value === false, lambda_checked)
        @printf("  realised lambda == pinned     : %d / %d checked%s\n",
                length(lambda_checked) - n_lambda_bad, length(lambda_checked),
                n_lambda_bad > 0 ? "   <-- $(n_lambda_bad) VIOLATION(S): the axis is not an axis" : "")
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
    results_dir::String = ""
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
    if isempty(results_dir)
        print(USAGE)
        error("no results directory given. Naming it is required — see above.")
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

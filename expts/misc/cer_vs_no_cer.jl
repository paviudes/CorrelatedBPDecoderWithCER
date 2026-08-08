# ============================================================================
# cer_vs_no_cer.jl — CER vs no-CER Neural BP, against the BP and BP-OSD baselines
# ============================================================================
# RUN FROM expts/ :
#
#     julia --project="./../" misc/cer_vs_no_cer.jl [results_dir] [--csv out.csv]
#
# Default results_dir: ./../data/update_23July2026/72q_BB_cycles_1/results
#
# WHAT IT READS
#
#   simulation_results_test_p_<p>_s_<s>_nlayers_<L>_epochs_<E>_trained_using_<src><ARM>.csv
#       Neural BP. <ARM> is whatever suffix follows the training source —
#       `_no_cer`, `_corr`, `_corr_rescaled`, ... — so new arms need no code
#       change. `num_failures` / `num_samples_per_error_rate`.
#
#   72q_BB_p_<p>_cycles_1_BP_failure_rates.txt
#   72q_BB_p_<p>_cycles_1_BP+OSD_failure_rates_OSD_E_order_2.txt
#       Per-p baselines, header + one row:
#         p sample number_of_failures total_number_of_trials LER standard_error
#
#   72q_BB_cycles_1_BP+OSD_failure_rates_OSD_E_order_2.txt
#       Aggregate baseline, same columns, MANY rows, no header.
#
# TRIAL COUNTS DIFFER BETWEEN THESE SOURCES — the aggregate BP-OSD file is
# 100,000 trials while the per-p files are 1,000,000. Everything is therefore
# compared as a RATE, never as a raw failure count, and where both sources cover
# the same p the higher-statistics one wins (with a warning if they disagree).
# Confusing those two is exactly how "BP-OSD = 298" once got compared against a
# Neural BP count of 303 out of ten times as many shots.
#
# THE STATISTIC. `z` is a two-proportion test of CER against no-CER on the SAME
# test samples. It measures TEST-set uncertainty only. Both arms here are single
# training runs, and run-to-run spread from the RNG alone was measured at ~2x on
# this code — larger than most entries in this table. So a large |z| says "these
# two trained networks differ", NOT "this configuration is better". Only a
# multi-seed comparison can say the latter.
# ============================================================================

using CSV
using DataFrames
using Printf
using Statistics

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "..", "..", "data",
                                     "update_23July2026", "72q_BB_cycles_1", "results")

# The training source itself contains dots (`train_p_0.0001_s_1`), so the source
# group must be a non-greedy `.+?` anchored on its trailing `_s_<n>`; anything
# after that is the arm tag.
const NBP_PATTERN = r"^simulation_results_test_p_([0-9.]+)_s_(\d+)_nlayers_(\d+)_epochs_(\d+)_trained_using_(.+?)_s_(\d+)(.*)\.csv$"
const PER_P_PATTERN = r"_p_([0-9.]+)_cycles"

"""
    read_neural_bp_results(results_dir) -> Dict{Float64, Dict{String, Tuple{Int, Int}}}

Map error rate -> arm tag -> (failures, trials). The arm tag is whatever the
filename carries after the training source; `""` becomes `"(untagged)"`.
"""
function read_neural_bp_results(results_dir::String)::Dict{Float64, Dict{String, Tuple{Int, Int}}}
    results::Dict{Float64, Dict{String, Tuple{Int, Int}}} = Dict()
    for filename in readdir(results_dir)
        filename_match::Union{RegexMatch, Nothing} = match(NBP_PATTERN, filename)
        if filename_match === nothing
            continue
        end
        error_rate::Float64 = parse(Float64, filename_match.captures[1])
        arm_tag::String = String(filename_match.captures[7])
        if isempty(arm_tag)
            arm_tag = "(untagged)"
        end
        row = first(CSV.File(joinpath(results_dir, filename)))
        failures::Int = Int(row.num_failures)
        trials::Int = Int(row.num_samples_per_error_rate)
        if !haskey(results, error_rate)
            results[error_rate] = Dict{String, Tuple{Int, Int}}()
        end
        results[error_rate][arm_tag] = (failures, trials)
    end
    return results
end

"""
    read_baseline_table(path) -> Dict{Float64, Tuple{Int, Int}}

Read one whitespace-separated baseline file (`p sample failures trials ...`),
skipping comments. Handles both the single-row per-p files and the multi-row
aggregate file.
"""
function read_baseline_table(path::String)::Dict{Float64, Tuple{Int, Int}}
    table::Dict{Float64, Tuple{Int, Int}} = Dict()
    for raw_line in eachline(path)
        line::String = strip(raw_line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        fields::Vector{String} = split(line)
        if length(fields) < 4
            continue
        end
        error_rate::Float64 = parse(Float64, fields[1])
        failures::Int = parse(Int, fields[3])
        trials::Int = parse(Int, fields[4])
        table[error_rate] = (failures, trials)
    end
    return table
end

"""
    read_baselines(results_dir, tag) -> Dict{Float64, Tuple{Int, Int}}

Collect a baseline decoder across every file that carries it. Per-p files are
preferred over the aggregate file because they have 10x the trials; a
disagreement in RATE beyond 15% is warned about rather than silently resolved.
"""
function read_baselines(results_dir::String, tag::String)::Dict{Float64, Tuple{Int, Int}}
    aggregate::Dict{Float64, Tuple{Int, Int}} = Dict()
    per_error_rate::Dict{Float64, Tuple{Int, Int}} = Dict()

    for filename in readdir(results_dir)
        if !endswith(filename, ".txt") || !occursin(tag, filename)
            continue
        end
        # `BP_failure_rates` must not also match the `BP+OSD_failure_rates` files.
        if tag == "_BP_failure_rates" && occursin("OSD", filename)
            continue
        end
        path::String = joinpath(results_dir, filename)
        if match(PER_P_PATTERN, filename) === nothing
            merge!(aggregate, read_baseline_table(path))
        else
            merge!(per_error_rate, read_baseline_table(path))
        end
    end

    combined::Dict{Float64, Tuple{Int, Int}} = copy(aggregate)
    for (error_rate, per_p_entry) in per_error_rate
        if haskey(aggregate, error_rate)
            aggregate_rate::Float64 = aggregate[error_rate][1] / aggregate[error_rate][2]
            per_p_rate::Float64 = per_p_entry[1] / per_p_entry[2]
            if per_p_rate > 0 && abs(aggregate_rate - per_p_rate) / per_p_rate > 0.15
                @warn "$(tag) at p=$(error_rate): aggregate file says $(aggregate_rate) " *
                      "but the per-p file says $(per_p_rate). Using the per-p value " *
                      "($(per_p_entry[2]) trials vs $(aggregate[error_rate][2]))."
            end
        end
        combined[error_rate] = per_p_entry
    end
    return combined
end

"""
    two_proportion_z(failures_a, trials_a, failures_b, trials_b) -> Float64

Pooled two-proportion z for rate(a) - rate(b). Positive => a is WORSE.
"""
function two_proportion_z(failures_a::Int, trials_a::Int, failures_b::Int, trials_b::Int)::Float64
    pooled_rate::Float64 = (failures_a + failures_b) / (trials_a + trials_b)
    standard_error::Float64 = sqrt(pooled_rate * (1 - pooled_rate) * (1 / trials_a + 1 / trials_b))
    if standard_error <= 0
        return 0.0
    end
    z_score::Float64 = (failures_a / trials_a - failures_b / trials_b) / standard_error
    return z_score
end

function build_comparison(results_dir::String;
                          baseline_arm::String = "_no_cer",
                          cer_arm::String = "_corr")::DataFrame
    neural_results = read_neural_bp_results(results_dir)
    plain_bp = read_baselines(results_dir, "_BP_failure_rates")
    bp_osd = read_baselines(results_dir, "BP+OSD_failure_rates")

    comparison::DataFrame = DataFrame(
        p = Float64[], nbp_no_cer = Float64[], nbp_cer = Float64[],
        cer_over_no_cer = Float64[], z = Float64[], verdict = String[],
        bp = Float64[], bp_osd = Float64[], cer_over_bp_osd = Float64[],
        cer_failures = Int[], no_cer_failures = Int[], trials = Int[],
    )

    for error_rate in sort(collect(keys(neural_results)))
        arms = neural_results[error_rate]
        if !haskey(arms, baseline_arm) || !haskey(arms, cer_arm)
            continue
        end
        (cer_failures, cer_trials) = arms[cer_arm]
        (base_failures, base_trials) = arms[baseline_arm]
        cer_rate::Float64 = cer_failures / cer_trials
        base_rate::Float64 = base_failures / base_trials
        z_score::Float64 = two_proportion_z(cer_failures, cer_trials, base_failures, base_trials)

        verdict::String = "="
        if z_score < -2
            verdict = "CER better"
        elseif z_score > 2
            verdict = "CER worse"
        end

        bp_rate::Float64 = NaN
        if haskey(plain_bp, error_rate)
            bp_rate = plain_bp[error_rate][1] / plain_bp[error_rate][2]
        end
        osd_rate::Float64 = NaN
        if haskey(bp_osd, error_rate)
            osd_rate = bp_osd[error_rate][1] / bp_osd[error_rate][2]
        end

        push!(comparison, (error_rate, base_rate, cer_rate, cer_rate / base_rate,
                           z_score, verdict, bp_rate, osd_rate, cer_rate / osd_rate,
                           cer_failures, base_failures, cer_trials))
    end
    return comparison
end

function print_comparison(comparison::DataFrame)::Nothing
    println(repeat("=", 108))
    @printf("%-9s %11s %11s %10s %8s %-11s %11s %11s %9s\n",
            "p", "NBP no-CER", "NBP CER", "CER/noCER", "z", "verdict", "BP", "BP-OSD", "CER/OSD")
    println(repeat("-", 108))
    for row in eachrow(comparison)
        @printf("%-9g %11.3e %11.3e %10.3f %8.2f %-11s %11.3e %11.3e %9.3f\n",
                row.p, row.nbp_no_cer, row.nbp_cer, row.cer_over_no_cer,
                row.z, row.verdict, row.bp, row.bp_osd, row.cer_over_bp_osd)
    end
    println(repeat("=", 108))

    n_better::Int = count(==("CER better"), comparison.verdict)
    n_worse::Int = count(==("CER worse"), comparison.verdict)
    n_tied::Int = nrow(comparison) - n_better - n_worse
    println()
    @printf("  CER better (z < -2): %2d of %d error rates\n", n_better, nrow(comparison))
    @printf("  CER worse  (z > +2): %2d\n", n_worse)
    @printf("  indistinguishable  : %2d\n", n_tied)

    beats_osd = comparison[.!isnan.(comparison.cer_over_bp_osd) .& (comparison.cer_over_bp_osd .< 1), :]
    if nrow(beats_osd) > 0
        @printf("\n  Neural BP (CER) beats BP-OSD at %d of %d error rates; best factor %.1fx at p=%g\n",
                nrow(beats_osd), nrow(comparison),
                1 / minimum(beats_osd.cer_over_bp_osd),
                beats_osd.p[argmin(beats_osd.cer_over_bp_osd)])
    end

    println()
    println("  CAVEAT: z is TEST-set uncertainty only. Both arms are single training")
    println("  runs, and RNG-only run-to-run spread on this code was measured at ~2x —")
    println("  larger than most ratios above. A large |z| means these two NETWORKS")
    println("  differ, not that the CONFIGURATION is better. Use several seeds.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    results_dir::String = DEFAULT_RESULTS_DIR
    output_csv::String = ""
    argument_index::Int = 1
    while argument_index <= length(ARGS)
        if ARGS[argument_index] == "--csv"
            output_csv = ARGS[argument_index + 1]
            argument_index += 2
        else
            results_dir = ARGS[argument_index]
            argument_index += 1
        end
    end

    if !isdir(results_dir)
        error("results directory not found: $(results_dir)")
    end
    println("[cer_vs_no_cer] reading $(results_dir)\n")

    comparison = build_comparison(results_dir)
    if nrow(comparison) == 0
        error("no paired (_no_cer, _corr) results found in $(results_dir)")
    end
    print_comparison(comparison)

    if isempty(output_csv)
        output_csv = joinpath(results_dir, "cer_vs_no_cer_summary.csv")
    end
    CSV.write(output_csv, comparison)
    println("\n  table written to $(output_csv)")
end

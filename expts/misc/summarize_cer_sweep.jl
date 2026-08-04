# ============================================================================
# summarize_cer_sweep.jl — read the CER sweep results and build comparison
#                          tables + a figure
# ============================================================================
#
# Reads every `simulation_results_*.csv` in a codename's results/ folder,
# recovers (test p, alpha4, alpha3) from the filename, and produces:
#
#   1. a long CSV, one row per sweep point, with the logical error rate, its
#      standard error, and two comparisons:
#        - vs the alpha4 = 0 CONTROL at the same (p, alpha3)  -> isolates the
#          correlation TERM (both arms have CER priors), and
#        - vs the no-CER baseline at the same p               -> the full
#          CER-vs-no-CER effect (priors + term together).
#   2. pivot tables printed to the console (rows alpha4, columns alpha3).
#   3. a figure: LER vs alpha4, one panel per p, one line per alpha3, with the
#      CER and no-CER baselines as horizontal references.
#
# WHY NOT `neural_gather`: `_logical_error_rate_vs_p` groups rows by the
# `test_p_<p>` tag and AVERAGES them. All 16 sweep points share only two p
# values, so gathering them would silently average across alpha4 and alpha3.
# This script keys on the run_tag instead.
#
# The comparison statistic is the two-count z score, z = (n1 - n0)/sqrt(n1 + n0),
# for independent failure counts at equal sample size. |z| > 3 is a real
# difference; note it compares two SINGLE trained models, so it does NOT account
# for training-seed variance (see the train@p vs train@0.002 analysis).
#
# NOTE ON ENVIRONMENT: needs the experiments environment (Plots + CSV +
# DataFrames):
#
#     julia --project=expts expts/misc/summarize_cer_sweep.jl
#     julia --project=expts expts/misc/summarize_cer_sweep.jl ./../data 72q_BB_cycles_1
#
# ============================================================================

using CSV
using DataFrames
using Plots
using Printf
using Statistics

# Matches both the sweep results and the baselines:
#   simulation_results_test_p_<tp>_s_<ts>_nlayers_<L>_epochs_<E>_trained_using_train_p_<rp>_s_<rs>[_no_cer][_a4<A>_a3<B>].csv
# Group 8 captures the WHOLE run_tag verbatim (e.g. "_u400_a40p1_a30p5_r7"). That
# is what lets a result be joined back to its row in models/directory.csv, since
# the registry is keyed on the hyperparameters filename, which is built from the
# run_tag: hyperparams_sweep[_nocer]<run_tag>.toml
const RESULT_REGEX = r"^simulation_results_test_p_([0-9.]+)_s_(\d+)_nlayers_(\d+)_epochs_(\d+)_trained_using_train_p_([0-9.]+)_s_(\d+)(_no_cer)?((?:_u(\d+))?(?:_c([0-9p]+))?_a4([0-9p]+)_a3([0-9p]+)(?:_r(\d+))?)?\.csv$"

function untag_value(tag::AbstractString)::Float64
    """
    Invert the filename-safe token written by sweep_correlation_weight.sh:
    "0" -> 0.0, "0p01" -> 0.01, "1p0" -> 1.0.
    """
    return parse(Float64, replace(String(tag), "p" => "."))
end

function parse_result_filename(filename::String)
    """
    Pull the run parameters out of a results filename. Returns a NamedTuple, or
    `nothing` if the name doesn't match (so unrelated CSVs are skipped).

    `is_sweep` is false for the baseline runs (no `_a4.._a3..` tag); their
    `alpha4`/`alpha3` are NaN. `use_cer` is false only for `_no_cer` files.
    `repeat_index` is 1 for points written without an `_r<k>` suffix.
    """
    m = match(RESULT_REGEX, filename)
    m === nothing && return nothing

    has_tags = m.captures[11] !== nothing
    n_epochs = parse(Int, m.captures[4])
    use_cer  = m.captures[7] === nothing
    run_tag  = m.captures[8] === nothing ? "" : String(m.captures[8])
    # `_u<N>` is omitted for single-rung sweeps; fall back to NaN so those runs
    # still aggregate (they simply have one budget).
    updates = m.captures[9] === nothing ? NaN : parse(Float64, m.captures[9])
    # Reconstruct the registry key. Mirrors sweep_hp_name() in _sweep_common.sh.
    hyperparams_file = isempty(run_tag) ? "" :
        "hyperparams_sweep" * (use_cer ? "" : "_nocer") * run_tag * ".toml"
    return (
        filename       = filename,
        test_p         = parse(Float64, m.captures[1]),
        test_seed      = parse(Int, m.captures[2]),
        n_layers       = parse(Int, m.captures[3]),
        n_epochs       = n_epochs,
        train_p        = parse(Float64, m.captures[5]),
        train_seed     = parse(Int, m.captures[6]),
        use_cer        = use_cer,
        is_sweep       = has_tags,
        run_tag        = run_tag,
        hyperparams_file = hyperparams_file,
        updates_per_epoch = updates,
        gradient_steps = isnan(updates) ? NaN : updates * n_epochs,
        prior_llr_clip = m.captures[10] === nothing ? 0.0 : untag_value(m.captures[10]),
        alpha4         = has_tags ? untag_value(m.captures[11]) : NaN,
        alpha3         = has_tags ? untag_value(m.captures[12]) : NaN,
        repeat_index   = m.captures[13] === nothing ? 1 : parse(Int, m.captures[13]),
    )
end

function read_run_directory(models_dir::String)::DataFrame
    """
    Load models/directory.csv — the registry written by sweep_train.sh recording,
    for every run_tag, the FULL hyperparameter set that produced it.

    Everything is read as String: the annealing schedules are comma-containing
    quoted fields ("2e0,5e0,0.7,down") and there is nothing to gain from coercing
    the numeric ones here. Returns an empty DataFrame (with a warning) if the
    registry is absent, so the rest of the summary still works without it.
    """
    path = joinpath(models_dir, "directory.csv")
    if !isfile(path)
        @warn "no run directory at $(path) — results will not carry their hyperparameters."
        return DataFrame()
    end
    return CSV.read(path, DataFrame; types = String)
end

function attach_directory_metadata(results::DataFrame, registry::DataFrame)::DataFrame
    """
    Left-join every result row to its registry entry on `hyperparams_file`, which
    is the registry's natural key (it distinguishes the CER and no-CER arms that
    share a run_tag). Registry columns already present in `results` — the ones the
    filename itself encodes, e.g. use_CER, alpha4, n_epochs — are skipped so the
    filename stays the single source of truth for those and any disagreement is
    caught by `check_directory_consistency` rather than silently overwritten.

    Unmatched rows (baselines from earlier campaigns, which have no run_tag) get
    empty strings rather than being dropped.
    """
    joined = copy(results)
    if nrow(registry) == 0
        return joined
    end

    lookup = Dict{String, Int}()
    for i in 1:nrow(registry)
        lookup[registry[i, :hyperparams_file]] = i
    end

    row_index = Vector{Int}(undef, nrow(joined))
    for r in 1:nrow(joined)
        row_index[r] = get(lookup, joined[r, :hyperparams_file], 0)
    end

    for column in names(registry)
        column in names(joined) && continue
        joined[!, column] = [row_index[r] == 0 ? "" : registry[row_index[r], column]
                             for r in 1:nrow(joined)]
    end

    n_matched = count(!=(0), row_index)
    n_sweep = count(joined.is_sweep)
    println("  registry join: $(n_matched)/$(n_sweep) sweep result(s) matched a directory.csv entry")
    if n_matched < n_sweep
        unmatched = unique(joined[(joined.is_sweep) .& (row_index .== 0), :hyperparams_file])
        @warn "some results have no registry entry" missing_keys=first(unmatched, 5)
    end
    return joined
end

function check_directory_consistency(joined::DataFrame)
    """
    Cross-check the two independent sources of truth: what the FILENAME says a run
    was, and what directory.csv says it was configured as. They are produced by
    different code paths (src/train.jl builds the filename; _sweep_common.sh writes
    the registry), so agreement is real evidence the join is sound — and a
    mismatch would mean a result is being attributed to the wrong settings.
    """
    ("alpha4_correlation_weight" in names(joined)) || return nothing
    checked = joined[joined.is_sweep .& (joined.alpha4_correlation_weight .!= ""), :]
    nrow(checked) == 0 && return nothing

    mismatches = 0
    for row in eachrow(checked)
        registry_alpha4 = tryparse(Float64, row.alpha4_correlation_weight)
        registry_alpha3 = tryparse(Float64, row.alpha3_sparsity_importance)
        registry_updates = tryparse(Float64, row.n_gradient_updates_per_epoch)
        registry_alpha4 === nothing && continue
        agrees = isapprox(registry_alpha4, row.alpha4; atol = 1e-9) &&
                 isapprox(registry_alpha3, row.alpha3; atol = 1e-9) &&
                 (isnan(row.updates_per_epoch) || isapprox(registry_updates, row.updates_per_epoch; atol = 1e-9))
        agrees || (mismatches += 1)
    end
    if mismatches == 0
        println("  consistency: filename and directory.csv agree on (alpha4, alpha3, updates) for all $(nrow(checked)) row(s)")
    else
        @warn "filename and directory.csv DISAGREE on $(mismatches) row(s) — results may be mis-attributed"
    end
    return nothing
end

function read_results_dir(results_dir::String)::DataFrame
    """
    Read every parseable `simulation_results_*.csv` in `results_dir` into one
    DataFrame (one row per file).

    The standard error is RECOMPUTED here as sqrt(mu (1 - mu) / N) rather than
    taken from the file's `std_logical_error_rate` column, so that results
    written before the SEM fix (which divided by n_layers instead of the sample
    count) are still summarised correctly.
    """
    isdir(results_dir) || error("no such directory: $(results_dir)")

    rows = NamedTuple[]
    for filename in sort(readdir(results_dir))
        endswith(filename, ".csv") || continue
        parsed = parse_result_filename(filename)
        parsed === nothing && continue

        table = CSV.read(joinpath(results_dir, filename), DataFrame)
        nrow(table) == 0 && continue
        record = table[1, :]

        num_failures = Int(record.num_failures)
        num_samples  = Int(record.num_samples_per_error_rate)
        logical_error_rate = num_samples > 0 ? num_failures / num_samples : NaN
        standard_error = (num_samples > 0 && 0.0 <= logical_error_rate <= 1.0) ?
            sqrt(logical_error_rate * (1 - logical_error_rate) / num_samples) : NaN
        runtime = hasproperty(record, :runtime) ? Float64(record.runtime) : NaN

        push!(rows, merge(parsed, (
            num_failures = num_failures,
            num_samples  = num_samples,
            logical_error_rate = logical_error_rate,
            standard_error = standard_error,
            runtime = runtime,
        )))
    end

    isempty(rows) && error("no parseable simulation_results_*.csv found in $(results_dir)")
    return DataFrame(rows)
end

function two_count_z(n1::Real, n0::Real)::Float64
    """
    z score for the difference of two independent failure counts at equal
    sample size: z = (n1 - n0) / sqrt(n1 + n0). Positive => `n1` is WORSE.
    """
    total = n1 + n0
    total <= 0 && return NaN
    return (n1 - n0) / sqrt(total)
end

function baseline_failures(results::DataFrame, test_p::Float64, want_cer::Bool)
    """
    Failure count of the non-sweep baseline at `test_p` (CER when `want_cer`,
    else no-CER). Returns `nothing` when that baseline isn't in the folder — the
    baselines are optional, so the summary still works without them.
    """
    mask = (.!results.is_sweep) .& (results.test_p .== test_p) .& (results.use_cer .== want_cer)
    subset = results[mask, :]
    nrow(subset) == 0 && return nothing
    return subset[1, :num_failures]
end

function build_comparison_table(results::DataFrame)::DataFrame
    """
    Restrict to the sweep points and attach the two comparisons described in the
    file header: against the alpha4 = 0 control at the same (p, alpha3), and
    against the no-CER baseline at the same p.
    """
    sweep = results[results.is_sweep, :]
    nrow(sweep) == 0 && error("no sweep points (files with an _a4.._a3.. tag) found.")
    sweep = sort(sweep, [:test_p, :alpha3, :alpha4])

    ratio_to_control = Float64[]
    z_to_control     = Float64[]
    ratio_to_nocer   = Float64[]
    z_to_nocer       = Float64[]

    for row in eachrow(sweep)
        # alpha4 = 0 control at the same (p, alpha3).
        control_mask = (sweep.test_p .== row.test_p) .& (sweep.alpha3 .== row.alpha3) .&
                       (sweep.alpha4 .== 0.0)
        control = sweep[control_mask, :]
        if nrow(control) == 1 && control[1, :num_failures] > 0
            push!(ratio_to_control, row.num_failures / control[1, :num_failures])
            push!(z_to_control, two_count_z(row.num_failures, control[1, :num_failures]))
        else
            push!(ratio_to_control, NaN)
            push!(z_to_control, NaN)
        end

        # no-CER baseline at the same p.
        nocer = baseline_failures(results, row.test_p, false)
        if nocer !== nothing && nocer > 0
            push!(ratio_to_nocer, row.num_failures / nocer)
            push!(z_to_nocer, two_count_z(row.num_failures, nocer))
        else
            push!(ratio_to_nocer, NaN)
            push!(z_to_nocer, NaN)
        end
    end

    sweep.ratio_vs_alpha4_0 = ratio_to_control
    sweep.z_vs_alpha4_0     = z_to_control
    sweep.ratio_vs_no_cer   = ratio_to_nocer
    sweep.z_vs_no_cer       = z_to_nocer
    return sweep
end

function aggregate_repeats(sweep::DataFrame)::DataFrame
    """
    Collapse repeats into one row per ARM = (test_p, use_cer, alpha4, alpha3),
    reporting the mean and sd of the per-repeat logical error rate.

    The sd here is TRAINING variance (different random initialisations), which is
    the uncertainty that actually matters when comparing arms. Pooling the raw
    failure counts instead would quote a Monte-Carlo error bar ~10x too small and
    make noise look significant — the trap that made the earlier single-seed
    comparisons unreadable.
    """
    keys = unique(sweep[:, [:test_p, :use_cer, :alpha4, :alpha3, :gradient_steps]])
    rows = NamedTuple[]
    for key in eachrow(keys)
        mask = (sweep.test_p .== key.test_p) .& (sweep.use_cer .== key.use_cer) .&
               (isnan(key.alpha4) ? isnan.(sweep.alpha4) : sweep.alpha4 .== key.alpha4) .&
               (isnan(key.alpha3) ? isnan.(sweep.alpha3) : sweep.alpha3 .== key.alpha3) .&
               (isnan(key.gradient_steps) ? isnan.(sweep.gradient_steps) :
                    sweep.gradient_steps .== key.gradient_steps)
        block = sweep[mask, :]
        rates = collect(Float64, block.logical_error_rate)
        push!(rows, (
            test_p = key.test_p, use_cer = key.use_cer,
            alpha4 = key.alpha4, alpha3 = key.alpha3,
            gradient_steps = key.gradient_steps,
            n_repeats = nrow(block),
            mean_ler = mean(rates),
            sd_ler = length(rates) > 1 ? std(rates) : NaN,
            min_ler = minimum(rates), max_ler = maximum(rates),
            mean_failures = mean(collect(Float64, block.num_failures)),
            total_failures = sum(block.num_failures),
            total_samples = sum(block.num_samples),
        ))
    end
    return sort(DataFrame(rows), [:test_p, :gradient_steps, :use_cer, :alpha4, :alpha3])
end

function welch_t(mean1::Float64, sd1::Float64, n1::Int, mean2::Float64, sd2::Float64, n2::Int)::Float64
    """
    Welch t statistic for two arms' repeat-level means. NaN when either arm has
    fewer than two repeats (no variance estimate available).
    """
    (n1 < 2 || n2 < 2 || !isfinite(sd1) || !isfinite(sd2)) && return NaN
    denominator = sqrt(sd1^2 / n1 + sd2^2 / n2)
    denominator == 0 && return NaN
    return (mean1 - mean2) / denominator
end

function compare_arms_to_no_cer(agg::DataFrame)::DataFrame
    """
    For each CER arm, compare against the no-CER arm at the same p: the ratio of
    mean logical error rates and a Welch t on the repeat-level means. Negative t
    means the CER arm is BETTER.
    """
    ratios = Float64[]
    tstats = Float64[]
    for row in eachrow(agg)
        # The no-CER reference must come from the SAME budget rung — comparing
        # across rungs would confound the arm effect with the training budget.
        same_budget = isnan(row.gradient_steps) ? isnan.(agg.gradient_steps) :
                          agg.gradient_steps .== row.gradient_steps
        reference = agg[(agg.test_p .== row.test_p) .& (.!agg.use_cer) .& same_budget, :]
        if row.use_cer && nrow(reference) == 1 && reference[1, :mean_ler] > 0
            push!(ratios, row.mean_ler / reference[1, :mean_ler])
            push!(tstats, welch_t(row.mean_ler, row.sd_ler, row.n_repeats,
                                  reference[1, :mean_ler], reference[1, :sd_ler],
                                  reference[1, :n_repeats]))
        else
            push!(ratios, NaN)
            push!(tstats, NaN)
        end
    end
    agg.ratio_vs_no_cer_arm = ratios
    agg.welch_t_vs_no_cer = tstats
    return agg
end

function print_arm_tables(agg::DataFrame)
    """
    One block per p: every arm's mean LER +/- training sd, and the CER-vs-no-CER
    verdict. This is the table that answers the experiment.
    """
    for test_p in sort(unique(agg.test_p))
      for steps in sort(unique(agg[agg.test_p .== test_p, :gradient_steps]); lt = (a, b) -> isnan(b) || (!isnan(a) && a < b))
        block = agg[(agg.test_p .== test_p) .&
                    (isnan(steps) ? isnan.(agg.gradient_steps) : agg.gradient_steps .== steps), :]
        nrow(block) == 0 && continue
        println()
        println("="^92)
        steps_label = isnan(steps) ? "unspecified" : string(Int(steps))
        @printf("p = %g   |   gradient steps = %s   —   mean +/- sd over training repeats\n",
                test_p, steps_label)
        println("="^92)
        @printf("  %-8s %-8s %-6s %10s %12s %10s %10s %8s\n",
                "use_CER", "alpha4", "reps", "mean LER", "sd (train)", "min", "max", "t vs noCER")
        for row in eachrow(block)
            alpha4_label = row.use_cer ? (isnan(row.alpha4) ? "-" : string(row.alpha4)) : "n/a"
            t_label = isfinite(row.welch_t_vs_no_cer) ? @sprintf("%8.2f", row.welch_t_vs_no_cer) : "       -"
            @printf("  %-8s %-8s %-6d %10.3e %12.2e %10.3e %10.3e %s\n",
                    string(row.use_cer), alpha4_label, row.n_repeats,
                    row.mean_ler, row.sd_ler, row.min_ler, row.max_ler, t_label)
        end

        cer_rows = block[block.use_cer, :]
        if nrow(cer_rows) > 0 && any(isfinite, cer_rows.ratio_vs_no_cer_arm)
            best = cer_rows[argmin(replace(cer_rows.ratio_vs_no_cer_arm, NaN => Inf)), :]
            println()
            @printf("  best CER arm: alpha4 = %s  ->  %.3fx the no-CER arm (Welch t = %.2f)\n",
                    string(best.alpha4), best.ratio_vs_no_cer_arm, best.welch_t_vs_no_cer)
            verdict = if !isfinite(best.welch_t_vs_no_cer)
                "need >= 2 repeats per arm to judge."
            elseif best.welch_t_vs_no_cer < -2.2
                "CER BEATS no-CER (t < -2.2)."
            elseif best.welch_t_vs_no_cer > 2.2
                "CER is WORSE than no-CER."
            else
                "no significant difference — CER neither helps nor hurts here."
            end
            println("  VERDICT: $(verdict)")
        end
      end
    end
    println()
end

function print_pivot_tables(sweep::DataFrame, results::DataFrame)
    """
    Print, per p: a failures pivot (rows alpha4, columns alpha3), the ratio to
    the alpha4 = 0 control, and the available baselines.
    """
    alpha4_values = sort(unique(sweep.alpha4))
    alpha3_values = sort(unique(sweep.alpha3))

    for test_p in sort(unique(sweep.test_p))
        block = sweep[sweep.test_p .== test_p, :]
        println()
        println("="^78)
        @printf("p = %g      (%d samples per point)\n", test_p, block[1, :num_samples])
        println("="^78)

        cer_baseline   = baseline_failures(results, test_p, true)
        nocer_baseline = baseline_failures(results, test_p, false)
        println("  baselines:  CER = $(cer_baseline === nothing ? "n/a" : cer_baseline)" *
                "   no-CER = $(nocer_baseline === nothing ? "n/a" : nocer_baseline)")
        println()

        # --- failures -------------------------------------------------------
        print("  num_failures      ")
        for a3 in alpha3_values
            @printf("  a3=%-8g", a3)
        end
        println()
        for a4 in alpha4_values
            @printf("    alpha4 = %-6g", a4)
            for a3 in alpha3_values
                cell = block[(block.alpha4 .== a4) .& (block.alpha3 .== a3), :]
                if nrow(cell) == 1
                    @printf("  %10d", cell[1, :num_failures])
                else
                    @printf("  %10s", "-")
                end
            end
            println()
        end

        # --- ratio to the alpha4 = 0 control --------------------------------
        println()
        print("  ratio vs alpha4=0 ")
        for a3 in alpha3_values
            @printf("  a3=%-8g", a3)
        end
        println()
        for a4 in alpha4_values
            @printf("    alpha4 = %-6g", a4)
            for a3 in alpha3_values
                cell = block[(block.alpha4 .== a4) .& (block.alpha3 .== a3), :]
                if nrow(cell) == 1 && isfinite(cell[1, :ratio_vs_alpha4_0])
                    @printf("  %10.3f", cell[1, :ratio_vs_alpha4_0])
                else
                    @printf("  %10s", "-")
                end
            end
            println()
        end

        # --- best point -----------------------------------------------------
        best = block[argmin(block.num_failures), :]
        println()
        @printf("  best: alpha4 = %g, alpha3 = %g  ->  %d failures (LER = %.3e)\n",
                best.alpha4, best.alpha3, best.num_failures, best.logical_error_rate)
    end
    println()
end

function log_safe_yerror(values::Vector{Float64}, errors::Vector{Float64}, floor_value::Float64)
    """
    Asymmetric (lower, upper) error-bar deltas that stay positive on a log axis:
    the lower delta is capped so `value - delta` never reaches `floor_value`.
    """
    upper = copy(errors)
    lower = similar(upper)
    for i in eachindex(values)
        lower_end = max(values[i] - errors[i], floor_value)
        lower[i] = values[i] - lower_end
    end
    return (lower, upper)
end

function plot_sweep(sweep::DataFrame, results::DataFrame, output_path::String)::String
    """
    One panel per p: logical error rate (log y) against alpha4 on a CATEGORICAL
    x axis — alpha4 = 0 is a real sweep point and cannot sit on a log axis, so
    the values are drawn at equal spacing and labelled instead. One line per
    alpha3, with the CER and no-CER baselines as horizontal references.
    """
    alpha4_values = sort(unique(sweep.alpha4))
    alpha3_values = sort(unique(sweep.alpha3))
    test_p_values = sort(unique(sweep.test_p))
    positions = Dict(value => index for (index, value) in enumerate(alpha4_values))
    colors = [:blue, :red, :green, :orange, :purple]

    # Shared y-limits across panels so the two p values are visually comparable.
    all_rates = filter(isfinite, sweep.logical_error_rate)
    all_rates = filter(>(0.0), all_rates)
    y_floor = isempty(all_rates) ? 1e-12 : 10.0^floor(log10(minimum(all_rates)))

    panels = Any[]
    for test_p in test_p_values
        block = sweep[sweep.test_p .== test_p, :]
        panel = plot(;
            title = "p = $(test_p)",
            xlabel = "\$\\alpha_4\$  (correlation_weight)",
            ylabel = "Logical Error Rate",
            yscale = :log10,
            xticks = (1:length(alpha4_values), string.(alpha4_values)),
            xlims = (0.5, length(alpha4_values) + 0.5),
            legend = :topleft,
        )

        for (series_index, a3) in enumerate(alpha3_values)
            series = sort(block[block.alpha3 .== a3, :], :alpha4)
            nrow(series) == 0 && continue
            xs = [positions[value] for value in series.alpha4]
            ys = collect(Float64, series.logical_error_rate)
            es = collect(Float64, series.standard_error)
            color = colors[mod1(series_index, length(colors))]
            plot!(panel, xs, ys;
                yerr = log_safe_yerror(ys, es, y_floor),
                label = "\$\\alpha_3\$ = $(a3)", color = color,
                marker = :circle, markersize = 4, linewidth = 2)
        end

        cer_baseline   = baseline_failures(results, test_p, true)
        nocer_baseline = baseline_failures(results, test_p, false)
        samples = block[1, :num_samples]
        if cer_baseline !== nothing
            hline!(panel, [cer_baseline / samples];
                   label = "CER baseline", color = :black, linestyle = :dash)
        end
        if nocer_baseline !== nothing
            hline!(panel, [nocer_baseline / samples];
                   label = "no-CER baseline", color = :gray, linestyle = :dot)
        end

        push!(panels, panel)
    end

    figure = plot(panels...; layout = (1, length(panels)), size = (560 * length(panels), 520))
    mkpath(dirname(output_path))
    savefig(figure, output_path)
    return output_path
end

# ----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    # Defaults; override positionally:
    #   julia --project=expts expts/misc/summarize_cer_sweep.jl <workdir> <codename>
    work_dir = length(ARGS) >= 1 ? ARGS[1] : "./../data"
    codename = length(ARGS) >= 2 ? ARGS[2] : "72q_BB_cycles_1"

    results_dir = joinpath(work_dir, codename, "results")
    println("[summarize_cer_sweep] reading $(results_dir)")

    results = read_results_dir(results_dir)
    n_sweep = count(results.is_sweep)
    println("  parsed $(nrow(results)) result file(s): $(n_sweep) sweep point(s), " *
            "$(nrow(results) - n_sweep) baseline(s)")

    # Join every result to the full hyperparameter set that produced it.
    models_dir = joinpath(work_dir, codename, "models")
    registry = read_run_directory(models_dir)
    results = attach_directory_metadata(results, registry)
    check_directory_consistency(results)

    sweep = build_comparison_table(results)
    print_pivot_tables(sweep, results)

    # Arm-level view: collapse repeats, compare each CER arm to the no-CER arm.
    aggregated = compare_arms_to_no_cer(aggregate_repeats(sweep))
    print_arm_tables(aggregated)

    output_csv = joinpath(results_dir, "cer_sweep_summary.csv")
    CSV.write(output_csv, sweep)
    println("  per-run CSV -> $(output_csv)   ($(ncol(sweep)) columns, incl. every " *
            "hyperparameter from directory.csv)")

    arms_csv = joinpath(results_dir, "cer_sweep_arms.csv")
    CSV.write(arms_csv, aggregated)
    println("  arm CSV     -> $(arms_csv)")

    output_plot = joinpath(work_dir, codename, "plots", "cer_sweep.pdf")
    plot_sweep(sweep, results, output_plot)
    println("  figure      -> $(output_plot)")
end

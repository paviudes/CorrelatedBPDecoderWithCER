# ============================================================================
# plot_standard_vs_neural.jl — logical error rate vs p, standard vs neural,
# for several codes on one figure.
# ============================================================================
#
# Each code is a colour; BP-OSD is a solid line and the neural decoder a dashed
# line, with an optional third decoder (plain BP) drawn dotted when its data is
# supplied. So the legend needs only the decoder entries (solid / dotted /
# dashed) while colour distinguishes the codes. A per-code comparison CSV is
# written alongside for inspection; the gain Δ uses plain BP as the baseline when
# present, else BP-OSD.
# ============================================================================

function _logical_error_rate_vs_p(stats_dataframe::DataFrame)::Matrix{Float64}
    """
    Extract (p, mean logical error rate, mean std) from a decoder-statistics DataFrame,
    grouping rows by p (from the `test_p_<p>_s_<sample>` tag in `error_model_parameters_description`)
    and sorting by p. Returns an (n_p x 3) matrix with columns [p, LER, std].
    """
    logerrs_by_p = Dict{Float64, Vector{Tuple{Float64, Float64}}}()
    for row in eachrow(stats_dataframe)
        description = String(row.error_model_parameters_description)
        # Anchor on `test_p_` so a `p_...` inside a codename can't be picked up.
        p_match = match(r"test_p_([0-9]*\.?[0-9]+)", description)
        if p_match === nothing
            continue
        end
        p_value = parse(Float64, p_match.captures[1])
        entry = get!(logerrs_by_p, p_value, Tuple{Float64, Float64}[])
        push!(entry, (Float64(row.average_logical_error_rate), Float64(row.std_logical_error_rate)))
    end

    p_values = sort(collect(keys(logerrs_by_p)))
    result = Matrix{Float64}(undef, length(p_values), 3)
    for (i, p_value) in enumerate(p_values)
        entries = logerrs_by_p[p_value]
        result[i, 1] = p_value
        result[i, 2] = mean(first.(entries))
        result[i, 3] = mean(last.(entries))
    end
    return result
end

function _comparison_dataframe(bposd_matrix::Matrix{Float64}, neural_matrix::Matrix{Float64};
        bp_matrix::Union{Nothing, Matrix{Float64}} = nothing)::DataFrame
    """
    Join the standard (BP-OSD), optional plain-BP, and neural [p, LER, std]
    matrices on p and build the comparison table. Columns:
        p,
        average_logical_error_rate_BP_OSD, standard_error_BP_OSD,
        [average_logical_error_rate_BP, standard_error_BP,]   # only if bp_matrix given
        average_logical_error_rate_Neural_BP, standard_error_Neural_BP,
        performance_gain, gain_baseline.
    The gain is LER(baseline) / LER(Neural BP), where the baseline is plain BP
    when `bp_matrix` is supplied, else BP-OSD (recorded per row in gain_baseline).
    The table is keyed on the baseline's p grid; only p values also present for
    the neural decoder are kept, and any other decoder's value missing at a kept
    p is written as NaN.
    """
    to_lookup(m) = Dict(m[r, 1] => (m[r, 2], m[r, 3]) for r in 1:size(m, 1))
    neural_by_p = to_lookup(neural_matrix)
    bposd_by_p  = to_lookup(bposd_matrix)
    bp_by_p     = bp_matrix === nothing ? nothing : to_lookup(bp_matrix)

    has_bp = bp_matrix !== nothing
    baseline_matrix = has_bp ? bp_matrix : bposd_matrix
    baseline_name = has_bp ? "BP" : "BP-OSD"

    p_values         = Float64[]
    ler_bposd        = Float64[]
    std_bposd        = Float64[]
    ler_bp           = Float64[]
    std_bp           = Float64[]
    ler_neural       = Float64[]
    std_neural       = Float64[]
    performance_gain = Float64[]
    for r in 1:size(baseline_matrix, 1)
        p_value = baseline_matrix[r, 1]
        haskey(neural_by_p, p_value) || continue
        (neural_ler, neural_std) = neural_by_p[p_value]
        (bposd_ler, bposd_std)   = get(bposd_by_p, p_value, (NaN, NaN))
        push!(p_values, p_value)
        push!(ler_bposd, bposd_ler)
        push!(std_bposd, bposd_std)
        push!(ler_neural, neural_ler)
        push!(std_neural, neural_std)
        if has_bp
            (bp_ler_value, bp_std_value) = bp_by_p[p_value]  # present: we iterate BP's grid
            push!(ler_bp, bp_ler_value)
            push!(std_bp, bp_std_value)
            baseline_ler = bp_ler_value
        else
            baseline_ler = bposd_ler
        end
        push!(performance_gain, baseline_ler / neural_ler)
    end

    comparison_df = DataFrame(
        p = p_values,
        average_logical_error_rate_BP_OSD = ler_bposd,
        standard_error_BP_OSD = std_bposd,
    )
    if has_bp
        comparison_df.average_logical_error_rate_BP = ler_bp
        comparison_df.standard_error_BP = std_bp
    end
    comparison_df.average_logical_error_rate_Neural_BP = ler_neural
    comparison_df.standard_error_Neural_BP = std_neural
    comparison_df.performance_gain = performance_gain
    comparison_df.gain_baseline = fill(baseline_name, length(p_values))
    return comparison_df
end

function _log_safe_yerror(values::AbstractVector{Float64}, stds::AbstractVector{Float64}, y_floor::Float64)
    """
    Build asymmetric (lower, upper) error-bar deltas that stay valid on a log
    y-axis. The upper delta is the std; the lower delta is capped so the lower
    end never drops to/below `y_floor` (value - std can be <= 0 when std > value,
    which breaks log-scale error bars and mangles the line).
    """
    upper = collect(stds)
    lower = similar(upper)
    for k in eachindex(values)
        lower_end = max(values[k] - stds[k], y_floor)
        lower[k] = values[k] - lower_end
    end
    return (lower, upper)
end

function _build_ler_panel(
    standard_matrices::Vector{Matrix{Float64}},
    neural_matrices::Vector{Matrix{Float64}},
    codename_labels::Vector{String},
    standard_label::String,
    neural_label::String,
    colors::Vector{Symbol},
    x_limits,
    xscale::Symbol,
    yscale::Symbol;
    bp_matrices::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    bp_label::String = "BP",
)
    """
    Build the TOP panel: logical error rate vs p. Each code is a colour;
    BP-OSD is a solid line + circle and neural-BP a dashed line + square. When
    `bp_matrices` is supplied, plain BP is added as a dotted line + diamond (same
    colour per code). The legend carries the decoder entries (`standard_label`
    solid, `bp_label` dotted if present, `neural_label` dashed) plus one coloured
    marker per code from `codename_labels`. Returns the `Plots.jl` plot. X tick
    labels are kept visible (both panels show them).
    """
    is_y_log = yscale in (:log10, :log2, :ln, :log)
    n_codes = length(standard_matrices)
    has_bp = bp_matrices !== nothing

    ler_panel = plot(;
        xlabel = "\nPhysical Error Probability \$p\$",
        ylabel = "Logical Error Rate\n",
        xscale = xscale,
        yscale = yscale,
        legend = :bottomright,
    )
    if x_limits !== nothing
        plot!(ler_panel; xlims = x_limits)
    end

    # LER y-limits (decade-snapped only on a log y-axis) + error-bar floor.
    ylim_matrices = has_bp ?
        Iterators.flatten((standard_matrices, neural_matrices, bp_matrices)) :
        Iterators.flatten((standard_matrices, neural_matrices))
    all_y = Float64[]
    for matrix in ylim_matrices
        for r in 1:size(matrix, 1)
            ler = matrix[r, 2]
            err = matrix[r, 3]
            if !is_y_log || ler > 0
                push!(all_y, ler)
                push!(all_y, ler + err)
            end
        end
    end
    y_floor = 1e-12
    if is_y_log && !isempty(all_y)
        y_low  = 10.0 ^ floor(log10(minimum(all_y)))
        y_high = 10.0 ^ ceil(log10(maximum(all_y)))
        y_floor = y_low
        plot!(ler_panel; ylims = (y_low, y_high))
    end

    # A single code's LER line, with log-safe error bars.
    yerr_for(matrix) = is_y_log ? _log_safe_yerror(matrix[:, 2], matrix[:, 3], y_floor) : matrix[:, 3]
    draw_line!(matrix, style, mark, color) = plot!(ler_panel, matrix[:, 1], matrix[:, 2];
        yerr = yerr_for(matrix), color = color, linestyle = style,
        marker = mark, markersize = 3, linewidth = 1, label = "")

    # Legend: the decoder entries (solid / dotted / dashed) + one coloured marker
    # per code.
    plot!(ler_panel, [NaN], [NaN]; label = standard_label, color = :black, linestyle = :solid, linewidth = 1)
    if has_bp
        plot!(ler_panel, [NaN], [NaN]; label = bp_label, color = :black, linestyle = :dot, linewidth = 1)
    end
    plot!(ler_panel, [NaN], [NaN]; label = neural_label, color = :black, linestyle = :dash, linewidth = 1)
    for i in 1:n_codes
        color = colors[mod1(i, length(colors))]
        scatter!(ler_panel, [NaN], [NaN]; label = codename_labels[i], color = color,
                 markershape = :circle, markersize = 5, markerstrokewidth = 0)

        draw_line!(standard_matrices[i], :solid, :circle, color)
        if has_bp
            draw_line!(bp_matrices[i], :dot, :diamond, color)
        end
        draw_line!(neural_matrices[i], :dash, :square, color)
    end
    return ler_panel
end

function _build_gain_panel(
    comparison_dataframes::Vector{DataFrame},
    codename_labels::Vector{String},
    colors::Vector{Symbol},
    x_limits,
    xscale::Symbol,
    gain_ylabel::String,
)
    """
    Build the BOTTOM panel: the performance gain Δ = LER(baseline) / LER(neural)
    vs p, one line per code (matching colour). A dotted Δ = 1 break-even reference
    is drawn (neural is better above it). The legend maps each coloured marker to
    its code label (`codename_labels`). The y-axis is always log. `gain_ylabel`
    names the axis after the baseline actually used (e.g. `\$\\Delta_{\\text{BP}}\$`
    when BP is present, `\$\\Delta_{\\text{BP-OSD}}\$` otherwise). Returns the
    `Plots.jl` plot.
    """
    n_codes = length(comparison_dataframes)

    gain_panel = plot(;
        xlabel = "\nPhysical Error Probability \$p\$",
        ylabel = gain_ylabel,
        xscale = xscale,
        yscale = :log10,
        legend = :topright,
    )
    if x_limits !== nothing
        plot!(gain_panel; xlims = x_limits)
    end

    # Gain y-limits (log), decade-snapped and always including Δ = 1.
    all_gain = Float64[]
    for comparison_df in comparison_dataframes
        for gain in comparison_df.performance_gain
            if isfinite(gain) && gain > 0
                push!(all_gain, gain)
            end
        end
    end
    if !isempty(all_gain)
        gain_low  = 10.0 ^ floor(log10(min(minimum(all_gain), 1.0)))
        gain_high = 10.0 ^ ceil(log10(maximum(all_gain)))
        plot!(gain_panel; ylims = (gain_low, gain_high))
    end

    # Δ = 1 break-even reference (neural is better above the line).
    hline!(gain_panel, [1.0]; color = :gray, linestyle = :dot, linewidth = 1, label = "")

    for i in 1:n_codes
        color = colors[mod1(i, length(colors))]
        comparison_df = comparison_dataframes[i]
        # Non-finite / non-positive gains -> NaN so the line breaks cleanly there.
        gains = Float64[]
        for gain in comparison_df.performance_gain
            if isfinite(gain) && gain > 0
                push!(gains, gain)
            else
                push!(gains, NaN)
            end
        end
        plot!(gain_panel, comparison_df.p, gains;
            color = color, linestyle = :solid, marker = :circle, markersize = 3,
            linewidth = 1, label = codename_labels[i])
    end
    return gain_panel
end

function plot_standard_vs_neural(
    standard_dataframes::Vector{DataFrame},
    neural_dataframes::Vector{DataFrame},
    codename_labels::Vector{String},
    standard_label::String,
    neural_label::String,
    output_path::String,
    comparison_csv_paths::Vector{String};
    bp_dataframes::Union{Nothing, Vector{DataFrame}} = nothing,
    bp_label::String = "BP",
    xscale::Symbol = :log10,
    yscale::Symbol = :log10,
)
    """
    Two-panel figure (shared x-axis) for several codes:
      - TOP: logical error rate vs physical error probability `p`. Each code is a
        colour; BP-OSD solid, neural-BP dashed, and — when `bp_dataframes` is
        given — plain BP dotted. Legend = the decoder entries (`standard_label`
        solid, `bp_label` dotted if present, `neural_label` dashed) plus one
        coloured-marker entry per code from `codename_labels`.
      - BOTTOM: the performance gain Δ = LER(baseline) / LER(neural) vs the same
        `p`, one line per code (matching colour), with a dotted Δ = 1 break-even
        reference (neural is better above the line). The baseline is plain BP when
        `bp_dataframes` is supplied, else BP-OSD.

    `standard_dataframes[i]` / `neural_dataframes[i]` (and optional
    `bp_dataframes[i]`) are the aggregated statistics for code `i` (columns
    `error_model_parameters_description`, `average_logical_error_rate`,
    `std_logical_error_rate`). `xscale`/`yscale` set the TOP panel's scales
    (:log10 or :identity); decade-snapping of limits is applied only on log axes,
    and the gain panel always uses a log y-axis. For each code the comparison
    table (see `_comparison_dataframe`) is written to `comparison_csv_paths[i]`.
    The figure is written to `output_path`, whose path is returned.
    """
    n_codes = length(standard_dataframes)
    if length(neural_dataframes) != n_codes
        error("standard and neural dataframe lists must have the same length " *
              "($(n_codes) vs $(length(neural_dataframes))).")
    end
    if length(codename_labels) != n_codes
        error("codename_labels must have one entry per code " *
              "($(length(codename_labels)) vs $(n_codes)).")
    end
    if length(comparison_csv_paths) != n_codes
        error("comparison_csv_paths must have one entry per code " *
              "($(length(comparison_csv_paths)) vs $(n_codes)).")
    end
    if bp_dataframes !== nothing && length(bp_dataframes) != n_codes
        error("bp_dataframes must have one entry per code " *
              "($(length(bp_dataframes)) vs $(n_codes)).")
    end

    colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown]

    # Per-code [p, LER, std] matrices and the joined comparison tables. Reused for
    # both panels and written out as CSVs. BP is optional (dotted curve + gain
    # baseline); when absent, `bp_matrices` stays `nothing` and the figure is the
    # usual BP-OSD-vs-neural comparison.
    standard_matrices = [_logical_error_rate_vs_p(df) for df in standard_dataframes]
    neural_matrices   = [_logical_error_rate_vs_p(df) for df in neural_dataframes]
    bp_matrices = bp_dataframes === nothing ? nothing :
                  [_logical_error_rate_vs_p(df) for df in bp_dataframes]
    comparison_dataframes = [_comparison_dataframe(standard_matrices[i], neural_matrices[i];
                                bp_matrix = (bp_matrices === nothing ? nothing : bp_matrices[i]))
                             for i in 1:n_codes]
    for i in 1:n_codes
        csv_path = comparison_csv_paths[i]
        mkpath(dirname(csv_path))
        CSV.write(csv_path, comparison_dataframes[i])
    end

    is_x_log = xscale in (:log10, :log2, :ln, :log)

    # Shared x-limits (decade-snapped only on a log x-axis; linear uses the data
    # range plus a small margin, which also stops the [NaN] legend dummies / hline
    # from driving a bogus auto-range like 0..2).
    xlim_matrices = bp_matrices === nothing ?
        Iterators.flatten((standard_matrices, neural_matrices)) :
        Iterators.flatten((standard_matrices, neural_matrices, bp_matrices))
    all_p = Float64[]
    for matrix in xlim_matrices
        for r in 1:size(matrix, 1)
            p_value = matrix[r, 1]
            if !is_x_log || p_value > 0
                push!(all_p, p_value)
            end
        end
    end
    x_limits = nothing
    if !isempty(all_p)
        if is_x_log
            x_limits = (10.0 ^ floor(log10(minimum(all_p))), 10.0 ^ ceil(log10(maximum(all_p))))
        else
            p_min = minimum(all_p)
            p_max = maximum(all_p)
            span = p_max - p_min
            if span == 0.0
                span = max(abs(p_max), 1.0)
            end
            margin = 0.05 * span
            x_limits = (p_min - margin, p_max + margin)
        end
    end

    # Build each panel separately, then stack them with a shared x-axis. Both
    # panels keep their x tick labels (easier to read off the gain against p).
    ler_panel = _build_ler_panel(standard_matrices, neural_matrices, codename_labels,
        standard_label, neural_label, colors, x_limits, xscale, yscale;
        bp_matrices = bp_matrices, bp_label = bp_label)
    # Name the gain axis after the baseline used: BP when present, else BP-OSD.
    gain_ylabel = bp_matrices === nothing ? "\$\\Delta_{\\text{BP-OSD}}\$\n" : "\$\\Delta_{\\text{BP}}\$\n"
    gain_panel = _build_gain_panel(comparison_dataframes, codename_labels, colors, x_limits, xscale, gain_ylabel)

    plt = plot(ler_panel, gain_panel;
        layout = @layout([a{0.68h}; b{0.32h}]),
        link = :x,
        size = (720, 760),
    )

    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

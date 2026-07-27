# ============================================================================
# plot_logical_error_rate.jl — single-decoder logical error rate vs p, for
# several codes on one figure.
# ============================================================================
#
# Unlike plot_standard_vs_neural.jl (which overlays BOTH decoders), this draws
# ONE decoder: one coloured line+marker per code, LER vs physical error
# probability p. Used for the standalone BP-OSD and Neural BP figures. Reuses the
# `_logical_error_rate_vs_p` and `_log_safe_yerror` helpers from
# plot_standard_vs_neural.jl (same module).
# ============================================================================

function plot_logical_error_rate(
    dataframes::Vector{DataFrame},
    codename_labels::Vector{String},
    decoder_label::String,
    output_path::String;
    xscale::Symbol = :log10,
    yscale::Symbol = :log10,
)
    """
    Single-decoder logical-error-rate figure: LER vs physical error probability
    `p`, one coloured line+marker per code.

    `dataframes[i]` is the aggregated decoder statistics for code `i` (columns
    `error_model_parameters_description`, `average_logical_error_rate`,
    `std_logical_error_rate`) and `codename_labels[i]` is its legend entry.
    `decoder_label` names the decoder and is used as the plot title (e.g.
    "BP-OSD" or "Neural BP"). `xscale`/`yscale` are :log10 or :identity;
    decade-snapping of the limits and the log-safe error bars are applied only on
    a log axis. The figure is written to `output_path`, whose path is returned.
    """
    n_codes = length(dataframes)
    if length(codename_labels) != n_codes
        error("codename_labels must have one entry per code " *
              "($(length(codename_labels)) vs $(n_codes)).")
    end

    colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown]
    matrices = [_logical_error_rate_vs_p(df) for df in dataframes]

    is_x_log = xscale in (:log10, :log2, :ln, :log)
    is_y_log = yscale in (:log10, :log2, :ln, :log)

    # x-limits: decade-snapped on a log axis; data range + small margin on linear.
    all_p = Float64[]
    for matrix in matrices
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

    plt = plot(;
        title = decoder_label,
        xlabel = "\nPhysical Error Probability \$p\$",
        ylabel = "Logical Error Rate\n",
        xscale = xscale,
        yscale = yscale,
        legend = :bottomright,
        size = (720, 540),
    )
    if x_limits !== nothing
        plot!(plt; xlims = x_limits)
    end

    # y-limits (decade-snapped only on a log y-axis) + error-bar floor.
    all_y = Float64[]
    for matrix in matrices
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
        plot!(plt; ylims = (y_low, y_high))
    end

    for i in 1:n_codes
        color = colors[mod1(i, length(colors))]
        matrix = matrices[i]
        if is_y_log
            yerr = _log_safe_yerror(matrix[:, 2], matrix[:, 3], y_floor)
        else
            yerr = matrix[:, 3]
        end
        plot!(plt, matrix[:, 1], matrix[:, 2];
            yerr = yerr, color = color, linestyle = :solid,
            marker = :circle, markersize = 3, linewidth = 1,
            label = codename_labels[i])
    end

    mkpath(dirname(output_path))
    savefig(plt, output_path)
    return output_path
end

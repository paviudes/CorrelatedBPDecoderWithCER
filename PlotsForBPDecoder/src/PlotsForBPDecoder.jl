module PlotsForBPDecoder

# ============================================================================
# PlotsForBPDecoder — analysis-only plotting helpers for CorrelatedBPDecoderWithCER
# ============================================================================
#
# This is a sister package to CorrelatedBPDecoderWithCER. It exists to keep
# the heavyweight Plots.jl + StatsPlots.jl stack out of the main package's
# dependency tree so that cluster jobs don't have to precompile or install
# them.
#
# Usage on your Mac:
#
#     julia --project=PlotsForBPDecoder
#     julia> using PlotsForBPDecoder
#     julia> using CSV, DataFrames
#     julia> df = CSV.read("decoder_statistics_ballistic.csv", DataFrame)
#     julia> plot_statistics_for_ballistic_error_model(df, [0.01], [0.001])
#
# First-time install:
#
#     cd CorrelatedBPDecoderWithCER
#     julia --project=PlotsForBPDecoder -e 'using Pkg; Pkg.instantiate()'
#
# Julia 1.11+'s `[sources]` in Project.toml handles the sibling-path resolution
# for CorrelatedBPDecoderWithCER — no manual `Pkg.develop` required.
# ============================================================================

using CSV
using DataFrames
using Statistics
using Plots
using Plots.PlotMeasures
using StatsPlots

# Symbols pulled from the main package. `fmt_probs` is the canonical
# (p, q) → filename-tag formatter; the collectors/savers are used by collect.jl.
using CorrelatedBPDecoderWithCER: fmt_probs, collect_decoder_statistics,
    save_decoder_dataframe, collect_decoder_statistics_correlated,
    collect_standard_decoder_statistics_correlated

# Ballistic error-rate curves (avg logical error rate vs. q).
include("error_rate_curves.jl")
export plot_statistics_for_ballistic_error_model

# Violin plot of per-sample performance gain (BP-OSD LER / Neural BP LER).
include("performance_spread.jl")
export plot_performance_spread

# Logical error rate vs p, standard (solid) vs neural (dashed), multiple codes.
include("plot_standard_vs_neural.jl")
export plot_standard_vs_neural

# Result aggregation / analysis entry points (moved from expts/).
include("collect.jl")
export postprocess_neuralbp_results, collect_results

end # module

# ============================================================================
# postprocess.jl — control panel dispatcher for post-processing operations
# ============================================================================
#
# The actual "control panel" is postprocess_panel.toml: flip `run = true` on the
# sections you want and fill their parameters. This script is a thin dispatcher —
# it reads the TOML and runs each enabled section's handler, in the order below.
#
# From the expts/ directory (its Project.toml has both packages via [sources]):
#   julia --project="./" postprocess.jl                       # run every section with run = true
#   julia --project="./" postprocess.jl --task=neural_gather  # run just one section
#   julia --project="./" postprocess.jl --config=other.toml   # a different panel file
#
# Paths: a global `workdir` is set in the TOML; each section names a `codename`.
# Files are resolved under <workdir>/<codename>/results and plots are written
# into <workdir>/<codename>/plots, so the TOML holds bare filenames.
#
# ADD A NEW OPERATION in three steps:
#   1. add a [section] to postprocess_panel.toml (with `run` + params),
#   2. write a handler `run_<id>(section, id, workdir)` below,
#   3. add ("<id>", run_<id>) to OPERATIONS.
#
# NOTE: pulls in PlotsForBPDecoder (for postprocess_neuralbp_results and the
# plotting handlers), which loads Plots/StatsPlots — fine on a Mac, never on a
# cluster job.
# ============================================================================

using ArgParse
using TOML
using DataFrames
using CorrelatedBPDecoderWithCER   # collect_standard_decoder_statistics, save_decoder_dataframe
using PlotsForBPDecoder            # postprocess_neuralbp_results, plotting helpers

# Fetch a required key from a section, with a clear error naming the section.
function _require(section::AbstractDict, key::String, id::String)
    if !haskey(section, key)
        error("[$(id)] missing required key '$(key)' in the config.")
    end
    value = section[key]
    return value
end

# ----------------------------------------------------------------------------
# Handlers — one per operation. Each takes its TOML section, the section id, and
# the global workdir. Files live under <workdir>/<codename>/results (inputs and
# aggregated outputs) and <workdir>/<codename>/plots (plot outputs).
# ----------------------------------------------------------------------------

function run_standard_bposd(section::AbstractDict, id::String, workdir::String)
    error_model_string = lowercase(strip(String(_require(section, "error_model", id))))
    if !(error_model_string in ("ising", "circuit"))
        error("[$(id)] error_model must be \"Ising\" or \"Circuit\", got \"$(error_model_string)\".")
    end
    error_model::Symbol = :Circuit
    if error_model_string == "ising"
        error_model = :Ising
    end

    codename = String(_require(section, "codename", id))
    prefix = joinpath(workdir, codename)
    failures_file = String(_require(section, "file", id))
    stats_df = collect_standard_decoder_statistics(
        error_model; prefix = prefix, standard_BP_output_file = failures_file)

    out_path = joinpath(prefix, "results", String(_require(section, "out", id)))
    saved_path = save_decoder_dataframe(stats_df, out_path)
    println("  standard ($(error_model)) -> $(saved_path)  ($(nrow(stats_df)) rows)")
    return nothing
end

function run_neural_gather(section::AbstractDict, id::String, workdir::String)
    codename = String(_require(section, "codename", id))
    results_dir = joinpath(workdir, codename, "results")

    filenames = String.(_require(section, "files", id))
    if isempty(filenames)
        error("[$(id)] `files` is empty.")
    end
    file_paths = [joinpath(results_dir, name) for name in filenames]

    out_path = joinpath(results_dir, String(_require(section, "out", id)))
    saved_path = postprocess_neuralbp_results(file_paths, out_path)
    println("  gathered $(length(file_paths)) file(s) -> $(saved_path)")
    return nothing
end

function run_plot_standard_vs_neural(section::AbstractDict, id::String, workdir::String)
    codename = String(_require(section, "codename", id))
    results_dir = joinpath(workdir, codename, "results")
    plots_dir = joinpath(workdir, codename, "plots")

    neural_csv = joinpath(results_dir, String(_require(section, "neural_csv", id)))
    standard_csv = joinpath(results_dir, String(_require(section, "standard_csv", id)))
    plot_path = joinpath(plots_dir, String(_require(section, "plot_file", id)))

    println("  [$(id)] not implemented yet — placeholder.")
    println("    would read : $(neural_csv), $(standard_csv)")
    println("    would write: $(plot_path)")
    # TODO: mkpath(plots_dir); load the two CSVs; call plot_performance_spread /
    # plot_statistics_for_ballistic_error_model from PlotsForBPDecoder.
    return nothing
end

# ----------------------------------------------------------------------------
# Registry — ordered so `run = true` sections execute in a sensible sequence.
# ----------------------------------------------------------------------------
const OPERATIONS = [
    ("standard_bposd",          run_standard_bposd),
    ("neural_gather",           run_neural_gather),
    ("plot_standard_vs_neural", run_plot_standard_vs_neural),
]

function main()
    s = ArgParseSettings(
        description = "Post-processing control panel. Edit postprocess_panel.toml " *
                      "(flip `run = true`, fill params), then run this.")
    @add_arg_table s begin
        "--config"
            help = "path to the control-panel TOML"
            arg_type = String
            default = "postprocess_panel.toml"
        "--task"
            help = "run only this section id (default: run every section with run = true)"
            arg_type = String
            default = ""
    end
    args = parse_args(s)

    config_path = args["config"]
    if !isfile(config_path)
        error("config not found: $(config_path) — see postprocess_panel.toml for the template.")
    end
    cfg = TOML.parsefile(config_path)

    workdir = "./../data"
    if haskey(cfg, "workdir")
        workdir = String(cfg["workdir"])
    end

    only = strip(args["task"])
    ran = 0
    for (id, handler) in OPERATIONS
        section = get(cfg, id, nothing)
        if !isempty(only)
            if id != only
                continue
            end
            if section === nothing
                error("--task=$(only): no [$(only)] section in $(config_path).")
            end
        else
            enabled = section !== nothing && get(section, "run", false) === true
            if !enabled
                continue
            end
        end
        println("[postprocess] running: $(id)")
        handler(section, id, workdir)
        ran += 1
    end

    if ran == 0
        if isempty(only)
            println("[postprocess] nothing to do — set `run = true` on a section in $(config_path).")
        else
            known_ids = join(first.(OPERATIONS), ", ")
            println("[postprocess] unknown --task=\"$(only)\". Known ids: $(known_ids).")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

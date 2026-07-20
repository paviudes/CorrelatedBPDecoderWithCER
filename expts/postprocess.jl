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
using CSV
using DataFrames
using CorrelatedBPDecoderWithCER   # collect_standard_decoder_statistics, save_decoder_dataframe
using PlotsForBPDecoder            # postprocess_neuralbp_results, plot_standard_vs_neural

# Fetch a required key from a section, with a clear error naming the section.
function _require(section::AbstractDict, key::String, id::String)
    if !haskey(section, key)
        error("[$(id)] missing required key '$(key)' in the config.")
    end
    value = section[key]
    return value
end

# Map a scale string ("log10" / "linear") to the Plots symbol (:log10 / :identity).
function _scale_symbol(scale_string::AbstractString)::Symbol
    s = lowercase(strip(String(scale_string)))
    if s in ("linear", "identity")
        return :identity
    end
    if s in ("log10", "log")
        return :log10
    end
    error("scale must be \"log10\" or \"linear\", got \"$(scale_string)\".")
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

    failures_files = String.(_require(section, "files", id))
    if isempty(failures_files)
        error("[$(id)] `files` is empty.")
    end

    # Parse each failures file (from <prefix>/results/) and stack the resulting
    # tables into one. Files that produce no rows (missing/empty) are skipped.
    per_file_frames = DataFrame[]
    for failures_file in failures_files
        file_df = collect_standard_decoder_statistics(
            error_model; prefix = prefix, standard_BP_output_file = failures_file)
        if nrow(file_df) > 0
            push!(per_file_frames, file_df)
        end
    end
    if isempty(per_file_frames)
        error("[$(id)] no rows collected from any of the $(length(failures_files)) file(s) in $(joinpath(prefix, "results")).")
    end
    stats_df = reduce(vcat, per_file_frames)

    out_path = joinpath(prefix, "results", String(_require(section, "output_bposd_file", id)))
    saved_path = save_decoder_dataframe(stats_df, out_path)
    println("  standard ($(error_model)) -> $(saved_path)  " *
            "($(nrow(stats_df)) rows from $(length(failures_files)) file(s))")
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

    out_path = joinpath(results_dir, String(_require(section, "output_neuralbp_file", id)))
    saved_path = postprocess_neuralbp_results(file_paths, out_path)
    println("  gathered $(length(file_paths)) file(s) -> $(saved_path)")
    return nothing
end

function run_plot_standard_vs_neural(section::AbstractDict, id::String, workdir::String)
    codenames = String.(_require(section, "codenames", id))
    if isempty(codenames)
        error("[$(id)] `codenames` is empty.")
    end
    codename_labels = String.(_require(section, "codename_labels", id))
    if length(codename_labels) != length(codenames)
        error("[$(id)] `codename_labels` must have one entry per codename " *
              "($(length(codename_labels)) vs $(length(codenames))).")
    end
    neural_csv = String(_require(section, "neural_csv", id))
    standard_csv = String(_require(section, "standard_csv", id))
    standard_label = String(_require(section, "standard_BP_plot_label", id))
    neural_label = String(_require(section, "neural_BP_plot_label", id))

    # Read the aggregate CSVs for every code (same bare filenames in each
    # <workdir>/<codename>/results). Both must already exist per codename.
    standard_dataframes = DataFrame[]
    neural_dataframes = DataFrame[]
    for codename in codenames
        results_dir = joinpath(workdir, codename, "results")
        standard_path = joinpath(results_dir, standard_csv)
        neural_path = joinpath(results_dir, neural_csv)
        for path in (standard_path, neural_path)
            if !isfile(path)
                error("[$(id)] missing input: $(path) — generate it with the " *
                      "standard_bposd / neural_gather sections.")
            end
        end
        push!(standard_dataframes, CSV.read(standard_path, DataFrame))
        push!(neural_dataframes, CSV.read(neural_path, DataFrame))
    end

    # Combined figure, written to <workdir>/<plot_file> (plot_file may include a
    # subdirectory, e.g. "<codename>/plots/foo.pdf"; missing dirs are created).
    plot_path = joinpath(workdir, String(_require(section, "plot_file", id)))

    # Per-code 6-column comparison CSVs, written into each code's results/ folder
    # as standard_vs_neural_<codename>.csv.
    comparison_csv_paths = [joinpath(workdir, codename, "results", "standard_vs_neural_$(codename).csv")
                            for codename in codenames]

    # Axis scales — default log10; set to "linear" in the TOML to turn off log.
    x_scale = _scale_symbol(get(section, "x_scale", "log10"))
    y_scale = _scale_symbol(get(section, "y_scale", "log10"))

    saved_path = plot_standard_vs_neural(
        standard_dataframes, neural_dataframes, codename_labels,
        standard_label, neural_label, plot_path, comparison_csv_paths;
        xscale = x_scale, yscale = y_scale)
    println("  plotted $(length(codenames)) code(s) -> $(saved_path)")
    println("  comparison CSVs -> $(comparison_csv_paths)")
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

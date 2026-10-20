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
#   julia --project="./" postprocess.jl --task=all_plots      # run a [task_groups] group (several sections)
#   julia --project="./" postprocess.jl --config=postprocess_configs/other.toml   # a different panel
#
# `--task=<x>` matches EITHER a single section id OR a key in the optional
# [task_groups] TOML table (which maps one key to a list of section ids); a group
# runs all its member sections in registry order, regardless of their `run` flag.
#
# Paths: a global `workdir` is set in the TOML; each section names a `codename`.
# Files are resolved under <workdir>/<codename>/results and plots are written
# into <workdir>/<codename>/plots, so the TOML holds bare filenames.
#
# Gather sections take `files` (bare names), `pattern` (shell-style, e.g.
# "simulation_results_*_seed_*.csv"), or both, and REPLACE their output file.
# For (p, sigma, sample) data they add the columns the file NAMES carry — the
# arm and the datasets for Neural BP, sigma for the reference tables; see
# postprocess_configs/postprocess_panel_sweep.toml.
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
using PlotsForBPDecoder            # postprocess_neuralbp_results, plot_standard_vs_neural, plot_logical_error_rate

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

# Output filename for a standard-decoder gather section: canonical key is
# `output_file`; `output_bposd_file` is accepted as a legacy fallback (the old
# BP-OSD-only name) so existing panels keep working without edits.
function _gather_output_filename(section::AbstractDict, id::String)::String
    if haskey(section, "output_file")
        return String(section["output_file"])
    end
    if haskey(section, "output_bposd_file")
        return String(section["output_bposd_file"])
    end
    error("[$(id)] missing required key 'output_file' (legacy alias 'output_bposd_file').")
end

# ----------------------------------------------------------------------------
# Which files a gather reads, and what their names say (2026-10-20).
# ----------------------------------------------------------------------------

"Characters a regex treats specially; a glob pattern escapes them."
const REGEX_SPECIAL_CHARACTERS = Set{Char}(collect("\\^\$.|+()[]{}"))

"""
    glob_regex(pattern) -> Regex

A shell-style file pattern as a regex over the whole name: `*` matches any run of
characters, `?` any one character, every other character itself.
"""
function glob_regex(pattern::String)::Regex
    regex_text::String = "^"
    for character in pattern
        if character == '*'
            regex_text *= ".*"
        elseif character == '?'
            regex_text *= "."
        elseif character in REGEX_SPECIAL_CHARACTERS
            regex_text *= "\\" * character
        else
            regex_text *= character
        end
    end
    regex_text *= "\$"
    pattern_regex::Regex = Regex(regex_text)
    return pattern_regex
end

"""
    gather_file_names(section, id, results_dir) -> Vector{String}

The bare names of the files a gather section reads, sorted, each once: its
`files` list, and every file in `results_dir` that matches its `pattern` (one
shell-style pattern, or a list of them), e.g. "simulation_results_*_seed_*.csv".
"""
function gather_file_names(section::AbstractDict, id::String, results_dir::String)::Vector{String}
    file_names::Vector{String} = String[String(name) for name in get(section, "files", String[])]
    pattern_setting::Any = get(section, "pattern", String[])
    patterns::Vector{String} = String[]
    if pattern_setting isa AbstractString
        push!(patterns, String(pattern_setting))
    else
        patterns = String[String(pattern) for pattern in pattern_setting]
    end
    if !isempty(patterns)
        if !isdir(results_dir)
            error("[$(id)] no results directory at $(results_dir).")
        end
        directory_names::Vector{String} = readdir(results_dir)
        for pattern in patterns
            pattern_regex::Regex = glob_regex(pattern)
            matched_names::Vector{String} = filter(name -> occursin(pattern_regex, name), directory_names)
            if isempty(matched_names)
                @warn "[$(id)] pattern $(repr(pattern)) matches no file in $(results_dir)."
            end
            append!(file_names, matched_names)
        end
    end
    unique_names::Vector{String} = sort(unique(file_names))
    if isempty(unique_names)
        error("[$(id)] nothing to gather: set `pattern` (e.g. \"simulation_results_*_seed_*.csv\") or `files`.")
    end
    return unique_names
end

"""
The name of a Neural BP results file on a (p, sigma, sample) dataset, as
neural_bp_experiments.jl and src/train.jl write it:

    simulation_results_test_<test key>_nlayers_<L>_epochs_<E>_trained_using_train_<train key>[_no_cer]<run tag>[_seed_<n>].csv

with <key> = p_<p>_sig_<sigma>_s_<sample>. `_no_cer` is train.jl's own mark of a
no-CER model, and the run tag names the arm: `_cer` / `_nocer` from batch_run.jl
(with `_<run_label>` after it when one was set), `_cer_histpair` and the like from
a sweep.
"""
const NEURAL_RESULTS_NAME = r"^simulation_results_test_(?<test_key>p_(?<test_p>[^_]+)_sig_(?<test_sigma>[^_]+)_s_(?<test_sample>\d+))_nlayers_(?<n_layers>\d+)_epochs_(?<n_epochs>\d+)_trained_using_train_(?<train_key>p_(?<train_p>[^_]+)_sig_(?<train_sigma>[^_]+)_s_(?<train_sample>\d+))(?<no_cer>_no_cer)?(?<run_tag>.*?)(?:_seed_(?<seed>\d+))?\.csv$"

"""
The name of a reference decoder's table for one (p, sigma, sample) dataset, as it
comes with the data: `<code>_<key>_BP_failure_rates.txt` and
`<code>_<key>BP+OSD_failure_rates_OSD_E_order_2.txt` (no underscore before BP+OSD).
"""
const STANDARD_RESULTS_NAME = r"^(?<code>.+?)_(?<test_key>p_(?<test_p>[^_]+)_sig_(?<test_sigma>[^_]+)_s_(?<test_sample>\d+))_?(?<decoder>BP\+OSD|BP)_failure_rates"

"Columns neural_gather adds from a results file's name, in this order, before the file's own."
const NEURAL_NAME_COLUMNS = ["arm", "use_CER", "run_tag", "test_key", "test_p", "test_sigma", "test_sample",
                             "train_key", "train_p", "train_sigma", "train_sample"]

"""
    neural_results_metadata(file_name) -> Union{Dict{String, Any}, Nothing}

What a Neural BP results file's name says (`NEURAL_RESULTS_NAME`), under the
names in `NEURAL_NAME_COLUMNS`: the arm (the run tag without its underscore —
`cer`, `nocer` — or, with no run tag, `cer` / `nocer` from the `_no_cer` mark),
whether the model used CER, and the test and training datasets' p, sigma and
sample. `nothing` for a name in an older format (no `sig_`).
"""
function neural_results_metadata(file_name::String)::Union{Dict{String, Any}, Nothing}
    name_match::Union{RegexMatch, Nothing} = match(NEURAL_RESULTS_NAME, file_name)
    if name_match === nothing
        return nothing
    end
    use_cer::Bool = name_match[:no_cer] === nothing
    run_tag::String = String(name_match[:run_tag])
    arm::String = "cer"
    if !use_cer
        arm = "nocer"
    end
    if !isempty(run_tag)
        arm = String(lstrip(run_tag, '_'))
    end
    metadata::Dict{String, Any} = Dict{String, Any}(
        "arm"          => arm,
        "use_CER"      => use_cer,
        "run_tag"      => run_tag,
        "test_key"     => String(name_match[:test_key]),
        "test_p"       => parse(Float64, name_match[:test_p]),
        "test_sigma"   => parse(Float64, name_match[:test_sigma]),
        "test_sample"  => parse(Int, name_match[:test_sample]),
        "train_key"    => String(name_match[:train_key]),
        "train_p"      => parse(Float64, name_match[:train_p]),
        "train_sigma"  => parse(Float64, name_match[:train_sigma]),
        "train_sample" => parse(Int, name_match[:train_sample]),
    )
    return metadata
end

# ----------------------------------------------------------------------------
# Handlers — one per operation. Each takes its TOML section, the section id, and
# the global workdir. Files live under <workdir>/<codename>/results (inputs and
# aggregated outputs) and <workdir>/<codename>/plots (plot outputs).
# ----------------------------------------------------------------------------

# Gather one or more standard-decoder failures files (plain BP *or* BP+OSD — the
# collector reads trials/average/std straight from the file, so the same handler
# serves both) into one aggregated CSV. Registered under both `standardbp_gather`
# (BP-OSD) and `bp_gather` (plain BP); the section id only changes which files it
# reads and where it writes.
#
# A file named for a (p, sigma, sample) dataset (`STANDARD_RESULTS_NAME`, e.g.
# 72q_BB_p_0.0005_sig_0.001_s_2_BP_failure_rates.txt) gets the columns
# results_file, decoder, test_key, test_p, test_sigma, test_sample from its name,
# and its error_model_parameters_description becomes testing_data/test_<key>.txt:
# the file's own table carries p and the sample but not sigma. The output file is
# REPLACED on every run (it used to be appended to, so a second run doubled it).
function run_standard_gather(section::AbstractDict, id::String, workdir::String)::Nothing
    error_model_text::String = lowercase(strip(String(_require(section, "error_model", id))))
    if !(error_model_text in ("ising", "circuit"))
        error("[$(id)] error_model must be \"Ising\" or \"Circuit\", got \"$(error_model_text)\".")
    end
    error_model::Symbol = :Circuit
    if error_model_text == "ising"
        error_model = :Ising
    end

    codename::String = String(_require(section, "codename", id))
    prefix::String = joinpath(workdir, codename)
    results_dir::String = joinpath(prefix, "results")
    failures_files::Vector{String} = gather_file_names(section, id, results_dir)

    # Parse each failures file and stack the tables. Files that produce no rows
    # (missing/empty) are skipped by the collector, with a warning.
    per_file_frames::Vector{DataFrame} = DataFrame[]
    for failures_file in failures_files
        file_frame::DataFrame = collect_standard_decoder_statistics(
            error_model; prefix = prefix, standard_BP_output_file = failures_file)
        if nrow(file_frame) == 0
            continue
        end
        name_match::Union{RegexMatch, Nothing} = match(STANDARD_RESULTS_NAME, failures_file)
        if name_match !== nothing
            test_key::String = String(name_match[:test_key])
            name_p::Float64 = parse(Float64, name_match[:test_p])
            name_sample::Int = parse(Int, name_match[:test_sample])
            # The table's own p and sample (in the description the collector built)
            # must agree with the name.
            for description in file_frame.error_model_parameters_description
                content_match::Union{RegexMatch, Nothing} = match(r"test_p_([^_]+)_s_(\d+)\.txt$", String(description))
                is_consistent::Bool = content_match !== nothing &&
                                      isapprox(parse(Float64, content_match.captures[1]), name_p) &&
                                      parse(Int, content_match.captures[2]) == name_sample
                if !is_consistent
                    @warn "[$(id)] $(failures_file): its table says $(description), its name p = $(name_p), " *
                          "sample $(name_sample)."
                end
            end
            n_rows::Int = nrow(file_frame)
            name_frame::DataFrame = DataFrame(
                results_file = fill(failures_file, n_rows),
                decoder      = fill(String(name_match[:decoder]), n_rows),
                test_key     = fill(test_key, n_rows),
                test_p       = fill(name_p, n_rows),
                test_sigma   = fill(parse(Float64, name_match[:test_sigma]), n_rows),
                test_sample  = fill(name_sample, n_rows),
            )
            file_frame.error_model_parameters_description = fill("$(prefix)/testing_data/test_$(test_key).txt", n_rows)
            file_frame = hcat(name_frame, file_frame; makeunique = true)
        end
        push!(per_file_frames, file_frame)
    end
    if isempty(per_file_frames)
        error("[$(id)] no rows collected from any of the $(length(failures_files)) file(s) in $(results_dir).")
    end
    stats_frame::DataFrame = reduce((left, right) -> vcat(left, right; cols = :union), per_file_frames)

    output_path::String = joinpath(results_dir, _gather_output_filename(section, id))
    CSV.write(output_path, stats_frame)
    println("  [$(id)] standard ($(error_model)) -> $(output_path)  " *
            "($(nrow(stats_frame)) rows from $(length(failures_files)) file(s))")
    return nothing
end

# Gather Neural BP results CSVs into one table: the files a `pattern` matches (e.g.
# "simulation_results_*_seed_*.csv") and/or a `files` list. Each row gets
# results_file and, for a (p, sigma, sample) name, the columns NEURAL_NAME_COLUMNS
# read from the name — the arm and the test / training datasets live only there.
# Rows are stacked with cols = :union (older files lack some columns) and sorted by
# file name. The output file is REPLACED on every run (it used to be appended to).
function run_neural_gather(section::AbstractDict, id::String, workdir::String)::Nothing
    codename::String = String(_require(section, "codename", id))
    results_dir::String = joinpath(workdir, codename, "results")
    file_names::Vector{String} = gather_file_names(section, id, results_dir)

    frames::Vector{DataFrame} = DataFrame[]
    missing_files::Vector{String} = String[]
    unparsed_names::Vector{String} = String[]
    for file_name in file_names
        file_path::String = joinpath(results_dir, file_name)
        if !isfile(file_path)
            push!(missing_files, file_name)
            continue
        end
        results_frame::DataFrame = CSV.read(file_path, DataFrame)
        n_rows::Int = nrow(results_frame)
        name_frame::DataFrame = DataFrame(results_file = fill(file_name, n_rows))
        metadata::Union{Dict{String, Any}, Nothing} = neural_results_metadata(file_name)
        if metadata === nothing
            push!(unparsed_names, file_name)
        else
            for column_name in NEURAL_NAME_COLUMNS
                name_frame[!, column_name] = fill(metadata[column_name], n_rows)
            end
        end
        push!(frames, hcat(name_frame, results_frame; makeunique = true))
    end
    if !isempty(missing_files)
        @warn "[$(id)] $(length(missing_files)) file(s) not in $(results_dir), skipped:\n  " *
              join(missing_files, "\n  ")
    end
    if !isempty(unparsed_names)
        @warn "[$(id)] $(length(unparsed_names)) file name(s) are not of the (p, sigma, sample) form; their rows " *
              "have no arm or dataset columns:\n  " * join(first(unparsed_names, 5), "\n  ")
    end
    if isempty(frames)
        error("[$(id)] none of the $(length(file_names)) file(s) could be read from $(results_dir).")
    end
    gathered_frame::DataFrame = reduce((left, right) -> vcat(left, right; cols = :union), frames)
    sort!(gathered_frame, :results_file)

    output_path::String = joinpath(results_dir, String(_require(section, "output_neuralbp_file", id)))
    CSV.write(output_path, gathered_frame)
    arm_counts_text::String = ""
    if "arm" in names(gathered_frame)
        arm_counts::Vector{String} = String[]
        for arm in sort(unique(collect(skipmissing(gathered_frame.arm))))
            push!(arm_counts, "$(arm) $(count(isequal(arm), gathered_frame.arm))")
        end
        arm_counts_text = " (" * join(arm_counts, ", ") * ")"
    end
    println("  [$(id)] gathered $(nrow(gathered_frame)) row(s) from $(length(frames)) file(s)$(arm_counts_text) " *
            "-> $(output_path)")
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

    # Optional third (plain-BP) curve. Present `bp_csv` => a dotted BP line is
    # added per code and the gain Δ is computed against BP instead of BP-OSD;
    # absent => the plot is the usual two-decoder (BP-OSD vs neural) figure.
    bp_csv = haskey(section, "bp_csv") ? String(section["bp_csv"]) : nothing
    bp_label = String(get(section, "bp_plot_label", "BP"))

    # Read the aggregate CSVs for every code (same bare filenames in each
    # <workdir>/<codename>/results). Both must already exist per codename.
    standard_dataframes = DataFrame[]
    neural_dataframes = DataFrame[]
    bp_dataframes = bp_csv === nothing ? nothing : DataFrame[]
    for codename in codenames
        results_dir = joinpath(workdir, codename, "results")
        standard_path = joinpath(results_dir, standard_csv)
        neural_path = joinpath(results_dir, neural_csv)
        for path in (standard_path, neural_path)
            if !isfile(path)
                error("[$(id)] missing input: $(path) — generate it with the " *
                      "standardbp_gather / neural_gather sections.")
            end
        end
        push!(standard_dataframes, CSV.read(standard_path, DataFrame))
        push!(neural_dataframes, CSV.read(neural_path, DataFrame))
        if bp_dataframes !== nothing
            bp_path = joinpath(results_dir, bp_csv)
            if !isfile(bp_path)
                error("[$(id)] missing BP input: $(bp_path) — generate it with the " *
                      "bp_gather section, or drop `bp_csv` to plot without BP.")
            end
            push!(bp_dataframes, CSV.read(bp_path, DataFrame))
        end
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
        bp_dataframes = bp_dataframes, bp_label = bp_label,
        xscale = x_scale, yscale = y_scale)
    baseline_note = bp_dataframes === nothing ? "gain vs BP-OSD" : "with BP (dotted); gain vs BP"
    println("  plotted $(length(codenames)) code(s) [$(baseline_note)] -> $(saved_path)")
    println("  comparison CSVs -> $(comparison_csv_paths)")
    return nothing
end

# Shared body for the two single-decoder plots (standard_bp_osd_plots and
# neuralbp_plots): one aggregated data file per code, one line per code.
# `data_file_key` names the TOML key holding the bare filename (in each
# <workdir>/<codename>/results) and `default_label` titles the figure unless the
# section overrides it with `plot_label`.
function _run_single_decoder_plot(section::AbstractDict, id::String, workdir::String;
        data_file_key::String, default_label::String)
    codenames = String.(_require(section, "codenames", id))
    if isempty(codenames)
        error("[$(id)] `codenames` is empty.")
    end
    codename_labels = String.(_require(section, "codename_labels", id))
    if length(codename_labels) != length(codenames)
        error("[$(id)] `codename_labels` must have one entry per codename " *
              "($(length(codename_labels)) vs $(length(codenames))).")
    end
    data_file = String(_require(section, data_file_key, id))
    decoder_label = String(get(section, "plot_label", default_label))

    # Read the aggregate data file for every code (same bare filename in each
    # <workdir>/<codename>/results — the file produced by the matching gather
    # step). It must already exist per codename.
    dataframes = DataFrame[]
    for codename in codenames
        data_path = joinpath(workdir, codename, "results", data_file)
        if !isfile(data_path)
            error("[$(id)] missing input: $(data_path) — generate it with the gather step.")
        end
        push!(dataframes, CSV.read(data_path, DataFrame))
    end

    # Figure written to <workdir>/<plot_file> (plot_file may include a subdir).
    plot_path = joinpath(workdir, String(_require(section, "plot_file", id)))

    x_scale = _scale_symbol(get(section, "x_scale", "log10"))
    y_scale = _scale_symbol(get(section, "y_scale", "log10"))

    saved_path = plot_logical_error_rate(
        dataframes, codename_labels, decoder_label, plot_path;
        xscale = x_scale, yscale = y_scale)
    println("  plotted $(length(codenames)) code(s) [$(decoder_label)] -> $(saved_path)")
    return nothing
end

function run_standard_bp_osd_plots(section::AbstractDict, id::String, workdir::String)
    _run_single_decoder_plot(section, id, workdir;
        data_file_key = "standard_bp_data_file", default_label = "BP-OSD")
    return nothing
end

function run_neuralbp_plots(section::AbstractDict, id::String, workdir::String)
    _run_single_decoder_plot(section, id, workdir;
        data_file_key = "neuralbp_data_file", default_label = "Neural BP")
    return nothing
end

# ----------------------------------------------------------------------------
# Registry — ordered so `run = true` sections execute in a sensible sequence.
# ----------------------------------------------------------------------------
const OPERATIONS = [
    ("standardbp_gather",       run_standard_gather),   # BP+OSD failures files
    ("bp_gather",               run_standard_gather),   # plain-BP failures files
    ("neural_gather",           run_neural_gather),
    ("standard_bp_osd_plots",   run_standard_bp_osd_plots),
    ("neuralbp_plots",          run_neuralbp_plots),
    ("plot_standard_vs_neural", run_plot_standard_vs_neural),
]

function main()
    s = ArgParseSettings(
        description = "Post-processing control panel. Edit postprocess_configs/postprocess_panel.toml " *
                      "(flip `run = true`, fill params), then run this.")
    @add_arg_table s begin
        "--config"
            help = "path to the control-panel TOML (default lives in postprocess_configs/)"
            arg_type = String
            default = "postprocess_configs/postprocess_panel.toml"
        "--task"
            help = "run only this section id (default: run every section with run = true)"
            arg_type = String
            default = ""
    end
    args = parse_args(s)

    config_path = args["config"]
    if !isfile(config_path)
        error("config not found: $(config_path) — see postprocess_configs/postprocess_panel.toml for the template.")
    end
    cfg = TOML.parsefile(config_path)

    workdir = "./../data"
    if haskey(cfg, "workdir")
        workdir = String(cfg["workdir"])
    end

    only = String(strip(args["task"]))
    known_ids = Set(first.(OPERATIONS))
    task_groups = get(cfg, "task_groups", Dict{String, Any}())

    # Resolve --task into an explicit list of section ids to run. `--task=<id>`
    # runs that one section; `--task=<group>` (a [task_groups] key) runs all its
    # members. Empty `only` => run every section with run = true. A group's
    # members and any unknown id are validated up front so typos fail loudly.
    selected_ids = String[]
    if !isempty(only)
        if haskey(task_groups, only)
            selected_ids = String.(task_groups[only])
            if isempty(selected_ids)
                error("--task=$(only): task group [task_groups].$(only) is empty.")
            end
        else
            selected_ids = [only]
        end
        for sid in selected_ids
            if !(sid in known_ids)
                error("--task=$(only): unknown id \"$(sid)\". Known ids: $(join(first.(OPERATIONS), ", ")).")
            end
        end
    end

    ran = 0
    for (id, handler) in OPERATIONS
        section = get(cfg, id, nothing)
        if !isempty(only)
            if !(id in selected_ids)
                continue
            end
            if section === nothing
                error("--task=$(only): no [$(id)] section in $(config_path).")
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

    if ran == 0 && isempty(only)
        println("[postprocess] nothing to do — set `run = true` on a section in $(config_path).")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

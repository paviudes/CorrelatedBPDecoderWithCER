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
# postprocess_configs/postprocess_panel_sweep.toml. neural_gather also selects
# its files by submit.sh's keys: train_instance, test_pairs, test_samples, seeds,
# cer_arms, run_label; the reference gathers by test_pairs and test_samples.
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

# The (p, sigma, sample) settings submit.sh uses — train_instance, test_pairs,
# test_samples — read by the same functions here, for neural_gather's selection.
include("submission/dataset_settings.jl")

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
sample; plus "seed" (`missing` without `_seed_<n>`), which selects files but is
not added as a column — the CSV has its own. `nothing` for a name in an older
format (no `sig_`).
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
    seed_text::Union{AbstractString, Nothing} = name_match[:seed]
    name_seed::Union{Int, Missing} = missing
    if seed_text !== nothing
        name_seed = parse(Int, seed_text)
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
        "seed"         => name_seed,
    )
    return metadata
end

"""
    same_dataset_number(first_value, second_value) -> Bool

Whether two values of p (or of sigma) are one number: equal to 1e-9 relative,
so a typed 0.0014 matches a file named for 2 × 0.0007 whatever its last digit.
0 matches only 0.
"""
function same_dataset_number(first_value::Float64, second_value::Float64)::Bool
    is_same::Bool = isapprox(first_value, second_value; rtol = 1e-9)
    return is_same
end

"""
The test datasets a gather section asks for, in submit.sh's format: the
`test_pairs = [[p, sigma], …]` at every index in `test_samples`. Either left out
(or empty) keeps every value. Read by neural_gather and the reference gathers
(standardbp_gather, bp_gather) alike.
"""
struct TestDatasetSelection
    test_pairs::Vector{DatasetPair}
    test_samples::Vector{Int}
end

"""
    test_dataset_selection(section, id) -> TestDatasetSelection

The `test_pairs` and `test_samples` of a gather section.
"""
function test_dataset_selection(section::AbstractDict, id::String)::TestDatasetSelection
    test_pairs::Vector{DatasetPair} = DatasetPair[]
    if haskey(section, "test_pairs")
        test_pairs = dataset_pairs(section["test_pairs"], "[$(id)] test_pairs")
    end
    test_samples::Vector{Int} = Int[]
    if haskey(section, "test_samples")
        test_samples = sample_indices(section["test_samples"], "[$(id)] test_samples")
    end
    selection::TestDatasetSelection = TestDatasetSelection(test_pairs, test_samples)
    return selection
end

"""
    admits_test_dataset(selection, test_p, test_sigma, test_sample) -> Bool

Whether a test dataset, read off a file name, is one the selection asks for; p
and sigma compared as numbers (`same_dataset_number`), not as text.
"""
function admits_test_dataset(selection::TestDatasetSelection, test_p::Float64, test_sigma::Float64,
                             test_sample::Int)::Bool
    is_admitted::Bool = true
    if !isempty(selection.test_pairs)
        is_admitted = any(pair -> same_dataset_number(test_p, pair[1]) && same_dataset_number(test_sigma, pair[2]),
                          selection.test_pairs)
    end
    if !isempty(selection.test_samples)
        is_admitted = is_admitted && test_sample in selection.test_samples
    end
    return is_admitted
end

"""
    requested_datasets_without_files(selection, found_datasets) -> Vector{String}

The keys of the requested test datasets (every pair at every sample) that none
of `found_datasets` is; empty unless both `test_pairs` and `test_samples` are set.
"""
function requested_datasets_without_files(selection::TestDatasetSelection,
                                          found_datasets::Vector{DatasetInstance})::Vector{String}
    missing_keys::Vector{String} = String[]
    if isempty(selection.test_pairs) || isempty(selection.test_samples)
        return missing_keys
    end
    for instance in test_instance_list(selection.test_pairs, selection.test_samples)
        is_found::Bool = any(found -> same_dataset_number(found[1], instance[1]) &&
                                      same_dataset_number(found[2], instance[2]) && found[3] == instance[3],
                             found_datasets)
        if !is_found
            push!(missing_keys, instance_key(instance))
        end
    end
    return missing_keys
end

"""
Which Neural BP results `neural_gather` keeps, from the same keys submit.sh
writes: `train_instance = [p, sigma, sample]`, `test_pairs = [[p, sigma], …]`,
`test_samples = […]`, `seeds = […]`, and `cer_arms` / `run_label`, which give the
run tags `_<arm>[_<run_label>]`. A key left out (or empty) keeps every value.
"""
struct NeuralResultsSelection
    train_instance::Union{DatasetInstance, Nothing}
    test_pairs::Vector{DatasetPair}
    test_samples::Vector{Int}
    seeds::Vector{Int}
    run_tags::Vector{String}
end

"""
    selection_is_open(selection) -> Bool

True when no key narrows the selection, so every results file is kept.
"""
function selection_is_open(selection::NeuralResultsSelection)::Bool
    is_open::Bool = selection.train_instance === nothing && isempty(selection.test_pairs) &&
                    isempty(selection.test_samples) && isempty(selection.seeds) && isempty(selection.run_tags)
    return is_open
end

"""
    setting_integers(spec, setting_name) -> Vector{Int}

A list of integers (seeds) as the settings file gives them, each once; a bare
number is a list of one.
"""
function setting_integers(spec::Any, setting_name::String)::Vector{Int}
    items::Vector{Any} = Any[spec]
    if spec isa AbstractVector
        items = collect(Any, spec)
    end
    integers::Vector{Int} = Int[]
    for item in items
        parsed_value::Union{Int, Nothing} = nothing
        if item isa Integer && !(item isa Bool)
            parsed_value = Int(item)
        elseif item isa AbstractFloat && isinteger(item)
            parsed_value = Int(item)
        elseif item isa AbstractString
            parsed_value = tryparse(Int, strip(item))
        end
        if parsed_value === nothing
            error("$(setting_name): $(repr(item)) is not an integer.")
        end
        push!(integers, parsed_value)
    end
    unique_integers::Vector{Int} = unique(integers)
    return unique_integers
end

"""
    neural_results_selection(section, id) -> NeuralResultsSelection

The selection a [neural_gather] section asks for. `cer_arms` takes "cer" and
"nocer" as in submit.sh, each optionally followed by its submission's run_label
("cer_enr": run_label = "enr"), so arms of several submissions gather together;
a `run_label` here is appended to every arm, and without `cer_arms` means both
arms of that experiment.
"""
function neural_results_selection(section::AbstractDict, id::String)::NeuralResultsSelection
    train_instance::Union{DatasetInstance, Nothing} = nothing
    train_instance_setting::Any = get(section, "train_instance", Any[])
    is_empty_setting::Bool = train_instance_setting isa AbstractVector && isempty(train_instance_setting)
    if !is_empty_setting
        train_instance = parse_dataset_instance(train_instance_setting, "[$(id)] train_instance")
    end
    test_datasets::TestDatasetSelection = test_dataset_selection(section, id)
    seeds::Vector{Int} = Int[]
    if haskey(section, "seeds")
        seeds = setting_integers(section["seeds"], "[$(id)] seeds")
    end
    run_label::String = String(get(section, "run_label", ""))
    arm_names::Vector{String} = String[]
    if haskey(section, "cer_arms")
        arms_setting::Any = section["cer_arms"]
        if arms_setting isa AbstractString
            push!(arm_names, String(arms_setting))
        else
            arm_names = String[String(arm) for arm in arms_setting]
        end
    end
    if isempty(arm_names) && !isempty(run_label)
        arm_names = ["cer", "nocer"]    # a label names an experiment; both of its arms
    end
    run_tags::Vector{String} = String[]
    for arm in arm_names
        # "cer" / "nocer", or one followed by the run_label of its submission:
        # "cer_enr" is the CER arm of a submission made with run_label = "enr"
        # (run tag _cer_enr), so one gather can hold several submissions' arms.
        arm_match::Union{RegexMatch, Nothing} = match(r"^(cer|nocer)(_[A-Za-z0-9][A-Za-z0-9_-]*)?$", arm)
        if arm_match === nothing || occursin("seed", lowercase(arm))
            error("[$(id)] cer_arms: \"$(arm)\" — use \"cer\" or \"nocer\", as in submit.sh, optionally followed by " *
                  "the run_label of its submission: \"cer_enr\" for the CER arm of run_label = \"enr\".")
        end
        run_tag::String = "_$(arm)"
        if !isempty(run_label)
            run_tag *= "_$(run_label)"
        end
        push!(run_tags, run_tag)
    end
    selection::NeuralResultsSelection = NeuralResultsSelection(train_instance, test_datasets.test_pairs,
                                                               test_datasets.test_samples, seeds, run_tags)
    return selection
end

"""
    selection_admits(selection, metadata) -> Bool

Whether a results file, by what its name says, is one the selection asks for.
p and sigma are compared as numbers (`same_dataset_number`), not as text.
"""
function selection_admits(selection::NeuralResultsSelection, metadata::Dict{String, Any})::Bool
    is_admitted::Bool = true
    if selection.train_instance !== nothing
        training::DatasetInstance = selection.train_instance
        is_admitted = is_admitted && same_dataset_number(metadata["train_p"], training[1]) &&
                      same_dataset_number(metadata["train_sigma"], training[2]) &&
                      metadata["train_sample"] == training[3]
    end
    test_datasets::TestDatasetSelection = TestDatasetSelection(selection.test_pairs, selection.test_samples)
    is_admitted = is_admitted && admits_test_dataset(test_datasets, metadata["test_p"], metadata["test_sigma"],
                                                     metadata["test_sample"])
    if !isempty(selection.seeds)
        is_admitted = is_admitted && metadata["seed"] !== missing && metadata["seed"] in selection.seeds
    end
    if !isempty(selection.run_tags)
        is_admitted = is_admitted && metadata["run_tag"] in selection.run_tags
    end
    return is_admitted
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
# the file's own table carries p and the sample but not sigma. `test_pairs` and
# `test_samples` (TestDatasetSelection, as in neural_gather) narrow the files to
# those test datasets; a name with no dataset in it is then left out. The output
# file is REPLACED on every run (it used to be appended to, so a second run
# doubled it).
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
    candidate_files::Vector{String} = gather_file_names(section, id, results_dir)

    # Keep the requested test datasets, read off each name.
    dataset_selection::TestDatasetSelection = test_dataset_selection(section, id)
    selects_datasets::Bool = !isempty(dataset_selection.test_pairs) || !isempty(dataset_selection.test_samples)
    failures_files::Vector{String} = String[]
    found_datasets::Vector{DatasetInstance} = DatasetInstance[]
    n_not_selected::Int = 0
    for candidate_file in candidate_files
        candidate_match::Union{RegexMatch, Nothing} = match(STANDARD_RESULTS_NAME, candidate_file)
        if candidate_match === nothing
            if selects_datasets
                n_not_selected += 1    # no dataset in the name to select it by
            else
                push!(failures_files, candidate_file)
            end
            continue
        end
        candidate_dataset::DatasetInstance = (parse(Float64, candidate_match[:test_p]),
                                              parse(Float64, candidate_match[:test_sigma]),
                                              parse(Int, candidate_match[:test_sample]))
        is_requested::Bool = admits_test_dataset(dataset_selection, candidate_dataset[1], candidate_dataset[2],
                                                 candidate_dataset[3])
        if is_requested
            push!(failures_files, candidate_file)
            push!(found_datasets, candidate_dataset)
        else
            n_not_selected += 1
        end
    end
    if isempty(failures_files)
        error("[$(id)] none of the $(length(candidate_files)) candidate file(s) in $(results_dir) is a requested " *
              "test dataset (test_pairs / test_samples).")
    end
    missing_keys::Vector{String} = requested_datasets_without_files(dataset_selection, found_datasets)
    if !isempty(missing_keys)
        more_text::String = ""
        if length(missing_keys) > 10
            more_text = "\n  … and $(length(missing_keys) - 10) more"
        end
        @warn "[$(id)] $(length(missing_keys)) requested test dataset(s) have no table in $(results_dir):\n  " *
              join(first(missing_keys, 10), "\n  ") * more_text
    end

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
    selection_text::String = ""
    if selects_datasets
        selection_text = "; $(n_not_selected) outside test_pairs / test_samples"
    end
    println("  [$(id)] standard ($(error_model)) -> $(output_path)  " *
            "($(nrow(stats_frame)) rows from $(length(failures_files)) file(s)$(selection_text))")
    return nothing
end

"""
    report_missing_combinations(selection, kept_metadata, id) -> Nothing

When the selection names every axis — `cer_arms`, `seeds`, `test_pairs` and
`test_samples` — warn about each requested (arm, seed, test dataset) with no
results file, and about any with more than one (two training datasets, say,
when `train_instance` is not set), since a seed-paired comparison needs exactly
one of each.
"""
function report_missing_combinations(selection::NeuralResultsSelection, kept_metadata::Vector{Dict{String, Any}},
                                     id::String)::Nothing
    names_every_axis::Bool = !isempty(selection.run_tags) && !isempty(selection.seeds) &&
                             !isempty(selection.test_pairs) && !isempty(selection.test_samples)
    if !names_every_axis
        return nothing
    end
    requested_instances::Vector{DatasetInstance} = test_instance_list(selection.test_pairs, selection.test_samples)
    files_per_combination::Dict{Tuple{String, Int, String}, Int} = Dict{Tuple{String, Int, String}, Int}()
    for metadata in kept_metadata
        # The requested test dataset this file was admitted as, matched by number
        # as selection_admits matches it, so a key whose text differs in the last
        # digit still fills its slot.
        matching_index::Union{Int, Nothing} = findfirst(
            instance -> same_dataset_number(metadata["test_p"], instance[1]) &&
                        same_dataset_number(metadata["test_sigma"], instance[2]) &&
                        metadata["test_sample"] == instance[3],
            requested_instances)
        if metadata["seed"] === missing || matching_index === nothing
            continue    # neither can happen once every axis selects; such a file has no slot
        end
        combination::Tuple{String, Int, String} = (metadata["run_tag"], metadata["seed"],
                                                   instance_key(requested_instances[matching_index]))
        files_per_combination[combination] = get(files_per_combination, combination, 0) + 1
    end
    missing_combinations::Vector{String} = String[]
    repeated_combinations::Vector{String} = String[]
    for run_tag in selection.run_tags
        for seed in selection.seeds
            for instance in requested_instances
                combination::Tuple{String, Int, String} = (run_tag, seed, instance_key(instance))
                n_files::Int = get(files_per_combination, combination, 0)
                label::String = "$(lstrip(run_tag, '_')), seed $(seed), test_$(combination[3])"
                if n_files == 0
                    push!(missing_combinations, label)
                elseif n_files > 1
                    push!(repeated_combinations, "$(label): $(n_files) files")
                end
            end
        end
    end
    if !isempty(missing_combinations)
        more_text::String = ""
        if length(missing_combinations) > 10
            more_text = "\n  … and $(length(missing_combinations) - 10) more"
        end
        @warn "[$(id)] $(length(missing_combinations)) requested (arm, seed, test dataset) combination(s) have no " *
              "results file (a pair the data lacks at a sample shows up here too):\n  " *
              join(first(missing_combinations, 10), "\n  ") * more_text
    end
    if !isempty(repeated_combinations)
        @warn "[$(id)] $(length(repeated_combinations)) combination(s) have more than one results file; set " *
              "train_instance (or narrow `pattern`) to keep one:\n  " * join(first(repeated_combinations, 10), "\n  ")
    end
    return nothing
end

# Gather Neural BP results CSVs into one table. Which files: `files` and/or
# `pattern` (default: every simulation_results_*.csv), then narrowed by the same
# keys submit.sh writes — train_instance, test_pairs, test_samples, seeds,
# cer_arms / run_label (NeuralResultsSelection) — read off each file's NAME. Each
# row gets results_file and, for a (p, sigma, sample) name, the columns
# NEURAL_NAME_COLUMNS from the name: the arm and the test / training datasets live
# only there. Rows are stacked with cols = :union (older files lack some columns)
# and sorted by file name. The output file is REPLACED on every run (it used to be
# appended to).
function run_neural_gather(section::AbstractDict, id::String, workdir::String)::Nothing
    codename::String = String(_require(section, "codename", id))
    results_dir::String = joinpath(workdir, codename, "results")
    candidate_section::AbstractDict = section
    if !haskey(section, "files") && !haskey(section, "pattern")
        candidate_section = Dict{String, Any}("pattern" => "simulation_results_*.csv")
    end
    file_names::Vector{String} = gather_file_names(candidate_section, id, results_dir)
    selection::NeuralResultsSelection = neural_results_selection(section, id)
    is_open::Bool = selection_is_open(selection)

    frames::Vector{DataFrame} = DataFrame[]
    kept_metadata::Vector{Dict{String, Any}} = Dict{String, Any}[]
    missing_files::Vector{String} = String[]
    unparsed_names::Vector{String} = String[]
    n_not_selected::Int = 0
    for file_name in file_names
        file_path::String = joinpath(results_dir, file_name)
        if !isfile(file_path)
            push!(missing_files, file_name)
            continue
        end
        metadata::Union{Dict{String, Any}, Nothing} = neural_results_metadata(file_name)
        if metadata === nothing
            push!(unparsed_names, file_name)
            if !is_open
                continue    # nothing to select it by
            end
        elseif !is_open && !selection_admits(selection, metadata)
            n_not_selected += 1
            continue
        end
        results_frame::DataFrame = CSV.read(file_path, DataFrame)
        n_rows::Int = nrow(results_frame)
        name_frame::DataFrame = DataFrame(results_file = fill(file_name, n_rows))
        if metadata !== nothing
            for column_name in NEURAL_NAME_COLUMNS
                name_frame[!, column_name] = fill(metadata[column_name], n_rows)
            end
            push!(kept_metadata, metadata)
        end
        push!(frames, hcat(name_frame, results_frame; makeunique = true))
    end
    if !isempty(missing_files)
        @warn "[$(id)] $(length(missing_files)) file(s) not in $(results_dir), skipped:\n  " *
              join(missing_files, "\n  ")
    end
    if !isempty(unparsed_names)
        unparsed_fate::String = "their rows have no arm or dataset columns"
        if !is_open
            unparsed_fate = "they are left out, having nothing to select them by"
        end
        @warn "[$(id)] $(length(unparsed_names)) file name(s) are not of the (p, sigma, sample) form; " *
              "$(unparsed_fate):\n  " * join(first(unparsed_names, 5), "\n  ")
    end
    if isempty(frames)
        error("[$(id)] nothing gathered from $(results_dir): $(length(file_names)) candidate file(s), " *
              "$(n_not_selected) outside the selection (train_instance / test_pairs / test_samples / seeds / " *
              "cer_arms / run_label).")
    end
    report_missing_combinations(selection, kept_metadata, id)
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
    selection_text::String = ""
    if !is_open
        selection_text = "; $(n_not_selected) outside the selection"
    end
    println("  [$(id)] gathered $(nrow(gathered_frame)) row(s) from $(length(frames)) file(s)$(arm_counts_text)" *
            "$(selection_text) -> $(output_path)")
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

# ============================================================================
# dataset_settings.jl — how the settings files name a dataset
# ============================================================================
#
# Every input of one dataset lives under <workdir>/<codename>/ and carries the
# key p_<p>_sig_<sigma>_s_<sample>, e.g. p_0.0015_sig_0.003_s_1, the numbers
# written the way the Python data generator writes them (its repr): 0.0015,
# 0.003, 0.0. Settings give those numbers — nothing symbolic ("2p") is read:
#
#     train_instance = [0.0015, 0.003, 1]                 (p, sigma, sample)
#     test_pairs     = [[0.0001, 0.0], [0.0001, 0.0001]]  (p, sigma) pairs
#     test_samples   = [1, 2]                              sample indices
#
# Shared by the submission driver (batch_commands.jl, for batch_run.jl) and the
# post-processing control panel (postprocess.jl), so both read the same keys the
# same way. Depends on nothing but Printf.
#
# Exposes:
#   python_float_text, dataset_key, DatasetPair, DatasetInstance, instance_key
#   setting_number, setting_sample, parse_dataset_pair, parse_dataset_instance
#   dataset_pairs, sample_indices, test_instance_list
# ============================================================================

using Printf

"""
    python_float_text(value) -> String

The text Python's `repr` gives `value`, which is how the data generator names
its files: `0.0015`, `0.003`, `0.0`, and `1e-05` below 1e-4. Julia's shortest
round-trip printing agrees with Python's digit for digit; only the exponent form
differs (Julia `1.0e-5`, Python `1e-05`), so that is rewritten.
"""
function python_float_text(value::Float64)::String
    julia_text::String = string(value)
    if !occursin('e', julia_text)
        return julia_text
    end
    exponent_parts::Vector{SubString{String}} = split(julia_text, 'e')
    mantissa_text::String = String(exponent_parts[1])
    if endswith(mantissa_text, ".0")
        mantissa_text = mantissa_text[1:end-2]
    end
    exponent_value::Int = parse(Int, exponent_parts[2])
    python_text::String = mantissa_text * "e" * @sprintf("%+03d", exponent_value)
    return python_text
end

"""
    dataset_key(p_value, sigma_value, sample_index) -> String

`p_<p>_sig_<sigma>_s_<sample>`, the key every input file of a dataset carries.
"""
function dataset_key(p_value::Float64, sigma_value::Float64, sample_index::Int)::String
    key::String = "p_$(python_float_text(p_value))_sig_$(python_float_text(sigma_value))_s_$(sample_index)"
    return key
end

"The noise parameters of a dataset, (p, sigma), as the numbers its file names carry."
const DatasetPair = Tuple{Float64, Float64}

"One dataset, (p, sigma, sample index): one key."
const DatasetInstance = Tuple{Float64, Float64, Int}

"""
    instance_key(instance) -> String

The key of one dataset, e.g. (0.0015, 0.003, 1) -> `p_0.0015_sig_0.003_s_1`.
"""
function instance_key(instance::DatasetInstance)::String
    key::String = dataset_key(instance[1], instance[2], instance[3])
    return key
end

"""
    setting_number(item, setting_name, quantity_name) -> Float64

One number of a dataset setting: a TOML number, or its text from the command
line. Anything else — "2p" included — is refused: p and sigma are written as
the numbers the file names carry (p_0.0015_sig_0.003 is [0.0015, 0.003]).
"""
function setting_number(item::Any, setting_name::String, quantity_name::String)::Float64
    parsed_value::Union{Float64, Nothing} = nothing
    if item isa Real && !(item isa Bool)
        parsed_value = Float64(item)
    elseif item isa AbstractString
        parsed_value = tryparse(Float64, strip(item))
    end
    if parsed_value === nothing || !isfinite(parsed_value)
        throw(ArgumentError("$(setting_name): $(quantity_name) $(repr(item)) is not a number. Write the value the " *
                            "file names carry, e.g. sigma 0.003 for p_0.0015_sig_0.003."))
    end
    numeric_value::Float64 = parsed_value
    if numeric_value == 0.0
        numeric_value = 0.0    # -0.0 would print as "-0.0" and name no file
    end
    return numeric_value
end

"""
    setting_sample(item, setting_name) -> Int

A sample index (the s of `..._s_<s>`): a positive integer, given as a TOML
integer — or a whole float, which an array mixing floats and integers can hand
over — or as text from the command line.
"""
function setting_sample(item::Any, setting_name::String)::Int
    parsed_value::Union{Int, Nothing} = nothing
    if item isa Integer && !(item isa Bool)
        parsed_value = Int(item)
    elseif item isa AbstractFloat && isinteger(item)
        parsed_value = Int(item)
    elseif item isa AbstractString
        parsed_value = tryparse(Int, strip(item))
    end
    if parsed_value === nothing || parsed_value < 1
        throw(ArgumentError("$(setting_name): sample $(repr(item)) is not a positive integer (the s of " *
                            "p_<p>_sig_<sigma>_s_<s>)."))
    end
    sample_index::Int = parsed_value
    return sample_index
end

"""
    setting_entries(spec, setting_name, expected_length) -> Vector{Any}

The entries of one [p, sigma] or [p, sigma, sample]: an array in the settings
file, comma-separated text ("0.0015,0.003,1") on the command line.
"""
function setting_entries(spec::Any, setting_name::String, expected_length::Int)::Vector{Any}
    entries::Vector{Any} = Any[]
    if spec isa AbstractVector
        entries = collect(Any, spec)
    elseif spec isa AbstractString
        entries = Any[String(strip(part)) for part in split(spec, ',')]
    end
    if length(entries) != expected_length
        example_text::String = "[0.0015, 0.003] in the settings file, or 0.0015,0.003 on the command line"
        if expected_length == 3
            example_text = "[0.0015, 0.003, 1] in the settings file, or 0.0015,0.003,1 on the command line"
        end
        throw(ArgumentError("$(setting_name): $(repr(spec)) does not have $(expected_length) entries. Write it as " *
                            "$(example_text)."))
    end
    return entries
end

"""
    parse_dataset_pair(pair_spec, setting_name) -> DatasetPair

One (p, sigma), both numbers: p > 0, sigma ≥ 0.
"""
function parse_dataset_pair(pair_spec::Any, setting_name::String)::DatasetPair
    entries::Vector{Any} = setting_entries(pair_spec, setting_name, 2)
    p_value::Float64 = setting_number(entries[1], setting_name, "p")
    sigma_value::Float64 = setting_number(entries[2], setting_name, "sigma")
    if p_value <= 0.0 || sigma_value < 0.0
        throw(ArgumentError("$(setting_name): $(repr(pair_spec)) needs p > 0 and sigma ≥ 0."))
    end
    parsed_pair::DatasetPair = (p_value, sigma_value)
    return parsed_pair
end

"""
    parse_dataset_instance(instance_spec, setting_name) -> DatasetInstance

One (p, sigma, sample) — the training dataset, `train_instance = [0.0015, 0.003, 1]`.
"""
function parse_dataset_instance(instance_spec::Any, setting_name::String)::DatasetInstance
    entries::Vector{Any} = setting_entries(instance_spec, setting_name, 3)
    pair::DatasetPair = parse_dataset_pair(entries[1:2], setting_name)
    sample_index::Int = setting_sample(entries[3], setting_name)
    parsed_instance::DatasetInstance = (pair[1], pair[2], sample_index)
    return parsed_instance
end

"""
    dataset_pairs(pairs_spec, setting_name) -> Vector{DatasetPair}

A list of (p, sigma): `[[0.0001, 0.0], [0.0001, 0.0001], ...]` in the settings
file, or `0.0001,0 0.0001,0.0001 ...` on the command line.
"""
function dataset_pairs(pairs_spec::Any, setting_name::String)::Vector{DatasetPair}
    if !(pairs_spec isa AbstractVector)
        throw(ArgumentError("$(setting_name) must be a list of [p, sigma] pairs, e.g. " *
                            "$(setting_name) = [[0.0001, 0.0], [0.0001, 0.0001]]; got $(repr(pairs_spec))."))
    end
    parsed_pairs::Vector{DatasetPair} = DatasetPair[]
    for pair_spec in pairs_spec
        is_pair_shaped::Bool = pair_spec isa AbstractVector || pair_spec isa AbstractString
        if !is_pair_shaped
            throw(ArgumentError("$(setting_name): every entry must be a [p, sigma] pair, and $(repr(pair_spec)) " *
                                "is not. One pair still needs the outer brackets: $(setting_name) = [[0.0015, 0.003]]."))
        end
        push!(parsed_pairs, parse_dataset_pair(pair_spec, setting_name))
    end
    return parsed_pairs
end

"""
    sample_indices(samples_spec, setting_name) -> Vector{Int}

A list of sample indices: `[1, 2, 3]` in the settings file, `1 2 3` on the
command line; a bare number is one index. Each index once, in the order given.
"""
function sample_indices(samples_spec::Any, setting_name::String)::Vector{Int}
    sample_items::Vector{Any} = Any[samples_spec]
    if samples_spec isa AbstractVector
        sample_items = collect(Any, samples_spec)
    end
    indices::Vector{Int} = Int[setting_sample(item, setting_name) for item in sample_items]
    unique_indices::Vector{Int} = unique(indices)
    return unique_indices
end

"""
    test_instance_list(test_pairs, test_samples) -> Vector{DatasetInstance}

Every (p, sigma) pair at every sample index, pair by pair, each instance once.
"""
function test_instance_list(test_pairs::AbstractVector{DatasetPair},
                            test_samples::AbstractVector{Int})::Vector{DatasetInstance}
    instances::Vector{DatasetInstance} = DatasetInstance[]
    for pair in test_pairs
        for sample_index in test_samples
            instance::DatasetInstance = (pair[1], pair[2], sample_index)
            if !(instance in instances)
                push!(instances, instance)
            end
        end
    end
    return instances
end

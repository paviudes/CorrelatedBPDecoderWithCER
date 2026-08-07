# logging.jl
# ----------------------------------------------------------------------------
# Console records whose bracket closes.
#
# Julia's stock `ConsoleLogger` renders a single-line record as
#
#     [ Info: message
#
# The `[` never closes because it is the degenerate case of the `┌ … └` box the
# logger draws for multi-line records — there is no `]` in the design at all.
#
# ----------------------------------------------------------------------------
# WHY THIS FILE PRINTS INSTEAD OF INSTALLING A LOGGER
# ----------------------------------------------------------------------------
# The obvious fix is a custom `AbstractLogger` installed as the global logger.
# `BracketedConsoleLogger` below is exactly that, and it works — EXCEPT in a
# process that also calls `Enzyme.autodiff`, which is to say the training path,
# which is most of what this package does.
#
# GPUCompiler (pulled in by Enzyme) queries the active logger through a
# `@generated` helper, `_invoked_min_enabled_level`, that reaches the logger via
# `invoke` at a PINNED WORLD AGE. A logger type defined in this package has its
# `min_enabled_level` method added at package-load time, which is newer than the
# world Enzyme's abstract interpreter runs in. Inference then fails inside
# `autodiff` with
#
#     MethodError: no method matching invoke min_enabled_level(::BracketedConsoleLogger)
#     The applicable method may be too new: running in world age 38790,
#     while current world is 38823.
#
# — and it fails during compilation of the training loss, so it is fatal, not
# cosmetic. Nothing we can do on our side moves our method into that older world.
#
# So the messages we control are simply PRINTED, pre-formatted, by `print_info`.
# No global state, no logging machinery, no interaction with Enzyme. `@warn` and
# `@error` are left on the stock logger, where the `┌ … └` box already closes
# itself and where log levels and `maxlog` still work.
# ----------------------------------------------------------------------------

using Logging

"""
    log_level_label(level) -> String

The word shown after the opening bracket. Mirrors `ConsoleLogger`, including its
one irregularity: the level is spelled `Warn` but prints as `Warning`.
"""
function log_level_label(level::Logging.LogLevel)::String
    label::String = "Info"
    if level < Logging.Info
        label = "Debug"
    elseif level < Logging.Warn
        label = "Info"
    elseif level < Logging.Error
        label = "Warning"
    else
        label = "Error"
    end
    return label
end

"""
    log_level_color(level) -> Symbol

Colour of the bracket and label, matching `ConsoleLogger`'s palette so the output
still looks like Julia.
"""
function log_level_color(level::Logging.LogLevel)::Symbol
    color::Symbol = :cyan
    if level < Logging.Info
        color = :blue
    elseif level < Logging.Warn
        color = :cyan
    elseif level < Logging.Error
        color = :yellow
    else
        color = :light_red
    end
    return color
end

"""
    print_bracketed_record(io, label, color, body_lines) -> Nothing

Write one record. A single line is bracketed on both sides:

    [ Info: message ]

Several lines keep the box, because a `]` sitting four lines below its `[` is
harder to read, not easier:

    ┌ Info: first line
    │ second line
    └ third line

The whole record is assembled in a buffer and written once, so concurrent
workers cannot interleave halfway through a line.
"""
function print_bracketed_record(
    io::IO,
    label::String,
    color::Symbol,
    body_lines::Vector{String}
)::Nothing
    output_buffer::IOBuffer = IOBuffer()
    buffered_io::IOContext = IOContext(output_buffer, io)

    if length(body_lines) == 1
        printstyled(buffered_io, "[ ", label, ": "; color = color, bold = true)
        print(buffered_io, body_lines[1])
        printstyled(buffered_io, " ]"; color = color, bold = true)
        println(buffered_io)
    else
        n_body_lines::Int = length(body_lines)
        for (line_index, body_line) in enumerate(body_lines)
            marker::String = "│ "
            if line_index == 1
                marker = "┌ "
            elseif line_index == n_body_lines
                marker = "└ "
            end
            printstyled(buffered_io, marker; color = color, bold = true)
            if line_index == 1
                printstyled(buffered_io, label, ": "; color = color, bold = true)
            end
            println(buffered_io, body_line)
        end
    end

    write(io, take!(output_buffer))
    return nothing
end

"""
    print_info(message; io = stdout) -> Nothing

Print `[ Info: message ]`.

Use this instead of `@info` for progress messages this package emits itself. It
is a plain `print`, so — unlike `@info` — it is inert with respect to the global
logger and cannot collide with Enzyme's world-age-pinned logger query (see the
header of this file).

`stdout`, not `stderr`: SLURM routes `stderr` to the `.err` file while `println`
output goes to `.out`, and splitting one run's transcript across two files in
flush order helps nobody.
"""
function print_info(message::AbstractString; io::IO = stdout)::Nothing
    body_lines::Vector{String} = split(chomp(String(message)), '\n')
    print_bracketed_record(io, "Info", log_level_color(Logging.Info), body_lines)
    return nothing
end

# ----------------------------------------------------------------------------
# The logger. Correct, and unused by default — see the header.
# ----------------------------------------------------------------------------

"""
    BracketedConsoleLogger(stream = stderr, min_level = Logging.Info)

A `ConsoleLogger` replacement that closes its bracket on single-line records.

!!! warning "Do not install this in a process that calls `Enzyme.autodiff`"
    GPUCompiler queries the global logger via an `invoke` pinned to an older
    world age than the one in which this package's `min_enabled_level` method is
    defined, so inference inside `autodiff` dies with "The applicable method may
    be too new". That means the entire training path. Use `print_info` instead.

    It is safe in a process that only does inference/prediction, or in the REPL.

`maxlog = n` is honoured exactly as `ConsoleLogger` honours it — a per-message-id
budget, decremented on each emission.
"""
struct BracketedConsoleLogger <: Logging.AbstractLogger
    stream::IO
    min_level::Logging.LogLevel
    message_limits::Dict{Any, Int}
end

function BracketedConsoleLogger(
    stream::IO = stderr,
    min_level::Logging.LogLevel = Logging.Info
)::BracketedConsoleLogger
    logger::BracketedConsoleLogger = BracketedConsoleLogger(stream, min_level, Dict{Any, Int}())
    return logger
end

function Logging.min_enabled_level(logger::BracketedConsoleLogger)::Logging.LogLevel
    return logger.min_level
end

function Logging.shouldlog(
    logger::BracketedConsoleLogger,
    level::Logging.LogLevel,
    _module,
    group,
    id
)::Bool
    # A message whose `maxlog` budget is spent is refused here, before its
    # arguments are even evaluated.
    remaining_logs::Int = get(logger.message_limits, id, 1)
    should_log::Bool = remaining_logs > 0
    return should_log
end

function Logging.catch_exceptions(logger::BracketedConsoleLogger)::Bool
    return true
end

function Logging.handle_message(
    logger::BracketedConsoleLogger,
    level::Logging.LogLevel,
    message,
    _module,
    group,
    id,
    filepath,
    line;
    kwargs...
)::Nothing
    # `maxlog = n`: remember a per-id budget and stop once it is exhausted.
    maximum_log_count = get(kwargs, :maxlog, nothing)
    if maximum_log_count isa Integer
        remaining_logs::Int = get!(logger.message_limits, id, Int(maximum_log_count))
        logger.message_limits[id] = remaining_logs - 1
        if remaining_logs <= 0
            return nothing
        end
    end

    body_lines::Vector{String} = String[]
    append!(body_lines, split(chomp(string(message)), '\n'))

    for (key, value) in kwargs
        if key !== :maxlog
            push!(body_lines, "  $(key) = $(value)")
        end
    end

    # Location is noise on an Info record; on a warning it is the first thing
    # you want.
    if level >= Logging.Warn
        source_file::String = basename(string(filepath))
        push!(body_lines, "@ $(_module) $(source_file):$(line)")
    end

    print_bracketed_record(logger.stream, log_level_label(level), log_level_color(level), body_lines)
    return nothing
end

"""
    install_bracketed_logger!(; stream = stdout, min_level = Logging.Info) -> AbstractLogger

Make a `BracketedConsoleLogger` the global logger; return the logger it replaced.

!!! warning
    Read the warning on [`BracketedConsoleLogger`](@ref) first. This breaks
    `Enzyme.autodiff`, so it must not be called from any entry point that trains.
    `expts/neural_bp_experiments.jl` deliberately does not call it.
"""
function install_bracketed_logger!(;
    stream::IO = stdout,
    min_level::Logging.LogLevel = Logging.Info
)::Logging.AbstractLogger
    logger::BracketedConsoleLogger = BracketedConsoleLogger(stream, min_level)
    previous_logger::Logging.AbstractLogger = Logging.global_logger(logger)
    return previous_logger
end

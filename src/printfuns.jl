# ============================================================================
#                          Console output for simulations
# ============================================================================

"Width of the separators printed by `print_console_rule`."
const CONSOLE_RULE_WIDTH = 78

"Columns of a results DataFrame that `print_simulation_results_to_console` needs,
in display order. They map to `error_file`, `epochs`, `layers`, `failures`,
`samples` respectively."
const SIMULATION_CONSOLE_SOURCE_COLUMNS = Symbol[
    :error_model_parameters_description,
    :n_epochs,
    :n_layers,
    :num_failures,
    :num_samples_per_error_rate,
]

"""
    print_console_rule([io]; char='-')

Print a horizontal rule. Used to separate the training phase from the testing
phase on the console, so a combined train+test run reads as two blocks rather
than one undifferentiated stream.
"""
function print_console_rule(io::IO = Base.stdout; rule_character::Char = '-')::Nothing
    rule::String = repeat(rule_character, CONSOLE_RULE_WIDTH)
    println(io, rule)
    return nothing
end

"""
    print_simulation_results_to_console(results::DataFrame; io=stdout) -> DataFrame

Print a simulation's results as a five-column table:

    error_file   epochs   layers   failures   samples

`error_file` is the BASENAME of `error_model_parameters_description` (the full
path is already in the CSV; on the console it is just noise).

This is purely cosmetic — the CSV written by `record_decoder_statistics` keeps
every column, including the logical error rate and its standard error. The point
is that neither the raw JSON dict nor the full DataFrame is readable at a glance
when you are watching a sweep scroll past.

Accepts the DataFrame returned by `record_decoder_statistics` (a fresh run) or
by `collect_decoder_statistics` (results loaded from disk), so both paths print
identically. Returns the tidied frame.
"""
function print_simulation_results_to_console(results::DataFrame; io::IO = Base.stdout)::DataFrame
    if nrow(results) == 0
        println(io, "No simulation results to display.")
        return results
    end

    absent_columns::Vector{Symbol} = [
        source_column
        for source_column in SIMULATION_CONSOLE_SOURCE_COLUMNS
        if !hasproperty(results, source_column)
    ]
    if !isempty(absent_columns)
        # Don't hide data we failed to recognise — fall back to the full frame.
        @warn "print_simulation_results_to_console: missing column(s) $(absent_columns); showing the full frame."
        show(io, results; summary = false, allcols = true)
        println(io)
        return results
    end

    error_file_names::Vector{String} =
        [basename(String(description)) for description in results.error_model_parameters_description]

    tidy_results::DataFrame = DataFrame(
        error_file = error_file_names,
        epochs     = results.n_epochs,
        layers     = results.n_layers,
        failures   = results.num_failures,
        samples    = results.num_samples_per_error_rate,
    )
    show(io, tidy_results; summary = false, eltypes = false, allrows = true, allcols = true)
    println(io)
    return tidy_results
end

"""
    print_simulation_results_to_console(results_file::String; io=stdout) -> DataFrame

Convenience overload: read a results CSV from disk and print the same table.
"""
function print_simulation_results_to_console(results_file::String; io::IO = Base.stdout)::DataFrame
    results::DataFrame = collect_decoder_statistics(results_file)
    tidy_results::DataFrame = print_simulation_results_to_console(results; io = io)
    return tidy_results
end

function print_neuralbp_info(bpnn::NeuralBP; io::IO=Base.stdout)
    """
    Print the key information about the NeuralBP model.
    """
    println(io, "NeuralBP Decoder Information:")
    println(io, "----------------------------")
    println(io, "Number of bits (n): ", size(bpnn.parity_check_matrix, 2))
    println(io, "Number of checks (m): ", size(bpnn.parity_check_matrix, 1))
    println(io, "Number of neurons per layer: ", bpnn.nb_neurons_per_layer)
    println(io, "----------------------------")
    # print the input layer to V2C adjacency matrix
    println(io, "Adjacency matrix from input layer to V2C layer ($(size(bpnn.adj_initialize_V2C, 1)) x $(size(bpnn.adj_initialize_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_initialize_V2C)
    println(io, "\n----------------------------")
    # print the V2C to C2V adjacency matrix
    println(io, "Adjacency matrix from V2C layer to C2V layer ($(size(bpnn.adj_V2C_C2V, 1)) x $(size(bpnn.adj_V2C_C2V, 2))):")
    show(io, "text/plain", bpnn.adj_V2C_C2V)
    println(io, "\n----------------------------")
    # print the C2V to V2C adjacency matrix
    println(io, "Adjacency matrix from C2V layer to V2C layer ($(size(bpnn.adj_C2V_V2C, 1)) x $(size(bpnn.adj_C2V_V2C, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_V2C)
    println(io, "\n----------------------------")
    # print the C2V to readout adjacency matrix
    println(io, "Adjacency matrix from C2V layer to readout layer ($(size(bpnn.adj_C2V_readout, 1)) x $(size(bpnn.adj_C2V_readout, 2))):")
    show(io, "text/plain", bpnn.adj_C2V_readout)
    println(io, "\n----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Weights from V2C to C2V layer $(size(bpnn.weights_v2c_c2v)):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Weights from C2V to V2C layer $(size(bpnn.weights_c2v_v2c)):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Weights from C2V to readout layer $(size(bpnn.weights_c2v_readout)):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
end

function print_neuralbp_summary(bpnn::NeuralBP; io::IO=Base.stdout, final_llrs::Matrix{Float32}=Float32[])
    """
    Print a summary of training the NeuralBP model.
    Print the fitted parameters of the model.
    1. Weights from V2C to C2V
    2. Weights from C2V to V2C
    3. Weights from C2V to readout
    4. Initial LLRs
    5. (Optional) Final LLRs after training
    """
    println(io, "NeuralBP Training Summary:")
    println(io, "----------------------------")
    # print the weights
    # print the weights from V2C to C2V
    println(io, "Fitted Weights from V2C to C2V layer $(size(bpnn.weights_v2c_c2v)):")
    show(io, "text/plain", bpnn.weights_v2c_c2v)
    println(io, "\n----------------------------")
    # print the weights from C2V to V2C
    println(io, "Fitted Weights from C2V to V2C layer $(size(bpnn.weights_c2v_v2c)):")
    show(io, "text/plain", bpnn.weights_c2v_v2c)
    println(io, "\n----------------------------")
    # print the weights from C2V to readout
    println(io, "Fitted Weights from C2V to readout layer $(size(bpnn.weights_c2v_readout)):")
    show(io, "text/plain", bpnn.weights_c2v_readout)
    println(io, "\n----------------------------")
    # print the initial LLRs
    println(io, "Initial LLRs ($(length(bpnn.initial_llrs))):")
    show(io, "text/plain", bpnn.initial_llrs)
    println(io, "\n----------------------------")
    # print the final LLRs if provided
    if length(final_llrs) > 0
        println(io, "Final LLRs after training ($(size(final_llrs'))):")
        show(io, "text/plain", final_llrs')
        println(io, "\n----------------------------")
    end
    #TODO: print the runtime for training the neural BP model.
end
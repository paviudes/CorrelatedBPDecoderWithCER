using DelimitedFiles
using LinearAlgebra
using Random

abstract type ErrorModel end
# All error models should include a `sample` method that generates an error vector according to the model.
sample(model::ErrorModel, n_qubits::Int) = error("The sampling function is not implemented for the error model: $(typeof(model)).")

struct ExplicitErrorModel <: ErrorModel
    name::String
    error_file_name::String
    parameters_description::String
    function ExplicitErrorModel(filename::String; name="Explicit Error Set")
        parameters_description = "errorfile_$(filename)"
        new(name, filename, parameters_description)
    end
end

function sample_errors(model::ExplicitErrorModel, nqubits::Int)::Matrix{Int}
    """
    Load errors from a specified file.
    The file should contain a matrix where each row represents an error vector for nqubits.
    """
    errors = readdlm("./../data/$(model.error_file_name).txt", Int)
    if size(errors, 2) != nqubits
        throw(DimensionMismatch("The number of qubits in the error file does not match the specified nqubits."))
    end
    return errors
end

struct IIDErrorModel <: ErrorModel
    name::String
    per_qubit_error_prob::Float64
    parameters_description::String
    function IIDErrorModel(per_qubit_error_prob::Float64; name::String="IID Error Model")
        if per_qubit_error_prob < 0.0 || per_qubit_error_prob > 1.0
            error("Error probability must be in [0, 1].")
        end
        new(name, per_qubit_error_prob, "per_qubit_error_prob=$(per_qubit_error_prob)")
    end
end

function sample_error(model::IIDErrorModel, nqubits::Int)::Vector{Int}
    """
    Sample an n-qubit error vector from independent and identically distributed (i.i.d.) error model.
    """
    random_values = rand(nqubits)
    error_vector = [rv < model.per_qubit_error_prob ? rand(1:3) : 0 for rv in random_values]
    return error_vector
end

struct BallisticErrorModel <: ErrorModel
    """
    A simple ballistic error model where a contiguous block of qubits experiences errors.
    In this model, every qubit has a probability `p` of an error, and if an error occurs, it affects all the qubits that are connected to this qubit.
    """
    name::String
    nqubits::Int
    per_qubit_error_prob::Float64
    neighbour_error_prob::Float64
    average_block_size::Float64
    parameters_description::String
    correlations::Matrix{Int}  # Pairs for qubits along which correlated errors can occur.
    connections::Vector{Vector{Int}}  # Adjacency list to specify qubit connectivity.
    function BallisticErrorModel(per_qubit_error_prob::Float64, neighbour_error_prob::Float64; correlations::Matrix{Int}, name::String="Ballistic Error Model")
        if per_qubit_error_prob < 0.0 || per_qubit_error_prob > 1.0
            throw(BoundsError("Error probability must be in [0, 1]."))
        end
        average_block_size = compute_mean([length(corr) for corr in eachrow(correlations)])
        # convert the list of edges provided as `correlations` into an adjacency list `connections`
        nqubits = maximum(correlations)
        connections = [Int[] for _ in 1:nqubits]
        for edge in eachrow(correlations)
            u, v = edge
            push!(connections[u], v)
            push!(connections[v], u)
        end
        new(name, nqubits, per_qubit_error_prob, neighbour_error_prob, average_block_size, "per_qubit_error_prob=$(per_qubit_error_prob), neighbour_error_prob=$(neighbour_error_prob)", correlations, connections)
    end
end

function sample_error(model::BallisticErrorModel, nqubits::Int)::Vector{Int}
    """
    Sample an n-qubit error vector from the ballistic error model.
    """
    error_vector = zeros(Int, nqubits)
    random_values = rand(nqubits)
    error_vector = [rv < model.per_qubit_error_prob ? rand(1:3) : 0 for rv in random_values]
    # Introduce correlated errors based on the connectivity
    for qubit in 1:nqubits
        if error_vector[qubit] == 1
            for neighbor in model.connections[qubit]
                if rand() < model.neighbour_error_prob
                    error_vector[neighbor] = 1
                end
            end
        end
    end
    return error_vector
end

function sample_errors(model::ErrorModel, nqubits::Int, nsamples::Int)::Matrix{Int}
    """
    Sample multiple errors according to either the IID error model or the Ballistic error model.
    """
    errors = zeros(Int, nsamples, nqubits)
    for s in 1:nsamples
        errors[s, :] = sample_error(model, nqubits)
    end
    return errors
end

function print_error_model_info(model::ErrorModel; io::IO=stdout)
    """
    Print the error model information in a readable format.
    """
    println(io, "** Error Model Information **")
    println(io, "Error Model: ", model.name)
    println(io, "Parameters: ", model.parameters_description)
    println(io, "----------------------------------------")
end

function compute_mean(values::Vector{T})::T where T<:Real
    """
    Compute the mean of a vector of Real values.
    """
    return sum(values) / length(values)
end

function separate_error_components(error::Vector{Int})::Tuple{Vector{Int}, Vector{Int}}
    """
    Separate a combined error vector into its X and Z components.
    """
    error_X = [(e == 2 || e == 1) ? 1 : 0 for e in error] # X or Y errors contribute to X component
    error_Z = [(e == 2 || e == 3) ? 1 : 0 for e in error] # Z or Y errors contribute to Z component
    return (error_X, error_Z)
end

function join_error_components(error_X::Vector{Int}, error_Z::Vector{Int})::Vector{Int}
    """
    Combine separate X and Z error components into a single error vector.
    """
    n = length(error_X)
    @assert n == length(error_Z) "Error components must have the same length."
    combined_error = Vector{Int}(undef, n)
    for i in 1:n
        if error_X[i] == 0 && error_Z[i] == 0
            combined_error[i] = 0 # No error
        elseif error_X[i] == 1 && error_Z[i] == 0
            combined_error[i] = 1 # X error
        elseif error_X[i] == 1 && error_Z[i] == 1
            combined_error[i] = 2 # Y error
        elseif error_X[i] == 0 && error_Z[i] == 1
            combined_error[i] = 3 # Z error
        else
            error("Invalid error components at index $i: (X=$(error_X[i]), Z=$(error_Z[i]))")
        end
    end
    return combined_error
end
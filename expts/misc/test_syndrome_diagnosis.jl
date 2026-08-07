# ============================================================================
# test_syndrome_diagnosis.jl — acceptance tests for `count_syndrome_satisfactions`
# ============================================================================
# RUN FROM expts/ :
#
#     julia --project="./../" misc/test_syndrome_diagnosis.jl
#
# Tests, matching the handoff:
#
#   1. BACKWARD COMPATIBILITY  diagnosis.is_correct == check_bp_solutions(...),
#                              elementwise, asserted not eyeballed.
#   2. PARTITION               n_correct + n_coset_failures +
#                              n_convergence_failures == n_samples
#   3. CONSISTENCY             n_samples - n_correct == the `num_failures` the
#                              existing reporting path would write
#   4. COMMIT INVARIANT        committed_layer[i] == 0  <=>  syndrome_cleared[i] == false
#
# Plus structural checks on the chunked path, since a real run scores in chunks
# and stitches them: concatenating per-chunk diagnoses must equal diagnosing the
# whole thing at once, or the batching silently corrupts the per-sample vectors
# that the paired McNemar test depends on.
#
# Runs on the real [[72,12,6]] code with SYNTHETIC recoveries, deliberately
# constructed to exercise all three outcome buckets — a real trained decoder
# succeeds on ~99.97% of samples, so it would barely populate buckets 2 and 3
# and the test would prove almost nothing.
# ============================================================================

using CorrelatedBPDecoderWithCER
using CSV
using DataFrames
using DelimitedFiles
using Random
using Test

const CODE_DIRECTORY = joinpath(@__DIR__, "..", "..", "data", "72q_BB_cycles_1", "code")

function build_test_case(; n_samples::Int = 400, n_layers::Int = 12, seed::Int = 20250807)
    Random.seed!(seed)

    parity_check_matrix::Matrix{Int} = readdlm(joinpath(CODE_DIRECTORY, "HZ.txt"), Int)
    logicals::Matrix{Int} = readdlm(joinpath(CODE_DIRECTORY, "LZ.txt"), Int)
    n_bits::Int = size(parity_check_matrix, 2)

    errors::BitMatrix = falses(n_bits, n_samples)
    proposed_recoveries::Array{Bool, 3} = falses(n_bits, n_samples, n_layers)

    # Three regimes, so every bucket is populated:
    #   - "exact"      recovery equals the error at some layer  -> SUCCESS
    #   - "coset"      recovery = error XOR a logical operator  -> COSET FAILURE
    #   - "divergent"  recovery never clears the syndrome       -> CONVERGENCE FAILURE
    for i in 1:n_samples
        weight::Int = rand(1:3)
        for _ in 1:weight
            errors[rand(1:n_bits), i] = true
        end
        regime::Int = mod(i, 3)
        clearing_layer::Int = rand(1:n_layers)

        for layer in 1:n_layers
            # Before the clearing layer, emit something that does not clear.
            proposed_recoveries[:, i, layer] .= false
            proposed_recoveries[rand(1:n_bits), i, layer] = true
        end

        if regime == 0
            proposed_recoveries[:, i, clearing_layer] .= errors[:, i]
        elseif regime == 1
            logical_row::Int = rand(1:size(logicals, 1))
            @views proposed_recoveries[:, i, clearing_layer] .=
                errors[:, i] .⊻ (logicals[logical_row, :] .!= 0)
        end
        # regime == 2: leave the random non-clearing recoveries in place.
    end

    return parity_check_matrix, logicals, errors, proposed_recoveries
end

function chunked_diagnosis(parity_check_matrix, logicals, errors, proposed_recoveries, chunk_size::Int)
    n_samples::Int = size(errors, 2)
    chunks = NamedTuple[]
    for start in 1:chunk_size:n_samples
        stop::Int = min(start + chunk_size - 1, n_samples)
        push!(chunks, count_syndrome_satisfactions(
            parity_check_matrix, logicals,
            errors[:, start:stop], proposed_recoveries[:, start:stop, :]
        ))
    end
    stitched::NamedTuple = concatenate_diagnoses(chunks)
    return stitched
end

function run_tests()::Bool
    parity_check_matrix, logicals, errors, proposed_recoveries = build_test_case()
    n_samples::Int = size(errors, 2)

    diagnosis = count_syndrome_satisfactions(parity_check_matrix, logicals, errors, proposed_recoveries)
    reference_is_correct = check_bp_solutions(parity_check_matrix, logicals, errors, proposed_recoveries)

    println("outcome mix on the synthetic case (all three buckets must be non-empty):")
    println("  n_samples              = $(diagnosis.n_samples)")
    println("  n_correct              = $(diagnosis.n_correct)")
    println("  n_coset_failures       = $(diagnosis.n_coset_failures)")
    println("  n_convergence_failures = $(diagnosis.n_convergence_failures)")
    println("  mean_committed_layer   = $(round(mean_committed_layer(diagnosis), digits = 3))")
    println()

    all_passed::Bool = true
    @testset "count_syndrome_satisfactions" begin
        @testset "1. backward compatibility with check_bp_solutions" begin
            @test diagnosis.is_correct == reference_is_correct
            @test all(diagnosis.is_correct .== reference_is_correct)
        end

        @testset "2. the three buckets partition the samples" begin
            @test diagnosis.n_correct + diagnosis.n_coset_failures +
                  diagnosis.n_convergence_failures == n_samples
        end

        @testset "3. consistent with the existing num_failures" begin
            existing_num_failures::Int = count(.!reference_is_correct)
            @test n_samples - diagnosis.n_correct == existing_num_failures
        end

        @testset "4. committed_layer == 0 iff syndrome not cleared" begin
            @test all((diagnosis.committed_layer .== 0) .== .!diagnosis.syndrome_cleared)
        end

        @testset "5. buckets are actually exercised (guards against a vacuous test)" begin
            @test diagnosis.n_correct > 0
            @test diagnosis.n_coset_failures > 0
            @test diagnosis.n_convergence_failures > 0
        end

        @testset "6. chunked scoring equals whole-run scoring" begin
            for chunk_size in (1, 7, 64, n_samples)
                stitched = chunked_diagnosis(parity_check_matrix, logicals, errors, proposed_recoveries, chunk_size)
                @test stitched.is_correct == diagnosis.is_correct
                @test stitched.syndrome_cleared == diagnosis.syndrome_cleared
                @test stitched.committed_layer == diagnosis.committed_layer
                @test stitched.min_syndrome_weight == diagnosis.min_syndrome_weight
                @test stitched.n_correct == diagnosis.n_correct
                @test stitched.n_coset_failures == diagnosis.n_coset_failures
                @test stitched.n_convergence_failures == diagnosis.n_convergence_failures
            end
        end

        @testset "7. min_syndrome_weight is 0 exactly when a layer cleared" begin
            @test all((diagnosis.min_syndrome_weight .== 0) .== diagnosis.syndrome_cleared)
        end

        @testset "8. failure/histogram files are lossless for what they claim" begin
            output_directory::String = mktempdir()
            results_file::String = joinpath(output_directory, "simulation_results_toy.csv")
            written = write_failure_diagnostics(diagnosis, results_file)

            failures = CSV.read(written.failures_file, DataFrame)
            profile  = CSV.read(written.layer_profile_file, DataFrame)

            # one row per failed sample, and only failed samples
            @test nrow(failures) == n_samples - diagnosis.n_correct
            @test all(.!diagnosis.is_correct[failures.sample_index])
            # the two failure kinds partition, and match the aggregates
            @test count(==("coset"), failures.failure_kind) == diagnosis.n_coset_failures
            @test count(==("convergence"), failures.failure_kind) == diagnosis.n_convergence_failures
            # convergence failures committed to no layer; coset failures did
            @test all(failures.committed_layer[failures.failure_kind .== "convergence"] .== 0)
            @test all(failures.committed_layer[failures.failure_kind .== "coset"] .> 0)
            # the histogram accounts for every success and every coset failure
            @test sum(profile.n_correct) == diagnosis.n_correct
            @test sum(profile.n_coset_failures) == diagnosis.n_coset_failures
            @test nrow(profile) == diagnosis.n_layers
            # and it reproduces the mean committed layer to floating-point equality
            total_committed::Int = sum(profile.committed_layer .* (profile.n_correct .+ profile.n_coset_failures))
            total_cleared::Int = sum(profile.n_correct .+ profile.n_coset_failures)
            @test total_committed / total_cleared ≈ mean_committed_layer(diagnosis)

            rm(output_directory; recursive = true)
        end

        @testset "9. McNemar discordant pairs recoverable from failure sets alone" begin
            # Simulate a second arm by flipping a handful of outcomes, then check
            # that b and c computed from failure INDEX SETS equal those computed
            # from the full per-sample vectors.
            other_is_correct::BitVector = copy(diagnosis.is_correct)
            for index in 1:7:n_samples
                other_is_correct[index] = !other_is_correct[index]
            end
            failures_a::Set{Int} = Set(findall(.!diagnosis.is_correct))
            failures_b::Set{Int} = Set(findall(.!other_is_correct))
            b_from_sets::Int = length(setdiff(failures_b, failures_a))
            c_from_sets::Int = length(setdiff(failures_a, failures_b))
            b_from_vectors::Int = count(diagnosis.is_correct .& .!other_is_correct)
            c_from_vectors::Int = count(.!diagnosis.is_correct .& other_is_correct)
            @test b_from_sets == b_from_vectors
            @test c_from_sets == c_from_vectors
        end
    end
    return all_passed
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

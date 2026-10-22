using CorrelatedBPDecoderWithCER
using Test

# =============================================================================
# Tests for the test-time α transfer (src/neuralbase.jl, "Carrying α to another
# test set"): α* of a CER file's rates, α_test = α_learned · α*(test)/α*(train),
# the CER file of the training key, and the mode switch. The α* values below were
# checked against Python (math.log) on 2026-10-22.
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_coupling_scale_transfer.jl")'
# =============================================================================

function uniform_rates(rate::Float64, n_qubits::Int)::Dict{Int, Float32}
    """
    `n_qubits` single-qubit rates, all equal to `rate`, keyed 1…n as parse_cer_data keys them.
    """
    rates::Dict{Int, Float32} = Dict{Int, Float32}(qubit => Float32(rate) for qubit in 1:n_qubits)
    return rates
end

function write_cer_file(path::String, rates::Vector{Float64}, couplings::Vector{Tuple{Int, Int, Float64}})::Nothing
    """
    A CER file in the format parse_cer_data reads: `i : rate` and `(i, k) : J` lines.
    """
    open(path, "w") do io
        for (qubit, rate) in enumerate(rates)
            println(io, "$(qubit) : $(rate)")
        end
        for (qubit_a, qubit_b, coupling) in couplings
            println(io, "($(qubit_a), $(qubit_b)) : $(coupling)")
        end
    end
    return nothing
end

@testset "coupling_scale_transfer_mode" begin
    @test coupling_scale_transfer_mode("none") == COUPLING_SCALE_TRANSFER_NONE
    @test coupling_scale_transfer_mode("alpha_star") == COUPLING_SCALE_TRANSFER_ALPHA_STAR
    @test coupling_scale_transfer_mode("  Alpha_Star ") == COUPLING_SCALE_TRANSFER_ALPHA_STAR
    @test_throws ArgumentError coupling_scale_transfer_mode("alpha")
    @test_throws ArgumentError coupling_scale_transfer_mode("")
    @test COUPLING_SCALE_TRANSFER_TAG == "_alphastar"
end

@testset "prior_softening_factor is the log-odds ratio of the prior rescale" begin
    # The training key's median raw rate (p_0.0015_sig_0.003_s_1): α* = 0.5378.
    training_like::Dict{Int, Float32} = uniform_rates(0.01653, 72)
    median_rate::Float64 = Float64(Float32(0.01653))
    expected_factor::Float64 = log(9.0) / log((1.0 - median_rate) / median_rate)
    @test prior_softening_factor(training_like, 0.1f0) ≈ expected_factor rtol = 1e-12
    @test prior_softening_factor(training_like, 0.1f0) ≈ 0.5378 atol = 1e-3
    # p_0.0001_sig_0.0_s_1's median (1.0e-3): α* = 0.318.
    @test prior_softening_factor(uniform_rates(1.0e-3, 72), 0.1f0) ≈ 0.318 atol = 1e-3

    # The median, not the mean: rates 1, 2, 9 (× 1e-3) have median 2e-3.
    spread::Dict{Int, Float32} = Dict{Int, Float32}(1 => 1f-3, 2 => 2f-3, 3 => 9f-3)
    spread_median::Float64 = Float64(2f-3)
    @test prior_softening_factor(spread, 0.1f0) ≈
          log(9.0) / log((1.0 - spread_median) / spread_median) rtol = 1e-12

    # It IS the rescale's softening: after rescale_single_qubit_error_rates the median
    # qubit's log-odds are α* times its raw log-odds.
    rescaled::Dict{Int, Float32} = rescale_single_qubit_error_rates(spread, 0.1f0; verbose = false)
    rescaled_median::Float64 = Float64(rescaled[2])
    @test log((1.0 - rescaled_median) / rescaled_median) ≈
          prior_softening_factor(spread, 0.1f0) * log((1.0 - spread_median) / spread_median) rtol = 1e-5

    # 1 wherever the rescale leaves the rates alone.
    @test prior_softening_factor(training_like, 0f0) == 1.0
    @test prior_softening_factor(Dict{Int, Float32}(), 0.1f0) == 1.0
    @test prior_softening_factor(Dict{Int, Float32}(1 => 1f-3, 2 => 0.2f0), 0.1f0) == 1.0
end

@testset "transfer_coupling_scale carries the learned α to the test key" begin
    directory::String = mktempdir()
    training_file::String = joinpath(directory, "correlated_weights_train_like.txt")
    test_file::String = joinpath(directory, "correlated_weights_test_like.txt")
    write_cer_file(training_file, fill(0.01653, 6), [(1, 2, 2.8), (2, 3, 2.7)])
    write_cer_file(test_file, fill(1.0e-3, 6), [(1, 2, 4.6), (2, 3, 4.5)])

    learned_coupling_scale::Float32 = 0.513f0
    transfer::CouplingScaleTransfer = transfer_coupling_scale(learned_coupling_scale, test_file, training_file, 0.1f0)
    test_median::Float64 = Float64(Float32(1.0e-3))
    training_median::Float64 = Float64(Float32(0.01653))
    expected_test_factor::Float64 = log(9.0) / log((1.0 - test_median) / test_median)
    expected_training_factor::Float64 = log(9.0) / log((1.0 - training_median) / training_median)
    @test transfer.learned_coupling_scale == learned_coupling_scale
    @test transfer.test_softening_factor ≈ expected_test_factor rtol = 1e-12
    @test transfer.training_softening_factor ≈ expected_training_factor rtol = 1e-12
    @test transfer.transferred_coupling_scale ≈
          Float32(0.513 * expected_test_factor / expected_training_factor) rtol = 1e-6
    @test transfer.transferred_coupling_scale ≈ 0.303f0 atol = 2f-3
    # The couplings J play no part: only the single-qubit rates set α*.
    write_cer_file(test_file, fill(1.0e-3, 6), [(1, 2, 9.0)])
    @test transfer_coupling_scale(learned_coupling_scale, test_file, training_file, 0.1f0).transferred_coupling_scale ==
          transfer.transferred_coupling_scale

    # The training key tested on itself keeps the learned α exactly.
    @test transfer_coupling_scale(learned_coupling_scale, training_file, training_file, 0.1f0).transferred_coupling_scale ==
          learned_coupling_scale
    # Without the rescale the priors are raw, like J: nothing to carry.
    @test transfer_coupling_scale(learned_coupling_scale, test_file, training_file, 0f0).transferred_coupling_scale ==
          learned_coupling_scale

    # The stored logit round-trips to the transferred α.
    @test coupling_scale_from_logit(logit_from_coupling_scale(transfer.transferred_coupling_scale)) ≈
          transfer.transferred_coupling_scale rtol = 1e-6

    @test_throws ArgumentError transfer_coupling_scale(learned_coupling_scale, joinpath(directory, "missing.txt"),
                                                       training_file, 0.1f0)
end

@testset "training_key_cer_data_file follows train_<key> to its CER file" begin
    prefix::String = mktempdir()
    mkpath(joinpath(prefix, "correlated_weights"))
    expected_file::String = "$(prefix)/correlated_weights/correlated_weights_p_0.0015_sig_0.003_s_1.txt"
    write_cer_file(expected_file, fill(0.01653, 3), Tuple{Int, Int, Float64}[])
    @test training_key_cer_data_file(prefix, "train_p_0.0015_sig_0.003_s_1") == expected_file
    @test_throws ArgumentError training_key_cer_data_file(prefix, "training_errors")
    @test_throws ArgumentError training_key_cer_data_file(prefix, "train_p_0.0005_sig_0.0_s_1")
end

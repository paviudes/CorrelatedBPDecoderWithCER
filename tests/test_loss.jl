using CorrelatedBPDecoderWithCER
using Test

"""
Exercise the two detached gates on the correlation term and the active-pair
normalisation, on a hand-built 4-qubit example where every value is checkable by
hand.
"""
function test_correlation_gates()::Nothing
    n_bits = 4
    n_samples = 3

    # Columns: a decided co-flipped pair, an undecided pair, a decided pair that
    # is not co-flipped.
    posterior_llrs = Float32[
        4.0   0.2   4.0
        4.0   0.2  -4.0
       -4.0  -4.0  -4.0
       -4.0  -4.0  -4.0
    ]
    connectivity = [1 2]
    correlation_strengths = Float32[3.0]
    certainty_threshold = 2.2f0

    rewards = ising_correlation_reward_per_sample(
        posterior_llrs, connectivity, correlation_strengths, certainty_threshold
    )

    sigmoid_of_four::Float32 = 1.0f0 / (1.0f0 + exp(-4.0f0))

    @testset "certainty gate" begin
        # Sample 1: both endpoints decided (|μ| = 4 > 2.2), both flagged.
        # One active pair, so the normaliser is 1 and r = -J σ σ.
        @test rewards[1] ≈ -3.0f0 * sigmoid_of_four^2 atol=1e-5
        # Sample 2: |μ| = 0.2 < 2.2, gate closed, no active pairs, r = 0.
        # Ungated this pair would have contributed -3 * 0.55^2 = -0.91.
        @test rewards[2] == 0.0f0
        # Sample 3: decided, so the gate is open, but qubit 2 is σ ≈ 0 and the
        # co-activation product is ~0. The gate admits it; the reward is ~0.
        @test rewards[3] ≈ 0.0f0 atol=1e-3
    end

    @testset "gate-open fraction" begin
        open_fraction = correlation_gate_open_fraction(
            posterior_llrs, connectivity, certainty_threshold
        )
        # Two of three samples clear the threshold on both endpoints.
        @test open_fraction ≈ 2.0f0 / 3.0f0 atol=1e-6
    end

    @testset "active-pair normalisation" begin
        # Two identical pairs must give the same per-sample reward as one, since
        # the reward is a MEAN over active pairs, not a sum over all edges.
        wide_llrs = Float32[4.0; 4.0; 4.0; 4.0;;]
        one_pair = ising_correlation_reward_per_sample(
            wide_llrs, [1 2], Float32[3.0], certainty_threshold
        )
        two_pairs = ising_correlation_reward_per_sample(
            wide_llrs, [1 2; 3 4], Float32[3.0, 3.0], certainty_threshold
        )
        @test one_pair[1] ≈ two_pairs[1] atol=1e-6
    end

    @testset "reward is non-positive for positive couplings" begin
        # Ordering safety: a gated solved sample can never score worse than a
        # failing one, whose base loss is ≳ 1 per broken check.
        @test all(rewards .<= 0.0f0)
    end
    return nothing
end

"""
The syndrome gate is an indicator on the soft H-weight, so it must be exactly 0
or 1 and must use the stabilizers alone.
"""
function test_syndrome_gate()::Nothing
    parity_check_matrix = BitMatrix([1 1 0 0; 0 0 1 1])
    expected_recoveries = BitMatrix([0 0; 0 0; 0 0; 0 0])
    # Sample 1 is a clean solution; sample 2 flips one qubit of the first check.
    posterior_llrs = Float32[-8.0 8.0; -8.0 -8.0; -8.0 -8.0; -8.0 -8.0]

    weights = soft_syndrome_weight_per_sample(
        posterior_llrs, expected_recoveries, parity_check_matrix
    )
    gate = syndrome_gate_per_sample(
        posterior_llrs, expected_recoveries, parity_check_matrix, 0.5f0
    )
    @testset "syndrome gate" begin
        @test weights[1] < 0.5f0
        @test weights[2] > 0.5f0
        @test gate == Float32[1.0, 0.0]
        @test all(g -> g == 0.0f0 || g == 1.0f0, gate)
    end
    return nothing
end

test_correlation_gates()
test_syndrome_gate()

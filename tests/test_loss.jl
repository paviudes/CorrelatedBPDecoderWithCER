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

# =============================================================================
#  Certainty penalty family: the point of the alternatives is the force at mu=0
# =============================================================================
@testset "certainty penalty family" begin
    # --- shape: all three are symmetric, peak at 0, decay to 0 ---------------
    for penalty_function in (binary_entropy_of_sigmoid,
                             exponential_certainty_penalty,
                             μ -> hinge_certainty_penalty(μ, 2.2f0))
        @test penalty_function(1.5f0) ≈ penalty_function(-1.5f0) atol = 1f-6
        @test penalty_function(0.0f0) > penalty_function(1.0f0)
        @test penalty_function(1.0f0) > penalty_function(4.0f0)
        @test penalty_function(20.0f0) ≈ 0.0f0 atol = 1f-5
    end

    # --- the whole reason these exist: force at a perfectly undecided qubit --
    # Central differences, so a cusp reports ~0 by symmetry; take the one-sided
    # slope just off zero, which is what the optimizer actually sees.
    step::Float32 = 1f-3
    entropy_force::Float32 =
        abs(binary_entropy_of_sigmoid(step) - binary_entropy_of_sigmoid(0.0f0)) / step
    exponential_force::Float32 =
        abs(exponential_certainty_penalty(step) - exponential_certainty_penalty(0.0f0)) / step
    hinge_force::Float32 =
        abs(hinge_certainty_penalty(step, 2.2f0) - hinge_certainty_penalty(0.0f0, 2.2f0)) / step
    @test entropy_force < 1f-2                     # vanishes, as symmetry demands
    @test exponential_force > 0.9f0                # ~1, its maximum
    @test hinge_force ≈ 1.0f0 / 2.2f0 rtol = 1f-2  # exactly 1/w

    # --- hinge is exactly inert beyond its width ----------------------------
    @test hinge_certainty_penalty(2.2f0, 2.2f0) == 0.0f0
    @test hinge_certainty_penalty(5.0f0, 2.2f0) == 0.0f0

    # --- code lookup is strict ----------------------------------------------
    @test certainty_penalty_code("entropy")      == CERTAINTY_PENALTY_ENTROPY
    @test certainty_penalty_code("  Hinge  ")    == CERTAINTY_PENALTY_HINGE
    @test_throws ArgumentError certainty_penalty_code("gaussian")

    # --- per-sample sum dispatches and stays finite --------------------------
    posterior_llrs::Matrix{Float32} = Float32[0.0 3.0; -0.5 -8.0; 1.0 0.2]
    for kind in (CERTAINTY_PENALTY_ENTROPY, CERTAINTY_PENALTY_EXPONENTIAL, CERTAINTY_PENALTY_HINGE)
        certainties::Vector{Float32} = certainty_per_sample(posterior_llrs, kind, 2.2f0)
        @test length(certainties) == 2
        @test all(isfinite, certainties)
        @test all(certainties .>= 0.0f0)
        # sample 1 is far less decided than sample 2, for every penalty
        @test certainties[1] > certainties[2]
    end
end

using CorrelatedBPDecoderWithCER
using Test

"""
Exercise the two detached gates on the correlation term and the active-pair
normalisation, on a hand-built 4-qubit example where every value is checkable by
hand.
"""
function test_correlation_gates()::Nothing
    # SIGN CONVENTION. `sigmoid(μ) = 1/(1 + exp(μ))` is the ERROR probability, so
    # μ = -4 is a qubit that almost certainly FLIPPED (σ ≈ 0.982) and μ = +4 is
    # one that almost certainly did not (σ ≈ 0.018). An earlier version of this
    # test used 1/(1 + exp(-μ)) and so asserted the mirror image of every case.
    n_bits = 4
    n_samples = 4

    # Pair under test is (qubit 1, qubit 2). Columns, in order:
    #   1  both FLIPPED and decided  -> gate open, co-activation ≈ 1
    #   2  undecided                 -> gate closed
    #   3  both CLEAN and decided    -> gate open, co-activation ≈ 0
    #   4  one flipped one clean     -> gate open, co-activation small
    posterior_llrs = Float32[
       -4.0   0.2   4.0  -4.0
       -4.0   0.2   4.0   4.0
       -4.0  -4.0  -4.0  -4.0
       -4.0  -4.0  -4.0  -4.0
    ]
    connectivity = [1 2]
    correlation_strengths = Float32[3.0]
    certainty_threshold = 2.2f0

    rewards = ising_correlation_reward_per_sample(
        posterior_llrs, connectivity, correlation_strengths, certainty_threshold
    )

    # Probability that a qubit at μ = ∓4 is flipped, in the CODE's convention.
    flipped_probability::Float32 = sigmoid(-4.0f0)   # ≈ 0.98201
    clean_probability::Float32   = sigmoid(4.0f0)    # ≈ 0.01799

    @testset "certainty gate" begin
        # Sample 1: both decided (|μ| = 4 > 2.2) and both flipped. One active
        # pair, so the normaliser is 1 and r = -J σ σ ≈ -3 × 0.982² = -2.893.
        @test rewards[1] ≈ -3.0f0 * flipped_probability^2 atol=1e-5
        # Sample 2: |μ| = 0.2 < 2.2, gate closed, no active pairs, r = 0 exactly.
        # Ungated this pair would have contributed ≈ -3 × 0.45² = -0.61.
        @test rewards[2] == 0.0f0
        # Sample 3: gate open, but neither qubit flipped, so the co-activation
        # product is ≈ 0.018² and the reward is ≈ -0.001.
        @test rewards[3] ≈ -3.0f0 * clean_probability^2 atol=1e-5
        @test abs(rewards[3]) < 1e-2
        # Sample 4: gate open, exactly one qubit flipped. The couplings reward
        # CO-activation, so a mixed pair earns far less than a co-flipped one.
        @test rewards[4] ≈ -3.0f0 * flipped_probability * clean_probability atol=1e-5
        @test abs(rewards[4]) < abs(rewards[1]) / 10
    end

    @testset "gate-open fraction" begin
        open_fraction = correlation_gate_open_fraction(
            posterior_llrs, connectivity, certainty_threshold
        )
        # Three of four samples clear the threshold on both endpoints.
        @test open_fraction ≈ 3.0f0 / 4.0f0 atol=1e-6
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

# =============================================================================
#  log-agreement L3: force at the DISCORDANT corners, none at the concordant ones
# =============================================================================

# Single-pair, single-sample wrappers so the corner cases below read as scalars.
# Named top-level functions rather than closures because an anonymous
# `function (args...)::T` is a syntax error in Julia — the return-type annotation
# is only allowed on a named definition — and the house style requires one.
function log_agreement_penalty_for_pair(
    mu_i::Float32,
    mu_k::Float32,
    coupling::Float32,
    certainty_threshold::Float32,
    agreement_floor::Float32
)::Float32
    posterior_llrs::Matrix{Float32} = reshape(Float32[mu_i, mu_k], 2, 1)
    pair_connectivity::Matrix{Int} = [1 2]
    penalties::Vector{Float32} = ising_log_agreement_penalty_per_sample(
        posterior_llrs, pair_connectivity, Float32[coupling],
        certainty_threshold, agreement_floor
    )
    return penalties[1]
end

function bilinear_reward_for_pair(
    mu_i::Float32,
    mu_k::Float32,
    coupling::Float32,
    certainty_threshold::Float32
)::Float32
    posterior_llrs::Matrix{Float32} = reshape(Float32[mu_i, mu_k], 2, 1)
    pair_connectivity::Matrix{Int} = [1 2]
    rewards::Vector{Float32} = ising_correlation_reward_per_sample(
        posterior_llrs, pair_connectivity, Float32[coupling], certainty_threshold
    )
    return rewards[1]
end

@testset "log agreement correlation term" begin
    # Certainty gate wide open: the gate is |μ| > threshold, so 0 admits every
    # decided qubit. (μ = 0 exactly still fails it, which the ≥ 0 grid below uses.)
    open_gate_threshold::Float32 = 0.0f0
    floor_value::Float32 = 1.0f-4
    saturated::Float32 = 12.0f0
    step::Float32 = 1.0f-3

    # (1 + t_i t_k)/2 IS P(agree): check against the explicit probability.
    for (mu_i, mu_k) in ((2.0f0, 3.0f0), (-1.0f0, 4.0f0), (0.0f0, 0.0f0))
        error_probability_i::Float32 = sigmoid(mu_i)
        error_probability_k::Float32 = sigmoid(mu_k)
        agreement::Float32 = 0.5f0 * (1.0f0 + tanh(mu_i / 2) * tanh(mu_k / 2))
        expected_agreement::Float32 =
            error_probability_i * error_probability_k +
            (1.0f0 - error_probability_i) * (1.0f0 - error_probability_k)
        @test agreement ≈ expected_agreement atol=1f-5
    end

    # POSITIVE coupling: concordant corners cost nothing, discordant ones cost.
    @test log_agreement_penalty_for_pair( saturated,  saturated, 1.5f0, open_gate_threshold, floor_value) ≈ 0.0f0 atol=1f-4
    @test log_agreement_penalty_for_pair(-saturated, -saturated, 1.5f0, open_gate_threshold, floor_value) ≈ 0.0f0 atol=1f-4
    @test log_agreement_penalty_for_pair( saturated, -saturated, 1.5f0, open_gate_threshold, floor_value) > 10.0f0
    # NEGATIVE coupling: the barrier moves to AGREEMENT, and stays a penalty.
    @test log_agreement_penalty_for_pair( saturated, -saturated, -0.6f0, open_gate_threshold, floor_value) ≈ 0.0f0 atol=1f-4
    @test log_agreement_penalty_for_pair( saturated,  saturated, -0.6f0, open_gate_threshold, floor_value) > 1.0f0

    # Never negative, for either sign of J — this is what a raw -J log A lacked:
    # with J < 0 that form is unbounded BELOW, and 24% of the real couplings are
    # negative, so the optimizer would have had a free escape to -Inf.
    for coupling in (2.0f0, -2.0f0), mu_i in (-saturated, 0.0f0, saturated), mu_k in (-saturated, 0.0f0, saturated)
        @test log_agreement_penalty_for_pair(mu_i, mu_k, coupling, open_gate_threshold, floor_value) >= 0.0f0
    end

    # Finite even where tanh saturates to exactly 1.0f0 (|μ| ≳ 18): without the
    # floor this is log(0) = -Inf, a NaN gradient and silently untrained weights.
    @test isfinite(log_agreement_penalty_for_pair(40.0f0, -40.0f0,  5.36f0, open_gate_threshold, floor_value))
    @test isfinite(log_agreement_penalty_for_pair(40.0f0,  40.0f0, -5.36f0, open_gate_threshold, floor_value))

    # Gradient: ~0 at the concordant corners, non-vanishing at the discordant ones.
    concordant_force::Float32 = abs(
        log_agreement_penalty_for_pair(saturated + step, saturated, 1.5f0, open_gate_threshold, floor_value) -
        log_agreement_penalty_for_pair(saturated,        saturated, 1.5f0, open_gate_threshold, floor_value)) / step
    discordant_force::Float32 = abs(
        log_agreement_penalty_for_pair(2.0f0 + step, -2.0f0, 1.5f0, open_gate_threshold, floor_value) -
        log_agreement_penalty_for_pair(2.0f0,        -2.0f0, 1.5f0, open_gate_threshold, floor_value)) / step
    @test concordant_force < 1f-3
    @test discordant_force > 0.1f0

    # The bilinear form vanishes at the discordant corner too — the defect.
    bilinear_discordant_force::Float32 = abs(
        bilinear_reward_for_pair(saturated + step, -saturated, 1.5f0, open_gate_threshold) -
        bilinear_reward_for_pair(saturated,        -saturated, 1.5f0, open_gate_threshold)) / step
    @test bilinear_discordant_force < 1f-3

    @test correlation_form_code("bilinear") == CORRELATION_FORM_BILINEAR
    @test correlation_form_code(" Log_Agreement ") == CORRELATION_FORM_LOG_AGREEMENT
    @test_throws ArgumentError correlation_form_code("ising")
end

# =============================================================================
#  tau_2: L2's own syndrome gate. The narrow hinge is DEAD under a shared gate.
# =============================================================================

function gated_certainty_contribution(
    posterior_llrs::Matrix{Float32},
    expected_recoveries::BitMatrix,
    parity_check_matrix::BitMatrix,
    syndrome_gate_threshold::Float32,
    certainty_syndrome_gate_threshold::Float32,
    certainty_penalty_kind::Int,
    certainty_hinge_width::Float32
)::Float32
    """
    The part of L2 that actually reaches the gradient: gate x penalty, summed.
    Testing the penalty VALUE alone is what let a structurally dead term ship --
    both hinge widths logged non-zero penalties while contributing exactly zero.
    """
    syndrome_weights::Vector{Float32} = soft_syndrome_weight_per_sample(
        posterior_llrs, expected_recoveries, parity_check_matrix)
    effective_threshold::Float32 = certainty_syndrome_gate_threshold
    if certainty_syndrome_gate_threshold < 0.0f0
        effective_threshold = syndrome_gate_threshold
    end
    certainty_gate::Vector{Float32} = Float32.(syndrome_weights .< effective_threshold)
    penalties::Vector{Float32} = certainty_per_sample(
        posterior_llrs, certainty_penalty_kind, certainty_hinge_width)
    total::Float32 = sum(certainty_gate .* penalties)
    return total
end

@testset "L2 gate decoupling" begin
    # Two weight-2 checks on 4 qubits. Sample 1 carries TWO undecided qubits
    # (mu = 0.0 and 0.4); sample 2 is fully decided and clean. mu = 0.4 matters:
    # at mu = 0 the hinge is 1.0 for EVERY width, so a fixture with only that
    # qubit could not tell two widths apart even with the gate open.
    parity_check_matrix = BitMatrix([1 1 0 0; 0 0 1 1])
    expected_recoveries = BitMatrix([0 0; 0 0; 0 0; 0 0])
    posterior_llrs = Float32[0.0 8.0; 0.4 8.0; 8.0 8.0; 8.0 8.0]

    syndrome_weights::Vector{Float32} = soft_syndrome_weight_per_sample(
        posterior_llrs, expected_recoveries, parity_check_matrix)
    # An undecided qubit lifts |s| far above tau = 0.5. That is the whole
    # mechanism: the shared gate drops exactly the sample L2 wants to fix.
    @test syndrome_weights[1] > 0.5f0     # ~0.989
    @test syndrome_weights[2] < 0.5f0     # ~0.002

    narrow::Float32 = 0.3f0
    wider::Float32  = 0.5f0
    inherit_tau::Float32 = -1.0f0
    opened::Float32 = 1.0f6

    shared_narrow::Float32 = gated_certainty_contribution(posterior_llrs,
        expected_recoveries, parity_check_matrix, 0.5f0, inherit_tau,
        CERTAINTY_PENALTY_HINGE, narrow)
    shared_wider::Float32 = gated_certainty_contribution(posterior_llrs,
        expected_recoveries, parity_check_matrix, 0.5f0, inherit_tau,
        CERTAINTY_PENALTY_HINGE, wider)
    opened_narrow::Float32 = gated_certainty_contribution(posterior_llrs,
        expected_recoveries, parity_check_matrix, 0.5f0, opened,
        CERTAINTY_PENALTY_HINGE, narrow)
    opened_wider::Float32 = gated_certainty_contribution(posterior_llrs,
        expected_recoveries, parity_check_matrix, 0.5f0, opened,
        CERTAINTY_PENALTY_HINGE, wider)

    # Under the INHERITED gate a narrow hinge contributes nothing at all, and the
    # two widths are indistinguishable -- the exact symptom that shipped 72 dead
    # runs whose weights came out bit-identical.
    @test shared_narrow == 0.0f0
    @test shared_wider == 0.0f0
    @test shared_narrow == shared_wider
    # DECOUPLED, it fires, and the widths separate.
    @test opened_narrow ≈ 1.0f0 atol = 1f-4
    @test opened_wider ≈ 1.2f0 atol = 1f-4
    @test opened_narrow != opened_wider

    # The entropy survives a shared gate: its tail reaches the decided sample.
    shared_entropy::Float32 = gated_certainty_contribution(posterior_llrs,
        expected_recoveries, parity_check_matrix, 0.5f0, inherit_tau,
        CERTAINTY_PENALTY_ENTROPY, narrow)
    @test shared_entropy > 0.0f0

    # tau_2 < 0 must reproduce the shared gate exactly, for every penalty: this
    # is what keeps every pre-split run bit-for-bit reproducible.
    for kind in (CERTAINTY_PENALTY_ENTROPY, CERTAINTY_PENALTY_EXPONENTIAL, CERTAINTY_PENALTY_HINGE)
        @test gated_certainty_contribution(posterior_llrs, expected_recoveries,
                  parity_check_matrix, 0.5f0, -1.0f0, kind, 2.2f0) ==
              gated_certainty_contribution(posterior_llrs, expected_recoveries,
                  parity_check_matrix, 0.5f0, 0.5f0, kind, 2.2f0)
    end
end

# =============================================================================
#  coflip L3: silent at concordant corners, kicks the CLEAN qubit of a
#  discordant pair toward errored, positive couplings only.
# =============================================================================

function coflip_penalty_for_pair(
    mu_i::Float32,
    mu_k::Float32,
    coupling::Float32,
    certainty_threshold::Float32,
    agreement_floor::Float32
)::Float32
    posterior_llrs::Matrix{Float32} = reshape(Float32[mu_i, mu_k], 2, 1)
    pair_connectivity::Matrix{Int} = [1 2]
    penalties::Vector{Float32} = ising_coflip_penalty_per_sample(
        posterior_llrs, pair_connectivity, Float32[coupling],
        certainty_threshold, agreement_floor
    )
    return penalties[1]
end

@testset "coflip correlation term" begin
    open_gate::Float32 = 0.0f0
    floor_value::Float32 = 1.0f-4
    # |mu| = 6 is DECIDED (sigma = 2.5e-3) but not hyper-saturated: the kick is
    # near full strength there. At |mu| = 14 (sigma = 8e-7 < eps) the eps floor
    # caps the 1/sigma divergence and the kick decays -- by design, since a
    # bounded loss cannot move a qubit that saturated anyway.
    decided::Float32 = 6.0f0
    step::Float32 = 1.0f-3

    # --- values: small at both concordant corners, large at discordance ------
    # At (clean, clean) the VALUE is ~0.044, not 0: each term is sigma*|log sigma|
    # ~ mu*exp(-mu), which only reaches 0 asymptotically. What the request was
    # about -- and what the gradient assertions below check -- is that the FORCE
    # is silent there. 0.044 against 8.9 at discordance is a 200x separation.
    @test coflip_penalty_for_pair( decided,  decided, 1.5f0, open_gate, floor_value) < 0.1f0
    @test coflip_penalty_for_pair(-decided, -decided, 1.5f0, open_gate, floor_value) ≈ 0.0f0 atol=1f-2
    @test coflip_penalty_for_pair( decided, -decided, 1.5f0, open_gate, floor_value) > 5.0f0
    @test coflip_penalty_for_pair(-decided,  decided, 1.5f0, open_gate, floor_value) > 5.0f0

    # --- the requested gradient profile --------------------------------------
    kick_on_clean::Float32 = abs(
        coflip_penalty_for_pair(decided + step, -decided, 1.5f0, open_gate, floor_value) -
        coflip_penalty_for_pair(decided,        -decided, 1.5f0, open_gate, floor_value)) / step
    kick_on_errored::Float32 = abs(
        coflip_penalty_for_pair(decided, -decided - step, 1.5f0, open_gate, floor_value) -
        coflip_penalty_for_pair(decided, -decided,        1.5f0, open_gate, floor_value)) / step
    silent_both_clean::Float32 = abs(
        coflip_penalty_for_pair(decided + step, decided, 1.5f0, open_gate, floor_value) -
        coflip_penalty_for_pair(decided,        decided, 1.5f0, open_gate, floor_value)) / step
    silent_both_errored::Float32 = abs(
        coflip_penalty_for_pair(-decided + step, -decided, 1.5f0, open_gate, floor_value) -
        coflip_penalty_for_pair(-decided,        -decided, 1.5f0, open_gate, floor_value)) / step
    @test kick_on_clean > 1.0f0            # ~ J * 0.956 = 1.43: the strong kick
    @test kick_on_errored < 0.1f0          # the errored partner is left alone
    @test silent_both_clean < 0.05f0       # concordant: silent
    @test silent_both_errored < 0.05f0     # concordant: silent

    # --- negative couplings are excluded entirely ----------------------------
    @test coflip_penalty_for_pair( decided, -decided, -0.6f0, open_gate, floor_value) == 0.0f0
    @test coflip_penalty_for_pair(-decided, -decided, -0.6f0, open_gate, floor_value) == 0.0f0
    # ...including from the active-pair normaliser: a mixed edge list must give
    # the same value as the positive edge alone.
    mixed_llrs::Matrix{Float32} = reshape(Float32[decided, -decided, decided, -decided], 4, 1)
    positive_only::Vector{Float32} = ising_coflip_penalty_per_sample(
        mixed_llrs, [1 2], Float32[1.5], open_gate, floor_value)
    with_negative::Vector{Float32} = ising_coflip_penalty_per_sample(
        mixed_llrs, [1 2; 3 4], Float32[1.5, -0.6], open_gate, floor_value)
    @test positive_only[1] ≈ with_negative[1] atol=1f-6

    # --- finite at Float32 sigmoid underflow (mu >= 104 gives sigma == 0.0f0) --
    @test isfinite(coflip_penalty_for_pair(120.0f0, -120.0f0, 5.36f0, open_gate, floor_value))
    @test coflip_penalty_for_pair(120.0f0, -120.0f0, 5.36f0, open_gate, floor_value) >= 0.0f0

    # --- certainty gate still applies ----------------------------------------
    @test coflip_penalty_for_pair(0.2f0, -0.2f0, 1.5f0, 2.2f0, floor_value) == 0.0f0

    # --- never negative over a grid, for either coupling sign ----------------
    for coupling in (2.0f0, -2.0f0), a in (-decided, 0.0f0, decided), b in (-decided, 0.0f0, decided)
        @test coflip_penalty_for_pair(a, b, coupling, open_gate, floor_value) >= 0.0f0
    end

    @test correlation_form_code("coflip") == CORRELATION_FORM_COFLIP
end

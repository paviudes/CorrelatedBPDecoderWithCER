using CorrelatedBPDecoderWithCER
using Test
using Enzyme

# =============================================================================
# Tests for the training loss, src/loss.jl. The loss is the softmin over scored
# layers of the base loss — the residue of e + σ(μ) against [H; L] — and nothing
# else, so these tests are short.
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_loss.jl")'
# =============================================================================

function two_check_dual()::BitMatrix
    """
    Four bits, two stabilizer rows and one logical row: [H; L].
    """
    dual::BitMatrix = BitMatrix([1 1 0 0; 0 0 1 1; 1 0 1 0])
    return dual
end

@testset "smooth_loss is the piecewise quadratic distance to the even integers" begin
    @test smooth_loss(0.0f0) == 0.0f0
    @test smooth_loss(2.0f0) == 0.0f0
    @test smooth_loss(4.0f0) == 0.0f0
    @test smooth_loss(1.0f0) == 1.0f0          # maximal at the odd integers
    @test smooth_loss(0.5f0) == 0.25f0
    @test smooth_loss(1.5f0) == 0.25f0         # symmetric about 1
    @test smooth_loss(2.5f0) == 0.25f0         # periodic with period 2
    @test smooth_loss(3.0f0) == 1.0f0
end

@testset "compute_smooth_loss_from_llrs is zero exactly at the right decode" begin
    dual::BitMatrix = two_check_dual()
    # Sample 1: error on bit 1. Sample 2: no error.
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 0; 0 0])
    # Confident correct decisions: σ(μ) ≈ 1 where the error is (μ ≪ 0), ≈ 0 elsewhere.
    correct::Matrix{Float32} = Float32[-30 30; 30 30; 30 30; 30 30]
    @test compute_smooth_loss_from_llrs(correct, expected, dual) < 1e-6
    # Confident WRONG decision on sample 1 (says bit 1 is clean): e + σ(μ) = 1 on
    # bit 1, which breaks check 1 and logical 1 -> two unit penalties for one of
    # two samples.
    wrong::Matrix{Float32} = Float32[30 30; 30 30; 30 30; 30 30]
    @test isapprox(compute_smooth_loss_from_llrs(wrong, expected, dual), 2.0f0 / 2; atol = 1e-5)
    # Undecided (μ = 0, σ = 0.5 everywhere). Sample 1: e + σ = [1.5, .5, .5, .5],
    # rows give 2.0, 1.0, 2.0 -> penalty 1. Sample 2: every row sums to 1.0 ->
    # penalty 3. Mean over the two samples: 2. Note this is WORSE than the
    # confidently wrong decode above: the penalty is periodic in the residue, so
    # half-decided bits can land on the odd integers where it is maximal.
    undecided::Matrix{Float32} = zeros(Float32, 4, 2)
    @test isapprox(compute_smooth_loss_from_llrs(undecided, expected, dual), 2.0f0; atol = 1e-5)
end

@testset "softmin_loss interpolates between the mean and the minimum" begin
    losses::Vector{Float32} = Float32[3.0, 1.0, 2.0]
    @test softmin_loss(Float32[1.5], 0.3f0) == 1.5f0                    # one layer: itself
    @test isapprox(softmin_loss(losses, 1.0f-3), 1.0f0; atol = 1e-2)   # cold: the minimum
    # This softmin sits BELOW min(L), by T·log(n) at zero spread, so it DECREASES
    # with temperature: cold is the minimum itself, warm is the minimum minus a
    # spread-dependent margin. Annealing T down therefore raises the reported
    # loss even when the decoder is improving -- which is why the training-loss
    # trajectory has to be read against the schedule, not on its own.
    @test softmin_loss(losses, 1.0f0) <= softmin_loss(losses, 1.0f-3)  # warmer is lower ...
    @test softmin_loss(losses, 1.0f0) <= minimum(losses)                # ... and never above the minimum
    @test softmin_loss(Float32[2.0, 2.0, 2.0], 0.5f0) < 2.0f0           # T·log(n) below at zero spread
    @test isapprox(softmin_loss(Float32[2.0, 2.0, 2.0], 0.5f0), 2.0f0 - 0.5f0 * log(3.0f0); atol = 1e-6)
end

@testset "compute_loss is the softmin of the per-layer base losses, after warmup" begin
    dual::BitMatrix = two_check_dual()
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 0; 0 0])
    # Three layers: wrong, undecided, correct.
    posteriors::Array{Float32, 3} = zeros(Float32, 4, 2, 3)
    posteriors[:, :, 1] .= Float32[30 30; 30 30; 30 30; 30 30]
    posteriors[:, :, 2] .= 0.0f0
    posteriors[:, :, 3] .= Float32[-30 30; 30 30; 30 30; 30 30]
    per_layer::Vector{Float32} = base_loss_per_layer(posteriors, expected, dual, 0)
    @test length(per_layer) == 3
    @test isapprox(per_layer[1], 1.0f0; atol = 1e-5)   # wrong (see the test above)
    @test isapprox(per_layer[2], 2.0f0; atol = 1e-5)   # undecided
    @test per_layer[3] < 1e-6                          # correct
    # A cold softmin commits to the correct layer whichever earlier layers are
    # still being scored.
    @test compute_loss(posteriors, expected, dual, 1.0f-3, 0) < 1e-2
    @test compute_loss(posteriors, expected, dual, 1.0f-3, 1) < 1e-2
    @test compute_loss(posteriors, expected, dual, 1.0f-3, 2) < 1e-2
    # Warmup drops the first layers from scoring.
    @test base_loss_per_layer(posteriors, expected, dual, 1) == per_layer[2:3]
    @test_throws ArgumentError base_loss_per_layer(posteriors, expected, dual, 3)
    for temperature in (0.01f0, 0.5f0, 5.0f0)
        @test compute_loss(posteriors, expected, dual, temperature, 0) ==
              softmin_loss(per_layer, temperature)
        @test compute_loss(posteriors, expected, dual, temperature, 1) ==
              softmin_loss(per_layer[2:3], temperature)
    end
end

@testset "Enzyme differentiates the training loss through the forward pass" begin
    # Bit 3 sits in BOTH checks, and that is load-bearing. `adj_C2V_V2C` connects
    # two neurons only when they share a VARIABLE and differ in CHECK, so on a
    # graph whose variables all have degree 1 -- [1 1 0 0; 0 0 1 1], the obvious
    # toy -- `nb_weights_c2v_v2c` is 0, `weights_c2v_v2c` is an empty vector, and
    # "some weight has a non-zero gradient" is vacuously false whatever the loss
    # does. The test would fail while reporting nothing about the loss.
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    initial_llrs::Vector{Float32} = fill(Float32(log(9)), 4)
    base::NeuralBPBase = NeuralBPBase(parity_check_matrix, parity_check_matrix_dual, initial_llrs, 3)
    @test base.nb_weights_c2v_v2c > 0
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.1f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.1f0),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.1f0),
    )
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 1; 0 0])
    syndromes::BitMatrix = BitMatrix(mod.(parity_check_matrix * Matrix{Int}(expected), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 2)

    grad_w_c2v_v2c::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_v2c))
    grad_w_llrs::Vector{Float32} = zeros(Float32, length(bpnn.weights_llrs))
    grad_w_readout::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_readout))
    grad_alpha::Vector{Float32} = zeros(Float32, 1)
    grad_schedule::Vector{Float32} = zeros(Float32, 2)
    temperature::Float32 = 1.0f0
    (_, loss_value) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        CorrelatedBPDecoderWithCER.get_loss_value,
        Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
        Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
        Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
        Enzyme.Duplicated(bpnn.coupling_logit, grad_alpha),
        Enzyme.Duplicated(bpnn.coupling_schedule, grad_schedule),
        Enzyme.Const(temperature),   # loss_layer_temperature
        Enzyme.Const(0),             # warmup_loss_layers
        Enzyme.Const(LOSS_LAYERS_SOFTMIN),              # loss_layer_selection
        Enzyme.Const(DEFAULT_LOSS_LAYER_RAMP_SHARPNESS), # loss_layer_ramp_sharpness (ignored here)
        Enzyme.Const(BASE_LOSS_SIN_RESIDUE),            # base_loss_selection
        Enzyme.Const(base),
        Enzyme.Const(llrs_batch),
        Enzyme.Const(syndromes),
        Enzyme.Const(expected)
    )
    @test isfinite(loss_value)
    # The total is a SOFTMIN, which sits below min(L): with per-layer losses
    # non-negative it lies in [-T·log(n_layers), min(L)], so a negative value is
    # normal, not a failure. Asserting `> 0` mistakes the softmin's offset for an
    # error -- the meaningful bound is the floor.
    @test loss_value >= -temperature * log(Float32(base.n_layers)) - 1.0f-4
    @test all(isfinite, grad_w_c2v_v2c)
    @test all(isfinite, grad_w_llrs)
    @test all(isfinite, grad_w_readout)
    # A gradient of exactly zero everywhere would mean the loss was detached
    # from the weights entirely -- silently untrainable.
    @test any(!iszero, grad_w_c2v_v2c)
    @test any(!iszero, grad_w_llrs)
    # The standard check node never reads alpha, so its gradient is exactly 0;
    # nor the layer schedule, which only scales alpha.
    @test grad_alpha[1] == 0.0f0
    @test grad_schedule == zeros(Float32, 2)
end

@testset "loss_layer_selection: softmin vs last vs mean" begin
    dual::BitMatrix = two_check_dual()
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 0; 0 0])
    # Three layers with deliberately different losses: wrong, undecided, correct.
    posteriors::Array{Float32, 3} = zeros(Float32, 4, 2, 3)
    posteriors[:, :, 1] .= Float32[30 30; 30 30; 30 30; 30 30]     # confidently wrong
    posteriors[:, :, 2] .= 0.0f0                                    # undecided
    posteriors[:, :, 3] .= Float32[-30 30; 30 30; 30 30; 30 30]     # correct
    per_layer::Vector{Float32} = base_loss_per_layer(posteriors, expected, dual, 0)

    # Names round-trip; an unknown one fails at configuration time.
    @test loss_layer_selection_code("softmin") == LOSS_LAYERS_SOFTMIN
    @test loss_layer_selection_code("last") == LOSS_LAYERS_LAST
    @test loss_layer_selection_code(" Mean ") == LOSS_LAYERS_MEAN
    @test loss_layer_selection_name(LOSS_LAYERS_LAST) == "last"
    @test loss_layer_selection_name(LOSS_LAYERS_MEAN) == "mean"
    @test_throws ArgumentError loss_layer_selection_code("argmin")
    @test_throws ArgumentError loss_layer_selection_name(9)
    @test_throws ArgumentError combine_layer_losses(per_layer, 1.0f0, 9)

    # LAST reads the final layer and nothing else. Here that layer is the correct
    # one, so the loss is ~0 even though layer 1 is confidently wrong -- which is
    # the point: `last` makes the earlier layers' quality invisible, and the
    # matching commit rule is the one that also only looks at the final layer.
    @test combine_layer_losses(per_layer, 1.0f0, LOSS_LAYERS_LAST) == per_layer[end]
    @test compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_LAST) < 1e-6
    # Temperature is ignored by `last`.
    for temperature in (0.01f0, 1.0f0, 100.0f0)
        @test compute_loss(posteriors, expected, dual, temperature, 0, LOSS_LAYERS_LAST) ==
              compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_LAST)
    end
    # ... and `warmup_loss_layers` cannot change it either, since the final layer
    # is the final layer however many earlier ones are dropped.
    @test compute_loss(posteriors, expected, dual, 1.0f0, 2, LOSS_LAYERS_LAST) ==
          compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_LAST)

    # MEAN is the plain average, so unlike softmin it CANNOT be satisfied by one
    # good layer: the confidently-wrong layer 1 keeps the total up.
    @test isapprox(combine_layer_losses(per_layer, 1.0f0, LOSS_LAYERS_MEAN),
                   sum(per_layer) / 3; atol = 1e-6)
    @test compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_MEAN) > 0.9f0
    @test compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_MEAN) >
          compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_LAST)

    # SOFTMIN is unchanged, and is the DEFAULT when no mode is given -- every
    # earlier run must reproduce bit for bit.
    for temperature in (0.01f0, 0.5f0, 5.0f0)
        @test combine_layer_losses(per_layer, temperature, LOSS_LAYERS_SOFTMIN) ==
              softmin_loss(per_layer, temperature)
        @test compute_loss(posteriors, expected, dual, temperature, 0) ==
              softmin_loss(per_layer, temperature)
        @test compute_loss(posteriors, expected, dual, temperature, 0, LOSS_LAYERS_SOFTMIN) ==
              compute_loss(posteriors, expected, dual, temperature, 0)
    end
    # A cold softmin commits to the correct layer; mean cannot. This is the
    # anti-alignment with first-to-clear testing, stated as a test.
    @test compute_loss(posteriors, expected, dual, 1.0f-3, 0, LOSS_LAYERS_SOFTMIN) < 1e-2
    @test compute_loss(posteriors, expected, dual, 1.0f-3, 0, LOSS_LAYERS_MEAN) > 0.9f0

    # One scored layer: all three modes coincide, since there is nothing to choose.
    single::Vector{Float32} = Float32[0.37]
    for mode in (LOSS_LAYERS_SOFTMIN, LOSS_LAYERS_LAST, LOSS_LAYERS_MEAN)
        @test combine_layer_losses(single, 0.5f0, mode) == 0.37f0
    end
end

@testset "Enzyme differentiates the loss under every layer selection" begin
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    initial_llrs::Vector{Float32} = fill(Float32(log(9)), 4)
    base::NeuralBPBase = NeuralBPBase(parity_check_matrix, parity_check_matrix_dual, initial_llrs, 3)
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.1f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.1f0),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.1f0),
    )
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 1; 0 0])
    syndromes::BitMatrix = BitMatrix(mod.(parity_check_matrix * Matrix{Int}(expected), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 2)

    for mode in (LOSS_LAYERS_SOFTMIN, LOSS_LAYERS_LAST, LOSS_LAYERS_MEAN, LOSS_LAYERS_RAMP)
        grad_w_c2v_v2c::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_v2c))
        grad_w_llrs::Vector{Float32} = zeros(Float32, length(bpnn.weights_llrs))
        grad_w_readout::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_readout))
        grad_alpha::Vector{Float32} = zeros(Float32, 1)
        grad_schedule::Vector{Float32} = zeros(Float32, 2)
        (_, loss_value) = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            CorrelatedBPDecoderWithCER.get_loss_value,
            Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
            Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
            Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
            Enzyme.Duplicated(bpnn.coupling_logit, grad_alpha),
            Enzyme.Duplicated(bpnn.coupling_schedule, grad_schedule),
            Enzyme.Const(1.0f0),         # loss_layer_temperature
            Enzyme.Const(0),             # warmup_loss_layers
            Enzyme.Const(mode),          # loss_layer_selection
            Enzyme.Const(3.0f0),         # loss_layer_ramp_sharpness
            Enzyme.Const(BASE_LOSS_SIN_RESIDUE),            # base_loss_selection
            Enzyme.Const(base),
            Enzyme.Const(llrs_batch),
            Enzyme.Const(syndromes),
            Enzyme.Const(expected)
        )
        @test isfinite(loss_value)
        @test all(isfinite, grad_w_c2v_v2c)
        @test all(isfinite, grad_w_llrs)
        @test all(isfinite, grad_w_readout)
        # Untrainable-by-accident is the failure this guards against.
        @test any(!iszero, grad_w_c2v_v2c)
        @test any(!iszero, grad_w_llrs)
        # `last`, `mean` and `ramp` are non-negative; only softmin can sit below
        # zero (it lies T*log(n) under min(L)).
        if mode != LOSS_LAYERS_SOFTMIN
            @test loss_value >= 0.0f0
        end
    end
end

@testset "loss_layer_selection: ramp" begin
    # ---- the weights themselves --------------------------------------------
    # Exactly 1 at the last scored layer, strictly increasing, and -> 0 at the
    # start -- for every sharpness.
    for sharpness in (0.1f0, 1.0f0, 3.0f0, 10.0f0)
        for n_scored in (1, 2, 7, 85)
            weights::Vector{Float32} = [ramp_layer_weight(i, n_scored, sharpness) for i in 1:n_scored]
            @test weights[end] == 1.0f0
            @test all(weights .> 0.0f0)
            @test issorted(weights)
            if n_scored > 1
                @test weights[1] < weights[end]
            end
        end
    end
    @test ramp_layer_weight(1, 1, 3.0f0) == 1.0f0
    @test_throws ArgumentError ramp_layer_weight(1, 0, 3.0f0)
    @test_throws ArgumentError ramp_layer_weight(1, 5, 0.0f0)
    @test_throws ArgumentError ramp_layer_weight(1, 5, -1.0f0)

    # The two limits. Small k: tanh(k u)/tanh(k) -> u, a linear ramp. Large k:
    # tanh saturates, so every layer past the first few weighs ~1 -- a step to
    # uniform-after-warmup.
    n_scored::Int = 85
    linear_like::Vector{Float32} = [ramp_layer_weight(i, n_scored, 0.01f0) for i in 1:n_scored]
    for i in (1, 20, 42, 85)
        @test isapprox(linear_like[i], Float32(i) / Float32(n_scored); atol = 1e-3)
    end
    step_like::Vector{Float32} = [ramp_layer_weight(i, n_scored, 50.0f0) for i in 1:n_scored]
    @test step_like[10] > 0.99f0
    @test step_like[1] < 0.6f0
    # The default at the production geometry: 90 layers, warmup 5 -> 85 scored.
    # w crosses 0.5 at scored index 16 (layer 21) and 0.9 at scored index 42
    # (layer 47); w[41] = 0.8996 sits just under 0.9, which is the sharp edge
    # this pins down.
    production::Vector{Float32} = [ramp_layer_weight(i, 85, DEFAULT_LOSS_LAYER_RAMP_SHARPNESS) for i in 1:85]
    @test production[16] >= 0.5f0 && production[15] < 0.5f0
    @test production[42] >= 0.9f0 && production[41] < 0.9f0

    # ---- the reduction -----------------------------------------------------
    # Hand-computed on three layers: weights for k = 3 over n = 3 are
    # tanh(1)/tanh(3), tanh(2)/tanh(3), 1.
    per_layer::Vector{Float32} = Float32[2.0, 1.0, 0.5]
    w1::Float32 = Float32(tanh(1.0) / tanh(3.0))
    w2::Float32 = Float32(tanh(2.0) / tanh(3.0))
    by_hand::Float32 = (w1 * 2.0f0 + w2 * 1.0f0 + 1.0f0 * 0.5f0) / (w1 + w2 + 1.0f0)
    @test isapprox(ramp_loss(per_layer, 3.0f0), by_hand; atol = 1e-6)
    @test isapprox(combine_layer_losses(per_layer, 1.0f0, LOSS_LAYERS_RAMP, 3.0f0), by_hand; atol = 1e-6)
    # It is a weighted MEAN, so it sits between min and max and is on the scale of
    # one layer's loss -- unlike a weighted SUM, which would grow with n_layers.
    @test minimum(per_layer) <= ramp_loss(per_layer, 3.0f0) <= maximum(per_layer)
    # Temperature is ignored by `ramp`.
    @test combine_layer_losses(per_layer, 0.01f0, LOSS_LAYERS_RAMP, 3.0f0) ==
          combine_layer_losses(per_layer, 100.0f0, LOSS_LAYERS_RAMP, 3.0f0)
    # One scored layer: ramp coincides with the others.
    @test combine_layer_losses(Float32[0.37], 0.5f0, LOSS_LAYERS_RAMP, 3.0f0) == 0.37f0
    # Names round-trip.
    @test loss_layer_selection_code("ramp") == LOSS_LAYERS_RAMP
    @test loss_layer_selection_code(" RAMP ") == LOSS_LAYERS_RAMP
    @test loss_layer_selection_name(LOSS_LAYERS_RAMP) == "ramp"

    # ---- what the ramp is FOR ------------------------------------------------
    # A profile shaped like the measured ones: a huge transient, a plateau at a
    # floor, then a rise at the end (the drift that costs cleared decodes).
    # `last` sees only the rise; `mean` is swamped by the transient; `ramp`
    # with the transient excluded by warmup sees the plateau AND the rise, and
    # weights the rise most.
    profile::Vector{Float32} = vcat(Float32[35.0, 7.5, 1.1, 0.24, 0.12],       # transient, layers 1-5
                                    fill(1.0f-5, 40),                           # plateau, layers 6-45
                                    Float32[0.05, 0.10, 0.20, 0.30, 0.40])      # the rise, layers 46-50
    scored::Vector{Float32} = profile[6:end]                                    # warmup 5
    ramp_value::Float32 = combine_layer_losses(scored, 1.0f0, LOSS_LAYERS_RAMP, 3.0f0)
    flat_value::Float32 = combine_layer_losses(profile, 1.0f0, LOSS_LAYERS_MEAN)
    last_value::Float32 = combine_layer_losses(profile, 1.0f0, LOSS_LAYERS_LAST)
    # The ramp's number is dominated by the rise, not the plateau...
    @test ramp_value > 1.0f-3
    # ...but is a mean, so it is well below the last layer alone...
    @test ramp_value < last_value
    # ...and nowhere near the transient-dominated flat mean.
    @test flat_value > 0.8f0
    @test ramp_value < 0.1f0 * flat_value
    # Remove the rise and the ramp drops to the floor: it is the rise it scores.
    no_rise::Vector{Float32} = vcat(fill(1.0f-5, 45))
    @test combine_layer_losses(no_rise, 1.0f0, LOSS_LAYERS_RAMP, 3.0f0) < 2.0f-5
    # And through compute_loss with warmup = 2 on the three-layer posteriors from
    # the earlier testset, `ramp` of a single scored layer is that layer.
    dual::BitMatrix = two_check_dual()
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 0; 0 0])
    posteriors::Array{Float32, 3} = zeros(Float32, 4, 2, 3)
    posteriors[:, :, 1] .= Float32[30 30; 30 30; 30 30; 30 30]
    posteriors[:, :, 2] .= 0.0f0
    posteriors[:, :, 3] .= Float32[-30 30; 30 30; 30 30; 30 30]
    @test compute_loss(posteriors, expected, dual, 1.0f0, 2, LOSS_LAYERS_RAMP, 3.0f0) ==
          compute_loss(posteriors, expected, dual, 1.0f0, 2, LOSS_LAYERS_LAST)
    # With all three scored the wrong first layer is still felt (unlike `last`)
    # but far less than under `mean`.
    ramp_all::Float32 = compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_RAMP, 3.0f0)
    mean_all::Float32 = compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_MEAN)
    last_all::Float32 = compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_LAST)
    @test last_all < ramp_all < mean_all
end

@testset "base_loss: sin_residue vs smooth_loss" begin
    # ---- names and values --------------------------------------------------
    @test base_loss_code("sin_residue") == BASE_LOSS_SIN_RESIDUE
    @test base_loss_code(" Smooth_Loss ") == BASE_LOSS_SMOOTH
    @test base_loss_name(BASE_LOSS_SIN_RESIDUE) == "sin_residue"
    @test base_loss_name(BASE_LOSS_SMOOTH) == "smooth_loss"
    @test_throws ArgumentError base_loss_code("sine")
    @test_throws ArgumentError base_loss_name(7)
    # Both residues are zero at even integers and one at odd integers; they
    # differ in between, and that is where the gradient at the peak comes from.
    for even_value in (0.0f0, 2.0f0, 4.0f0)
        @test smooth_loss(even_value) == 0.0f0
        @test sine_residue_loss(even_value) < 1.0f-6
    end
    for odd_value in (1.0f0, 3.0f0)
        @test smooth_loss(odd_value) == 1.0f0
        @test isapprox(sine_residue_loss(odd_value), 1.0f0; atol = 1e-6)
    end
    @test smooth_loss(0.5f0) == 0.25f0
    @test smooth_loss(1.5f0) == 0.25f0
    @test isapprox(sine_residue_loss(0.5f0), Float32(sqrt(0.5)); atol = 1e-6)

    # ---- the batch loss under each, on the standard three-layer posteriors ----
    dual::BitMatrix = two_check_dual()
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 0; 0 0])
    posteriors::Array{Float32, 3} = zeros(Float32, 4, 2, 3)
    posteriors[:, :, 1] .= Float32[30 30; 30 30; 30 30; 30 30]     # confidently wrong
    posteriors[:, :, 2] .= 0.0f0                                    # undecided
    posteriors[:, :, 3] .= Float32[-30 30; 30 30; 30 30; 30 30]     # correct
    # The default is the sine, so every earlier call is reproduced exactly.
    @test compute_smooth_loss_from_llrs(posteriors[:, :, 1], expected, dual) ==
          compute_smooth_loss_from_llrs(posteriors[:, :, 1], expected, dual, BASE_LOSS_SIN_RESIDUE)
    @test base_loss_per_layer(posteriors, expected, dual, 0) ==
          base_loss_per_layer(posteriors, expected, dual, 0, BASE_LOSS_SIN_RESIDUE)
    @test compute_loss(posteriors, expected, dual, 1.0f0, 0) ==
          compute_loss(posteriors, expected, dual, 1.0f0, 0, LOSS_LAYERS_SOFTMIN,
                       DEFAULT_LOSS_LAYER_RAMP_SHARPNESS, BASE_LOSS_SIN_RESIDUE)
    @test_throws ArgumentError compute_smooth_loss_from_llrs(posteriors[:, :, 1], expected, dual, 7)
    for residue in (BASE_LOSS_SIN_RESIDUE, BASE_LOSS_SMOOTH)
        per_layer::Vector{Float32} = base_loss_per_layer(posteriors, expected, dual, 0, residue)
        @test per_layer[3] < 1.0f-6          # the correct layer clears both residues
        @test per_layer[1] > 0.9f0           # the wrong one is near its peak under both
        @test per_layer[2] > per_layer[3]
    end
    # At saturation (|mu| = 30) the two agree exactly: every x is an integer.
    @test isapprox(base_loss_per_layer(posteriors, expected, dual, 0, BASE_LOSS_SIN_RESIDUE)[1],
                   base_loss_per_layer(posteriors, expected, dual, 0, BASE_LOSS_SMOOTH)[1]; atol = 1e-5)

    # ---- the gradient at a stuck check: the reason smooth_loss exists ----------
    # One sample, bit 1 in error, every LLR at +8 ("no error", moderately
    # confident; this codebase's sigmoid is 1/(1+exp(x)), so +8 -> sigma = 3.4e-4).
    # Rows 1 and 3 of [H; L] are then VIOLATED at x = 1 + 2*sigma(8), row 2 is
    # SATISFIED at x = 2*sigma(8). Bit 1 sits only in the violated rows; bit 4
    # only in the satisfied one. Measured on the real runs: informative batches
    # plateau at exactly this configuration (x within 1e-3 of 1) from layer ~40.
    stuck_llrs::Matrix{Float32} = fill(8.0f0, 4, 1)
    stuck_expected::BitMatrix = BitMatrix([1; 0; 0; 0;;])
    gradients::Dict{Int, Vector{Float32}} = Dict{Int, Vector{Float32}}()
    for residue in (BASE_LOSS_SIN_RESIDUE, BASE_LOSS_SMOOTH)
        grad_llrs::Matrix{Float32} = zeros(Float32, 4, 1)
        (_, residue_value) = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            compute_smooth_loss_from_llrs,
            Enzyme.Duplicated(stuck_llrs, grad_llrs),
            Enzyme.Const(stuck_expected),
            Enzyme.Const(dual),
            Enzyme.Const(residue)
        )
        @test isfinite(residue_value)
        @test isapprox(residue_value, 2.0f0; atol = 1e-2)     # two violated rows
        @test all(isfinite, grad_llrs)
        gradients[residue] = vec(grad_llrs)
    end
    sin_gradient::Vector{Float32} = gradients[BASE_LOSS_SIN_RESIDUE]
    smooth_gradient::Vector{Float32} = gradients[BASE_LOSS_SMOOTH]
    # Bit 1, only in VIOLATED checks: the sine's (pi/2)cos(pi x/2) vanishes at
    # x = 1, the quadratic's subgradient is 2. Ratio ~1200 analytically.
    @test abs(smooth_gradient[1]) > 100.0f0 * abs(sin_gradient[1])
    @test abs(smooth_gradient[1]) > 1.0f-4
    # Bit 4, only in the SATISFIED check: the sine has a cusp of slope pi/2 at
    # x = 0 and the quadratic is flat there. The sine spends its gradient driving
    # an already-correct bit to saturate harder. Ratio ~1200 the other way.
    @test abs(sin_gradient[4]) > 100.0f0 * abs(smooth_gradient[4])
    @test abs(sin_gradient[4]) > 1.0f-4
    # Both push a violated check away from x = 1 in the same direction: toward
    # flipping bit 1 (lower its LLR so sigma rises and x climbs to 2).
    @test sign(smooth_gradient[1]) == sign(sin_gradient[1]) || sin_gradient[1] == 0.0f0
end

@testset "Enzyme differentiates the full training loss under smooth_loss" begin
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    initial_llrs::Vector{Float32} = fill(Float32(log(9)), 4)
    base::NeuralBPBase = NeuralBPBase(parity_check_matrix, parity_check_matrix_dual, initial_llrs, 3)
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.1f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.1f0),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.1f0),
    )
    expected::BitMatrix = BitMatrix([1 0; 0 0; 0 1; 0 0])
    syndromes::BitMatrix = BitMatrix(mod.(parity_check_matrix * Matrix{Int}(expected), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 2)
    for mode in (LOSS_LAYERS_SOFTMIN, LOSS_LAYERS_LAST, LOSS_LAYERS_RAMP)
        grad_w_c2v_v2c::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_v2c))
        grad_w_llrs::Vector{Float32} = zeros(Float32, length(bpnn.weights_llrs))
        grad_w_readout::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_readout))
        grad_alpha::Vector{Float32} = zeros(Float32, 1)
        grad_schedule::Vector{Float32} = zeros(Float32, 2)
        (_, loss_value) = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            CorrelatedBPDecoderWithCER.get_loss_value,
            Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
            Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
            Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
            Enzyme.Duplicated(bpnn.coupling_logit, grad_alpha),
            Enzyme.Duplicated(bpnn.coupling_schedule, grad_schedule),
            Enzyme.Const(1.0f0),                # loss_layer_temperature
            Enzyme.Const(0),                    # warmup_loss_layers
            Enzyme.Const(mode),                 # loss_layer_selection
            Enzyme.Const(3.0f0),                # loss_layer_ramp_sharpness
            Enzyme.Const(BASE_LOSS_SMOOTH),     # base_loss_selection
            Enzyme.Const(base),
            Enzyme.Const(llrs_batch),
            Enzyme.Const(syndromes),
            Enzyme.Const(expected)
        )
        @test isfinite(loss_value)
        @test all(isfinite, grad_w_c2v_v2c)
        @test all(isfinite, grad_w_llrs)
        @test all(isfinite, grad_w_readout)
        @test any(!iszero, grad_w_c2v_v2c)
        @test any(!iszero, grad_w_llrs)
    end
end

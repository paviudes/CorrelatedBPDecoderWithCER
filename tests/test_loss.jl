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

    for mode in (LOSS_LAYERS_SOFTMIN, LOSS_LAYERS_LAST, LOSS_LAYERS_MEAN)
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
        # `last` and `mean` are non-negative; only softmin can sit below zero
        # (it lies T*log(n) under min(L)).
        if mode != LOSS_LAYERS_SOFTMIN
            @test loss_value >= 0.0f0
        end
    end
end

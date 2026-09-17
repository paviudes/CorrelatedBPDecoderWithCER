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
    @test softmin_loss(losses, 1.0f-3) <= softmin_loss(losses, 1.0f0)  # colder is lower ...
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
    parity_check_matrix::Matrix{Int} = [1 1 0 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 0 0; 0 0 1 1; 1 0 1 0]
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

    grad_w_c2v_v2c::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_v2c))
    grad_w_llrs::Vector{Float32} = zeros(Float32, length(bpnn.weights_llrs))
    grad_w_readout::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_readout))
    grad_alpha::Vector{Float32} = zeros(Float32, 1)
    (_, loss_value) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        CorrelatedBPDecoderWithCER.get_loss_value,
        Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
        Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
        Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
        Enzyme.Duplicated(bpnn.coupling_scale, grad_alpha),
        Enzyme.Const(1.0f0),         # loss_layer_temperature
        Enzyme.Const(0),             # warmup_loss_layers
        Enzyme.Const(base),
        Enzyme.Const(llrs_batch),
        Enzyme.Const(syndromes),
        Enzyme.Const(expected)
    )
    @test isfinite(loss_value)
    @test loss_value > 0.0f0
    @test all(isfinite, grad_w_c2v_v2c)
    @test all(isfinite, grad_w_llrs)
    @test all(isfinite, grad_w_readout)
    # A gradient of exactly zero everywhere would mean the loss was detached
    # from the weights entirely -- silently untrainable.
    @test any(!iszero, grad_w_c2v_v2c)
    @test any(!iszero, grad_w_llrs)
    # The standard check node never reads alpha, so its gradient is exactly 0.
    @test grad_alpha[1] == 0.0f0
end

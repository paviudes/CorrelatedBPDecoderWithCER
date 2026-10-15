using CorrelatedBPDecoderWithCER
using Test
using Random
using LinearAlgebra
using Enzyme

# =============================================================================
# Tests for src/base_loss.jl: the residue options, their derivatives, and the
# batch base loss with its gradient with respect to the posterior LLRs. Enzyme
# is the reference for every derivative.
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_base_loss.jl")'
# =============================================================================

function enzyme_scalar_derivative(scalar_function::Function, x::Float32)::Float32
    """
    d f / d x at x, by Enzyme in reverse mode, for a Float32 -> Float32 function.
    """
    derivative::Float32 = Enzyme.autodiff(Enzyme.Reverse, scalar_function, Enzyme.Active, Enzyme.Active(x))[1][1]
    return derivative
end

function base_loss_test_dual()::BitMatrix
    """
    Four bits, two stabilizer rows and one logical row: [H; L].
    """
    dual::BitMatrix = BitMatrix([1 1 1 0; 0 0 1 1; 1 0 1 0])
    return dual
end

# Points away from the residues' kinks (the even integers for the sine, the odd
# integers for the quadratic), on both sides of zero and past one period.
const BASE_LOSS_TEST_POINTS::Vector{Float32} = Float32[-1.7, -0.6, -0.25, 0.1, 0.37, 0.8, 0.999, 1.001, 1.3, 1.75, 2.2, 2.9, 3.6]

@testset "residue derivatives: closed forms" begin
    # smooth_loss: 2 × the signed distance to the nearest even integer.
    @test smooth_loss_derivative(0.25f0) == 0.5f0
    @test smooth_loss_derivative(1.5f0) == -1.0f0
    @test smooth_loss_derivative(2.25f0) == 0.5f0          # period 2
    @test smooth_loss_derivative(-0.5f0) == -1.0f0         # -0.5 mod 2 = 1.5
    @test smooth_loss_derivative(1.0f0) == 2.0f0           # the kink takes the left branch, as smooth_loss does
    # sine: zero at the peak, ±π/2 slope at the cusp, odd about zero.
    @test isapprox(sine_residue_derivative(1.0f0), 0.0f0; atol = 1e-6)
    @test isapprox(sine_residue_derivative(0.5f0), Float32(pi / 2 * cos(pi / 4)); rtol = 1e-5)
    @test isapprox(sine_residue_derivative(-0.5f0), -Float32(pi / 2 * cos(pi / 4)); rtol = 1e-5)
    @test isapprox(sine_residue_derivative(1.5f0), -Float32(pi / 2 * cos(pi / 4)); rtol = 1e-5)
    # This codebase's sigmoid is 1/(1 + e^x): decreasing, slope -1/4 at 0.
    @test sigmoid_derivative(0.0f0) == -0.25f0
    @test sigmoid_derivative(3.0f0) < 0.0f0
    @test sigmoid_derivative(-3.0f0) == sigmoid_derivative(3.0f0)
    # Never NaN, where the e^x/(1 + e^x)^2 form is Inf/Inf.
    @test isfinite(sigmoid_derivative(100.0f0))
    @test isfinite(sigmoid_derivative(-100.0f0))
end

@testset "residue derivatives equal Enzyme's" begin
    for x in BASE_LOSS_TEST_POINTS
        @test isapprox(smooth_loss_derivative(x), enzyme_scalar_derivative(smooth_loss, x); rtol = 1e-5, atol = 1e-6)
        @test isapprox(sine_residue_derivative(x), enzyme_scalar_derivative(sine_residue_loss, x); rtol = 1e-5, atol = 1e-6)
    end
    # The sigmoid, out to where the e^x/(1 + e^x)^2 form Enzyme evaluates still
    # holds (|x| < 44); the product form is accurate on the whole range.
    for x in Float32[-40, -15, -10, -2, -0.3, 0, 0.3, 2, 10, 15, 40]
        @test isapprox(sigmoid_derivative(x), enzyme_scalar_derivative(sigmoid, x); rtol = 1e-5, atol = 1e-30)
    end
end

@testset "compute_base_loss is the computation it replaced" begin
    Random.seed!(21)
    dual::BitMatrix = base_loss_test_dual()
    posterior::Matrix{Float32} = 3.0f0 .* randn(Float32, 4, 6)
    expected::BitMatrix = bitrand(4, 6)
    for residue in (BASE_LOSS_SIN_RESIDUE, BASE_LOSS_SMOOTH)
        @test compute_base_loss(posterior, expected, dual, residue) ==
              compute_smooth_loss_from_llrs(posterior, expected, dual, residue)
    end
    # The default is the sine, as before.
    @test compute_base_loss(posterior, expected, dual) == compute_base_loss(posterior, expected, dual, BASE_LOSS_SIN_RESIDUE)
    @test_throws ArgumentError compute_base_loss(posterior, expected, dual, 7)
    @test_throws ArgumentError compute_base_loss_gradient(posterior, expected, dual, 7)
end

@testset "compute_base_loss_gradient equals Enzyme's gradient of compute_base_loss" begin
    Random.seed!(22)
    # The toy [H; L] and a random 18 × 40 one, so every LLR sits in several rows.
    duals::Vector{BitMatrix} = [base_loss_test_dual(), BitMatrix(rand(18, 40) .< 0.2)]
    for dual in duals
        n_bits::Int = size(dual, 2)
        n_samples::Int = 7
        # Moderate LLRs, both signs: the region where the gradient is not ~0.
        posterior::Matrix{Float32} = 3.0f0 .* randn(Float32, n_bits, n_samples)
        expected::BitMatrix = BitMatrix(rand(n_bits, n_samples) .< 0.15)
        for residue in (BASE_LOSS_SIN_RESIDUE, BASE_LOSS_SMOOTH)
            enzyme_gradient::Matrix{Float32} = zeros(Float32, n_bits, n_samples)
            (_, enzyme_loss) = Enzyme.autodiff(
                Enzyme.ReverseWithPrimal,
                compute_base_loss,
                Enzyme.Duplicated(posterior, enzyme_gradient),
                Enzyme.Const(expected),
                Enzyme.Const(dual),
                Enzyme.Const(residue)
            )
            manual_gradient::Matrix{Float32} = compute_base_loss_gradient(posterior, expected, dual, residue)
            # Enzyme's primal is its own compiled copy of the sum: a few ulp apart
            # is expected (see check_gradients_agree in test_gradient.jl).
            @test isapprox(enzyme_loss, compute_base_loss(posterior, expected, dual, residue); rtol = 1e-5)
            @test size(manual_gradient) == size(posterior)
            @test all(isfinite, manual_gradient)
            @test any(!iszero, manual_gradient)
            @test isapprox(manual_gradient, enzyme_gradient; rtol = 1e-4)
            largest::Float32 = maximum(abs, enzyme_gradient)
            @test all(abs.(manual_gradient .- enzyme_gradient) .<= 1f-4 * largest)
        end
    end
end

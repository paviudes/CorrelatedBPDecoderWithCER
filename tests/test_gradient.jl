using CorrelatedBPDecoderWithCER
using Test
using Random
using LinearAlgebra
using Enzyme

# =============================================================================
# Tests for src/gradient.jl and src/gradient_gpu.jl: the hand-written gradient of
# the training loss against Enzyme's, which stays the oracle, for the standard
# (tanh) check node. Every layer rule × residue × warmup (and two softmin
# temperatures), on the toy graph and on the [[72,12,6]] BB code, in a moderate
# regime and in the saturating one of the real CER priors. Then the GPU gradient
# against the CPU one (on Metal on a Mac, whether or not USE_GPU is set), and
# training with `manual_backprop = true` against training with Enzyme.
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_gradient.jl")'
# =============================================================================

function gradient_test_two_check_base(n_layers::Int)::NeuralBPBase
    """
    Four bits, two checks: check 1 = {1,2,3}, check 2 = {3,4}; bit 3 is in both,
    so the variable-node rule has weights.
    """
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    initial_llrs::Vector{Float32} = fill(Float32(log(9)), 4)
    base::NeuralBPBase = NeuralBPBase(parity_check_matrix, parity_check_matrix_dual, initial_llrs, n_layers)
    return base
end

function gradient_test_bb_base(n_layers::Int, check_node::String)::NeuralBPBase
    """
    The [[72,12,6]] BB code (Z checks) with its CER priors (LLRs ≈ 5.4).
    """
    data_directory::String = "./../data/72q_BB_cycles_1_soft_constraints"
    base::NeuralBPBase = load_base_BP_model(
        "$(data_directory)/code/HZ.txt",
        "$(data_directory)/code/LZ.txt",
        n_layers;
        cer_data_file = "$(data_directory)/correlated_weights/correlated_weights_p_0.0005_sig_0.0_s_1.txt",
        use_cer = true,
        check_node = check_node,
    )
    return base
end

function gradient_test_model(base::NeuralBPBase, scale::Float32)::NachmaniNeuralBP
    """
    Random weights around one, as training initialises them.
    """
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = scale),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = scale),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = scale),
    )
    return bpnn
end

function gradient_test_batch(base::NeuralBPBase, n_samples::Int, error_weight::Int)::Tuple{BitMatrix, BitMatrix}
    """
    `n_samples` random errors of exactly `error_weight` bits, and their syndromes.
    """
    expected::BitMatrix = falses(base.code_n_bits, n_samples)
    for j in 1:n_samples
        expected[randperm(base.code_n_bits)[1:error_weight], j] .= true
    end
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(base.parity_check_matrix) * Matrix{Int}(expected), 2) .== 1)
    batch::Tuple{BitMatrix, BitMatrix} = (expected, syndromes)
    return batch
end

function enzyme_loss_and_gradients(
    bpnn::NachmaniNeuralBP,
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int,
    loss_layer_selection::Int,
    loss_layer_ramp_sharpness::Float32,
    base_loss_selection::Int,
    llrs_batch::Matrix{Float32},
    syndromes_batch::BitMatrix,
    expected_recoveries::BitMatrix
)::LossAndGradients
    """
    The oracle: exactly the call `train_neuralbp_enzyme!` makes.
    """
    gradient_weights_c2v_v2c::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_v2c))
    gradient_weights_llrs::Vector{Float32} = zeros(Float32, length(bpnn.weights_llrs))
    gradient_weights_c2v_readout::Vector{Float32} = zeros(Float32, length(bpnn.weights_c2v_readout))
    gradient_coupling_logit::Vector{Float32} = zeros(Float32, length(bpnn.coupling_logit))
    (_, loss_value) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        CorrelatedBPDecoderWithCER.get_loss_value,
        Enzyme.Duplicated(bpnn.weights_c2v_v2c, gradient_weights_c2v_v2c),
        Enzyme.Duplicated(bpnn.weights_llrs, gradient_weights_llrs),
        Enzyme.Duplicated(bpnn.weights_c2v_readout, gradient_weights_c2v_readout),
        Enzyme.Duplicated(bpnn.coupling_logit, gradient_coupling_logit),
        Enzyme.Const(loss_layer_temperature),
        Enzyme.Const(warmup_loss_layers),
        Enzyme.Const(loss_layer_selection),
        Enzyme.Const(loss_layer_ramp_sharpness),
        Enzyme.Const(base_loss_selection),
        Enzyme.Const(bpnn.base),
        Enzyme.Const(llrs_batch),
        Enzyme.Const(syndromes_batch),
        Enzyme.Const(expected_recoveries)
    )
    result::LossAndGradients = LossAndGradients(
        Float32(loss_value),
        gradient_weights_c2v_v2c,
        gradient_weights_llrs,
        gradient_weights_c2v_readout,
        gradient_coupling_logit,
    )
    return result
end

function manual_loss_and_gradients(
    bpnn::NachmaniNeuralBP,
    loss_layer_temperature::Float32,
    warmup_loss_layers::Int,
    loss_layer_selection::Int,
    loss_layer_ramp_sharpness::Float32,
    base_loss_selection::Int,
    llrs_batch::Matrix{Float32},
    syndromes_batch::BitMatrix,
    expected_recoveries::BitMatrix
)::LossAndGradients
    """
    The hand-written gradient, called with `get_loss_value`'s arguments.
    """
    result::LossAndGradients = compute_loss_and_gradients(
        bpnn.weights_c2v_v2c,
        bpnn.weights_llrs,
        bpnn.weights_c2v_readout,
        bpnn.coupling_logit,
        loss_layer_temperature,
        warmup_loss_layers,
        loss_layer_selection,
        loss_layer_ramp_sharpness,
        base_loss_selection,
        bpnn.base,
        llrs_batch,
        syndromes_batch,
        expected_recoveries,
    )
    return result
end

function relative_difference(manual::Vector{Float32}, reference::Vector{Float32})::Float64
    """
    ‖manual - reference‖ / max(‖manual‖, ‖reference‖), in Float64.
    """
    manual_64::Vector{Float64} = Float64.(manual)
    reference_64::Vector{Float64} = Float64.(reference)
    scale::Float64 = max(norm(manual_64), norm(reference_64), 1e-30)
    difference::Float64 = norm(manual_64 .- reference_64) / scale
    return difference
end

function check_gradients_agree(
    manual::LossAndGradients,
    reference::LossAndGradients,
    norm_tolerance::Float64,
    component_tolerance::Float64
)::Float64
    """
    The loss, and each of the three trained parameter vectors, agree: in norm to
    `norm_tolerance` (relative), and every component to `component_tolerance` ×
    the largest component of Enzyme's. α's gradient is exactly zero on both
    sides (the tanh node never reads it). Returns the largest relative norm
    difference, for the summary line.

    The loss against ENZYME'S PRIMAL is compared to 1e-5, not exactly: Enzyme
    returns the value of its own compiled copy of `get_loss_value`, whose @simd
    sums over ~400 residues per layer may associate differently. Measured
    2026-10-11 on the saturating BB batch: 0.035526216 vs 0.035526276, 1.7e-6
    relative (~14 ulp). The exact check is against a plain `get_loss_value` call,
    in `compare_every_mode`.
    """
    @test isapprox(manual.loss, reference.loss; rtol = 1e-5)
    worst::Float64 = 0.0
    for (manual_vector, reference_vector) in (
        (manual.weights_c2v_v2c, reference.weights_c2v_v2c),
        (manual.weights_llrs, reference.weights_llrs),
        (manual.weights_c2v_readout, reference.weights_c2v_readout),
    )
        @test length(manual_vector) == length(reference_vector)
        @test all(isfinite, manual_vector)
        @test any(!iszero, manual_vector) == any(!iszero, reference_vector)
        difference::Float64 = relative_difference(manual_vector, reference_vector)
        @test difference <= norm_tolerance
        largest::Float32 = maximum(abs, reference_vector)
        @test all(abs.(manual_vector .- reference_vector) .<= Float32(component_tolerance) * largest)
        worst = max(worst, difference)
    end
    @test all(iszero, manual.coupling_logit)
    @test all(iszero, reference.coupling_logit)
    return worst
end

const GRADIENT_TEST_SELECTIONS::Vector{Int} = [LOSS_LAYERS_SOFTMIN, LOSS_LAYERS_LAST, LOSS_LAYERS_MEAN, LOSS_LAYERS_RAMP]
const GRADIENT_TEST_RESIDUES::Vector{Int} = [BASE_LOSS_SIN_RESIDUE, BASE_LOSS_SMOOTH]

function compare_every_mode(
    bpnn::NachmaniNeuralBP,
    llrs_batch::Matrix{Float32},
    syndromes_batch::BitMatrix,
    expected_recoveries::BitMatrix,
    warmups::Vector{Int},
    norm_tolerance::Float64,
    component_tolerance::Float64
)::Float64
    """
    Every layer rule × residue × warmup, and softmin at two temperatures (the
    softmax weights sharpen as T falls). Returns the largest relative difference.
    """
    worst::Float64 = 0.0
    for selection in GRADIENT_TEST_SELECTIONS, residue in GRADIENT_TEST_RESIDUES, warmup in warmups
        temperatures::Vector{Float32} = Float32[1.0]
        if selection == LOSS_LAYERS_SOFTMIN
            temperatures = Float32[1.0, 0.2]
        end
        for temperature in temperatures
            reference::LossAndGradients = enzyme_loss_and_gradients(
                bpnn, temperature, warmup, selection, 3.0f0, residue, llrs_batch, syndromes_batch, expected_recoveries)
            manual::LossAndGradients = manual_loss_and_gradients(
                bpnn, temperature, warmup, selection, 3.0f0, residue, llrs_batch, syndromes_batch, expected_recoveries)
            # Same functions on bit-identical posteriors: the loss must be EXACTLY
            # the plain (non-Enzyme) `get_loss_value`.
            plain_loss::Float32 = CorrelatedBPDecoderWithCER.get_loss_value(
                bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, bpnn.coupling_logit,
                temperature, warmup, selection, 3.0f0, residue, bpnn.base,
                llrs_batch, syndromes_batch, expected_recoveries)
            @test manual.loss == plain_loss
            worst = max(worst, check_gradients_agree(manual, reference, norm_tolerance, component_tolerance))
        end
    end
    return worst
end

@testset "forward_pass_with_tape reproduces forward_pass_with_weights bit for bit" begin
    Random.seed!(31)
    base::NeuralBPBase = gradient_test_bb_base(4, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 5
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    syndromes_matrix::Matrix{Bool} = Matrix{Bool}(syndromes)
    reference::Array{Float32, 3} = forward_pass_with_weights(bpnn, llrs_batch, syndromes_matrix)
    tape::ForwardTape = forward_pass_with_tape(
        bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, bpnn.coupling_logit,
        base, llrs_batch, syndromes_matrix)
    @test tape.posterior_llrs == reference
    @test length(tape.messages_v2c) == base.n_layers
    @test size(tape.activated_c2v_signs[1]) == (base.nb_neurons_per_layer, n_samples)
    @test all(iszero, tape.incoming_messages_c2v[1])
    for layer in 1:(base.n_layers - 1)
        @test tape.incoming_messages_c2v[layer + 1] === tape.outgoing_messages_c2v[layer]
    end
end

@testset "compute_loss_and_gradients equals Enzyme: two-check graph, every mode" begin
    Random.seed!(32)
    base::NeuralBPBase = gradient_test_two_check_base(3)
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    expected::BitMatrix = BitMatrix([1 0 0; 0 0 1; 0 1 0; 0 0 1])
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(base.parity_check_matrix) * Matrix{Int}(expected), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 3)
    worst::Float64 = compare_every_mode(bpnn, llrs_batch, syndromes, expected, [0, 1], 1e-4, 1e-4)
    println("  two-check graph: largest relative difference, manual vs Enzyme = $(worst)")
end

@testset "compute_loss_and_gradients equals Enzyme: BB code, moderate LLRs, every mode" begin
    Random.seed!(33)
    base::NeuralBPBase = gradient_test_bb_base(3, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 6
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = fill(Float32(log(9)), base.code_n_bits, n_samples)
    worst::Float64 = compare_every_mode(bpnn, llrs_batch, syndromes, expected, [0, 2], 1e-3, 1e-3)
    println("  BB code, LLR log 9: largest relative difference, manual vs Enzyme = $(worst)")
end

@testset "compute_loss_and_gradients equals Enzyme: BB code, CER priors (saturating)" begin
    # LLRs ≈ 5.4 saturate messages within a few layers, so the activation clamps
    # are active and the derivative factors are at their worst-conditioned. The
    # tolerance allows for Enzyme and the hand-written factors evaluating
    # 1/(1 - y²) and 1/cosh² in different orders there (a Float32 emulation put
    # the spread at ≤ 2e-4 for 3 layers).
    Random.seed!(34)
    base::NeuralBPBase = gradient_test_bb_base(3, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 6
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    worst::Float64 = compare_every_mode(bpnn, llrs_batch, syndromes, expected, [0], 5e-3, 1e-2)
    println("  BB code, CER priors: largest relative difference, manual vs Enzyme = $(worst)")
end

@testset "compute_loss_and_gradients refuses the enriched check node" begin
    Random.seed!(35)
    base::NeuralBPBase = gradient_test_bb_base(3, "enriched")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, 2, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 2)
    @test_throws ArgumentError manual_loss_and_gradients(
        bpnn, 1.0f0, 0, LOSS_LAYERS_SOFTMIN, 3.0f0, BASE_LOSS_SIN_RESIDUE, llrs_batch, syndromes, expected)
end

# -----------------------------------------------------------------------------
# The GPU gradient (src/gradient_gpu.jl) against the CPU one, which the
# testsets above hold to Enzyme.
# -----------------------------------------------------------------------------

function compare_gpu_with_cpu(
    bpnn::NachmaniNeuralBP,
    llrs_batch::Matrix{Float32},
    syndromes_batch::BitMatrix,
    expected_recoveries::BitMatrix,
    warmups::Vector{Int},
    loss_tolerance::Float64,
    norm_tolerance::Float64
)::Float64
    """
    `compute_loss_and_gradients_gpu` against `compute_loss_and_gradients` for every
    layer rule × residue × warmup at T = 1: the loss to `loss_tolerance` and each
    weight gradient to `norm_tolerance` in norm, both relative; θ's gradient is
    zero on both sides. The two forwards differ only by the device's Float32
    arithmetic. Returns the largest relative gradient difference.
    """
    worst::Float64 = 0.0
    for selection in GRADIENT_TEST_SELECTIONS, residue in GRADIENT_TEST_RESIDUES, warmup in warmups
        cpu::LossAndGradients = manual_loss_and_gradients(
            bpnn, 1.0f0, warmup, selection, 3.0f0, residue, llrs_batch, syndromes_batch, expected_recoveries)
        gpu::LossAndGradients = compute_loss_and_gradients_gpu(
            bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, bpnn.coupling_logit,
            1.0f0, warmup, selection, 3.0f0, residue, bpnn.base,
            llrs_batch, syndromes_batch, expected_recoveries)
        @test isapprox(gpu.loss, cpu.loss; rtol = loss_tolerance, atol = 1e-6)
        for (gpu_vector, cpu_vector) in (
            (gpu.weights_c2v_v2c, cpu.weights_c2v_v2c),
            (gpu.weights_llrs, cpu.weights_llrs),
            (gpu.weights_c2v_readout, cpu.weights_c2v_readout),
        )
            @test length(gpu_vector) == length(cpu_vector)
            @test all(isfinite, gpu_vector)
            difference::Float64 = relative_difference(gpu_vector, cpu_vector)
            @test difference <= norm_tolerance
            worst = max(worst, difference)
        end
        @test all(iszero, gpu.coupling_logit)
    end
    return worst
end

@testset "forward_pass_gpu_with_tape reproduces forward_pass_gpu" begin
    Random.seed!(38)
    base::NeuralBPBase = gradient_test_bb_base(4, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 5
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = fill(Float32(log(9)), base.code_n_bits, n_samples)
    reference::Array{Float32, 3} = forward_pass_gpu(bpnn, llrs_batch, syndromes)
    gpu_state::GPUState = build_gpu_state(bpnn, syndromes)
    initial_llrs_gpu::CorrelatedBPDecoderWithCER.ArrayT{Float32, 2} = CorrelatedBPDecoderWithCER.ArrayT{Float32, 2}(llrs_batch)
    tape::GPUForwardTape = forward_pass_gpu_with_tape(gpu_state, initial_llrs_gpu)
    # The same layer function on the same device, in the same order.
    @test tape.posterior_llrs == reference
    @test length(tape.messages_v2c) == base.n_layers
    @test all(iszero, Array(tape.incoming_messages_c2v[1]))
    for layer in 1:(base.n_layers - 1)
        @test tape.incoming_messages_c2v[layer + 1] === tape.outgoing_messages_c2v[layer]
    end
    release_gpu_forward_tape!(tape)
    release_gpu_state!(gpu_state)
end

@testset "compute_loss_and_gradients_gpu equals the CPU gradient: two-check graph" begin
    Random.seed!(40)
    base::NeuralBPBase = gradient_test_two_check_base(3)
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    expected::BitMatrix = BitMatrix([1 0 0; 0 0 1; 0 1 0; 0 0 1])
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(base.parity_check_matrix) * Matrix{Int}(expected), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 3)
    worst::Float64 = compare_gpu_with_cpu(bpnn, llrs_batch, syndromes, expected, [0, 1], 1e-4, 1e-3)
    println("  two-check graph: largest relative difference, GPU vs CPU = $(worst)")
end

@testset "compute_loss_and_gradients_gpu equals the CPU gradient: BB code, LLR log 9" begin
    Random.seed!(39)
    base::NeuralBPBase = gradient_test_bb_base(3, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 6
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = fill(Float32(log(9)), base.code_n_bits, n_samples)
    worst::Float64 = compare_gpu_with_cpu(bpnn, llrs_batch, syndromes, expected, [0, 2], 1e-4, 1e-3)
    println("  BB code, LLR log 9: largest relative difference, GPU vs CPU = $(worst)")
end

@testset "compute_loss_and_gradients_gpu equals the CPU gradient: BB code, CER priors (saturating)" begin
    # Unlike Enzyme against the CPU wrapper, the two sides here do not share
    # their forward values: the device rounds its own exp/log/tanh/atanh and
    # matmuls. Near saturation the tanh rule turns that last-digit difference
    # into a visible one — 2 atanh(exp(M)) with exp(M) a few ulp below 1 — so
    # the posteriors differ by up to ~1e-3 relative (the GPU forward test in
    # test_soft_constraints.jl allows exactly that), and the loss with them.
    # Measured 2026-10-13: losses 2–3e-4 apart, gradients 7.7e-4.
    Random.seed!(42)
    base::NeuralBPBase = gradient_test_bb_base(3, "tanh")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    n_samples::Int = 6
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, n_samples, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    worst::Float64 = compare_gpu_with_cpu(bpnn, llrs_batch, syndromes, expected, [0], 1e-3, 5e-3)
    println("  BB code, CER priors: largest relative difference, GPU vs CPU = $(worst)")
end

@testset "compute_loss_and_gradients_gpu refuses the enriched check node" begin
    Random.seed!(43)
    base::NeuralBPBase = gradient_test_bb_base(3, "enriched")
    bpnn::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, 2, 3)
    expected::BitMatrix = batch[1]
    syndromes::BitMatrix = batch[2]
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, 2)
    @test_throws ArgumentError compute_loss_and_gradients_gpu(
        bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, bpnn.coupling_logit,
        1.0f0, 0, LOSS_LAYERS_SOFTMIN, 3.0f0, BASE_LOSS_SIN_RESIDUE, base,
        llrs_batch, syndromes, expected)
end

# -----------------------------------------------------------------------------
# Training with the hand-written gradient (`manual_backprop = true` in
# `train_neuralbp_enzyme!`) against training with Enzyme.
# -----------------------------------------------------------------------------

function training_test_hyperparameters(manual_backprop::Bool)::Dict{String, Any}
    """
    The default hyperparameters, cut down to one epoch of two batches of two on
    the two-check graph. No weight decay, so the optimiser is Adam with
    eps = 1e-4, whose step g/(|g| + eps) is smooth in g: a gradient that differs
    by round-off moves the weights by round-off.
    """
    hyperparameters::Dict{String, Any} = parse_hyper_parameters("")
    hyperparameters["n_epochs"] = 1
    hyperparameters["batch_size"] = 2
    hyperparameters["warmup_layers"] = 0
    hyperparameters["learning_rate"] = 1.0f-2
    hyperparameters["weight_decay"] = 0.0f0
    hyperparameters["manual_backprop"] = manual_backprop
    return hyperparameters
end

@testset "train_neuralbp_enzyme! with manual_backprop follows Enzyme's updates" begin
    base::NeuralBPBase = gradient_test_two_check_base(3)
    expected::BitMatrix = BitMatrix([1 0 0 1; 0 0 1 0; 0 1 0 0; 0 0 1 1])
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(base.parity_check_matrix) * Matrix{Int}(expected), 2) .== 1)
    Random.seed!(44)
    initial::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)

    trained_models::Dict{String, NachmaniNeuralBP} = Dict{String, NachmaniNeuralBP}()
    for (label, manual_backprop, use_gpu_setting) in (
        ("Enzyme", false, "0"),
        ("hand-written, CPU", true, "0"),
        ("hand-written, GPU", true, "1"),
    )
        hyperparameters::Dict{String, Any} = training_test_hyperparameters(manual_backprop)
        # The same batch shuffles for every run.
        Random.seed!(45)
        trained::NachmaniNeuralBP = withenv("USE_GPU" => use_gpu_setting) do
            train_neuralbp_enzyme!(deepcopy(initial), syndromes, expected, hyperparameters; is_quiet = true)
        end
        trained_models[label] = trained
    end

    reference::NachmaniNeuralBP = trained_models["Enzyme"]
    # Two updates moved the weights, so agreeing after them means something.
    @test relative_difference(reference.weights_c2v_v2c, initial.weights_c2v_v2c) > 1e-4
    for label in ("hand-written, CPU", "hand-written, GPU")
        trained::NachmaniNeuralBP = trained_models[label]
        worst::Float64 = 0.0
        for (trained_vector, reference_vector) in (
            (trained.weights_c2v_v2c, reference.weights_c2v_v2c),
            (trained.weights_llrs, reference.weights_llrs),
            (trained.weights_c2v_readout, reference.weights_c2v_readout),
        )
            difference::Float64 = relative_difference(trained_vector, reference_vector)
            @test difference <= 1e-5
            worst = max(worst, difference)
        end
        # α is frozen under the tanh node, whichever gradient trains.
        @test trained.coupling_logit == initial.coupling_logit
        println("  training, $(label) vs Enzyme: largest relative weight difference after two updates = $(worst)")
    end
end

@testset "manual_backprop is validated before training" begin
    @test manual_backprop_setting(Dict{String, Any}(), CHECK_NODE_TANH) == false
    @test manual_backprop_setting(Dict{String, Any}("manual_backprop" => true), CHECK_NODE_TANH) == true
    @test_throws ArgumentError manual_backprop_setting(Dict{String, Any}("manual_backprop" => "yes"), CHECK_NODE_TANH)
    # The hand-written gradient does not cover the enriched check node: refused
    # before the first batch, not trained with Enzyme behind the user's back.
    Random.seed!(46)
    base::NeuralBPBase = gradient_test_bb_base(3, "enriched")
    @test_throws ArgumentError manual_backprop_setting(
        Dict{String, Any}("manual_backprop" => true), base.check_node_kind)
    model::NachmaniNeuralBP = gradient_test_model(base, 0.3f0)
    batch::Tuple{BitMatrix, BitMatrix} = gradient_test_batch(base, 2, 3)
    @test_throws ArgumentError train_neuralbp_enzyme!(
        model, batch[2], batch[1], training_test_hyperparameters(true); is_quiet = true)
end

using CorrelatedBPDecoderWithCER
using Test
using Enzyme
using JSON
using Random
using DelimitedFiles   # readdlm, for the LX file the coset-failure planting needs

# =============================================================================
# Tests for the enriched (correlation-adapted) check node, src/soft_constraints.jl.
#
# Run from the `tests/` directory:  julia --project="./../" -e 'include("test_soft_constraints.jl")'
#
# What is checked, in order:
#   1. The tables built from the BB code's HZ and its CER file: every one of the
#      540 pairs lands in exactly one of the 36 weight-6 checks, energies are
#      the right sums of J, slots point at the right neurons.
#   2. At alpha = 0 the enriched rule reproduces the standard tanh rule.
#   3. On a hand-built two-check example, prior + message equals the exact
#      brute-force posterior of the coupled prior, and the rows of the check
#      that carries no coupling are left untouched.
#   4. Enzyme differentiates the CPU kernel, with respect to alpha and to the
#      incoming messages, and agrees with central finite differences.
#   5. The dense (GPU) formulation equals the loop kernel on plain Arrays, and
#      the full GPU forward pass equals the CPU one on an enriched model.
#   6. The standard ("tanh") path is unchanged: it equals the legacy forward
#      pass, which knows nothing about the enriched node.
#   7. The weights file round-trips alpha and stays readable without it.
#   8-13. The layer schedule on alpha (alpha_t = alpha * d(t)); see the block
#      at the end of the file for the list.
# =============================================================================

function debug_data_directory()::String
    return "./../data/72q_BB_cycles_1_soft_constraints"
end

function debug_cer_file()::String
    data_dir = debug_data_directory()
    return "$(data_dir)/correlated_weights/correlated_weights_p_0.0005_sig_0.0_s_1.txt"
end

function load_bb_base(check_node::String, n_layers::Int)::NeuralBPBase
    """
    The [[72,12,6]] BB code with its CER priors and couplings.
    """
    base::NeuralBPBase = load_base_BP_model(
        "$(debug_data_directory())/code/HZ.txt",
        "$(debug_data_directory())/code/LZ.txt",
        n_layers;
        cer_data_file = debug_cer_file(),
        use_cer = true,
        check_node = check_node,
    )
    return base
end

function standard_check_messages(
    messages_v2c::Matrix{Float32},
    syndromes_batch::AbstractMatrix{Bool},
    base::NeuralBPBase
)::Matrix{Float32}
    """
    The standard tanh-rule check-to-variable messages for a matrix of raw
    variable-to-check messages, via the same `v2c_to_c2v!` the forward pass uses.
    """
    magnitudes_v2c::Matrix{Float32} = similar(messages_v2c)
    signs_v2c::Matrix{Bool} = falses(size(messages_v2c))
    safe_log_tanh_split!(magnitudes_v2c, signs_v2c, messages_v2c)
    messages_c2v::Matrix{Float32} = zeros(Float32, size(messages_v2c))
    magnitudes_c2v::Matrix{Float32} = similar(messages_v2c)
    signs_c2v::Matrix{Bool} = falses(size(messages_v2c))
    v2c_to_c2v!(messages_c2v, magnitudes_c2v, signs_c2v, magnitudes_v2c, signs_v2c, syndromes_batch, base)
    return messages_c2v
end

function two_check_base(check_node::String)::NeuralBPBase
    """
    Four bits, two checks. Check 1 = {1,2,3} carries three couplings; check 2 =
    {3,4} carries none, so only check 1 is enriched.
    """
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    error_probabilities::Vector{Float64} = [0.12, 0.05, 0.25, 0.08]
    initial_llrs::Vector{Float32} = Float32.(log.((1 .- error_probabilities) ./ error_probabilities))
    connectivity::Matrix{Int} = [1 2; 1 3; 2 3]
    correlation_strengths::Vector{Float32} = Float32[1.2, -0.7, 0.9]
    base::NeuralBPBase = NeuralBPBase(
        parity_check_matrix, parity_check_matrix_dual, initial_llrs, 1;
        connectivity = connectivity,
        correlation_strengths = correlation_strengths,
        check_node = check_node,
    )
    return base
end

function brute_force_posterior_llr(
    target_bit::Int,
    syndrome_bit::Int,
    log_q_zero::Vector{Float64},
    log_q_one::Vector{Float64},
    connectivity::Matrix{Int},
    correlation_strengths::Vector{Float32}
)::Float64
    """
    ln Pr(x_v = 0 | s) / Pr(x_v = 1 | s) under the coupled prior
    Π q_i(e_i) · exp(Σ J e_i e_k) on the bits 1..length(log_q_zero), restricted
    to configurations of the given parity. Float64, all 2^n configurations.
    """
    n_bits::Int = length(log_q_zero)
    numerator::Float64 = 0.0
    denominator::Float64 = 0.0
    for integer_code in 0:(2^n_bits - 1)
        configuration::Vector{Int} = [(integer_code >> (bit - 1)) & 1 for bit in 1:n_bits]
        if sum(configuration) % 2 != syndrome_bit
            continue
        end
        log_weight::Float64 = 0.0
        for bit in 1:n_bits
            if configuration[bit] == 1
                log_weight += log_q_one[bit]
            else
                log_weight += log_q_zero[bit]
            end
        end
        for pair_index in 1:size(connectivity, 1)
            log_weight += Float64(correlation_strengths[pair_index]) *
                configuration[connectivity[pair_index, 1]] *
                configuration[connectivity[pair_index, 2]]
        end
        if configuration[target_bit] == 0
            numerator += exp(log_weight)
        else
            denominator += exp(log_weight)
        end
    end
    posterior_llr::Float64 = log(numerator / denominator)
    return posterior_llr
end

function enriched_probe_loss(
    coupling_scale::Vector{Float32},
    messages_v2c::Matrix{Float32},
    messages_c2v_seed::Matrix{Float32},
    syndromes_batch::Matrix{Bool},
    tables::SoftCheckTables,
    probe::Matrix{Float32}
)::Float32
    """
    A scalar function of the kernel's output, for differentiation: the
    probe-weighted sum of every check-to-variable message after the enriched
    rows have been overwritten.
    """
    messages_c2v::Matrix{Float32} = copy(messages_c2v_seed)
    apply_enriched_checks!(messages_c2v, messages_v2c, syndromes_batch, coupling_scale, tables)
    loss::Float32 = sum(messages_c2v .* probe)
    return loss
end

# -----------------------------------------------------------------------------

@testset "Soft check tables on the BB code" begin
    base::NeuralBPBase = load_bb_base("enriched", 2)
    tables::SoftCheckTables = base.soft_check_tables
    @test base.check_node_kind == CHECK_NODE_ENRICHED
    @test tables.n_enriched_checks == 36
    @test tables.check_degree == 6
    @test tables.n_configurations == 64
    @test tables.n_pairs_assigned == size(base.connectivity, 1) == 540
    @test tables.n_pairs_unassigned == 0
    @test size(tables.pair_energy) == (64, 36)
    @test size(tables.edge_neurons) == (6, 36)

    # Configuration 1 is all zeros: no coupling is active.
    @test all(iszero, tables.pair_energy[1, :])
    # Configuration 64 is all ones: every coupling inside the check is active,
    # so the energy is the plain sum of that check's J values.
    for (enriched_index, check) in enumerate(tables.enriched_check_ids)
        support::Vector{Int} = findall(base.parity_check_matrix[check, :])
        expected_total::Float32 = 0.0f0
        for pair_index in 1:size(base.connectivity, 1)
            if base.connectivity[pair_index, 1] in support && base.connectivity[pair_index, 2] in support
                expected_total += base.correlation_strengths[pair_index]
            end
        end
        @test isapprox(tables.pair_energy[64, enriched_index], expected_total; atol = 1e-4)
        # Slots point at the neurons of (check, support[slot]).
        for slot in 1:6
            @test base.neuron_to_check_variable[tables.edge_neurons[slot, enriched_index]] == (check, support[slot])
        end
    end

    # Parities match the bit table.
    for configuration_index in 1:64
        parity_from_bits::Float32 = Float32(sum(tables.configuration_bits[configuration_index, :]) % 2)
        @test tables.configuration_parity[configuration_index] == parity_from_bits
    end

    # A "tanh" base carries empty tables, and an enriched base without
    # couplings is refused rather than silently equal to tanh.
    tanh_base::NeuralBPBase = load_bb_base("tanh", 2)
    @test tanh_base.check_node_kind == CHECK_NODE_TANH
    @test tanh_base.soft_check_tables.n_enriched_checks == 0
    @test_throws ArgumentError NeuralBPBase(
        Matrix{Int}(base.parity_check_matrix), Matrix{Int}(base.parity_check_matrix_dual),
        base.initial_llrs, 2; check_node = "enriched")
    @test_throws ArgumentError check_node_code("sigmoid")
end

@testset "Enriched rule reduces to the tanh rule at alpha = 0" begin
    Random.seed!(11)
    base::NeuralBPBase = load_bb_base("enriched", 2)
    n_samples::Int = 8
    messages_v2c::Matrix{Float32} = 2.5f0 .* randn(Float32, base.nb_neurons_per_layer, n_samples)
    syndromes_batch::Matrix{Bool} = rand(Bool, base.code_n_checks, n_samples)

    standard_messages::Matrix{Float32} = standard_check_messages(messages_v2c, syndromes_batch, base)
    enriched_messages::Matrix{Float32} = copy(standard_messages)
    apply_enriched_checks!(enriched_messages, messages_v2c, syndromes_batch, Float32[0.0], base.soft_check_tables)

    # atanh amplifies Float32 round-off when the product of tanh's is near 1,
    # so the tolerance is loose in absolute terms and tight in relative terms.
    @test all(isapprox.(enriched_messages, standard_messages; atol = 1e-2, rtol = 1e-3))
    @test all(isfinite, enriched_messages)

    # With the couplings switched on the messages must actually move.
    coupled_messages::Matrix{Float32} = copy(standard_messages)
    apply_enriched_checks!(coupled_messages, messages_v2c, syndromes_batch, Float32[1.0], base.soft_check_tables)
    @test maximum(abs.(coupled_messages .- standard_messages)) > 1e-2
    @test all(abs.(coupled_messages) .<= ENRICHED_MESSAGE_CAP)
end

@testset "Two-check example: prior + message equals the brute-force posterior" begin
    base::NeuralBPBase = two_check_base("enriched")
    tables::SoftCheckTables = base.soft_check_tables
    @test tables.n_enriched_checks == 1
    @test tables.enriched_check_ids == [1]
    @test tables.check_degree == 3
    @test tables.n_pairs_assigned == 3

    # First BP round: every variable-to-check message is the channel prior.
    messages_v2c::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, 1)
    for neuron in 1:base.nb_neurons_per_layer
        (_, bit) = base.neuron_to_check_variable[neuron]
        messages_v2c[neuron, 1] = base.initial_llrs[bit]
    end
    syndromes_batch::Matrix{Bool} = reshape(Bool[true, false], 2, 1)

    standard_messages::Matrix{Float32} = standard_check_messages(messages_v2c, syndromes_batch, base)
    enriched_messages::Matrix{Float32} = copy(standard_messages)
    apply_enriched_checks!(enriched_messages, messages_v2c, syndromes_batch, Float32[1.0], tables)

    # Check 1 (bits 1..3, syndrome 1): compare against the exact posterior of
    # the coupled prior over the three bits.
    log_q_zero::Vector{Float64} = [-log1p(exp(-Float64(base.initial_llrs[bit]))) for bit in 1:3]
    log_q_one::Vector{Float64} = [-Float64(base.initial_llrs[bit]) - log1p(exp(-Float64(base.initial_llrs[bit]))) for bit in 1:3]
    for slot in 1:3
        neuron::Int = tables.edge_neurons[slot, 1]
        (_, bit) = base.neuron_to_check_variable[neuron]
        exact_posterior::Float64 = brute_force_posterior_llr(
            bit, 1, log_q_zero, log_q_one, base.connectivity, base.correlation_strengths)
        @test isapprox(Float64(base.initial_llrs[bit]) + Float64(enriched_messages[neuron, 1]), exact_posterior; atol = 1e-4)
        # The coupled message differs from the uncoupled one on this check.
        @test abs(enriched_messages[neuron, 1] - standard_messages[neuron, 1]) > 1e-3
    end

    # Check 2 carries no coupling: its rows are exactly the standard messages.
    for neuron in 1:base.nb_neurons_per_layer
        (check, _) = base.neuron_to_check_variable[neuron]
        if check == 2
            @test enriched_messages[neuron, 1] == standard_messages[neuron, 1]
        end
    end
end

@testset "Enzyme differentiates the enriched kernel" begin
    Random.seed!(23)
    # (a) Finite-difference agreement on the small example, where the
    #     probe-weighted sum is O(1) and Float32 differences are clean.
    base::NeuralBPBase = two_check_base("enriched")
    tables::SoftCheckTables = base.soft_check_tables
    n_neurons::Int = base.nb_neurons_per_layer
    messages_v2c::Matrix{Float32} = 1.5f0 .* randn(Float32, n_neurons, 2)
    seed_messages::Matrix{Float32} = randn(Float32, n_neurons, 2)
    syndromes_batch::Matrix{Bool} = Bool[true false; false true]
    probe::Matrix{Float32} = randn(Float32, n_neurons, 2)
    coupling_scale::Vector{Float32} = Float32[0.8]

    gradient_coupling_scale::Vector{Float32} = zeros(Float32, 1)
    gradient_messages::Matrix{Float32} = zeros(Float32, n_neurons, 2)
    Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        enriched_probe_loss,
        Enzyme.Duplicated(coupling_scale, gradient_coupling_scale),
        Enzyme.Duplicated(messages_v2c, gradient_messages),
        Enzyme.Const(seed_messages),
        Enzyme.Const(syndromes_batch),
        Enzyme.Const(tables),
        Enzyme.Const(probe)
    )
    @test all(isfinite, gradient_messages)
    @test isfinite(gradient_coupling_scale[1])

    finite_difference_step::Float32 = 1.0f-2
    loss_plus::Float32 = enriched_probe_loss(Float32[0.8 + finite_difference_step], messages_v2c, seed_messages, syndromes_batch, tables, probe)
    loss_minus::Float32 = enriched_probe_loss(Float32[0.8 - finite_difference_step], messages_v2c, seed_messages, syndromes_batch, tables, probe)
    finite_difference_alpha::Float32 = (loss_plus - loss_minus) / (2.0f0 * finite_difference_step)
    @test isapprox(gradient_coupling_scale[1], finite_difference_alpha; atol = 2e-3, rtol = 2e-2)

    for neuron in (1, 3, 5)
        for sample in 1:2
            perturbed_up::Matrix{Float32} = copy(messages_v2c)
            perturbed_up[neuron, sample] += finite_difference_step
            perturbed_down::Matrix{Float32} = copy(messages_v2c)
            perturbed_down[neuron, sample] -= finite_difference_step
            finite_difference_message::Float32 = (
                enriched_probe_loss(coupling_scale, perturbed_up, seed_messages, syndromes_batch, tables, probe) -
                enriched_probe_loss(coupling_scale, perturbed_down, seed_messages, syndromes_batch, tables, probe)
            ) / (2.0f0 * finite_difference_step)
            @test isapprox(gradient_messages[neuron, sample], finite_difference_message; atol = 2e-3, rtol = 2e-2)
        end
    end
    # Rows of the uncoupled check are pass-through, so their probe weight is
    # their whole gradient contribution: the seed's gradient is not requested,
    # but the messages feeding check 2 must receive zero from the kernel.
    for neuron in 1:n_neurons
        (check, _) = base.neuron_to_check_variable[neuron]
        if check == 2
            @test gradient_messages[neuron, :] == zeros(Float32, 2)
        end
    end

    # (b) The real forward pass on the BB code: gradient of the training loss
    #     with respect to alpha exists, is finite and is not identically zero.
    bb_base::NeuralBPBase = load_bb_base("enriched", 3)
    bpnn::NachmaniNeuralBP = unit_weight_neuralbp(bb_base; coupling_scale = 1.0f0)
    n_samples::Int = 4
    expected_recoveries::BitMatrix = falses(bb_base.code_n_bits, n_samples)
    for sample in 1:n_samples
        expected_recoveries[rand(1:bb_base.code_n_bits), sample] = true
    end
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(bb_base.parity_check_matrix) * Matrix{Int}(expected_recoveries), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(bb_base.initial_llrs, 1, n_samples)
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
        Enzyme.Const(bb_base),
        Enzyme.Const(llrs_batch),
        Enzyme.Const(syndromes),
        Enzyme.Const(expected_recoveries)
    )
    @test isfinite(loss_value)
    @test all(isfinite, grad_w_c2v_v2c)
    @test all(isfinite, grad_w_llrs)
    @test all(isfinite, grad_w_readout)
    @test isfinite(grad_alpha[1])
    @test grad_alpha[1] != 0.0f0
    # Under the CONSTANT schedule the two schedule parameters are never read,
    # so their gradient is exactly zero -- not small, zero.
    @test grad_schedule == zeros(Float32, 2)
end

@testset "Dense formulation equals the loop kernel; GPU forward pass equals CPU" begin
    Random.seed!(37)
    base::NeuralBPBase = load_bb_base("enriched", 3)
    tables::SoftCheckTables = base.soft_check_tables
    n_samples::Int = 6
    messages_v2c::Matrix{Float32} = 2.0f0 .* randn(Float32, base.nb_neurons_per_layer, n_samples)
    syndromes_batch::Matrix{Bool} = rand(Bool, base.code_n_checks, n_samples)
    standard_messages::Matrix{Float32} = standard_check_messages(messages_v2c, syndromes_batch, base)

    loop_messages::Matrix{Float32} = copy(standard_messages)
    apply_enriched_checks!(loop_messages, messages_v2c, syndromes_batch, Float32[0.9], tables)

    # The dense algebra on plain Arrays, no device involved.
    state::GPUSoftCheckState = build_gpu_soft_check_state(tables, syndromes_batch, base.nb_neurons_per_layer, identity)
    dense_messages::Matrix{Float32} = apply_enriched_checks_gpu(standard_messages, messages_v2c, 0.9f0, state)
    @test all(isapprox.(dense_messages, loop_messages; atol = 1e-3, rtol = 1e-4))

    # The full forward pass, GPU (Metal on Apple, CUDA on Linux) against CPU.
    bpnn::NachmaniNeuralBP = unit_weight_neuralbp(base; coupling_scale = 1.0f0)
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    syndromes_bits::BitMatrix = BitMatrix(syndromes_batch)
    cpu_posteriors::Array{Float32, 3} = forward_pass_with_weights(bpnn, llrs_batch, syndromes_bits)
    gpu_posteriors::Array{Float32, 3} = forward_pass_gpu(bpnn, llrs_batch, syndromes_bits)
    @test size(gpu_posteriors) == size(cpu_posteriors)
    @test all(isapprox.(gpu_posteriors, cpu_posteriors; atol = 5e-2, rtol = 1e-3))
end

@testset "The tanh path is unchanged" begin
    Random.seed!(41)
    base::NeuralBPBase = load_bb_base("tanh", 3)
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.1f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.1f0),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.1f0),
    )
    @test isapprox(effective_coupling_scale(bpnn), 1.0f0; atol = 1e-5)
    n_samples::Int = 5
    syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, n_samples))
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    # legacy.jl's callable forward pass predates the enriched node entirely.
    legacy_posteriors::Array{Float32, 3} = bpnn(llrs_batch, syndromes)
    current_posteriors::Array{Float32, 3} = forward_pass_with_weights(bpnn, llrs_batch, syndromes)
    @test all(isapprox.(legacy_posteriors, current_posteriors; atol = 1e-5))
    # ... and alpha is genuinely inert on this path, as is the schedule vector.
    other_alpha::NachmaniNeuralBP = NachmaniNeuralBP(
        base, bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout,
        Float32[5.0], Float32[-3.0, 2.0])
    @test forward_pass_with_weights(other_alpha, llrs_batch, syndromes) == current_posteriors
end

@testset "Weights file round-trips alpha and stays readable without it" begin
    base::NeuralBPBase = load_bb_base("enriched", 2)
    bpnn::NachmaniNeuralBP = unit_weight_neuralbp(base; coupling_scale = 0.7f0)
    weights_file::String = tempname() * ".json"
    save_trained_neuralbp_model(weights_file, bpnn; seed = 3)
    loaded::NachmaniNeuralBP = load_trained_neuralbp_model(weights_file, bpnn)
    @test isapprox(effective_coupling_scale(loaded), 0.7f0; atol = 1e-5)
    @test loaded.coupling_logit == bpnn.coupling_logit        # exact parameter round trip
    @test loaded.weights_llrs == bpnn.weights_llrs
    saved_contents::Dict{String, Any} = JSON.parsefile(weights_file)
    @test saved_contents["check_node"] == "enriched"
    @test saved_contents["seed"] == 3

    # A file written before the enriched node existed has neither key.
    legacy_file::String = tempname() * ".json"
    open(legacy_file, "w") do io
        JSON.print(io, Dict(
            "weights_c2v_v2c" => bpnn.weights_c2v_v2c,
            "weights_llrs" => bpnn.weights_llrs,
            "weights_c2v_readout" => bpnn.weights_c2v_readout,
        ))
    end
    # Loading it into an enriched model works, defaults alpha to 1, and warns
    # that the weights were trained under the other rule.
    loaded_legacy::NachmaniNeuralBP = @test_logs (:warn, r"trained with check_node") load_trained_neuralbp_model(legacy_file, bpnn)
    @test isapprox(effective_coupling_scale(loaded_legacy), 1.0f0; atol = 1e-5)
    rm(weights_file)
    rm(legacy_file)
end

@testset "Vectorised scoring equals the per-sample reference" begin
    # `count_syndrome_satisfactions` is now vectorised over samples (one BLAS
    # call per layer instead of an integer matmul per sample); `check_bp_solutions`
    # is deliberately still the per-sample reference. They must agree exactly,
    # including on the awkward cases: no clearing layer, a layer that clears the
    # syndrome but carries a logical, and ties broken by taking the FIRST
    # clearing layer.
    Random.seed!(67)
    base::NeuralBPBase = load_bb_base("tanh", 8)
    parity_check_matrix::Matrix{Int} = convert.(Int, base.parity_check_matrix)
    n_checks::Int = size(parity_check_matrix, 1)
    logicals::Matrix{Int} = convert.(Int, base.parity_check_matrix_dual[n_checks + 1:end, :])
    n_bits::Int = base.code_n_bits
    n_samples::Int = 200
    n_layers::Int = 8

    errors::BitMatrix = falses(n_bits, n_samples)
    for sample in 1:n_samples
        for _ in 1:rand(0:3)
            errors[rand(1:n_bits), sample] = true
        end
    end

    # A coset failure needs a residual w with HZ·w = 0 (syndrome cleared) AND
    # `logicals`·w ≠ 0 (wrong coset). The rows of `logicals` are NOT such a w:
    # `logicals` is the tail of the dual, i.e. LZ, which is the DETECTOR, and
    # HZ·LZᵀ ≠ 0 -- XORing with an LZ row changes the syndrome, so the sample
    # scores as a convergence failure and the coset bucket stays empty. The
    # representatives live in LX, which is the matrix symplectically paired with
    # LZ (LZ·LXᵀ = I).
    coset_representatives::Matrix{Int} =
        readdlm("$(debug_data_directory())/code/LX.txt", Int)
    coset_shift::Vector{Bool} = coset_representatives[1, :] .== 1
    @test all(iszero, mod.(parity_check_matrix * Int.(coset_shift), 2))   # clears the syndrome
    @test any(!iszero, mod.(logicals * Int.(coset_shift), 2))             # but is detected

    # A mixture: random recoveries (mostly non-clearing), plus planted exact and
    # logical-carrying recoveries so all three outcome buckets are populated.
    recoveries::Array{Bool, 3} = rand(Bool, n_bits, n_samples, n_layers)
    for sample in 1:3:n_samples
        planted_layer::Int = rand(1:n_layers)
        recoveries[:, sample, planted_layer] .= errors[:, sample]
    end
    for sample in 2:3:n_samples
        planted_layer::Int = rand(1:n_layers)
        # errors XOR a logical operator: clears the syndrome, wrong coset.
        recoveries[:, sample, planted_layer] .= errors[:, sample] .⊻ coset_shift
    end

    diagnosis::NamedTuple = count_syndrome_satisfactions(
        parity_check_matrix, logicals, errors, recoveries)
    reference_is_correct::BitVector = check_bp_solutions(
        parity_check_matrix, logicals, errors, recoveries)

    @test diagnosis.is_correct == reference_is_correct
    @test count(diagnosis.is_correct) > 0                 # successes present
    @test diagnosis.n_coset_failures > 0                  # coset failures present
    @test diagnosis.n_convergence_failures > 0            # convergence failures present
    @test diagnosis.n_syndrome_cleared ==
          diagnosis.n_correct + diagnosis.n_coset_failures
    @test diagnosis.n_convergence_failures ==
          n_samples - diagnosis.n_syndrome_cleared
    @test diagnosis.error_weight == vec(sum(errors, dims = 1))

    # Committed layer and minimum syndrome weight, against a direct recomputation.
    for sample in 1:n_samples
        layer_weights::Vector{Int} = [
            sum(mod.(parity_check_matrix * (errors[:, sample] .⊻ recoveries[:, sample, layer]), 2))
            for layer in 1:n_layers
        ]
        @test diagnosis.min_syndrome_weight[sample] == minimum(layer_weights)
        expected_layer::Int = something(findfirst(==(0), layer_weights), 0)
        @test diagnosis.committed_layer[sample] == expected_layer
        @test diagnosis.syndrome_cleared[sample] == (expected_layer > 0)
    end
end

@testset "GPU state reuse and on-device thresholding are exact" begin
    # Both optimisations must be bit-for-bit invisible: a reused state must
    # decode a chunk identically to a freshly built one, and thresholding on the
    # device must equal thresholding on the host. Run for BOTH check nodes,
    # since the reuse path also repoints the enriched syndromes.
    Random.seed!(53)
    for check_node in ("tanh", "enriched")
        base::NeuralBPBase = load_bb_base(check_node, 6)
        bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
            base;
            weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.1f0),
            weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.1f0),
            weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.1f0),
            coupling_scale = 0.8f0,
        )
        n_samples::Int = 5
        first_syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, n_samples))
        second_syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, n_samples))
        llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)

        # Device thresholding against the host thresholding it replaces.
        fresh_state::GPUState = build_gpu_state(bpnn, first_syndromes)
        device_recoveries::Array{Bool, 3} = predict_recoveries_gpu(fresh_state, llrs_batch)
        host_recoveries::Array{Bool, 3} =
            Array(forward_pass_with_weights(bpnn, llrs_batch, first_syndromes) .< 0)
        @test device_recoveries == host_recoveries

        # A REUSED state (repointed at new syndromes) must equal a fresh one.
        reused_state::GPUState = reusable_gpu_state(fresh_state, bpnn, second_syndromes)
        @test reused_state === fresh_state          # reused, not rebuilt
        reused_recoveries::Array{Bool, 3} = predict_recoveries_gpu(reused_state, llrs_batch)
        rebuilt_state::GPUState = build_gpu_state(bpnn, second_syndromes)
        rebuilt_recoveries::Array{Bool, 3} = predict_recoveries_gpu(rebuilt_state, llrs_batch)
        @test reused_recoveries == rebuilt_recoveries
        # ... and it really is the second chunk, not a stale first one.
        @test reused_recoveries != device_recoveries

        # A differently sized chunk forces a rebuild rather than a bad reuse.
        short_syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, 3))
        short_state::GPUState = reusable_gpu_state(reused_state, bpnn, short_syndromes)
        @test short_state !== reused_state
        @test short_state.n_samples == 3
        @test_throws DimensionMismatch update_gpu_state_syndromes!(short_state, base, first_syndromes)

        release_gpu_state!(rebuilt_state)
        release_gpu_state!(short_state)
    end
end

@testset "alpha is confined to (0,1) by the logistic link" begin
    # alpha = 1/(1+exp(-theta)). The point of the link is that NO value of the
    # trained parameter can put alpha outside the physically meaningful range:
    # below 0 it would invert every coupling (J carries the sign already), above
    # 1 it would claim more coupling than CER measured.
    for logit in (-40.0f0, -5.0f0, -0.3228f0, 0.0f0, 1.0f0, 40.0f0)
        alpha::Float32 = coupling_scale_from_logit(logit)
        @test 0.0f0 <= alpha <= 1.0f0
        @test isfinite(alpha)
    end
    @test coupling_scale_from_logit(0.0f0) == 0.5f0
    # Monotone increasing, so a larger parameter always means more coupling.
    logits::Vector{Float32} = Float32[-6, -2, -0.5, 0, 0.5, 2, 6]
    alphas::Vector{Float32} = coupling_scale_from_logit.(logits)
    @test all(alphas[i] < alphas[i + 1] for i in 1:(length(alphas) - 1))
    # Round trip, to the precision the margin allows.
    for alpha_target in (0.05f0, 0.2f0, 0.42f0, 0.5f0, 0.8f0, 0.95f0)
        @test isapprox(coupling_scale_from_logit(logit_from_coupling_scale(alpha_target)),
                       alpha_target; atol = 1e-5)
    end
    # The endpoints of the CLOSED range are nudged inside rather than sent to
    # +/-Inf: an infinite parameter makes the gradient NaN and would NaN-skip
    # every batch. They are IN range, so the nudge is silent -- alpha = 1 is the
    # Bayesian value and the default, and every weights file that predates the
    # coupling field loads with it.
    @test isfinite(logit_from_coupling_scale(0.0f0))
    @test isfinite(logit_from_coupling_scale(1.0f0))
    @test coupling_scale_from_logit(logit_from_coupling_scale(0.0f0)) < 1e-5
    @test coupling_scale_from_logit(logit_from_coupling_scale(1.0f0)) > 1 - 1e-5
    @test_logs logit_from_coupling_scale(0.0f0)      # no log records at all
    @test_logs logit_from_coupling_scale(1.0f0)
    # Genuinely OUT of range is a statement about the physics the run cannot
    # honour, and is reported every time.
    @test_logs (:warn, r"lies outside") logit_from_coupling_scale(-0.5f0)
    @test_logs (:warn, r"lies outside") logit_from_coupling_scale(1.5f0)

    # A model built with alpha = 0.42 uses 0.42, and the parameter it trains is
    # that value's logit.
    base::NeuralBPBase = load_bb_base("enriched", 2)
    model::NachmaniNeuralBP = unit_weight_neuralbp(base; coupling_scale = 0.42f0)
    @test isapprox(effective_coupling_scale(model), 0.42f0; atol = 1e-5)
    @test isapprox(model.coupling_logit[1], log(0.42f0 / 0.58f0); atol = 1e-5)
    # Driving the parameter far out never leaves the range. In exact arithmetic
    # alpha only approaches 1; in Float32 it rounds to exactly 1 above theta
    # ~= 16.6, which is benign -- alpha = 1 is the Bayesian value, and the link's
    # derivative alpha*(1-alpha) rounds to 0 there too, so a runaway theta parks
    # alpha at the top of the range instead of producing anything infinite.
    model.coupling_logit[1] = 12.0f0
    @test effective_coupling_scale(model) < 1.0f0
    @test effective_coupling_scale(model) > 0.99f0
    model.coupling_logit[1] = 50.0f0
    @test effective_coupling_scale(model) == 1.0f0     # Float32 saturation, not a bug
    model.coupling_logit[1] = -50.0f0
    @test effective_coupling_scale(model) >= 0.0f0
    @test effective_coupling_scale(model) < 1.0f-6
end

@testset "Prediction batch size accounts for the enriched kernel" begin
    # The kernel holds two (2^d x n_enriched x batch) Float32 tensors per layer
    # — on Metal these come out of system RAM, and the uncapped 16384 fallback
    # was OOM-killed on a 10^6-sample run.
    enriched_base::NeuralBPBase = load_bb_base("enriched", 90)
    tanh_base::NeuralBPBase = load_bb_base("tanh", 90)
    enriched_bytes::Int = enriched_kernel_bytes_per_sample(enriched_base.soft_check_tables)
    @test enriched_bytes == 64 * 36 * 4 * 4
    @test enriched_kernel_bytes_per_sample(tanh_base.soft_check_tables) == 0

    # The cap is a power of two, fits the budget, and never raises a batch size.
    capped::Int = cap_batch_size_for_enriched_kernel(1 << 20, enriched_base.soft_check_tables)
    @test capped < (1 << 20)
    @test capped > 0
    @test ispow2(capped)
    @test capped * enriched_bytes <= ENRICHED_KERNEL_MEMORY_BUDGET_BYTES
    @test cap_batch_size_for_enriched_kernel(64, enriched_base.soft_check_tables) == 64
    # The standard check node is untouched at any size.
    @test cap_batch_size_for_enriched_kernel(1 << 20, tanh_base.soft_check_tables) == (1 << 20)

    # ... and the cap is actually applied on the path that has no memory budget,
    # which is where the OOM happened: no `gpu_memory`, no `prediction_batch_size`.
    enriched_model::NachmaniNeuralBP = unit_weight_neuralbp(enriched_base)
    tanh_model::NachmaniNeuralBP = unit_weight_neuralbp(tanh_base)
    withenv("GPU_MEMORY" => nothing, "SLURM_MEM_PER_GPU" => nothing) do
        # At the 16384 fallback and degree 6 the cap does not bite, so both
        # check nodes get the same batch size; the cap exists for wider checks.
        @test resolve_prediction_batch_size(enriched_model) ==
              cap_batch_size_for_enriched_kernel(16384, enriched_base.soft_check_tables)
        @test resolve_prediction_batch_size(tanh_model) == 16384
        # An explicit request still wins outright.
        @test resolve_prediction_batch_size(enriched_model; batch_size = 65536) == 65536
    end
end

# =============================================================================
# The layer schedule on alpha: alpha_t = alpha * d(t), d(t) = 1/(1+exp((t-T0)/w)),
# with [T0, log(w - W_MIN)] as two trainable parameters (src/soft_constraints.jl,
# `coupling_schedule_*`). What is checked:
#   8.  The step itself: 1/2 at T0, monotone decreasing, saturates cleanly at
#       both ends, the width floor holds, human units round-trip.
#   9.  The CONSTANT schedule reproduces the schedule-free forward pass bit for
#       bit, and the schedule parameters are inert under it.
#  10.  A STEP schedule with T0 far past the last layer is bit-identical to the
#       constant one; a step inside the network changes the posteriors and
#       does so only from the layers the step touches.
#  11.  The three definitions of alpha_t agree: `effective_coupling_scale_at_layer`,
#       the GPU state's per-layer vector, and the CPU path (via 10).
#  12.  Enzyme's gradient reaches [T0, rho] under the step schedule and is
#       exactly zero under the constant one; finite differences agree.
#  13.  The weights file round-trips the schedule and its kind, and a file
#       without either loads with the caller's defaults.
# =============================================================================

function load_bb_base_with_schedule(check_node::String, coupling_schedule::String, n_layers::Int)::NeuralBPBase
    """
    `load_bb_base` with the layer schedule selected as well.
    """
    base::NeuralBPBase = load_base_BP_model(
        "$(debug_data_directory())/code/HZ.txt",
        "$(debug_data_directory())/code/LZ.txt",
        n_layers;
        cer_data_file = debug_cer_file(),
        use_cer = true,
        check_node = check_node,
        coupling_schedule = coupling_schedule,
    )
    return base
end

@testset "The coupling schedule step: shape, floor, human-unit round trip" begin
    # d(T0) = 1/2 exactly, whatever the width.
    for (step_layer, step_width) in ((12.0f0, 3.0f0), (5.0f0, 1.0f0), (40.0f0, 8.0f0))
        parameters::Vector{Float32} = coupling_schedule_parameters(step_layer, step_width)
        @test length(parameters) == 2
        @test parameters[1] == step_layer
        @test isapprox(coupling_schedule_damping(parameters, round(Int, step_layer)), 0.5f0; atol = 1e-6)
        # Human units come back out.
        @test isapprox(coupling_schedule_width_from_parameter(parameters[2]), step_width; atol = 1e-5)
    end
    # Monotone decreasing in the layer, 1 at the front, 0 at the back. Strictly
    # decreasing while unsaturated; once tanh reaches exactly 1 in Float32
    # (around (t - T0)/2w > 9, i.e. layer ~66 here) consecutive layers tie at
    # exactly 0, so the tail is tested non-strictly. Those exact zeros are the
    # point of the tanh form: 1/(1 + exp(u)) reaches the same values through
    # exp = Inf, whose reverse-mode derivative is Inf·0 = NaN.
    parameters::Vector{Float32} = coupling_schedule_parameters(12.0f0, 3.0f0)
    dampings::Vector{Float32} = [coupling_schedule_damping(parameters, layer) for layer in 1:90]
    @test all(dampings[layer] > dampings[layer + 1] for layer in 1:50)
    @test all(dampings[layer] >= dampings[layer + 1] for layer in 1:89)
    @test dampings[1] > 0.97f0
    @test dampings[90] == 0.0f0
    @test all(0.0f0 .<= dampings .<= 1.0f0)
    @test all(isfinite, dampings)
    # Saturation is clean in Float32: T0 far off either end gives exactly 1 or 0,
    # never NaN, for every layer.
    far_future::Vector{Float32} = coupling_schedule_parameters(1.0f6, 3.0f0)
    far_past::Vector{Float32} = coupling_schedule_parameters(-1.0f6, 3.0f0)
    @test all(coupling_schedule_damping(far_future, layer) == 1.0f0 for layer in 1:90)
    @test all(coupling_schedule_damping(far_past, layer) == 0.0f0 for layer in 1:90)
    # ... and, at the width FLOOR with a step in the middle, the layers far past
    # it are exactly 0 too. This is the configuration whose gradient would be
    # NaN under the exp form; the test below on Enzyme covers the gradient, this
    # one pins the value.
    narrow::Vector{Float32} = coupling_schedule_parameters(12.0f0, 0.51f0)
    @test coupling_schedule_damping(narrow, 90) == 0.0f0
    @test coupling_schedule_damping(narrow, 1) == 1.0f0
    # The width can never reach the floor, however far the stored parameter goes.
    @test coupling_schedule_width_from_parameter(-50.0f0) >= COUPLING_SCHEDULE_MIN_WIDTH
    @test coupling_schedule_width_from_parameter(-50.0f0) < COUPLING_SCHEDULE_MIN_WIDTH + 1.0f-6
    @test isfinite(coupling_schedule_width_from_parameter(50.0f0))
    # Asking for a width at or below the floor is nudged above it, with a warning.
    @test_logs (:warn, r"at or below the floor") coupling_schedule_parameters(12.0f0, 0.5f0)
    @test_logs (:warn, r"at or below the floor") coupling_schedule_parameters(12.0f0, 0.0f0)
    nudged::Vector{Float32} = @test_logs (:warn, r"at or below") coupling_schedule_parameters(12.0f0, 0.1f0)
    @test coupling_schedule_width_from_parameter(nudged[2]) > COUPLING_SCHEDULE_MIN_WIDTH
    # A comfortable width is silent.
    @test_logs coupling_schedule_parameters(12.0f0, 3.0f0)
    # Names round-trip and unknown ones fail at configuration time.
    @test coupling_schedule_code("constant") == COUPLING_SCHEDULE_CONSTANT
    @test coupling_schedule_code("step") == COUPLING_SCHEDULE_STEP
    @test coupling_schedule_code(" Step ") == COUPLING_SCHEDULE_STEP
    @test coupling_schedule_name(COUPLING_SCHEDULE_CONSTANT) == "constant"
    @test coupling_schedule_name(COUPLING_SCHEDULE_STEP) == "step"
    @test_throws ArgumentError coupling_schedule_code("ramp")
    @test_throws ArgumentError coupling_schedule_name(7)
    # A schedule on the tanh rule would be silently inert; NeuralBPBase refuses.
    @test_throws ArgumentError load_bb_base_with_schedule("tanh", "step", 2)
end

@testset "The constant schedule is the schedule-free forward pass, bit for bit" begin
    Random.seed!(53)
    base::NeuralBPBase = load_bb_base_with_schedule("enriched", "constant", 4)
    @test base.coupling_schedule_kind == COUPLING_SCHEDULE_CONSTANT
    bpnn::NachmaniNeuralBP = unit_weight_neuralbp(base; coupling_scale = 0.503f0)
    n_samples::Int = 5
    syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, n_samples))
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    reference::Array{Float32, 3} = forward_pass_with_weights(bpnn, llrs_batch, syndromes)
    # Whatever the schedule vector holds, the constant schedule never reads it.
    for other_schedule in (Float32[-3.0, 2.0], Float32[2.0, -4.0], Float32[1.0f6, 0.0])
        other::NachmaniNeuralBP = NachmaniNeuralBP(
            base, bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout,
            bpnn.coupling_logit, other_schedule)
        @test forward_pass_with_weights(other, llrs_batch, syndromes) == reference
        @test all(effective_coupling_scale_at_layer(other, layer) == effective_coupling_scale(other) for layer in 1:4)
    end
    # The default schedule the keyword constructor writes is the last layer with
    # width 3, so even a step schedule left at defaults barely differs.
    (default_layer, default_width) = effective_coupling_schedule(bpnn)
    @test default_layer == Float32(base.n_layers)
    @test isapprox(default_width, 3.0f0; atol = 1e-5)
end

@testset "A step schedule far past the last layer equals constant; one inside the network does not" begin
    Random.seed!(59)
    n_layers::Int = 6
    constant_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "constant", n_layers)
    step_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "step", n_layers)
    @test step_base.coupling_schedule_kind == COUPLING_SCHEDULE_STEP
    n_samples::Int = 6
    syndromes::BitMatrix = BitMatrix(rand(Bool, constant_base.code_n_checks, n_samples))
    llrs_batch::Matrix{Float32} = repeat(constant_base.initial_llrs, 1, n_samples)

    constant_model::NachmaniNeuralBP = unit_weight_neuralbp(constant_base; coupling_scale = 0.503f0)
    constant_posteriors::Array{Float32, 3} = forward_pass_with_weights(constant_model, llrs_batch, syndromes)

    # (a) T0 far beyond the last layer: d(t) == 1 exactly for every layer, so
    #     the posteriors are bit-identical to the constant schedule's.
    far_model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = 1.0f6, coupling_schedule_width = 3.0f0)
    @test all(effective_coupling_scale_at_layer(far_model, layer) == effective_coupling_scale(far_model) for layer in 1:n_layers)
    @test forward_pass_with_weights(far_model, llrs_batch, syndromes) == constant_posteriors

    # (b) T0 far BEFORE the first layer: alpha_t == 0 everywhere, which is the
    #     tanh rule (checked elsewhere at alpha = 0), so it differs from (a).
    off_model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = -1.0f6, coupling_schedule_width = 3.0f0)
    @test all(effective_coupling_scale_at_layer(off_model, layer) == 0.0f0 for layer in 1:n_layers)
    off_posteriors::Array{Float32, 3} = forward_pass_with_weights(off_model, llrs_batch, syndromes)
    @test off_posteriors != constant_posteriors

    # (c) A step INSIDE the network. Layers before it see (nearly) full
    #     couplings, so their posteriors agree with the constant schedule to
    #     Float32 noise; layers at and after it differ.
    step_layer::Float32 = 4.0f0
    inside_model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = step_layer, coupling_schedule_width = 0.6f0)
    inside_posteriors::Array{Float32, 3} = forward_pass_with_weights(inside_model, llrs_batch, syndromes)
    @test isapprox(effective_coupling_scale_at_layer(inside_model, 4), 0.503f0 / 2; atol = 1e-5)
    @test effective_coupling_scale_at_layer(inside_model, 1) > 0.99f0 * 0.503f0
    @test effective_coupling_scale_at_layer(inside_model, 6) < 0.05f0 * 0.503f0
    @test all(isapprox.(inside_posteriors[:, :, 1], constant_posteriors[:, :, 1]; atol = 2e-3))
    @test !all(isapprox.(inside_posteriors[:, :, 6], constant_posteriors[:, :, 6]; atol = 1e-3))
    # ... and the differing late layers still produce finite LLRs.
    @test all(isfinite, inside_posteriors)
end

@testset "alpha_t has one definition: struct, GPU state and CPU path agree" begin
    step_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "step", 8)
    model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = 4.5f0, coupling_schedule_width = 1.5f0)
    # The GPU state's per-layer vector is the same function of the parameters as
    # `effective_coupling_scale_at_layer`, computed by different code.
    per_layer::Vector{Float32} = coupling_scale_per_layer(model.coupling_logit, model.coupling_schedule, step_base)
    @test length(per_layer) == step_base.n_layers
    for layer in 1:step_base.n_layers
        @test per_layer[layer] == effective_coupling_scale_at_layer(model, layer)
    end
    # Decreasing, and both ends of the step are where they should be.
    @test all(per_layer[layer] > per_layer[layer + 1] for layer in 1:(step_base.n_layers - 1))
    @test per_layer[1] > 0.85f0 * 0.503f0
    @test per_layer[8] < 0.15f0 * 0.503f0
    # Under the constant schedule the vector is flat at alpha.
    constant_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "constant", 8)
    constant_model::NachmaniNeuralBP = unit_weight_neuralbp(constant_base; coupling_scale = 0.503f0)
    flat::Vector{Float32} = coupling_scale_per_layer(constant_model.coupling_logit, constant_model.coupling_schedule, constant_base)
    @test all(flat .== effective_coupling_scale(constant_model))
end

@testset "Enzyme differentiates the training loss with respect to the step schedule" begin
    Random.seed!(61)
    step_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "step", 6)
    # A step in the middle of the network with a width that keeps d(t) away
    # from both saturations at every scored layer, so the gradient has
    # somewhere to be non-zero.
    model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = 3.5f0, coupling_schedule_width = 2.0f0)
    n_samples::Int = 4
    expected_recoveries::BitMatrix = falses(step_base.code_n_bits, n_samples)
    for sample in 1:n_samples
        expected_recoveries[rand(1:step_base.code_n_bits), sample] = true
        expected_recoveries[rand(1:step_base.code_n_bits), sample] = true
    end
    syndromes::BitMatrix = BitMatrix(mod.(Matrix{Int}(step_base.parity_check_matrix) * Matrix{Int}(expected_recoveries), 2) .== 1)
    llrs_batch::Matrix{Float32} = repeat(step_base.initial_llrs, 1, n_samples)

    function schedule_loss(schedule_parameters::Vector{Float32})::Float32
        loss::Float32 = CorrelatedBPDecoderWithCER.get_loss_value(
            model.weights_c2v_v2c, model.weights_llrs, model.weights_c2v_readout,
            model.coupling_logit, schedule_parameters,
            1.0f0, 0, step_base, llrs_batch, syndromes, expected_recoveries)
        return loss
    end

    grad_w_c2v_v2c::Vector{Float32} = zeros(Float32, length(model.weights_c2v_v2c))
    grad_w_llrs::Vector{Float32} = zeros(Float32, length(model.weights_llrs))
    grad_w_readout::Vector{Float32} = zeros(Float32, length(model.weights_c2v_readout))
    grad_alpha::Vector{Float32} = zeros(Float32, 1)
    grad_schedule::Vector{Float32} = zeros(Float32, 2)
    (_, loss_value) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        CorrelatedBPDecoderWithCER.get_loss_value,
        Enzyme.Duplicated(model.weights_c2v_v2c, grad_w_c2v_v2c),
        Enzyme.Duplicated(model.weights_llrs, grad_w_llrs),
        Enzyme.Duplicated(model.weights_c2v_readout, grad_w_readout),
        Enzyme.Duplicated(model.coupling_logit, grad_alpha),
        Enzyme.Duplicated(model.coupling_schedule, grad_schedule),
        Enzyme.Const(1.0f0),
        Enzyme.Const(0),
        Enzyme.Const(step_base),
        Enzyme.Const(llrs_batch),
        Enzyme.Const(syndromes),
        Enzyme.Const(expected_recoveries)
    )
    @test isfinite(loss_value)
    @test all(isfinite, grad_schedule)
    @test all(isfinite, grad_alpha)
    # Both schedule parameters receive gradient under the step schedule.
    @test grad_schedule[1] != 0.0f0
    @test grad_schedule[2] != 0.0f0
    # ... and it agrees with central finite differences in each parameter.
    for parameter_index in 1:2
        step::Float32 = 2.0f-2
        up::Vector{Float32} = copy(model.coupling_schedule); up[parameter_index] += step
        down::Vector{Float32} = copy(model.coupling_schedule); down[parameter_index] -= step
        finite_difference::Float32 = (schedule_loss(up) - schedule_loss(down)) / (2.0f0 * step)
        @test isapprox(grad_schedule[parameter_index], finite_difference; atol = 5e-3, rtol = 5e-2)
    end
end

@testset "Weights file round-trips the schedule and stays readable without it" begin
    step_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "step", 3)
    model::NachmaniNeuralBP = unit_weight_neuralbp(
        step_base; coupling_scale = 0.503f0, coupling_schedule_layer = 11.0f0, coupling_schedule_width = 2.5f0)
    # Perturb the stored parameters so the round trip is not of the defaults.
    model.coupling_schedule[1] = 13.25f0
    model.coupling_schedule[2] = 0.7f0
    weights_file::String = tempname() * ".json"
    save_trained_neuralbp_model(weights_file, model; seed = 5)
    loaded::NachmaniNeuralBP = load_trained_neuralbp_model(weights_file, model)
    @test loaded.coupling_schedule == model.coupling_schedule          # exact parameter round trip
    @test loaded.coupling_logit == model.coupling_logit
    (loaded_layer, loaded_width) = effective_coupling_schedule(loaded)
    @test loaded_layer == 13.25f0
    @test isapprox(loaded_width, coupling_schedule_width_from_parameter(0.7f0); atol = 1e-6)
    saved_contents::Dict{String, Any} = JSON.parsefile(weights_file)
    @test saved_contents["coupling_schedule_kind"] == "step"
    @test length(saved_contents["coupling_schedule"]) == 2
    @test isapprox(saved_contents["coupling_schedule_layer"], 13.25; atol = 1e-6)
    @test isapprox(saved_contents["coupling_schedule_width"], loaded_width; atol = 1e-5)
    @test saved_contents["check_node"] == "enriched"

    # A file that predates the schedule has none of the keys: it loads with the
    # CALLER's schedule defaults, and warns that the kinds differ.
    legacy_file::String = tempname() * ".json"
    open(legacy_file, "w") do io
        JSON.print(io, Dict(
            "weights_c2v_v2c" => model.weights_c2v_v2c,
            "weights_llrs" => model.weights_llrs,
            "weights_c2v_readout" => model.weights_c2v_readout,
            "coupling_logit" => model.coupling_logit,
            "coupling_scale" => [effective_coupling_scale(model)],
            "check_node" => "enriched",
        ))
    end
    loaded_legacy::NachmaniNeuralBP = @test_logs (:warn, r"trained with coupling_schedule") load_trained_neuralbp_model(legacy_file, model)
    (legacy_layer, legacy_width) = effective_coupling_schedule(loaded_legacy)
    @test legacy_layer == 13.25f0                                        # the caller's, not a default
    @test isapprox(legacy_width, coupling_schedule_width_from_parameter(0.7f0); atol = 1e-6)
    @test loaded_legacy.weights_llrs == model.weights_llrs
    # Loading the same legacy file into a CONSTANT-schedule model is silent
    # about the schedule (both are "constant").
    constant_base::NeuralBPBase = load_bb_base_with_schedule("enriched", "constant", 3)
    constant_model::NachmaniNeuralBP = unit_weight_neuralbp(constant_base; coupling_scale = 0.503f0)
    @test_logs load_trained_neuralbp_model(legacy_file, constant_model)
    rm(weights_file)
    rm(legacy_file)
end

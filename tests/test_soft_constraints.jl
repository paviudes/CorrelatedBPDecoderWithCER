using CorrelatedBPDecoderWithCER
using Test
using Enzyme
using JSON
using Random

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
# =============================================================================

function debug_data_directory()::String
    return "./../data/72q_BB_cycles_1_debug"
end

function debug_cer_file()::String
    return "$(debug_data_directory())/correlated_weights/correlated_weights_p_0.0005_s_1.txt"
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
    (_, loss_value) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        CorrelatedBPDecoderWithCER.get_loss_value,
        Enzyme.Duplicated(bpnn.weights_c2v_v2c, grad_w_c2v_v2c),
        Enzyme.Duplicated(bpnn.weights_llrs, grad_w_llrs),
        Enzyme.Duplicated(bpnn.weights_c2v_readout, grad_w_readout),
        Enzyme.Duplicated(bpnn.coupling_scale, grad_alpha),
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
    @test bpnn.coupling_scale == Float32[1.0]
    n_samples::Int = 5
    syndromes::BitMatrix = BitMatrix(rand(Bool, base.code_n_checks, n_samples))
    llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    # legacy.jl's callable forward pass predates the enriched node entirely.
    legacy_posteriors::Array{Float32, 3} = bpnn(llrs_batch, syndromes)
    current_posteriors::Array{Float32, 3} = forward_pass_with_weights(bpnn, llrs_batch, syndromes)
    @test all(isapprox.(legacy_posteriors, current_posteriors; atol = 1e-5))
    # ... and alpha is genuinely inert on this path.
    other_alpha::NachmaniNeuralBP = NachmaniNeuralBP(
        base, bpnn.weights_c2v_v2c, bpnn.weights_llrs, bpnn.weights_c2v_readout, Float32[5.0])
    @test forward_pass_with_weights(other_alpha, llrs_batch, syndromes) == current_posteriors
end

@testset "Weights file round-trips alpha and stays readable without it" begin
    base::NeuralBPBase = load_bb_base("enriched", 2)
    bpnn::NachmaniNeuralBP = unit_weight_neuralbp(base; coupling_scale = 0.7f0)
    weights_file::String = tempname() * ".json"
    save_trained_neuralbp_model(weights_file, bpnn; seed = 3)
    loaded::NachmaniNeuralBP = load_trained_neuralbp_model(weights_file, bpnn)
    @test loaded.coupling_scale == Float32[0.7]
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
    @test loaded_legacy.coupling_scale == Float32[1.0]
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
        recoveries[:, sample, planted_layer] .= errors[:, sample] .⊻ (logicals[1, :] .== 1)
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
            coupling_scale = Float32[0.8],
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

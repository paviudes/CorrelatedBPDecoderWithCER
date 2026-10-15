using CorrelatedBPDecoderWithCER
using Test
using Random
using LinearAlgebra

# =============================================================================
# Tests for the hand-written backward pass, src/backward_pass_with_weights.jl
# and src/backward_pass_gpu.jl, one forward rule at a time.
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_backward_pass.jl")'
# =============================================================================

# Device arrays of whichever backend the package loaded (Metal on Apple, CUDA on
# Linux, plain Arrays otherwise). On a Mac the GPU testsets run on Metal whether
# or not USE_GPU is set.
const DeviceVector = CorrelatedBPDecoderWithCER.ArrayT{Float32, 1}
const DeviceMatrix = CorrelatedBPDecoderWithCER.ArrayT{Float32, 2}

function two_check_backward_base()::NeuralBPBase
    """
    Four bits, two checks: check 1 = {1,2,3}, check 2 = {3,4}. Five edges.
    """
    parity_check_matrix::Matrix{Int} = [1 1 1 0; 0 0 1 1]
    parity_check_matrix_dual::Matrix{Int} = [1 1 1 0; 0 0 1 1; 1 0 0 1]
    initial_llrs::Vector{Float32} = fill(Float32(log(9)), 4)
    base::NeuralBPBase = NeuralBPBase(parity_check_matrix, parity_check_matrix_dual, initial_llrs, 3)
    return base
end

function bb_backward_base()::NeuralBPBase
    """
    The [[72,12,6]] BB code (Z checks), standard check node, 3 layers.
    """
    data_directory::String = "./../data/72q_BB_cycles_1_soft_constraints"
    base::NeuralBPBase = load_base_BP_model(
        "$(data_directory)/code/HZ.txt",
        "$(data_directory)/code/LZ.txt",
        3;
        cer_data_file = "$(data_directory)/correlated_weights/correlated_weights_p_0.0005_sig_0.0_s_1.txt",
        use_cer = true,
        check_node = "tanh",
    )
    return base
end

function reference_backward_v2c_to_c2v(
    adjoint_activated_m_c2v::Matrix{Float32},
    base::NeuralBPBase
)::Matrix{Float32}
    """
    Rule 1̄ written out as it is derived, edge by edge:
        Ā_{v'->c} = ∑_{v ∈ N(c) - v'} Ā_{c->v}
    using only the neuron -> (check, bit) maps, never an adjacency matrix.
    """
    n_edges::Int = base.nb_neurons_per_layer
    reference::Matrix{Float32} = zeros(Float32, size(adjoint_activated_m_c2v))
    for input_edge in 1:n_edges
        for output_edge in 1:n_edges
            same_check::Bool = base.neuron_to_checks[output_edge] == base.neuron_to_checks[input_edge]
            other_variable::Bool = base.neuron_to_bits[output_edge] != base.neuron_to_bits[input_edge]
            if same_check && other_variable
                reference[input_edge, :] .+= adjoint_activated_m_c2v[output_edge, :]
            end
        end
    end
    return reference
end

@testset "backward_v2c_to_c2v!: rule 1̄ edge by edge, on the two-check graph" begin
    base::NeuralBPBase = two_check_backward_base()
    @test base.nb_neurons_per_layer == 5
    # One sample; edges are (c1,v1) (c1,v2) (c1,v3) (c2,v3) (c2,v4) in constructor order.
    adjoint_c2v::Matrix{Float32} = reshape(Float32[10, 20, 30, 40, 50], 5, 1)
    adjoint_v2c::Matrix{Float32} = zeros(Float32, 5, 1)
    backward_v2c_to_c2v!(adjoint_v2c, adjoint_c2v, base)
    # Edge (c1,v1) is consumed by the outputs (c1,v2) and (c1,v3): 20 + 30.
    @test adjoint_v2c[1, 1] == 50.0f0
    @test adjoint_v2c[2, 1] == 40.0f0     # (c1,v1) + (c1,v3) = 10 + 30
    @test adjoint_v2c[3, 1] == 30.0f0     # (c1,v1) + (c1,v2) = 10 + 20
    @test adjoint_v2c[4, 1] == 50.0f0     # check 2: the other edge (c2,v4)
    @test adjoint_v2c[5, 1] == 40.0f0     # check 2: the other edge (c2,v3)
    @test adjoint_v2c == reference_backward_v2c_to_c2v(adjoint_c2v, base)
end

@testset "backward_v2c_to_c2v!: equals the edge-by-edge rule on the BB code, batched" begin
    Random.seed!(5)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 7
    adjoint_c2v::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    adjoint_v2c::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, n_samples)
    backward_v2c_to_c2v!(adjoint_v2c, adjoint_c2v, base)
    reference::Matrix{Float32} = reference_backward_v2c_to_c2v(adjoint_c2v, base)
    @test all(isapprox.(adjoint_v2c, reference; atol = 1e-5, rtol = 1e-6))
    # Every edge of a weight-6 check collects exactly five terms.
    ones_in::Matrix{Float32} = ones(Float32, base.nb_neurons_per_layer, 1)
    count_out::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, 1)
    backward_v2c_to_c2v!(count_out, ones_in, base)
    @test all(count_out .== 5.0f0)
end

@testset "backward_v2c_to_c2v! is the adjoint of the forward aggregation" begin
    # The defining identity of an adjoint: <J x, y> = <x, Jᵀ y> for every x, y,
    # with J the forward map A_in -> adj_V2C_C2V * A_in of `v2c_to_c2v!`.
    Random.seed!(6)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 4
    x::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    y::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    forward_of_x::Matrix{Float32} = zeros(Float32, size(x))
    mul!(forward_of_x, base.adj_V2C_C2V, x)
    backward_of_y::Matrix{Float32} = zeros(Float32, size(y))
    backward_v2c_to_c2v!(backward_of_y, y, base)
    @test isapprox(sum(forward_of_x .* y), sum(x .* backward_of_y); rtol = 1e-4)

    # The chain rule it implements, as a number: for L(A_in) = ∑ G ⊙ (J A_in),
    # ∂L/∂A_in[k] is the finite difference in direction k, and equals (Jᵀ G)[k].
    gradient_weights::Matrix{Float64} = Float64.(y)
    loss_of(a_in::Matrix{Float64})::Float64 = sum(gradient_weights .* (Float64.(base.adj_V2C_C2V) * a_in))
    base_point::Matrix{Float64} = Float64.(x)
    step::Float64 = 1e-3
    for k in (1, 17, base.nb_neurons_per_layer)
        perturbed::Matrix{Float64} = copy(base_point)
        perturbed[k, 1] += step
        finite_difference::Float64 = (loss_of(perturbed) - loss_of(base_point)) / step
        @test isapprox(finite_difference, Float64(backward_of_y[k, 1]); atol = 1e-4)
    end

    # The adjacency is symmetric, so the transpose the backward multiplies by is
    # the forward's own matrix. A fact to know, not one the code relies on.
    @test transpose(base.adj_V2C_C2V) == base.adj_V2C_C2V
end

@testset "backward_v2c_to_c2v_gpu equals the CPU version" begin
    Random.seed!(7)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 9
    adjoint_c2v::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    cpu_adjoint_v2c::Matrix{Float32} = zeros(Float32, size(adjoint_c2v))
    backward_v2c_to_c2v!(cpu_adjoint_v2c, adjoint_c2v, base)

    # The state also holds the model's transposed per-layer weights (rule 2̄), so
    # it is built from a model; rule 1̄ reads only the adjacency, so unit weights do.
    state::GPUBackwardState = build_gpu_backward_state(unit_weight_neuralbp(base))
    @test state.nb_neurons == base.nb_neurons_per_layer
    @test size(state.adj_V2C_C2V_transposed) == (base.nb_neurons_per_layer, base.nb_neurons_per_layer)
    device_adjoint_c2v::DeviceMatrix = DeviceMatrix(adjoint_c2v)
    device_adjoint_v2c::DeviceMatrix = backward_v2c_to_c2v_gpu(device_adjoint_c2v, state)
    gpu_adjoint_v2c::Matrix{Float32} = Array(device_adjoint_v2c)
    @test size(gpu_adjoint_v2c) == size(cpu_adjoint_v2c)
    @test all(isapprox.(gpu_adjoint_v2c, cpu_adjoint_v2c; atol = 1e-4, rtol = 1e-5))
    release_gpu_backward_state!(state)
end

# -----------------------------------------------------------------------------
# Rule 2̄: the variable-node rule, layer t back to layer t-1, and its weights.
# -----------------------------------------------------------------------------

function forward_messages_v2c(
    weights_c2v_v2c_layer::Vector{Float32},
    weights_llr::Vector{Float32},
    messages_c2v_previous::Matrix{Float32},
    initial_llrs_batch::Matrix{Float32},
    base::NeuralBPBase
)::Matrix{Float32}
    """
    Rule 2 through the production `c2v_to_v2c_with_weights!`: the raw v2c
    messages of one layer from the previous layer's c2v messages.
    """
    n_samples::Int = size(messages_c2v_previous, 2)
    messages_v2c::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, n_samples)
    activated_magnitudes::Matrix{Float32} = similar(messages_v2c)
    activated_signs::Matrix{Bool} = falses(size(messages_v2c))
    weighted_channel_llrs::Matrix{Float32} = zeros(Float32, base.code_n_bits, n_samples)
    weights_llr_layer::Matrix{Float32} = weights_llr .* ones(Float32, 1, n_samples)
    c2v_to_v2c_with_weights!(
        activated_magnitudes, activated_signs, messages_v2c, messages_c2v_previous,
        weighted_channel_llrs, weights_llr_layer, weights_c2v_v2c_layer, initial_llrs_batch, base
    )
    return messages_v2c
end

function reference_backward_c2v_to_v2c(
    adjoint_messages_v2c::Matrix{Float32},
    weights_c2v_v2c_layer::Vector{Float32},
    base::NeuralBPBase
)::Matrix{Float32}
    """
    Rule 2̄ as derived, pair by pair: every weight k joins the output edge
    rows[k] = (v,c) to the input edge cols[k] = (c',v), so the input edge collects
    weight × the output edge's adjoint.
    """
    reference::Matrix{Float32} = zeros(Float32, size(adjoint_messages_v2c))
    for k in eachindex(weights_c2v_v2c_layer)
        output_edge::Int = base.non_zero_rows_C2V_V2C[k]
        input_edge::Int = base.non_zero_cols_C2V_V2C[k]
        @test base.neuron_to_bits[output_edge] == base.neuron_to_bits[input_edge]
        @test base.neuron_to_checks[output_edge] != base.neuron_to_checks[input_edge]
        reference[input_edge, :] .+= weights_c2v_v2c_layer[k] .* adjoint_messages_v2c[output_edge, :]
    end
    return reference
end

@testset "backward_c2v_to_v2c!: rule 2̄ by hand on the two-check graph" begin
    base::NeuralBPBase = two_check_backward_base()
    # Only bit 3 sits in two checks, so the variable-node rule has exactly two
    # weighted pairs: (c1,v3) <- (c2,v3) and (c2,v3) <- (c1,v3), edges 3 and 4.
    @test base.nb_weights_c2v_v2c == 2
    @test sort(base.non_zero_rows_C2V_V2C) == [3, 4]
    @test sort(base.non_zero_cols_C2V_V2C) == [3, 4]
    weights::Vector{Float32} = Float32[0.5, 2.0]
    adjoint_v2c::Matrix{Float32} = reshape(Float32[1, 2, 3, 4, 5], 5, 1)
    adjoint_c2v_previous::Matrix{Float32} = zeros(Float32, 5, 1)
    backward_c2v_to_v2c!(adjoint_c2v_previous, adjoint_v2c, weights, base)
    # The weight k on the pair (rows[k] <- cols[k]) sends rows[k]'s adjoint to cols[k].
    for k in 1:2
        @test adjoint_c2v_previous[base.non_zero_cols_C2V_V2C[k], 1] == weights[k] * adjoint_v2c[base.non_zero_rows_C2V_V2C[k], 1]
    end
    # Edges 1, 2 and 5 belong to bits of degree one: nothing flows back to them.
    @test adjoint_c2v_previous[[1, 2, 5], 1] == zeros(Float32, 3)
end

@testset "backward_c2v_to_v2c! is the adjoint of the forward rule on the BB code" begin
    Random.seed!(8)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 5
    weights::Vector{Float32} = random_values_around_one([base.nb_weights_c2v_v2c]; scale = 0.3f0)
    weights_llr::Vector{Float32} = random_values_around_one([base.code_n_bits]; scale = 0.3f0)
    messages_c2v_previous::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    initial_llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    adjoint_v2c::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)

    adjoint_c2v_previous::Matrix{Float32} = zeros(Float32, size(adjoint_v2c))
    backward_c2v_to_v2c!(adjoint_c2v_previous, adjoint_v2c, weights, base)
    @test all(isapprox.(adjoint_c2v_previous, reference_backward_c2v_to_v2c(adjoint_v2c, weights, base); atol = 1e-5, rtol = 1e-5))

    # <rule 2 (x), y> = <x, rule 2̄ (y)> for the part of rule 2 that is linear in
    # its input: the channel term is the same whatever x is, so it cancels.
    x::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    zero_input::Matrix{Float32} = zeros(Float32, size(x))
    forward_of_x::Matrix{Float32} = forward_messages_v2c(weights, weights_llr, x, initial_llrs_batch, base) .-
                                    forward_messages_v2c(weights, weights_llr, zero_input, initial_llrs_batch, base)
    backward_of_y::Matrix{Float32} = zeros(Float32, size(x))
    backward_c2v_to_v2c!(backward_of_y, adjoint_v2c, weights, base)
    @test isapprox(sum(Float64.(forward_of_x) .* Float64.(adjoint_v2c)),
                   sum(Float64.(x) .* Float64.(backward_of_y)); rtol = 1e-4, atol = 1e-3)
end

@testset "weight_gradients_c2v_to_v2c!: W̄ and b̄ against finite differences of the forward rule" begin
    Random.seed!(9)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 3
    weights::Vector{Float32} = random_values_around_one([base.nb_weights_c2v_v2c]; scale = 0.3f0)
    weights_llr::Vector{Float32} = random_values_around_one([base.code_n_bits]; scale = 0.3f0)
    messages_c2v_previous::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    initial_llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)
    # A loss that is linear in this rule's output: L = ∑ G ⊙ m_v2c, so ∂L/∂m_v2c = G.
    loss_weights::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    # Summed in Float64. In Float32 the rounding of this 648-term sum, divided by
    # a small step, swamps the difference: simulated, 19 of 30 seeds exceeded
    # 2e-3 at step 1e-2. The rule is linear in every parameter, so the difference
    # quotient is exact at any step, and a large one costs nothing.
    loss_of(w::Vector{Float32}, b::Vector{Float32})::Float64 =
        sum(Float64.(loss_weights) .* Float64.(forward_messages_v2c(w, b, messages_c2v_previous, initial_llrs_batch, base)))

    gradient_weights::Vector{Float32} = zeros(Float32, base.nb_weights_c2v_v2c)
    gradient_llr::Vector{Float32} = zeros(Float32, base.code_n_bits)
    weight_gradients_c2v_to_v2c!(gradient_weights, gradient_llr, loss_weights, messages_c2v_previous, initial_llrs_batch, base)

    # Closed forms, pair by pair and bit by bit.
    for k in (1, 50, base.nb_weights_c2v_v2c)
        expected::Float32 = sum(loss_weights[base.non_zero_rows_C2V_V2C[k], :] .* messages_c2v_previous[base.non_zero_cols_C2V_V2C[k], :])
        @test isapprox(gradient_weights[k], expected; atol = 1e-5, rtol = 1e-5)
    end
    for bit in (1, 36, base.code_n_bits)
        edges_of_bit::Vector{Int} = findall(==(bit), base.neuron_to_bits)
        expected_llr::Float32 = sum(initial_llrs_batch[bit, :] .* vec(sum(loss_weights[edges_of_bit, :]; dims = 1)))
        @test isapprox(gradient_llr[bit], expected_llr; atol = 1e-4, rtol = 1e-5)
    end
    # Finite differences through the production forward. The rule is linear in
    # every parameter, so the difference quotient is exact up to rounding.
    step::Float32 = 0.5f0
    for k in (1, 50, base.nb_weights_c2v_v2c)
        perturbed::Vector{Float32} = copy(weights)
        perturbed[k] += step
        finite_difference::Float64 = (loss_of(perturbed, weights_llr) - loss_of(weights, weights_llr)) / step
        @test isapprox(finite_difference, Float64(gradient_weights[k]); atol = 1e-3)
    end
    for bit in (1, 36, base.code_n_bits)
        perturbed_llr::Vector{Float32} = copy(weights_llr)
        perturbed_llr[bit] += step
        finite_difference_llr::Float64 = (loss_of(weights, perturbed_llr) - loss_of(weights, weights_llr)) / step
        @test isapprox(finite_difference_llr, Float64(gradient_llr[bit]); atol = 1e-3)
    end
    # Accumulation: a second call adds the same again.
    weight_gradients_c2v_to_v2c!(gradient_weights, gradient_llr, loss_weights, messages_c2v_previous, initial_llrs_batch, base)
    @test isapprox(gradient_weights[1], 2 * sum(loss_weights[base.non_zero_rows_C2V_V2C[1], :] .* messages_c2v_previous[base.non_zero_cols_C2V_V2C[1], :]); rtol = 1e-5, atol = 1e-5)
end

@testset "rule 2̄ and its weight gradients: GPU equals CPU" begin
    Random.seed!(10)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 6
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.3f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.3f0),
        weights_c2v_readout = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.3f0),
    )
    layer::Int = 2
    weights_layer::Vector{Float32} =
        bpnn.weights_c2v_v2c[(layer - 1) * base.nb_weights_c2v_v2c + 1 : layer * base.nb_weights_c2v_v2c]
    adjoint_v2c::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    messages_c2v_previous::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    initial_llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)

    cpu_adjoint_c2v_previous::Matrix{Float32} = zeros(Float32, size(adjoint_v2c))
    backward_c2v_to_v2c!(cpu_adjoint_c2v_previous, adjoint_v2c, weights_layer, base)
    cpu_gradient_weights::Vector{Float32} = zeros(Float32, base.nb_weights_c2v_v2c)
    cpu_gradient_llr::Vector{Float32} = zeros(Float32, base.code_n_bits)
    weight_gradients_c2v_to_v2c!(cpu_gradient_weights, cpu_gradient_llr, adjoint_v2c, messages_c2v_previous, initial_llrs_batch, base)

    state::GPUBackwardState = build_gpu_backward_state(bpnn)
    @test length(state.W_msg_transposed_layers) == base.n_layers
    @test state.n_bits == base.code_n_bits
    # The transposed edge <- bit map is the readout's adjacency, as a fact.
    @test transpose(base.adj_initialize_V2C) == base.adj_C2V_readout

    device_adjoint_v2c::DeviceMatrix = DeviceMatrix(adjoint_v2c)
    device_messages_c2v_previous::DeviceMatrix = DeviceMatrix(messages_c2v_previous)
    device_llrs::DeviceMatrix = DeviceMatrix(initial_llrs_batch)

    gpu_adjoint_c2v_previous::Matrix{Float32} = Array(backward_c2v_to_v2c_gpu(device_adjoint_v2c, state, layer))
    @test all(isapprox.(gpu_adjoint_c2v_previous, cpu_adjoint_c2v_previous; atol = 1e-4, rtol = 1e-4))

    # The weight gradient comes out flat, in the weight vector's own order.
    device_gradients::Tuple{DeviceVector, DeviceVector} =
        weight_gradients_c2v_to_v2c_gpu(device_adjoint_v2c, device_messages_c2v_previous, device_llrs, state)
    gpu_gradient_weights::Vector{Float32} = Array(device_gradients[1])
    gpu_gradient_llr::Vector{Float32} = Array(device_gradients[2])
    @test length(gpu_gradient_weights) == base.nb_weights_c2v_v2c
    @test all(isapprox.(gpu_gradient_weights, cpu_gradient_weights; atol = 1e-4, rtol = 1e-4))
    @test all(isapprox.(gpu_gradient_llr, cpu_gradient_llr; atol = 1e-4, rtol = 1e-4))
    release_gpu_backward_state!(state)
end

# -----------------------------------------------------------------------------
# The readout and the two activation conversions, CPU against GPU.
# -----------------------------------------------------------------------------

@testset "readout adjoint and its weight gradients: GPU equals CPU" begin
    Random.seed!(11)
    base::NeuralBPBase = bb_backward_base()
    n_samples::Int = 6
    weights_readout::Vector{Float32} = random_values_around_one([base.nb_weights_c2v_readout]; scale = 0.3f0)
    bpnn::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = random_values_around_one([base.nb_weights_c2v_v2c * base.n_layers]; scale = 0.3f0),
        weights_llrs = random_values_around_one([base.code_n_bits * base.n_layers]; scale = 0.3f0),
        weights_c2v_readout = weights_readout,
    )
    adjoint_posterior::Matrix{Float32} = randn(Float32, base.code_n_bits, n_samples)
    messages_c2v::Matrix{Float32} = randn(Float32, base.nb_neurons_per_layer, n_samples)
    initial_llrs_batch::Matrix{Float32} = repeat(base.initial_llrs, 1, n_samples)

    cpu_adjoint_c2v::Matrix{Float32} = zeros(Float32, base.nb_neurons_per_layer, n_samples)
    backward_readout!(cpu_adjoint_c2v, adjoint_posterior, weights_readout, base)
    cpu_gradient_readout::Vector{Float32} = zeros(Float32, base.nb_weights_c2v_readout)
    cpu_gradient_llr::Vector{Float32} = zeros(Float32, base.code_n_bits)
    weight_gradients_readout!(cpu_gradient_readout, cpu_gradient_llr, adjoint_posterior, messages_c2v, initial_llrs_batch, base)

    state::GPUBackwardState = build_gpu_backward_state(bpnn)
    device_adjoint_posterior::DeviceMatrix = DeviceMatrix(adjoint_posterior)
    device_messages_c2v::DeviceMatrix = DeviceMatrix(messages_c2v)
    device_llrs::DeviceMatrix = DeviceMatrix(initial_llrs_batch)
    gpu_adjoint_c2v::Matrix{Float32} = Array(backward_readout_gpu(device_adjoint_posterior, state))
    @test all(isapprox.(gpu_adjoint_c2v, cpu_adjoint_c2v; atol = 1e-4, rtol = 1e-4))
    device_gradients::Tuple{DeviceVector, DeviceVector} =
        weight_gradients_readout_gpu(device_adjoint_posterior, device_messages_c2v, device_llrs, state)
    @test all(isapprox.(Array(device_gradients[1]), cpu_gradient_readout; atol = 1e-4, rtol = 1e-4))
    @test all(isapprox.(Array(device_gradients[2]), cpu_gradient_llr; atol = 1e-4, rtol = 1e-4))
    release_gpu_backward_state!(state)
end

@testset "activation adjoints: the scalar factors, CPU loops and GPU broadcasts agree" begin
    Random.seed!(12)
    n_neurons::Int = 40
    n_samples::Int = 5
    # Spread over the unclamped range and into both clamps.
    messages_v2c::Matrix{Float32} = 6.0f0 .* randn(Float32, n_neurons, n_samples)
    messages_v2c[1, 1] = 0.0f0          # tanh(0) = 0 < eps: clamped
    messages_v2c[2, 1] = 40.0f0         # tanh(20) rounds to 1: clamped
    adjoint_magnitudes::Matrix{Float32} = randn(Float32, n_neurons, n_samples)
    magnitudes_c2v::Matrix{Float32} = -abs.(3.0f0 .* randn(Float32, n_neurons, n_samples))
    magnitudes_c2v[1, 1] = 0.0f0        # exp(0) = 1 > 1 - eps: clamped
    magnitudes_c2v[2, 1] = -40.0f0      # exp(-40) < eps: clamped
    signs_c2v::Matrix{Bool} = rand(Bool, n_neurons, n_samples)
    adjoint_messages::Matrix{Float32} = randn(Float32, n_neurons, n_samples)

    cpu_adjoint_v2c::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)
    backward_log_tanh_split!(cpu_adjoint_v2c, adjoint_magnitudes, messages_v2c)
    cpu_adjoint_magnitudes::Matrix{Float32} = zeros(Float32, n_neurons, n_samples)
    backward_atanh_exp_signed!(cpu_adjoint_magnitudes, adjoint_messages, magnitudes_c2v, signs_c2v)
    # The loops are the scalar factors, entry by entry; the clamps give exact zeros.
    @test cpu_adjoint_v2c[1, 1] == 0.0f0 && cpu_adjoint_v2c[2, 1] == 0.0f0
    @test cpu_adjoint_magnitudes[1, 1] == 0.0f0 && cpu_adjoint_magnitudes[2, 1] == 0.0f0
    @test cpu_adjoint_v2c[3, 2] == log_tanh_split_adjoint(adjoint_magnitudes[3, 2], messages_v2c[3, 2])
    @test cpu_adjoint_magnitudes[3, 2] == atanh_exp_signed_adjoint(adjoint_messages[3, 2], magnitudes_c2v[3, 2], signs_c2v[3, 2])

    gpu_adjoint_v2c::Matrix{Float32} = Array(backward_log_tanh_split_gpu(DeviceMatrix(adjoint_magnitudes), DeviceMatrix(messages_v2c)))
    gpu_adjoint_magnitudes::Matrix{Float32} = Array(backward_atanh_exp_signed_gpu(
        DeviceMatrix(adjoint_messages), DeviceMatrix(magnitudes_c2v), DeviceMatrix(Float32.(signs_c2v))))
    @test all(isapprox.(gpu_adjoint_v2c, cpu_adjoint_v2c; atol = 1e-5, rtol = 1e-4))
    @test all(isapprox.(gpu_adjoint_magnitudes, cpu_adjoint_magnitudes; atol = 1e-5, rtol = 1e-4))
    @test gpu_adjoint_v2c[1, 1] == 0.0f0 && gpu_adjoint_magnitudes[2, 1] == 0.0f0
end

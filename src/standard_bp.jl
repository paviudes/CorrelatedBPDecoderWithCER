# =============================================================================
# Standard belief propagation by setting all the weights in Neural BP to 1.0f0
# =============================================================================
#
function unit_weight_neuralbp(
    base::NeuralBPBase;
    coupling_scale::Float32 = 1.0f0,
    coupling_schedule_layer::Float32 = Float32(base.n_layers),
    coupling_schedule_width::Float32 = 3.0f0
)::NachmaniNeuralBP
    """
    Wrap a `NeuralBPBase` in a `NachmaniNeuralBP` whose every weight is 1.0f0,
    which makes its forward pass standard belief propagation.

    The layer count comes from `base.n_layers`, so for standard BP that field
    carries the number of BP iterations. `standard_bp_experiments.jl` accepts it
    under the name `--n_iterations_BP` for exactly that reason.

    `coupling_scale` is α for the enriched check node (`base.check_node_kind ==
    CHECK_NODE_ENRICHED`): a fixed hyperparameter here, where nothing is
    learned. α = 1 is the exact Bayesian message for the pairwise prior; α = 0
    reproduces the standard rule. Ignored by the standard check node.

    Stored internally as its logit, so an α of exactly 0 or 1 is nudged inside
    (0, 1) by `COUPLING_SCALE_LINK_MARGIN` = 1e-6 — far below any decodable
    difference. For an exact α = 0, use `check_node = "tanh"`.

    `coupling_schedule_layer` and `coupling_schedule_width` are the step
    schedule's (T₀, w) in layers, read only when `base.coupling_schedule_kind ==
    COUPLING_SCHEDULE_STEP`. Fixed here as well: this is how the classical scan
    over (T₀, w) is run, with nothing trained.
    """
    n_weights_c2v_v2c::Int = base.nb_weights_c2v_v2c * base.n_layers
    n_weights_llrs::Int = base.code_n_bits * base.n_layers
    n_weights_c2v_readout::Int = base.nb_weights_c2v_readout
    unit_weight_network::NachmaniNeuralBP = NachmaniNeuralBP(
        base;
        weights_c2v_v2c = ones(Float32, n_weights_c2v_v2c),
        weights_llrs = ones(Float32, n_weights_llrs),
        weights_c2v_readout = ones(Float32, n_weights_c2v_readout),
        coupling_scale = coupling_scale,
        coupling_schedule_layer = coupling_schedule_layer,
        coupling_schedule_width = coupling_schedule_width
    )
    return unit_weight_network
end

function standard_bp_test_predictions(
    base::NeuralBPBase,
    test_errors_file::String;
    coupling_scale::Float32 = 1.0f0,
    coupling_schedule_layer::Float32 = Float32(base.n_layers),
    coupling_schedule_width::Float32 = 3.0f0,
    batch_size::Int = 0,
    gpu_memory::AbstractString = "",
    diagnose::Bool = false,
)::Union{BitVector, NamedTuple}
    """
    Decode the test set with standard BP and score the predictions.

    Same arguments and same return contract as `neuralbp_test_predictions`, with
    a `NeuralBPBase` in place of a trained `NeuralBP`:

      `diagnose = false`  returns the `BitVector` of per-sample correctness.
      `diagnose = true`   returns the `count_syndrome_satisfactions` NamedTuple,
                          splitting failures into coset and convergence failures.

    Both cost exactly one forward pass. `batch_size` / `gpu_memory` are resolved
    by the same `resolve_prediction_batch_size` the neural path uses, so the GPU
    is used whenever `USE_GPU` is set, with no extra plumbing here.

    `coupling_scale`, `coupling_schedule_layer` and `coupling_schedule_width`
    are passed to `unit_weight_neuralbp`; see there.
    """
    unit_weight_network::NachmaniNeuralBP = unit_weight_neuralbp(
        base;
        coupling_scale = coupling_scale,
        coupling_schedule_layer = coupling_schedule_layer,
        coupling_schedule_width = coupling_schedule_width
    )
    prediction_outcome::Union{BitVector, NamedTuple} = neuralbp_test_predictions(
        unit_weight_network,
        test_errors_file;
        batch_size = batch_size,
        gpu_memory = gpu_memory,
        diagnose = diagnose
    )
    return prediction_outcome
end

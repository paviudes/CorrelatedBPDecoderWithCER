# =============================================================================
#   Which samples of the training pool to train on
# =============================================================================
# `filter_training_samples` turns the pool read from the training file into the
# set the batches are drawn from. Two hyperparameters control it:
#
#   training_samples          size of that set; 0 = the whole pool
#   failure_weight_boundary   λ of a Poisson over the ERROR WEIGHT; 0 = no weighting
#
# With λ > 0 the set is drawn so its error weights follow Poisson(λ), restricted
# to the weights the pool actually contains. Within one weight the draw is
# uniform over the pool, so the samples are still the error model's own (the
# CER correlations included) — only the weight distribution is reshaped. This
# is the same distribution as rejection from the pool with acceptance
# ∝ Pois(λ; w) / p_pool(w), without the rejections.
#
# Measured 2026-10-09 (CER, 50 layers): weights 0–2 never fail, weight 3 fails
# 0.3% of the time, weights 4–8 carry ~80% of the failures. A uniform draw puts
# 0.3% of batch samples on a failure; Poisson(5) puts 3.4%.

# Lowest error weight at which the decoder was measured to fail at a non-trivial
# rate (≥ 1.7%); only used to summarise a selection, never to make one.
const FAILURE_BAND_MINIMUM_WEIGHT = 4

function error_weights(expected_recoveries::AbstractMatrix{Bool})::Vector{Int}
    """
    Hamming weight of every error pattern (one per column).
    """
    weights::Vector{Int} = vec(sum(expected_recoveries; dims = 1))
    return weights
end

function log_poisson_probability(mean_weight::Float64, weight::Int)::Float64
    """
    log( e^{-λ} λ^w / w! ). log w! is summed directly; w never exceeds n_bits.
    """
    log_factorial::Float64 = 0.0
    for k in 2:weight
        log_factorial += log(k)
    end
    log_probability::Float64 = -mean_weight + weight * log(mean_weight) - log_factorial
    return log_probability
end

function poisson_over_present_weights(mean_weight::Float64, present_weights::Vector{Int})::Vector{Float64}
    """
    Pois(λ; w) for each weight in `present_weights`, renormalised over them. Weights
    the pool does not contain get no mass, so a draw can never land on an empty bin.
    """
    log_probabilities::Vector{Float64} = [log_poisson_probability(mean_weight, w) for w in present_weights]
    largest::Float64 = maximum(log_probabilities)
    probabilities::Vector{Float64} = exp.(log_probabilities .- largest)
    probabilities ./= sum(probabilities)
    return probabilities
end

function filter_training_samples(
    expected_recoveries::AbstractMatrix{Bool},
    training_samples::Int,
    failure_weight_boundary::Float64;
    rng::AbstractRNG = Random.default_rng()
)::Vector{Int}
    """
    Column indices of the training set to draw batches from.

      boundary = 0, samples = 0   every column, in order (no RNG consumed)
      boundary = 0, samples = N   a uniform subset of N columns without replacement
                                  (the whole pool if N ≥ its size)
      boundary = λ > 0            N columns with weights ~ Poisson(λ) over the
                                  weights present in the pool, uniform within a
                                  weight, WITH replacement; N must be > 0

    With replacement so the set always has exactly N entries; the rare heavy
    weights repeat when N × Pois(λ; w) exceeds the pool's count at w. The draw
    comes from `rng` (the global RNG by default), so under `seed` the set is
    reproducible, and it is taken after the model is initialised, so two runs
    at one seed share their initial weights whatever this chooses.
    """
    n_pool::Int = size(expected_recoveries, 2)
    if n_pool == 0
        throw(ArgumentError("filter_training_samples: the training pool is empty."))
    end
    if training_samples < 0
        throw(ArgumentError("training_samples = $(training_samples) must be ≥ 0 (0 = the whole pool)."))
    end
    if !(failure_weight_boundary >= 0.0)
        throw(ArgumentError("failure_weight_boundary = $(failure_weight_boundary) must be ≥ 0 (0 = no weighting)."))
    end

    if failure_weight_boundary == 0.0
        if training_samples == 0 || training_samples >= n_pool
            every_column::Vector{Int} = collect(1:n_pool)
            return every_column
        end
        uniform_subset::Vector{Int} = randperm(rng, n_pool)[1:training_samples]
        return uniform_subset
    end

    if training_samples == 0
        throw(ArgumentError(
            "failure_weight_boundary = $(failure_weight_boundary) needs training_samples > 0: " *
            "a weighted draw has no natural size."))
    end

    weights::Vector{Int} = error_weights(expected_recoveries)
    columns_by_weight::Dict{Int, Vector{Int}} = Dict{Int, Vector{Int}}()
    for (column, weight) in enumerate(weights)
        push!(get!(columns_by_weight, weight, Int[]), column)
    end
    present_weights::Vector{Int} = sort(collect(keys(columns_by_weight)))
    probabilities::Vector{Float64} = poisson_over_present_weights(failure_weight_boundary, present_weights)
    cumulative::Vector{Float64} = cumsum(probabilities)

    selected::Vector{Int} = Vector{Int}(undef, training_samples)
    for i in 1:training_samples
        draw::Float64 = rand(rng)
        bin_position::Int = min(searchsortedfirst(cumulative, draw), length(present_weights))
        bin::Vector{Int} = columns_by_weight[present_weights[bin_position]]
        selected[i] = bin[rand(rng, 1:length(bin))]
    end
    return selected
end

function describe_training_selection(selected::Vector{Int}, pool_weights::Vector{Int})::String
    """
    One line for the console and the debug log: how the selected set's weights
    compare with the pool's.
    """
    selected_weights::Vector{Int} = pool_weights[selected]
    n_distinct::Int = length(unique(selected))
    mean_selected::Float64 = sum(selected_weights) / length(selected_weights)
    mean_pool::Float64 = sum(pool_weights) / length(pool_weights)
    band_selected::Float64 = count(w -> w >= FAILURE_BAND_MINIMUM_WEIGHT, selected_weights) / length(selected_weights)
    band_pool::Float64 = count(w -> w >= FAILURE_BAND_MINIMUM_WEIGHT, pool_weights) / length(pool_weights)
    summary::String =
        "training set: $(length(selected)) draws, $(n_distinct) distinct, of a pool of $(length(pool_weights)); " *
        "mean error weight $(round(mean_selected; digits = 2)) (pool $(round(mean_pool; digits = 2))); " *
        "weight ≥ $(FAILURE_BAND_MINIMUM_WEIGHT): $(round(100 * band_selected; digits = 1))% " *
        "(pool $(round(100 * band_pool; digits = 1))%)"
    return summary
end

function write_training_selection(path::String, selected::Vector{Int}, pool_weights::Vector{Int})::Nothing
    """
    The selected set as a two-column CSV (pool column, error weight), in draw
    order, so a run's training set can be inspected or re-used without re-drawing.
    """
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "pool_column,error_weight")
        for column in selected
            println(io, "$(column),$(pool_weights[column])")
        end
    end
    return nothing
end

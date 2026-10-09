using CorrelatedBPDecoderWithCER
using Test
using Random

# =============================================================================
# Tests for src/sample_selection.jl: which samples of the training pool the
# batches are drawn from. The Poisson numbers below were checked against a
# Python reference (math.log / math.exp, 2026-10-09).
#
# Run from `tests/`:  julia --project="./../" -e 'include("test_sample_selection.jl")'
# =============================================================================

function pool_with_weights(column_weights::Vector{Int}; n_bits::Int = 12)::Matrix{Bool}
    """
    A pool whose column j has Hamming weight column_weights[j]: the first
    column_weights[j] bits set, the rest clear.
    """
    pool::Matrix{Bool} = zeros(Bool, n_bits, length(column_weights))
    for (column, weight) in enumerate(column_weights)
        pool[1:weight, column] .= true
    end
    return pool
end

@testset "error_weights is the column Hamming weight" begin
    pool::Matrix{Bool} = pool_with_weights([0, 1, 3, 12, 2])
    @test error_weights(pool) == [0, 1, 3, 12, 2]
    @test error_weights(falses(4, 3)) == [0, 0, 0]
end

@testset "log_poisson_probability" begin
    # Pois(5; 0) = e^-5, Pois(5; 5) = e^-5 5^5 / 120
    @test log_poisson_probability(5.0, 0) ≈ -5.0
    @test log_poisson_probability(5.0, 5) ≈ -1.7403021806115442
    @test exp(log_poisson_probability(5.0, 5)) ≈ 0.17546736976785068
    @test exp(log_poisson_probability(2.5, 3)) ≈ exp(-2.5) * 2.5^3 / 6
end

@testset "poisson_over_present_weights renormalises over the weights the pool has" begin
    # Only weights 0 and 2 present at λ = 2: Pois(2;0) = e^-2, Pois(2;2) = 2e^-2 -> 1/3, 2/3.
    probabilities::Vector{Float64} = poisson_over_present_weights(2.0, [0, 2])
    @test probabilities ≈ [1 / 3, 2 / 3]
    @test sum(poisson_over_present_weights(5.0, [0, 3, 7])) ≈ 1.0
    @test poisson_over_present_weights(5.0, [0, 3, 7]) ≈ [0.02679, 0.55802, 0.41519] atol = 1e-5
    # A single present weight gets everything, whatever λ.
    @test poisson_over_present_weights(40.0, [1]) == [1.0]
end

@testset "defaults return the whole pool, in order, without touching the RNG" begin
    pool::Matrix{Bool} = pool_with_weights([0, 1, 3, 2, 0, 5])
    rng::Xoshiro = Xoshiro(11)
    @test filter_training_samples(pool, 0, 0.0; rng = rng) == collect(1:6)
    # The RNG was not consumed: its next draw is the first draw of a fresh copy.
    @test rand(rng) == rand(Xoshiro(11))
    # training_samples at or above the pool size is the whole pool too.
    @test filter_training_samples(pool, 6, 0.0; rng = Xoshiro(1)) == collect(1:6)
    @test filter_training_samples(pool, 600, 0.0; rng = Xoshiro(1)) == collect(1:6)
end

@testset "boundary 0: a uniform subset without replacement" begin
    pool::Matrix{Bool} = pool_with_weights(collect(0:49); n_bits = 50)
    subset::Vector{Int} = filter_training_samples(pool, 20, 0.0; rng = Xoshiro(3))
    @test length(subset) == 20
    @test length(unique(subset)) == 20
    @test all(1 .<= subset .<= 50)
    # Reproducible under the seed, different under another.
    @test filter_training_samples(pool, 20, 0.0; rng = Xoshiro(3)) == subset
    @test filter_training_samples(pool, 20, 0.0; rng = Xoshiro(4)) != subset
end

@testset "bad inputs are refused" begin
    pool::Matrix{Bool} = pool_with_weights([0, 1, 2])
    @test_throws ArgumentError filter_training_samples(falses(4, 0), 0, 0.0)
    @test_throws ArgumentError filter_training_samples(pool, -1, 0.0)
    @test_throws ArgumentError filter_training_samples(pool, 2, -0.5)
    # A weighted draw has no natural size.
    @test_throws ArgumentError filter_training_samples(pool, 0, 3.0)
end

@testset "boundary λ > 0: exactly N draws, only from present weights, with replacement" begin
    # Weights 0, 3 and 7 only (λ = 5 would put most of its mass on 4-6, which the
    # pool lacks); 5 columns at each weight.
    column_weights::Vector{Int} = repeat([0, 3, 7], inner = 5)
    pool::Matrix{Bool} = pool_with_weights(column_weights)
    selected::Vector{Int} = filter_training_samples(pool, 400, 5.0; rng = Xoshiro(5))
    @test length(selected) == 400
    @test all(1 .<= selected .<= 15)
    # The three present weights get the Poisson's mass renormalised over THEM
    # (0.027 / 0.558 / 0.415), not Pois(5; w) itself, which sums to 0.17 here.
    # 400 draws: binomial sd <= 0.025, so 0.12 is a ~5σ tolerance.
    for (weight, target) in zip((0, 3, 7), (0.02679, 0.55802, 0.41519))
        @test abs(count(==(weight), column_weights[selected]) / 400 - target) < 0.12
    end
    # 400 draws from 15 columns: with replacement by necessity.
    @test length(unique(selected)) <= 15
    # Reproducible under the seed.
    @test filter_training_samples(pool, 400, 5.0; rng = Xoshiro(5)) == selected

    # A heavy weight with two columns at a λ that wants nearly all of its draws
    # there: the set is still exactly N long, so N is guaranteed, and the two
    # columns simply repeat.
    thin_pool::Matrix{Bool} = pool_with_weights(vcat(fill(0, 10), fill(9, 2)))
    thin_selected::Vector{Int} = filter_training_samples(thin_pool, 100, 9.0; rng = Xoshiro(6))
    @test length(thin_selected) == 100
    @test count(column -> column in (11, 12), thin_selected) >= 90
end

@testset "the drawn weights follow the Poisson restricted to the present weights" begin
    # 50 columns at each weight 0..10, λ = 4, 20,000 draws. The truncated Poisson
    # puts at most 0.196 on a weight, so the binomial sd of a frequency is
    # <= 0.0028 and 0.015 is a 5σ tolerance.
    column_weights::Vector{Int} = repeat(collect(0:10), inner = 50)
    pool::Matrix{Bool} = pool_with_weights(column_weights)
    n_draws::Int = 20000
    selected::Vector{Int} = filter_training_samples(pool, n_draws, 4.0; rng = Xoshiro(7))
    drawn_weights::Vector{Int} = column_weights[selected]
    expected::Vector{Float64} = poisson_over_present_weights(4.0, collect(0:10))
    @test expected ≈ [0.01837, 0.07347, 0.14694, 0.19592, 0.19592, 0.15674,
                      0.10449, 0.05971, 0.02985, 0.01327, 0.00531] atol = 1e-5
    for (position, weight) in enumerate(0:10)
        frequency::Float64 = count(==(weight), drawn_weights) / n_draws
        @test abs(frequency - expected[position]) < 0.015
    end
    # Uniform within a weight: every column at weight 4 (~3,900 draws over 50
    # columns) is hit, and no column is hit more than 2.5x its expectation.
    columns_at_four::Vector{Int} = findall(==(4), column_weights)
    hits::Vector{Int} = [count(==(column), selected) for column in columns_at_four]
    @test all(hits .> 0)
    @test maximum(hits) < 2.5 * n_draws * expected[5] / 50
    # The mean weight lands on the target's, not the pool's.
    @test abs(sum(drawn_weights) / n_draws - sum((0:10) .* expected)) < 0.1
end

@testset "the global RNG default is what `seed` reproduces" begin
    pool::Matrix{Bool} = pool_with_weights(repeat(collect(0:8), inner = 20))
    Random.seed!(2026)
    first_draw::Vector{Int} = filter_training_samples(pool, 300, 3.0)
    Random.seed!(2026)
    second_draw::Vector{Int} = filter_training_samples(pool, 300, 3.0)
    @test first_draw == second_draw
    Random.seed!(2027)
    @test filter_training_samples(pool, 300, 3.0) != first_draw
end

@testset "describe_training_selection and write_training_selection" begin
    column_weights::Vector{Int} = [0, 0, 1, 4, 5, 7]
    pool_weights::Vector{Int} = error_weights(pool_with_weights(column_weights))
    selected::Vector{Int} = [4, 5, 5, 6]
    summary::String = describe_training_selection(selected, pool_weights)
    @test occursin("4 draws", summary)
    @test occursin("3 distinct", summary)
    @test occursin("pool of 6", summary)
    # mean weight of the draws: (4 + 5 + 5 + 7) / 4 = 5.25; of the pool: 17/6 = 2.83
    @test occursin("5.25", summary)
    @test occursin("2.83", summary)
    # all four draws are at weight >= FAILURE_BAND_MINIMUM_WEIGHT (4); 3 of 6 pool columns are.
    @test occursin("100.0%", summary)
    @test occursin("50.0%", summary)

    mktempdir() do directory
        path::String = joinpath(directory, "logs", "training_selection_test.csv")
        write_training_selection(path, selected, pool_weights)
        lines::Vector{String} = readlines(path)
        @test lines[1] == "pool_column,error_weight"
        @test lines[2:end] == ["4,4", "5,5", "5,5", "6,7"]
    end
end

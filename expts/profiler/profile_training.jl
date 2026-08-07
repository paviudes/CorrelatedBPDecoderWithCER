# profile_training.jl
# ----------------------------------------------------------------------------
# Profile a short Neural BP training run, using the same CLI conventions as
# `neural_bp_experiments.jl` (so all the data paths, file conventions, and
# hyperparameter loading match production exactly).
#
# Differences vs. neural_bp_experiments.jl:
#   - Skips the testing phase. Only profiles `train_neuralbp_enzyme!`.
#   - Runs a small JIT-warmup training call first, then profiles the real one.
#     This keeps JIT compilation time out of the profile.
#   - Saves four text reports in the directory you run from:
#       profile_flat.txt        — leaf functions ranked by self-time
#       profile_tree.txt        — call tree with self+inclusive time
#       profile_allocations.txt — top allocation sites by total bytes
#       profile_timings.txt     — wall-clock summary
#
# Example invocation (from the expts/ directory, same flags as the real
# experiment):
#
#   julia --project="./../" profile_training.jl \
#       --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
#       --n_hidden_layers 20 \
#       --correlation_strengths_file correlated_weights_p_0.010_q_0.001_s_1.txt \
#       --train training_errors.txt \
#       --hyperparams default_hyperparams.json
#
# To keep the profile short, you can:
#   - Use `--n_samples 200` to subset training data, or
#   - Point `--hyperparams` at a JSON/TOML with `n_epochs = 1`.
# ----------------------------------------------------------------------------

using Profile
using Printf
using DelimitedFiles
using DataFrames
using CorrelatedBPDecoderWithCER

if abspath(PROGRAM_FILE) == @__FILE__

    if length(ARGS) == 0
        println("No command-line arguments provided. Use the same flags as neural_bp_experiments.jl.")
        println("Example:")
        println("  julia --project=\"./../\" profile_training.jl \\")
        println("      --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \\")
        println("      --n_hidden_layers 20 \\")
        println("      --correlation_strengths_file correlated_weights_p_0.010_q_0.001_s_1.txt \\")
        println("      --train training_errors.txt \\")
        println("      --hyperparams default_hyperparams.json")
        exit(1)
    end

    # ----------------------------------------------------------------------------
    # 1. Parse CLI args, identical to neural_bp_experiments.jl.
    # ----------------------------------------------------------------------------
    args_dict = parse_command_line_args_NN(; prefix = "./../data")

    prefix                     = "./../data/$(args_dict["codename"])"
    parity_check_matrix_file   = "$(prefix)/code/HZ.txt"
    logicals_file              = "$(prefix)/code/LZ.txt"
    correlation_strengths_file = "$(prefix)/correlated_weights/$(args_dict["cer_data"])"
    training_errors_file       = "$(prefix)/training_data/$(args_dict["train"])"
    n_hidden_layers            = args_dict["n_hidden_layers"]
    n_samples_override         = args_dict["n_samples"]   # -1 = use all
    is_debug                   = args_dict["debug"]
    is_quiet                   = args_dict["quiet"]

    hyperparams_file = args_dict["hyperparams"]
    hyperparams      = parse_hyper_parameters(hyperparams_file; prefix = prefix)
    n_epochs         = hyperparams["n_epochs"]
    batch_size       = hyperparams["batch_size"]

    println("=" ^ 70)
    println("Profile configuration")
    println("  codename:           $(args_dict["codename"])")
    println("  n_hidden_layers:    $(n_hidden_layers)")
    println("  n_epochs:           $(n_epochs)")
    println("  batch_size:         $(batch_size)")
    println("  training file:      $(training_errors_file)")
    println("  n_samples override: $(n_samples_override == -1 ? "all" : string(n_samples_override))")
    println("=" ^ 70)

    # ----------------------------------------------------------------------------
    # 2. Build the base BP model and the Neural BP wrapper, identical to
    #    neural_bp_experiments.jl.
    # ----------------------------------------------------------------------------
    base = load_base_BP_model(
        parity_check_matrix_file,
        logicals_file,
        n_hidden_layers;
        cer_data_file = correlation_strengths_file,
    )

    initial_conditions = Dict{String, Vector{Float32}}(
        "weights_c2v_v2c"     => random_values_around_one(
            [base.nb_weights_c2v_v2c * base.n_layers];
            scale = hyperparams["initial_conditions_scale"],
        ),
        "weights_llrs"        => random_values_around_one(
            [base.code_n_bits * base.n_layers];
            scale = hyperparams["initial_conditions_scale"],
        ),
        "weights_c2v_readout" => random_values_around_one(
            [base.nb_weights_c2v_readout];
            scale = hyperparams["initial_conditions_scale"],
        ),
    )

    bpnn = NachmaniNeuralBP(
        base;
        weights_c2v_v2c     = initial_conditions["weights_c2v_v2c"],
        weights_llrs        = initial_conditions["weights_llrs"],
        weights_c2v_readout = initial_conditions["weights_c2v_readout"],
    )

    # ----------------------------------------------------------------------------
    # 3. Read the training data from file.
    # ----------------------------------------------------------------------------
    expected_recoveries = convert.(Bool, readdlm(training_errors_file, Int))

    # Optional subset for fast profiling runs.
    if n_samples_override > 0 && n_samples_override < size(expected_recoveries, 2)
        expected_recoveries = expected_recoveries[:, 1:n_samples_override]
        println("Using first $(n_samples_override) training samples for the profile.")
    end

    n_samples = size(expected_recoveries, 2)
    training_syndromes = convert.(Bool, mod.(base.parity_check_matrix * expected_recoveries, 2))

    training_source = splitext(basename(training_errors_file))[1]

    # ----------------------------------------------------------------------------
    # 4. JIT warmup. The very first call to `train_neuralbp_enzyme!` compiles
    #    the entire Enzyme reverse-mode path — slow and not what we want to
    #    measure. Run a tiny throwaway training to amortize that cost.
    # ----------------------------------------------------------------------------
    println()
    println("=== Warmup pass (JIT compilation) ===")

    let
        # Throwaway bpnn so the warmup doesn't perturb the weights we'll profile against.
        warmup_bpnn = NachmaniNeuralBP(
            base;
            weights_c2v_v2c     = copy(initial_conditions["weights_c2v_v2c"]),
            weights_llrs        = copy(initial_conditions["weights_llrs"]),
            weights_c2v_readout = copy(initial_conditions["weights_c2v_readout"]),
        )
        warmup_n   = min(2 * batch_size, n_samples)
        warmup_syndromes = training_syndromes[:,  1:warmup_n]
        warmup_errors = expected_recoveries[:, 1:warmup_n]

        # Run the warmup with n_epochs=1 regardless of what hyperparams says,
        # by deep-copying and overriding.
        warmup_hp = deepcopy(hyperparams)
        warmup_hp["n_epochs"] = 1

        global warmup_t = @elapsed train_neuralbp_enzyme!(
            warmup_bpnn,
            warmup_syndromes,
            warmup_errors,
            warmup_hp;
            debugging_logfile = tempname(),
            is_debug          = false,
            is_quiet          = true,
        )
    end
    @printf("warmup time: %.3f s (this is JIT-heavy, not representative)\n", warmup_t)

    # ----------------------------------------------------------------------------
    # 5. Profiled training run — this is the one whose stats we'll analyze.
    # ----------------------------------------------------------------------------
    println()
    println("=== Profiling pass ===")

    Profile.clear()
    Profile.init(n = 10_000_000, delay = 0.001)   # 1 ms sample rate

    debugging_logfile = "$(prefix)/logs/debugging_$(training_source)_profile"
    profiled_t = @elapsed @profile train_neuralbp_enzyme!(
        bpnn,
        training_syndromes,
        expected_recoveries,
        hyperparams;
        debugging_logfile = debugging_logfile,
        is_debug          = is_debug,
        is_quiet          = is_quiet,
    )
    @printf("profiled training time: %.3f s\n", profiled_t)

    # ----------------------------------------------------------------------------
    # 6. Write profile reports.
    # ----------------------------------------------------------------------------

    # --- 6a. Flat profile (leaf hot spots) ---
    open("$(prefix)/logs/profile_flat.txt", "w") do io
        @printf(io, "# Flat profile — leaf functions ranked by self-time\n")
        @printf(io, "# codename=%s  n_layers=%d  batch=%d  n_samples=%d  n_epochs=%d\n",
                args_dict["codename"], n_hidden_layers, batch_size, n_samples, n_epochs)
        @printf(io, "# wall-clock (profiled run, excluding warmup): %.3f s\n\n",
                profiled_t)
        Profile.print(io; format = :flat, sortedby = :count, mincount = 20)
    end

    # --- 6b. Tree profile (call hierarchy) ---
    open("$(prefix)/logs/profile_tree.txt", "w") do io
        @printf(io, "# Tree profile — call hierarchy with sample counts\n")
        @printf(io, "# codename=%s  n_layers=%d  batch=%d  n_samples=%d  n_epochs=%d\n",
                args_dict["codename"], n_hidden_layers, batch_size, n_samples, n_epochs)
        @printf(io, "# wall-clock (profiled run): %.3f s\n\n", profiled_t)
        Profile.print(io; format = :tree, mincount = 50)
    end

    # --- 6c. Allocation profile (separate, shorter run) ---
    println()
    println("=== Allocation profiling (short separate run) ===")

    Profile.Allocs.clear()
    let
        alloc_bpnn = NachmaniNeuralBP(
            base;
            weights_c2v_v2c     = copy(bpnn.weights_c2v_v2c),
            weights_llrs        = copy(bpnn.weights_llrs),
            weights_c2v_readout = copy(bpnn.weights_c2v_readout),
        )
        alloc_n   = min(4 * batch_size, n_samples)
        alloc_syndromes = training_syndromes[:,  1:alloc_n]
        alloc_errors = expected_recoveries[:, 1:alloc_n]
        alloc_hp  = deepcopy(hyperparams)
        alloc_hp["n_epochs"] = 1

        Profile.Allocs.@profile sample_rate = 0.05 train_neuralbp_enzyme!(
            alloc_bpnn,
            alloc_syndromes,
            alloc_errors,
            alloc_hp;
            debugging_logfile = tempname(),
            is_debug          = false,
            is_quiet          = true,
        )
    end

    open("$(prefix)/logs/profile_allocations.txt", "w") do io
        @printf(io, "# Top allocation sites, aggregated by source location\n")
        @printf(io, "# codename=%s  n_layers=%d  batch=%d  (4 batches profiled)\n\n",
                args_dict["codename"], n_hidden_layers, batch_size)

        allocs = Profile.Allocs.fetch()
        sites  = Dict{String, Tuple{Int, Int}}()   # site → (total_bytes, count)
        for a in allocs.allocs
            key = "<unknown>"
            for sf in a.stacktrace
                s = string(sf)
                # Skip stdlib internals to surface user-code call sites.
                if !occursin("Base.", s) && !occursin("Core.", s) && !occursin("./Base/", s)
                    key = s
                    break
                end
            end
            prev = get(sites, key, (0, 0))
            sites[key] = (prev[1] + a.size, prev[2] + 1)
        end

        sorted = sort(collect(sites); by = x -> x[2][1], rev = true)
        @printf(io, "%-14s %-10s %s\n", "total_bytes", "count", "site")
        for (site, (bytes, count)) in first(sorted, 40)
            @printf(io, "%-14d %-10d %s\n", bytes, count, site)
        end
    end

    # --- 6d. Coarse timing summary ---
    open("$(prefix)/logs/profile_timings.txt", "w") do io
        n_batches = cld(n_samples, batch_size) * n_epochs
        @printf(io, "# Wall-clock summary\n\n")
        @printf(io, "config:\n")
        @printf(io, "  codename:               %s\n", args_dict["codename"])
        @printf(io, "  n_layers:               %d\n", n_hidden_layers)
        @printf(io, "  n_epochs:               %d\n", n_epochs)
        @printf(io, "  batch_size:             %d\n", batch_size)
        @printf(io, "  n_training_samples:     %d\n", n_samples)
        @printf(io, "  total_batches:          %d\n", n_batches)
        @printf(io, "  weights:                %d (c2v_v2c) + %d (llrs) + %d (readout) = %d\n",
                length(bpnn.weights_c2v_v2c),
                length(bpnn.weights_llrs),
                length(bpnn.weights_c2v_readout),
                length(bpnn.weights_c2v_v2c) + length(bpnn.weights_llrs) + length(bpnn.weights_c2v_readout))
        @printf(io, "\n")
        @printf(io, "timings:\n")
        @printf(io, "  warmup (JIT-heavy):       %.3f s\n", warmup_t)
        @printf(io, "  profiled training:        %.3f s\n", profiled_t)
        @printf(io, "  ⇒ avg per-batch:          %.4f s\n", profiled_t / n_batches)
        @printf(io, "  ⇒ extrapolated to 10⁴ × n_epochs (matched batch):\n")
        @printf(io, "      %.1f s (≈ %.1f min)\n",
                profiled_t / n_samples * 10_000,
                profiled_t / n_samples * 10_000 / 60)
    end

    println()
    println("=" ^ 70)
    println("Done. Reports written:")
    println("  $(prefix)/logs/profile_flat.txt        (share this first)")
    println("  $(prefix)/logs/profile_timings.txt     (share this)")
    println("  $(prefix)/logs/profile_tree.txt        (share first ~200 lines if asked)")
    println("  $(prefix)/logs/profile_allocations.txt (share this)")
    println("=" ^ 70)

end

#=
To run this file, copy and paste this command into the terminal (from the expts/ directory):
julia --project="./../" profile_training.jl \
  --codename 72q_BB_p_0.010_q_0.001_std_0.01_data \
  --n_hidden_layers 20 \
  --n_samples 200 \
  --hyperparams default_hyperparams.toml \
  --correlation_strengths_file correlated_weights_p_0.01_q_0.001_s_1.txt \
  --train train_ballistic_p_0.01_q_0.001_s_1.txt
=#
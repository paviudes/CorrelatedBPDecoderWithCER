module CorrelatedBPDecoderWithCER

#===============================================================================
                                    IMPORTS
===============================================================================#

# Core Julia packages
using LinearAlgebra     # Matrix operations, decompositions, linear algebra functions
using SparseArrays      # Sparse matrix representations for efficiency
using Base.Threads      # Multi-threading for parallel computations
using Random            # Random number generation for error sampling and initialization
using JSON              # Reading/writing JSON configuration and data files
using TOML              # TOML file parsing for configuration management
using DelimitedFiles    # File I/O for CSV-like data formats
using CSV               # CSV file reading and writing
using ProgressMeter     # Progress bars for training and other long operations
using ArgParse          # Command-line argument parsing for flexible execution
using Printf            # Formatted printing for error model parameters

# Set BLAS to use the same number of threads as Julia
BLAS.set_num_threads(Threads.nthreads())

# GPU acceleration.
#
# Two concerns are deliberately SEPARATED:
#
#   1. Which backend PACKAGE is loaded — decided by PLATFORM (Metal on Apple,
#      CUDA on Linux). The platform is identical at precompile time and at run
#      time, so a precompiled image is valid no matter what the environment was
#      when it was built. The backend package always loads (CUDA.jl / Metal.jl
#      load fine even with no usable device — `functional()` just returns false);
#      `CUDA_LOADED` / `METAL_LOADED` record which one is present.
#
#   2. Whether the GPU is actually USED — a RUNTIME switch, `use_gpu()`, read
#      from the `USE_GPU` env var on every call. `export USE_GPU=1` (or 0) takes
#      effect WITHOUT recompiling.
#
# This split fixes a silent CPU-fallback bug: the old design baked
# `const USE_GPU = get(ENV, ...)` at PRECOMPILE time, so a CPU-only train build
# (USE_GPU=0) sharing the depot could poison a GPU test job's precompile cache
# and force it onto the CPU. USE_GPU is now never compiled into the image.
#
#   export USE_GPU=1     # use the GPU (CUDA on Linux HPC, Metal on Apple Silicon)
#   export USE_GPU=0     # CPU only; falls back to forward_pass_with_weights
@static if Sys.isapple()
    import Metal
    const GPU_BACKEND  = "metal"
    const METAL_LOADED = true
    const CUDA_LOADED  = false
elseif Sys.islinux()
    import CUDA
    const GPU_BACKEND  = "cuda"
    const METAL_LOADED = false
    const CUDA_LOADED  = true
else
    const GPU_BACKEND  = "none"
    const METAL_LOADED = false
    const CUDA_LOADED  = false
end

# Runtime GPU on/off switch — read fresh from the environment on each call, never
# baked into the precompiled image.
use_gpu()::Bool = get(ENV, "USE_GPU", "0") == "1"

# The GPU is actually engaged only when the user asked for it (USE_GPU=1) AND a
# backend is compiled in for this platform.
gpu_active()::Bool = use_gpu() && (CUDA_LOADED || METAL_LOADED)

# Data manipulation and analysis
using DataFrames        # Tabular data structures for decoder statistics
using Statistics        # Statistical functions for analyzing decoder performance

# Visualization
# NOTE: Plots.jl / StatsPlots.jl are intentionally NOT dependencies of this
# module. They're heavyweight (~1 GB of precompile) and only used by the
# analysis-only helpers in src/plot.jl, which the cluster path never touches.
# When you want to plot on your Mac: `include(joinpath(pkgdir(
# CorrelatedBPDecoderWithCER), "src", "plot.jl"))` from a session that has
# Plots + StatsPlots available (install them in your @v1.12 global env).

# Machine learning framework
# NOTE: we intentionally do NOT depend on Flux or Zygote. Enzyme is the sole
# autodiff engine, and Optimisers.jl + Functors.jl together provide everything
# we need for parameter iteration/updates. Removing Flux+Zygote cuts precompile
# time significantly and shrinks the loaded-module RAM footprint per worker
# (which was OOM-killing MIG jobs at 15 GB).
using Functors          # @functor, for making struct fields trainable
using Optimisers        # OptimiserChain(ClipGrad, Adam/AdamW) etc.
using Enzyme            # Reverse-mode AD

# Base method extensions
import Base: eltype, length, sort!

#===============================================================================
                               MODULE INCLUDES
===============================================================================#

# Core data structures and graph representations
include("tanner_graph.jl")
export TannerGraph, print_tanner_graph, QuantumCode, measure_syndrome, print_code_info

# Configuration constants
export use_gpu, gpu_active, GPU_BACKEND, METAL_LOADED, CUDA_LOADED

# Quantum error correction codes
include("hypergraph_product_code.jl")
export get_hypergraph_product_code_H

# Error models and sampling
include("error_model.jl")
export ErrorModel, IIDErrorModel, sample_error, print_error_model_info, 
       separate_error_components, join_error_components, BallisticErrorModel, 
       ExplicitErrorModel, sample_errors, sweep_error_parameters, assign_error_model,
       RandomWalkErrorModel, RegenerativeErrorModel

# Data collection and postprocessing
include("postprocessing.jl")
export NeuralBPDecoderStatistics, collect_decoder_statistics, collect_standard_decoder_statistics,
       save_decoder_dataframe, record_decoder_statistics, write_failure_diagnostics

# Visualization and plotting are intentionally NOT part of this module — see
# the note under "Visualization" above and the header of src/plot.jl.

# Classical error correction codes
include("hamming.jl")
export generate_Hamming_Parity_Check_Matrix

# Quantum code operations
include("quantum_code.jl")
export ClassicalCode, QuantumCode, measure_syndrome_quantum_code, 
       print_quantum_code_info, quantum_belief_propagation_decoder, is_valid

# Belief propagation algorithms
include("bp_algo.jl")
export BPSettings, print_bp_settings, belief_propagation_decoder, 
       save_bp_settings, load_bp_settings, is_decoder_failure, 
       trim_constraints, run_bp

# Command line interface
include("command_line.jl")
export parse_command_line_args_BP, parse_command_line_args_NN, print_arguments, generate_runs, parse_hyper_parameters, disable_retrain_in_hyperparams,
       hyperparameter_seed, seed_tag_for, apply_training_seed!

# Neural belief propagation
include("neuralbase.jl")
export NeuralBPBase, NeuralBP, add_soft_constraints_to_neuralbpbase, parse_cer_data,
       rescale_single_qubit_error_rates

# Nachmani Neural BP model
include("nachmani.jl")
export NachmaniNeuralBP

# Explicit-weight forward pass helpers
include("forward_pass_weights.jl")
export forward_pass_with_weights, c2v_to_v2c_with_weights!, readout_with_weights!, v2c_to_c2v!

# GPU-accelerated forward pass
include("forward_gpu.jl")
export forward_pass_gpu

include("legacy.jl") # these are solely for debugging and testing, not intended for external use.
export forward_pass, c2v_to_v2c, v2c_to_c2v, readout,
       StandardBPDecoderStatistics, check_valid_fields_StandardBPDecoderStatistics,
       extract_collected_data, collect_decoder_statistics_correlated,
       collect_standard_decoder_statistics_correlated

# Console output. `print_info` is the one to use: a plain print, safe alongside
# Enzyme. `BracketedConsoleLogger` is defined but NOT installed — installing a
# custom global logger breaks `Enzyme.autodiff` (world age); see the file header.
include("logging.jl")
export print_info, print_bracketed_record, log_level_label, log_level_color,
       BracketedConsoleLogger, install_bracketed_logger!

# Print functions for Neural BP models
include("printfuns.jl")
export print_neuralbp_info, print_neuralbp_summary, print_simulation_results_to_console, print_console_rule

# Load and save Neural BP models
include("log_model.jl")
export load_trained_neuralbp_model, save_trained_neuralbp_model, load_base_BP_model

# Extraction and saving/loading of weights for BP
include("nbp_weights.jl")
export save_trained_weights, extract_weights_for_BP, save_extracted_weights_for_BP,
       load_extracted_weights_for_BP, load_trained_weights

# Loss functions
include("loss.jl")
export compute_additional_loss_from_ising_correlations, compute_loss_including_correlations, linear_ramp_loss,
       compute_quadratic_residue_loss_from_llrs, syndrome_loss_regularizer, sparsity_penalty, compute_sine_residue_loss_from_llrs

# Training routines
include("train.jl")
export train_neuralbp!, train_neuralbp_enzyme!, train_Nachmani_neuralbp

# Predictions with trained models
include("predict.jl")
export predict_neuralbp, check_bp_solutions, predict_and_check_neuralbp, neuralbp_test_predictions,
       resolve_prediction_batch_size,
       count_syndrome_satisfactions, predict_and_diagnose_neuralbp, concatenate_diagnoses,
       mean_committed_layer

# Utility functions
include("utils.jl")
export safe_atanh_exp_signed, safe_atanh_exp_signed!, safe_log_tanh_split, safe_log_tanh_split!, random_values_around_one, compute_std_assuming_bernoulli,
       debug_log_tanh_split, xor_affine!, sparse_multiply!, sigmoid, binary_entropy, binary_entropy_of_sigmoid, fmt_probs,
       parse_memory, compute_optimal_batch_size_for

#===============================================================================
                               END OF MODULE
===============================================================================#

end # module CorrelatedBPDecoderWithCER
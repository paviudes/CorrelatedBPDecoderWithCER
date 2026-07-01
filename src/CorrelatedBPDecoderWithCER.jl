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

# GPU acceleration (conditionally loaded based on environment variables)
#
# Two env vars control the GPU path:
#   USE_GPU       = "1" to enable any GPU code path. Anything else disables it.
#   GPU_BACKEND   = "metal" (default, Apple Silicon) or "cuda" (NVIDIA, HPC).
#
# Only one backend is loaded per Julia session — choose at startup with:
#   export USE_GPU="1" GPU_BACKEND="cuda"     # Linux HPC with NVIDIA GPUs
#   export USE_GPU="1" GPU_BACKEND="metal"    # macOS / Apple Silicon (default)
#   export USE_GPU="0"                      # CPU only; falls back to forward_pass_with_weights
const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
const GPU_BACKEND = lowercase(get(ENV, "GPU_BACKEND", "metal"))
@static if USE_GPU
    if GPU_BACKEND == "cuda"
        @eval begin
            import CUDA
            const METAL_LOADED = false
            const CUDA_LOADED  = true
        end
    elseif GPU_BACKEND == "metal"
        @eval begin
            import Metal
            const METAL_LOADED = true
            const CUDA_LOADED  = false
        end
    else
        error("Unknown GPU_BACKEND=\"$(GPU_BACKEND)\". Must be \"metal\" or \"cuda\".")
    end
else
    const METAL_LOADED = false
    const CUDA_LOADED  = false
end

# Data manipulation and analysis
using DataFrames        # Tabular data structures for decoder statistics
using Statistics        # Statistical functions for analyzing decoder performance

# Visualization
using Plots              # Plotting decoder performance and error rates
using Plots.PlotMeasures # For adding margins around the plot area
using StatsPlots         # Additional plotting recipes for statistical visualizations

# Machine learning framework
using Flux              # Neural network framework for neural belief propagation
using Functors          # Making neural network parameters trainable
using Optimisers        # Optimization algorithms for training neural networks
using Zygote            # Automatic differentiation for gradient computation
using Enzyme            # Alternative AD for potentially faster gradient computation

# Base method extensions
import Base: eltype, length, sort!

#===============================================================================
                               MODULE INCLUDES
===============================================================================#

# Core data structures and graph representations
include("tanner_graph.jl")
export TannerGraph, print_tanner_graph, QuantumCode, measure_syndrome, print_code_info

# Configuration constants
export USE_GPU, GPU_BACKEND, METAL_LOADED, CUDA_LOADED

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
export DecoderStatistics, collect_decoder_statistics, save_decoder_dataframe, 
       check_valid_fields_DecoderStatistics, record_decoder_statistics, 
       extract_collected_data, collect_decoder_statistics_for_ballistic_data, collect_standard_decoder_statistics_for_ballistic_data

# Visualization and plotting
include("plot.jl")
export plot_statistics_for_ballistic_error_model, plot_performance_spread

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
export parse_command_line_args_BP, parse_command_line_args_NN, print_arguments, generate_runs, parse_hyper_parameters, disable_retrain_in_hyperparams

# Neural belief propagation
include("neuralbase.jl")
export NeuralBPBase, NeuralBP, add_soft_constraints_to_neuralbpbase, parse_cer_data

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
export forward_pass, c2v_to_v2c, v2c_to_c2v, readout

# Print functions for Neural BP models
include("printfuns.jl")
export print_neuralbp_info, print_neuralbp_summary

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
export predict_neuralbp, check_bp_solutions, predict_and_check_neuralbp, neuralbp_test_predictions

# Utility functions
include("utils.jl")
export safe_atanh_exp_signed, safe_atanh_exp_signed!, safe_log_tanh_split, safe_log_tanh_split!, random_values_around_one, compute_std_assuming_bernoulli,
       debug_log_tanh_split, xor_affine!, sparse_multiply!, sigmoid, binary_entropy, binary_entropy_of_sigmoid

#===============================================================================
                               END OF MODULE
===============================================================================#

end # module CorrelatedBPDecoderWithCER
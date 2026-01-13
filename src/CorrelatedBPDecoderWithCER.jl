module CorrelatedBPDecoderWithCER

#===============================================================================
                                    IMPORTS
===============================================================================#

# Core Julia packages
using LinearAlgebra     # Matrix operations, decompositions, linear algebra functions
using Base.Threads           # Multi-threading for parallel computations
BLAS.set_num_threads(Threads.nthreads()) # Set BLAS to use the same number of threads as Julia
using Random            # Random number generation for error sampling and initialization
using JSON              # Reading/writing JSON configuration and data files
using DelimitedFiles    # File I/O for CSV-like data formats
using CSV               # CSV file reading and writing

# Data manipulation and analysis
using DataFrames        # Tabular data structures for decoder statistics

# Visualization
using Plots             # Plotting decoder performance and error rates

# Machine learning framework
using Flux              # Neural network framework for neural belief propagation
using Functors          # Making neural network parameters trainable
using Optimisers        # Optimization algorithms for training neural networks
using Zygote            # Automatic differentiation for gradient computation

# Base method extensions
import Base: eltype, length, sort!

#===============================================================================
                               MODULE INCLUDES
===============================================================================#

# Core data structures and graph representations
include("tanner_graph.jl")
export TannerGraph, print_tanner_graph, QuantumCode, measure_syndrome, print_code_info

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
       extract_collected_data

# Visualization and plotting
include("plot.jl")
export plot_statistics_for_ballistic_error_model

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
export parse_command_line_args_BP, parse_command_line_args_NN, print_arguments, generate_runs

# Neural belief propagation
include("neuralbp.jl")
export NeuralBPBase, NachmaniNeuralBP

# Print functions for Neural BP models
include("printfuns.jl")
export print_neuralbp_info, print_neuralbp_summary

# Load and save Neural BP models
include("log_model.jl")
export load_trained_neuralbp_model, save_trained_neuralbp_model

# Extraction and saving/loading of weights for BP
include("nbp_weights.jl")
export save_trained_weights, extract_weights_for_BP, save_extracted_weights_for_BP,
       load_extracted_weights_for_BP, load_trained_weights

# Loss functions
include("loss.jl")
export compute_loss_error_from_llrs, compute_additional_loss_from_ising_correlations, 
       compute_loss_including_correlations

# Training routines
include("train.jl")
export train_neuralbp!, generate_training_data

# Predictions with trained models
include("predict.jl")
export predict_neuralbp, check_bp_solutions

# Utility functions
include("utils.jl")
export safe_atanh_exp, safe_log_tanh, random_values_around_one

#===============================================================================
                               END OF MODULE
===============================================================================#

end # module CorrelatedBPDecoderWithCER
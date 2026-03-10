module CorrelatedBPDecoderWithCER

#===============================================================================
                                    IMPORTS
===============================================================================#

# Core Julia packages
using LinearAlgebra     # Matrix operations, decompositions, linear algebra functions
using SparseArrays     # Sparse matrix representations for efficiency
using Base.Threads           # Multi-threading for parallel computations
BLAS.set_num_threads(Threads.nthreads()) # Set BLAS to use the same number of threads as Julia
using Random            # Random number generation for error sampling and initialization
using JSON              # Reading/writing JSON configuration and data files
using DelimitedFiles    # File I/O for CSV-like data formats
using CSV               # CSV file reading and writing
using ProgressMeter     # Progress bars for training and other long operations

# Data manipulation and analysis
using DataFrames        # Tabular data structures for decoder statistics

# Visualization
using Plots             # Plotting decoder performance and error rates

#=
# Machine learning framework
using Flux              # Neural network framework for neural belief propagation
using Functors          # Making neural network parameters trainable
using Optimisers        # Optimization algorithms for training neural networks
using Optimisers: OptimiserChain, ClipGrad, WeightDecay, ADAM # Specific optimizers used in training
# using Zygote            # Automatic differentiation for gradient computation
=#

# Optimization tools for training neural networks
using Optimisers        # Optimization algorithms for training neural networks
using Optimisers: OptimiserChain, ClipGrad, WeightDecay, ADAM # Specific optimizers used in training

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
include("neuralbase.jl")
export NeuralBPBase, NeuralBP

# Nachmani Neural BP model
include("nachmani.jl")
export NachmaniNeuralBP, reset_to_standard_BP, unpack_params!, pack_params, 
       build_vectorization_maps!, NachmaniLearnableParameterGroup
include("forwprop.jl")

# Gradients for the Nachmani Neural BP model
include("nachmani_grads.jl")
export grad_message_c2v_wrt_weight, grad_llrs_wrt_weight, 
       grad_message_c2v_wrt_bias, grad_llrs_wrt_bias

include("backprop.jl")
export nachmani_loss_jacobian, derivative_loss_at_layer_t_wrt_weight, 
       derivative_total_loss_wrt_weight, derivative_loss_at_layer_t_wrt_bias, 
       derivative_total_loss_wrt_bias, nachmani_loss_jacobian_wrt_c2v_readout_weights

# Load and save Neural BP models
include("log_model.jl")
export load_NBP, save_NBP

# Loss functions
include("loss.jl")
export compute_loss_error_from_llrs, compute_loss_from_llrs, compute_loss_per_layer, compute_loss_including_correlations

# Training routines
include("train.jl")
export train_step!, train!, generate_training_data, train_minibatch!

# Predictions with trained models
include("predict.jl")
export predict_neuralbp, check_bp_solutions, predict_and_validate

# Complex number structure for handling contributions of the syndrome in the belief propagation algorithm
include("complex.jl")
export ComplexPi32, safe_sinh, safe_cosech, safe_negative_cosech, safe_atanh_exp, safe_log_tanh

# Utility functions
include("utils.jl")
export sigmoid

#===============================================================================
                               END OF MODULE
===============================================================================#

end # module CorrelatedBPDecoderWithCER
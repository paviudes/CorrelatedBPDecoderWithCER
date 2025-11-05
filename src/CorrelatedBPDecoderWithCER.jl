module CorrelatedBPDecoderWithCER

# Core Julia packages
using LinearAlgebra    # Matrix operations, decompositions, linear algebra functions
using Random           # Random number generation for error sampling and initialization
using JSON             # Reading/writing JSON configuration and data files
using DelimitedFiles   # File I/O for CSV-like data formats
using CSV              # CSV file reading and writing

# Data manipulation and analysis
using DataFrames       # Tabular data structures for decoder statistics

# Visualization
using Plots            # Plotting decoder performance and error rates

# Machine learning framework
using Flux             # Neural network framework for neural belief propagation
using Functors         # Making neural network parameters trainable
using Optimisers       # Optimization algorithms for training neural networks

import Base: eltype, length, sort!

include("tanner_graph.jl")
export TannerGraph, print_tanner_graph, QuantumCode, measure_syndrome, print_code_info

include("hypergraph_product_code.jl")
export get_hypergraph_product_code_H

include("error_model.jl")
export ErrorModel, IIDErrorModel, sample_error, print_error_model_info, separate_error_components, join_error_components, BallisticErrorModel, ExplicitErrorModel, sample_errors, sweep_error_parameters, assign_error_model

include("postprocessing.jl")
export DecoderStatistics, collect_decoder_statistics, save_decoder_dataframe, check_valid_fields_DecoderStatistics, record_decoder_statistics, extract_collected_data

include("plot.jl")
export plot_statistics_for_ballistic_error_model

include("hamming.jl")
export generate_Hamming_Parity_Check_Matrix

include("quantum_code.jl")
export ClassicalCode, QuantumCode, measure_syndrome_quantum_code, print_quantum_code_info, quantum_belief_propagation_decoder, is_valid

include("bp_algo.jl")
export BPSettings, print_bp_settings, belief_propagation_decoder, save_bp_settings, load_bp_settings, is_decoder_failure, trim_constraints, run_bp

include("command_line.jl")
export parse_command_line_args, print_arguments, generate_runs

include("neuralbp.jl")
export NeuralBP, print_neuralbp_info, generate_training_data, train_neuralbp!, predict_neuralbp, compute_loss_error_from_llrs

end # module CorrelatedBPDecoderWithCER
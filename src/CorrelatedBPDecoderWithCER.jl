module CorrelatedBPDecoderWithCER

using LinearAlgebra, JSON, Random, DelimitedFiles, DataFrames, Plots, CSV

import Base: eltype, length, sort!

include("tanner_graph.jl")
export TannerGraph, print_tanner_graph, QuantumCode, measure_syndrome, print_code_info

include("hypergraph_product_code.jl")
export get_hypergraph_product_code_H

include("error_model.jl")
export ErrorModel, IIDErrorModel, sample_error, print_error_model_info, separate_error_components, join_error_components, BallisticErrorModel, ExplicitErrorModel, sample_errors, sweep_error_parameters

include("postprocessing.jl")
export DecoderStatistics, print_decoder_statistics, save_decoder_statistics, get_output_filename, collect_decoder_statistics, save_dataframe

include("plot.jl")
export plot_statistics_for_ballistic_error_model, print_collected_data

include("hamming.jl")
export generate_Hamming_Parity_Check_Matrix

include("quantum_code.jl")
export ClassicalCode, QuantumCode, measure_syndrome_quantum_code, print_quantum_code_info, quantum_belief_propagation_decoder, is_valid

include("bp_algo.jl")
export BPSettings, print_bp_settings, belief_propagation_decoder, save_bp_settings, load_bp_settings, is_decoder_failure, trim_constraints

end # module BP
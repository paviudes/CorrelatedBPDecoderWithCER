include("test_forward.jl")
include("test_bp_algo.jl")
include("test_bp_utils.jl")
include("test_loss.jl")
include("test_classical_bp.jl")

"""
Main test runner for all Neural BP tests.

Available test functions:
- Forward propagation tests: test_forward_propagation()
- BP algorithm tests: test_neural_BP(), test_training_Nachmani_BP()
- BP utilities tests: test_activation_functions(), test_messages_c2v_to_v2c(), test_messages_v2c_c2v(), test_readout()
- Loss function tests: test_loss(), test_correlation_loss()
- Classical BP tests: test_classical_BP()
"""

function run_all_tests()
    """Run all neural BP tests."""
    println("="^60)
    println("Running all Neural BP tests...")
    println("="^60)
    
    # BP algorithm tests
    println("\n--- BP ALGORITHM TESTS ---")
    try
        test_neural_BP()
        println("✓ test_neural_BP passed")
    catch e
        println("✗ test_neural_BP failed: $e")
    end
    
    try
        test_training_Nachmani_BP()
        println("✓ test_training_Nachmani_BP passed")
    catch e
        println("✗ test_training_Nachmani_BP failed: $e")
    end
    
    # BP utilities tests
    println("\n--- BP UTILITIES TESTS ---")
    try
        test_activation_functions()
        println("✓ test_activation_functions passed")
    catch e
        println("✗ test_activation_functions failed: $e")
    end
    
    try
        test_messages_c2v_to_v2c()
        println("✓ test_messages_c2v_to_v2c passed")
    catch e
        println("✗ test_messages_c2v_to_v2c failed: $e")
    end
    
    try
        test_messages_v2c_c2v()
        println("✓ test_messages_v2c_c2v passed")
    catch e
        println("✗ test_messages_v2c_c2v failed: $e")
    end
    
    try
        test_readout()
        println("✓ test_readout passed")
    catch e
        println("✗ test_readout failed: $e")
    end
    
    # Loss function tests
    println("\n--- LOSS FUNCTION TESTS ---")
    try
        test_loss()
        println("✓ test_loss passed")
    catch e
        println("✗ test_loss failed: $e")
    end
    
    try
        test_correlation_loss()
        println("✓ test_correlation_loss passed")
    catch e
        println("✗ test_correlation_loss failed: $e")
    end
    
    # Classical BP tests
    println("\n--- CLASSICAL BP TESTS ---")
    try
        test_classical_BP()
        println("✓ test_classical_BP passed")
    catch e
        println("✗ test_classical_BP failed: $e")
    end
    
    println("\n" * "="^60)
    println("All tests completed!")
    println("="^60)
end

function run_bp_algo_tests()
    """Run BP algorithm tests only."""
    println("Running BP algorithm tests...")
    println("Test if the Neural BP decoder with all weights set to 1.0 produces the same LLRs as the standard BP decoder after K iterations.")
    test_neural_BP()
    println("-"^60)
    println("Test if a trained BP decoder is able to correctly decode single qubit errors")
    test_training_Nachmani_BP()
    println("-"^60)
end

function run_bp_utils_tests()
    """Run BP utilities tests only."""
    println("Running BP utilities tests...")
    #
    println("Test the activation functions used in the BP decoder.")
    test_activation_functions()
    #
    println("-"^60)
    #
    println("Test the messages from check nodes to variable nodes (c2v).")
    test_messages_c2v_to_v2c()
    #
    println("-"^60)
    #
    println("Test the messages from variable nodes to check nodes (v2c).")
    test_messages_v2c_c2v()
    #
    println("-"^60)
    #
    println("Test the readout function that computes the final LLRs for each qubit after K iterations.")
    test_readout()
    #
    println("-"^60)
    #
end

# Uncomment the line below to run all tests when this file is executed
# run_all_tests()
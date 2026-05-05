using LinearAlgebra
using CorrelatedBPDecoderWithCER

function test_classical_BP()
    """
    Test the classical belief propagation decoder on a simple parity-check matrix of the Hamming code.
    H = [0 0 0 1 1 1 1;
         0 1 1 0 0 1 1;
         1 0 1 0 1 0 1]
    syndrome = [1, 0, 1] (indicating errors on qubits 1, 3, and 4)
    The expected output is the decoded message [1, 0, 1, 1, 0, 0, 0].

    We will not only verify the output but also the LLRs at each step.
    After 2 iterations, the LLRs should be approximately:
    [1.0663514264498881, 3.3280977282225512, 2.19722457733622, 1.0663514264498881, -0.06452172443644333, 2.1972245773362196, 1.0663514264498881]
    """
    H = [0 0 0 1 1 1 1;
         0 1 1 0 0 1 1;
         1 0 1 0 1 0 1]
    # Run the classical BP decoder on the H matrix
    initial_llrs = log.((1 .- 0.1) ./ 0.1) .* ones(7)  # Assuming an initial bit-flip probability of 0.1
    
    # Case 1: syndrome = [1, 0, 1], convergence in Joshka's code after 1 iteration.
    syndrome = [1, 0, 1]  # Indicates errors on qubits 1, 3, and 4
    n_iterations = 1
    expected_llrs = [1.0663514264498881, 3.3280977282225512, 2.19722457733622, 1.0663514264498881, -0.06452172443644333, 2.1972245773362196, 1.0663514264498881]
    (final_llrs, _) = run_bp(H, 4, syndrome, initial_llrs, n_iterations)
    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs, expected_llrs, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values.")
    else
        println("LLRs after $(n_iterations) iterations do not match the expected values.")
        println("Expected: ", expected_llrs)
        println("Got: ", final_llrs)
    end

    # Case 2: syndrome = [1, 0, 0], convergence in Joshka's code after 2 iterations.
    syndrome = [1, 0, 0]  # Indicates errors on qubits 1 and 4
    n_iterations = 2
    expected_llrs = [2.9584134938935067, 2.9584134938935067, 3.4891275557835617, -0.29005600177239454, 1.7230087983796607, 1.7230087983796611, 2.011992335845365]
    (final_llrs, _) = run_bp(H, 4, syndrome, initial_llrs, n_iterations)
    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs, expected_llrs, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values.")
    else
        println("LLRs after $(n_iterations) iterations do not match the expected values.")
        println("Expected: ", expected_llrs)
        println("Got: ", final_llrs)
    end

    # Case 3: syndrome = [0, 1, 0], convergence in Joshka's code after 2 iterations.
    syndrome = [0, 1, 0]  # Indicates errors on qubits 2 and 5
    n_iterations = 2
    expected_llrs = [2.9584134938935067, -0.29005600177239454, 1.7230087983796607, 2.9584134938935067, 3.4891275557835617, 1.7230087983796611, 2.011992335845365]
    (final_llrs, _) = run_bp(H, 4, syndrome, initial_llrs, n_iterations)
    # Check if the final LLRs match the expected values
    println("Syndrome: ", syndrome)
    if all(isapprox.(final_llrs, expected_llrs, atol=1e-6))
        println("LLRs after $(n_iterations) iterations match the expected values.")
    else
        println("LLRs after $(n_iterations) iterations do not match the expected values.")
        println("Expected: ", expected_llrs)
        println("Got: ", final_llrs)
    end

    return nothing
end
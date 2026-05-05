using SparseArrays
using DelimitedFiles
using CorrelatedBPDecoderWithCER

function test_loss()
    """
    Test the computation of the loss function for the NeuralBP model.
    We will define a parity-check matrix, a syndrome, initial LLRs, and expected recoveries.
    We will then compute the loss using the `compute_loss_error_from_llrs` function.
    H = [
        0 0 0 1 1 1 1 0;
        0 1 1 0 0 1 1 0;
        1 0 1 0 1 0 1 0
        ]
    H^⟂ = [
            1 0 0 0 0 1 1 0;
            0 1 0 0 1 0 1 0;
            0 0 1 0 1 1 0 0;
            0 0 0 1 1 1 1 0
          ]
    syndrome = [1, 0, 1] (indicating errors on qubits 1, 3, and 4)
    errors = [1, 0, 1, 1, 0, 0, 0] (the actual error pattern)
    posterior_llrs = [1.0663514264498881; 1.0663514264498881; 3.3280977282225512; 2.1972245773362196; 1.0663514264498881; -0.06452172443644333; 2.1972245773362196; 1.0663514264498881]
    The Loss function is given by
        L(μ, e) = ∑_i  f ( ∑_(jk) H^⟂_ij M_(jk) [ e_k + σ(μ_k)])
    where
        - σ(μ_k) = 1 / (1 + exp(μ_k))
        - f(x) = |sin(π x / 2)|
        - M = [0 I ; I 0] is the symplectic matrix
        - H^⟂ is the parity-check matrix of the dual code.
    """
    # Define the parity-check matrix
    H = [0 0 0 1 1 1 1 0;
         0 1 1 0 0 1 1 0;
         1 0 1 0 1 0 1 0]
    H_dual = [1 0 0 0 0 1 1 0;
              0 1 0 0 1 0 1 0;
              0 0 1 0 1 1 0 0;
              0 0 0 1 1 1 1 0]
    # Example syndrome
    # syndrome = [1; 0; 1;;]  # Indicates errors on qubits 1, 3, and 4
    # Expected recovery
    errors = convert.(Bool, [1; 0; 1; 1; 0; 0; 0; 0;;])
    # Initial LLRs
    posterior_llrs = convert.(Float32, [2.0663514264498881; 1.0663514264498881; -3.3280977282225512; -2.1972245773362196; -0.0663514264498881; -0.06452172443644333; 2.1972245773362196; 1.0663514264498881;;])
    
    # Compute the Loss function.
    actual_loss = compute_loss_error_from_llrs(posterior_llrs, errors, convert.(Bool, H_dual))

    println("Computed Loss:", actual_loss)
end

function test_correlation_loss()
    """
    Test the computation of the loss function that comes from encouraging correlations between errors.
    The correlation loss is given by
        L_corr(μ) = λ * ∑_((qi, qj) ∈ C) [ e_(qi) XOR e_(qj) ]
    where
        - λ is a hyperparameter that controls the strength of the correlation penalty.
        - C is the set of correlated qubit index pairs.
        - e_(qi) is the predicted error at qubit `qi`.
    
    We will define a small set of correlated qubit pairs, predicted LLRs, and compute the correlation loss explicitly using the formula above.
    Additionally, we will compute the correlation loss using the `compute_additional_loss_from_ising_correlations` function and compare the results.
    """
    H_dual = [1 0 0 0 0 1 1 0;
              0 1 0 0 1 0 1 0;
              0 0 1 0 1 1 0 0;
              0 0 0 1 1 1 1 0]
    posterior_llrs = convert.(Float32, [
        2.0663514264498881;     # qubit 1
        1.0663514264498881;     # qubit 2
        -3.3280977282225512;    # qubit 3
        -2.1972245773362196;    # qubit 4
        -0.0663514264498881;    # qubit 5
        -0.06452172443644333;   # qubit 6
        2.1972245773362196;     # qubit 7
        1.0663514264498881;;    # qubit 8
    ])
    # Define correlated qubit pairs
    connectivity_matrix = [1 2; 3 4; 5 6; 7 8]
    correlation_strength = 0.5f0
    # Expected recoveries
    expected_recoveries = convert.(Bool, [1; 0; 1; 1; 0; 0; 0; 0;;])
    
    # Compute the correlation loss explicitly
    expected_corr_loss = 0.0f0
    for (qi, qj) in eachrow(connectivity_matrix)
        e_qi = 1 / (1 + exp(posterior_llrs[qi]))
        e_qj = 1 / (1 + exp(posterior_llrs[qj]))
        xor_value = e_qi + e_qj - 2 * e_qi * e_qj
        expected_corr_loss += xor_value
    end
    expected_corr_loss *= correlation_strength
    
    expected_bare_loss = compute_loss_error_from_llrs(
        posterior_llrs, 
        expected_recoveries,
        convert.(Bool, H_dual)
    )
    expected_loss = expected_bare_loss + expected_corr_loss
    
    # Now compute the correlation loss using the function
    actual_loss = compute_loss_including_correlations(
        posterior_llrs, 
        expected_recoveries,
        convert.(Bool, H_dual),
        connectivity_matrix, 
        correlation_strength,
        true
    )

    # Compare the two results
    if isapprox(actual_loss, expected_loss, atol=1e-6)
        println("Correlation loss computed by the function matches the expected value.")
    else
        println("Correlation loss computed by the function does not match the expected value.")
        println("Explicitly computed loss: ", expected_loss)
        println("Computed loss from function: ", actual_loss)
    end
end
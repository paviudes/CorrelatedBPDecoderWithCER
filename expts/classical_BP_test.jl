using LinearAlgebra
using CorrelatedBPDecoderWithCER

function test_Hamming_Code()
    r = 3  # Change this value for different Hamming codes
    H = generate_Hamming_Parity_Check_Matrix(r)
    println("Parity-Check Matrix for Hamming Code (r=$r):\n", H)
    hamming_tanner = TannerGraph(H)
    

    # Generate an example error
    error = [1, 0, 1, 0, 0, 0, 0]  # Example error vector (single-bit error)
    syndrome = mod.(H * error, 2)
    println("Syndrome for the error vector ", error, " is ", syndrome)
    
    n_iterations = 10
    # Example initial probabilities (for a BSC with crossover probability p)
    p = 0.1  # Crossover prior probability
    prior_probabilities = [1 - p for _ in 1:hamming_tanner.nv]  # Assuming all-zero codeword sent
    bp_settings = run_bp(hamming_tanner, error, syndrome, prior_probabilities, n_iterations; convergence_threshold=1e-6, verbose=true)
    println(bp_settings)
    save_bp_settings(bp_settings, "./../data/hamming_bp_settings.json")

    # Determine if the decoder has failed.
    G = [1 0 0 0 0 1 1;
         0 1 0 0 1 0 1;
         0 0 1 0 1 1 0;
         0 0 0 1 1 1 1]  # Generator matrix for (7,4) Hamming code
    is_fail = is_decoder_failure(error, bp_settings.recovery_hard_decision, G)
    println("Decoder failure: ", is_fail)
end
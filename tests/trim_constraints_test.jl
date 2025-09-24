using CorrelatedBPDecoderWithCER

function test_trim_constraints()
    """
    Test the `trim_constraints` function that removes constraints that only involve a single bit.
    The function is expected to fix the single bit parity-check equations by updating the LLRs and removing the corresponding rows and columns from the parity-check matrix and updating the syndrome.
    
    Example:
    H = [1 1 0 0 0 1;
         0 1 0 0 0 0;
         0 1 0 1 0 0;
         1 0 0 1 0 1;
         1 1 0 0 0 0] # Added an extra row to represent a soft constraint
    s = [1, 0, 1, 0, 0]

    1. Note that the 2nd row of H has only one non-zero entry (the 2nd bit). Therefore, we can directly infer the value of the 2nd bit from the syndrome.
    2. The syndrome for the 2nd row is 0, which means the 2nd bit must be 0 to satisfy the parity-check equation.
    3. We fix the 2nd bit to 0, set its LLR to a high positive value (indicating high confidence in it being 0), and set the 2nd row and 2nd column of H to zero.
    4. The 2nd bit also appears in the first and the third parity-check equations. So, we need to update the syndrome for these equations by removing the contribution of the 2nd bit. In other words, the second and third bits of the syndrome will be added with the value of the 2nd bit (which is 0 in this case, so the syndrome remains unchanged).
    5. Since the 2nd bit also appears in the last parity-check equation, which is a soft constraint, we can remove this row entirely and update the syndrome accordingly.
    5. Now the updated H and syndrome are:
    H = [1 0 0 0 0 1;
         0 0 0 0 0 0;
         0 0 0 1 0 0;
         1 0 0 1 0 1;
         0 0 0 0 0 0]
    s = [1, 0, 1, 0, 0]
    5. Now, the 3rd row of the updated H has only one non-zero entry (the 4th bit). The syndrome for the 3rd row is 1, which means the 4th bit must be 1 to satisfy the parity-check equation.
    6. We fix the 4th bit to 1, set its LLR to a high negative value (indicating high confidence in it being 1), and set the 3rd row and 4th column of H to zero.
    7. The 4th bit also appears in the fourth parity-check equation. So, we need to update the syndrome for this equation by removing the contribution of the 4th bit. In other words, the fourth bit of the syndrome will be added with the value of the 4th bit (which is 1 in this case, so the syndrome becomes 0).
    8. The final updated H and syndrome are:
    H = [1 0 0 0 0 1;
         0 0 0 0 0 0;
         0 0 0 0 0 0;
         1 0 0 0 0 1;
         0 0 0 0 0 0]
    s = [1, 0, 0, 1, 0]
    8. There are no more rows with a single non-zero entry, so we stop here.
    The function should return:
    fixed_bits = [2, 4]
    llrs_for_fixed_bits = [4.0, -4.0] # Assuming a confidence threshold of 4.0
    updated_syndrome = [1, 0, 0, 1, 0]
    """
    # Define a simple parity-check matrix and syndrome
    H = [1 1 0 0 0 1;
         0 1 0 0 0 0;
         0 1 0 1 0 0;
         1 0 0 1 0 1;
         1 1 0 0 0 0] # Added an extra row to represent a soft constraint
    s = [1, 0, 1, 0, 0]

    confidence_threshold = 4.0 # A value to set for the fixed bits.

    soft_constraint_start = 5 # First two rows are hard constraints, last two are soft constraints.

    # Call the trim_constraints function
    (updated_H, updated_syndrome, fixed_bits, llrs_for_fixed_bits) = trim_constraints(H, s, soft_constraint_start, confidence_threshold)

    # Expected results
    expected_H = [1 0 0 0 0 1;
                  0 0 0 0 0 0;
                  0 0 0 0 0 0;
                  1 0 0 0 0 1;
                  0 0 0 0 0 0]
    expected_fixed_bits = [2, 4]
    expected_llrs_for_fixed_bits = [confidence_threshold, -confidence_threshold]
    expected_updated_syndrome = [1, 0, 0, 1, 0]

    # Check if the results match the expected values
    if fixed_bits == expected_fixed_bits
        println("Fixed bits computed correctly.")
    else
        println("Actual Fixed Bits: ", fixed_bits)
        println("Expected Fixed Bits: ", expected_fixed_bits)
    end
    if llrs_for_fixed_bits == expected_llrs_for_fixed_bits
        println("LLRs for Fixed Bits computed correctly.")
    else
        println("Actual LLRs for Fixed Bits: ", llrs_for_fixed_bits)
        println("Expected LLRs for Fixed Bits: ", expected_llrs_for_fixed_bits)
    end
    if updated_syndrome == expected_updated_syndrome
        println("Updated Syndrome computed correctly.")
    else
        println("Actual Updated Syndrome: ", updated_syndrome)
        println("Expected Updated Syndrome: ", expected_updated_syndrome)
    end
    if updated_H == expected_H
        println("Updated H matrix computed correctly.")
    else
        println("Actual Updated H Matrix: ")
        println(updated_H)
        println("Expected Updated H Matrix: ")
        println(expected_H)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_trim_constraints()
end
import sys
import numpy as np
from ldpc import BpDecoder
from ldpc.bp_decoder import BpDecoder
from ldpc import BpOsdDecoder

def decode_ballistic_error(p_value, q_mean, n_samples, n_errors):
    # ----------------------------------------------------------------
    # Decoder (initialized once)
    # ----------------------------------------------------------------
    PCM_Z = np.loadtxt("PCM_Z_15q.txt", dtype=int)
    Log_Z = np.loadtxt("Log_Z_15q.txt", dtype=int)

    decoder_x = BpOsdDecoder(
        pcm=PCM_Z,
        error_rate=0.1,
        max_iter=500,
        bp_method="product_sum",
        schedule="parallel"
    )

    # Z stabilizers + logicals (for residual error check)
    H_perp = np.vstack([PCM_Z, Log_Z])

    results = []
    for sample in range(1, n_samples + 1):

        filename = f"aps_15q_Hamm_code_data/testing_data/test_ballistic_p_{p_value}_q_{q_mean}_s_{sample}.txt"

        # Load errors (shape: n_errors x n_qubits)
        sample_errors_x = np.loadtxt(filename, dtype=int).T

        # Compute syndromes for each error pattern
        syndromes_z = [(PCM_Z @ e) % 2 for e in sample_errors_x]

        # Run decoding
        success = 0
        for syndrome, error in zip(syndromes_z, sample_errors_x):
            bp_out = decoder_x.decode(syndrome)
            residual_error = (error + bp_out) % 2
            if np.all((H_perp @ residual_error) % 2 == 0):
                success += 1

        failures = n_errors - success
        results.append((p_value, q_mean, sample, failures))
    
    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    results = np.array(results)

    np.savetxt(
        f"aps_15q_Hamm_code_data/BP_OSD_failure_rates_p_{p_value}_q_{q_mean}.txt",
        results,
        header="p_value  q_mean  sample  logical_failures",
        fmt=["%.3f", "%.2f", "%d", "%d"]
    )

    print(f"Results saved to aps_15q_Hamm_code_data/BP_OSD_failure_rates_p_{p_value}_q_{q_mean}.txt")
    return None

if __name__ == "__main__":
    # Read `p_value` and `q_mean` from command line arguments
    p_value = float(sys.argv[1])
    q_mean = float(sys.argv[2])
    
    n_samples = 10
    n_qubits  = 241
    n_errors  = 100000

    decode_ballistic_error(p_value, q_mean, n_samples, n_errors)
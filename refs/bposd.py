import time
import numpy as np
import itertools as itools
# from ldpc import BpDecoder
# from ldpc.bp_decoder import BpDecoder
from ldpc import BpOsdDecoder

def run_decoder(error_file, decoder, parity_check_matrix, dual_parity_check_matrix):
    '''
    Run the BP decoder on the given error file and return the number of logical failures.
    '''
    # Load errors (shape: n_errors x n_qubits)
    sample_errors_x = np.loadtxt(error_file, dtype=int).T

    # Compute syndromes for each error pattern
    syndromes_z = [(parity_check_matrix @ error_pattern) % 2 for error_pattern in sample_errors_x]

    # Run decoding
    failures: int = 0
    for (syndrome, error) in zip(syndromes_z, sample_errors_x):
        bp_out = decoder.decode(syndrome)
        residual_error = (error + bp_out) % 2
        if np.any((dual_parity_check_matrix @ residual_error) % 2 == 1):
            failures += 1
    
    return failures

def get_error_file(codename, p_value, q_mean, sample, prefix="./../data"):
    '''
    Construct the error file name based on the codename, p_value, q_mean, and sample number.
    '''
    filename = f"{prefix}/{codename}/testing_data/test_ballistic_p_{p_value}_q_{q_mean}_s_{sample}.txt"
    return filename

def run_decoder_on_ballistic_noise(codename, p_values, q_values, samples, results_filename, prefix="./../data"):
    '''
    Run the decoder on ballistic noise for a range of p_values, q_values and samples.
    '''
    # Load parity check matrices
    pcm_filename = f"{prefix}/{codename}/code/HZ.txt"
    parity_check_matrix = np.loadtxt(pcm_filename, dtype=int)
    log_filename = f"{prefix}/{codename}/code/LZ.txt"
    logicals = np.loadtxt(log_filename, dtype=int)
    dual_parity_check_matrix = np.vstack([parity_check_matrix, logicals])

    # Initialize decoder
    decoder = BpOsdDecoder(
        pcm=parity_check_matrix,
        error_rate=0.1,
        max_iter=500,
        bp_method="product_sum",
        schedule="parallel"
    )

    n_parameters = len(p_values) * len(q_values) * len(samples)
    failures_count = np.zeros((n_parameters, 4))  # Columns: p_value, q_mean, sample, failures
    parameters = itools.product(p_values, q_values, samples)
    
    # Run decoder for each combination of parameters
    start = time.time()
    for idx, (p_value, q_mean, sample) in enumerate(parameters):
        error_file = get_error_file(codename, p_value, q_mean, sample, prefix=prefix)
        # Time the decoding process
        print("Starting decoding for parameters: p_value = {}, q_mean = {}, sample = {}".format(p_value, q_mean, sample))
        failures_count_value = run_decoder(error_file, decoder, parity_check_matrix, dual_parity_check_matrix)
        elapsed = time.time() - start
        print("[{:.2f} seconds] Done, failures: {}".format(elapsed, failures_count_value))
        # Record the results
        failures_count[idx] = [p_value, q_mean, sample, failures_count_value]
    
    # Save results
    np.savetxt(
        results_filename,
        failures_count,
        header="p_value  q_mean  sample  logical_failures",
        fmt=["%.3f", "%.2f", "%d", "%d"]
    )

    print(f"Results saved to {results_filename}")
    return None

if __name__ == "__main__":
    prefix = "./../data"
    codename = "90q_BB_p_0.008_q_0.2_std_0.2_data"
    p_values = [0.008]
    q_values = [0.2]
    samples = [1]
    results_filename = f"{prefix}/{codename}/BP_OSD_failure_rates.txt"
    run_decoder_on_ballistic_noise(codename, p_values, q_values, samples, results_filename, prefix=prefix)
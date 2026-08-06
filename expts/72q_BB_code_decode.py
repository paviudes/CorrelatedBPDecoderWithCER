import numpy as np
from ldpc import BpDecoder
import ldpc.codes
import ldpc.code_util
from ldpc.bp_decoder import BpDecoder
from ldpc import BpOsdDecoder
import time
from bposd.hgp import hgp
import os

import numpy as np
from tqdm import tqdm

# Parameters
p_values  = [0.01]
q_means   = [0.001]
n_samples = 64
n_qubits  = 72
n_errors  = 100000
PCM_Z=np.loadtxt("data/72q_BB_p_0.010_std_0.01_q_0.001_std_0.01_data/code/HZ.txt", dtype=int)
Log_Z=np.loadtxt("data/72q_BB_p_0.010_std_0.01_q_0.001_std_0.01_data/code/LZ.txt", dtype=int)
results_dir = "data/72q_BB_p_0.010_std_0.01_q_0.001_std_0.01_data/results"
os.makedirs(results_dir, exist_ok=True)
test_data_dir = "data/72q_BB_p_0.010_std_0.01_q_0.001_std_0.01_data/testing_data"
total_files = len(p_values) * len(q_means) * n_samples  

# Decoder (initialized once)
decoder_x = BpOsdDecoder(
    pcm=PCM_Z,
    error_rate=0.1,
    max_iter=500,
    bp_method="product_sum",
    schedule="parallel",
    osd_method='OSD_E',
    osd_order=2

)

# Z stabilizers + logicals (for residual error check)
H_perp = np.vstack([PCM_Z, Log_Z])

# Main loop to process all files and compute failure rates
results = []

with tqdm(total=total_files, desc="Decoding", unit="file") as pbar:

    for p_value in p_values:
        for q_mean in q_means:
            for sample in range(1, n_samples + 1):

                filename = f"{test_data_dir}/test_p_{p_value}_q_{q_mean}_s_{sample}.txt"

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

                # Update bar to show current file info
                pbar.set_postfix(p=p_value, q=q_mean, s=sample)
                pbar.update(1)


# Save results

results = np.array(results)

np.savetxt(
    f"{results_dir}/72q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt",
    results,
    header="p_value  q_mean  sample  logical_failures",
    fmt=["%.3f", "%.3f", "%d", "%d"]
)

print("Results saved to 72q_BB_p_0.010_std_0.01_q_0.001_std_0.01_data/72q_BB_BP+OSD_failure_rates_OSD_E_order_2.txt")
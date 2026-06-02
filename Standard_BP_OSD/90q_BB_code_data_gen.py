import numpy as np
from ldpc import BpDecoder
import ldpc.codes
import ldpc.code_util
from ldpc.bp_decoder import BpDecoder
from ldpc import BpOsdDecoder
import time
from bposd.hgp import hgp
import os
import itertools
#Write BB_code matrix generations here
#Writing the important functions
# Geometry: Ladder grid R x C
R, C = 2, 45
n_qubits = R * C

def idx(r, c):
    return r * C + c + 1  # qubit labels from 1

# Get neighboring qubits for each qubit in the ladder grid
def neighboring_qubits(R, C):
    neighbors = {idx(r, c): set() for r in range(R) for c in range(C)}
    for r in range(R):
        for c in range(C):
            q = idx(r, c)
            # horizontal neighbors with periodic boundary
            if c == 0:
                neighbors[q].add(idx(r, C-1))
            if c == C-1:
                neighbors[q].add(idx(r, 0))
            if c > 0:
                neighbors[q].add(idx(r, c-1))
            if c < C-1:
                neighbors[q].add(idx(r, c+1))
            # vertical neighbors (no periodic boundary)
            if r > 0:
                neighbors[q].add(idx(r-1, c))
            if r < R-1:
                neighbors[q].add(idx(r+1, c))
    return neighbors

# Get the neighboring qubits for each qubit in the ladder grid
neighbors = neighboring_qubits(R, C)

#making pairs of neighboring qubits 
def neighboring_pairs(R, C):
    nbrs = neighboring_qubits(R, C)
    pairs = {q: set() for q in nbrs}
    for q, nbs in nbrs.items():
        for nb in nbs:
            pairs[q].add((q, nb))
    return pairs
pairs = neighboring_pairs(R, C)

# Neighbors of NN pairs (macro nodes)
def get_neighbors_of_macro_nodes():
    neighbors_of_pairs = {}
    for q, qubit_pairs in pairs.items():
        for pair in qubit_pairs:
            q_i, q_j = pair
            neighbors_of_pairs[pair] = (neighbors[q_i] - {q_j}, neighbors[q_j] - {q_i})#Removing the pair qubits from their respective neighbor sets
    return neighbors_of_pairs
neighbors_of_macro_nodes = get_neighbors_of_macro_nodes()

# ----------------------------------------------------------------
# Error probability assignment per sample
# Each directed edge (q, neighbor) gets its own independent q draw,
# but undirected pairs share the same value: q_(i,j) == q_(j,i)
# ----------------------------------------------------------------
def assign_error_probabilities(p_value, q_mean):
    independent_error_prob = {}
    conditional_nb_flip_prob = {}
    # Pre-sample one q_value per undirected pair so both directions share it
    undirected_pair_q = {}

    for qubit in range(1, n_qubits + 1):
        independent_error_prob[qubit] = p_value
        for neighbor in neighbors[qubit]:
            undirected_key = (min(qubit, neighbor), max(qubit, neighbor))
            if undirected_key not in undirected_pair_q:
                q_val = np.random.normal(loc=q_mean, scale=0.1)# Sample q_val from a normal distribution around q_mean with std-dev=scale
                q_val=np.clip(q_val, 0, 1)  # Ensure q_val is in [0,1]
                undirected_pair_q[undirected_key] = q_val
            conditional_nb_flip_prob[(qubit, neighbor)] = undirected_pair_q[undirected_key]

    return  independent_error_prob, conditional_nb_flip_prob

# CER computation (pair-specific). See the detailed steps in the handwritten notes
def compute_CER(independent_error_prob, conditional_nb_flip_prob):
    CER = {}
    for pair, (neighbors_i, neighbors_j) in neighbors_of_macro_nodes.items():
        q_i, q_j = pair
        # Step 1: pair-specific P_00, P_01, P_10, P_11
        p_i = independent_error_prob[q_i]
        p_j = independent_error_prob[q_j]
        P_00 = (1 - p_i) * (1 - p_j)
        P_01 = (1 - p_i) * p_j
        P_10 = p_i * (1 - p_j)
        P_11 = p_i * p_j

        # Step 2: KL contributions
        q_k, q_l = neighbors_i
        p_k = independent_error_prob[q_k]
        p_l = independent_error_prob[q_l]
        q_ik = conditional_nb_flip_prob.get((q_i, q_k), 0)
        q_il = conditional_nb_flip_prob.get((q_i, q_l), 0)
        KL = {
            "KL_01": (1 - p_k) * p_l * q_il,
            "KL_10": p_k * (1 - p_l) * q_ik,
            "KL_11": p_k * p_l * ((1 - q_il) * q_ik + (1 - q_ik) * q_il)
        }

        # Step 3: MN contributions
        q_m, q_n = neighbors_j
        p_m = independent_error_prob[q_m]
        p_n = independent_error_prob[q_n]
        q_jm = conditional_nb_flip_prob.get((q_j, q_m), 0)
        q_jn = conditional_nb_flip_prob.get((q_j, q_n), 0)
        MN = {
            "MN_01": (1 - p_m) * p_n * q_jn,
            "MN_10": p_m * (1 - p_n) * q_jm,
            "MN_11": p_m * p_n * ((1 - q_jn) * q_jm + (1 - q_jm) * q_jn)
        }

        # Step 4: Multiply contributions
        KLMN = {f"{k}_{m}": KL[k]*MN[m] for k in KL for m in MN}

        # Step 5: Compute CER for this pair
        CER[pair] = P_00*sum(KLMN.values()) + P_01*sum(KL.values()) + P_10*sum(MN.values()) + P_11

    # Step 6: Normalize CER values
    total = sum(CER.values())
    CER_normalized = {k: v/total for k, v in CER.items()}

    return CER_normalized



#-----------------------------------------------------------------
# I am adding a new function that computes the one-qubit marginals that can be used as LLRs for BP
#See handwritten notes for the detailed steps. The main idea is to analyze all configurations of independently flipped neighbors (active neighbors) that can cause a flip on the qubit and compute the total probability of a flip from neighbors. We then combine this with the independent error probability to get the final marginal.
def prob_exactly_one_trigger(active_neighbors, qubit,conditional_nb_flip_prob):
    """computes the probability that exactly one neighbor in active_neighbors causes a flip on qubit
    where by active neighbors we mean those neighboring qubits that have an error due to independent flip"""
    if len(active_neighbors) == 0:
        return 0.0
     # Since your assigned q values are symmetric,
     # conditional_nb_flip_prob[(qubit, nb)] and [(nb, qubit)] are equal.
    qs=[conditional_nb_flip_prob.get((qubit, nb), 0) for nb in active_neighbors]
    prob=0.0
    for trigger_index in range(len(qs)):
        term=1.0
        for k, nflip_prob_value in enumerate(qs):#enumerates only over the active neighbors, not non-active ones
            if k == trigger_index:
                term *= nflip_prob_value
            else:
                term *= (1 - nflip_prob_value)
        prob += term
    return prob

def compute_one_qubit_marginals(independent_error_prob, conditional_nb_flip_prob, normalize=True):#Maybe I shouldnt normalize and send the unnormalized as LLRs?
    one_body_marginals = {}
    for qubit in range(1, n_qubits + 1):
        p_i=independent_error_prob[qubit]
        nbs=sorted(neighbors[qubit]) 

        neighbor_contribution=0.0
        #Now we have to analyze all configurations of independently flipped neighbore(active neighbors) that can cause a flip on the qubit 
        for bits in itertools.product([0, 1], repeat=len(nbs)):
            config_prob=1.0
            active_neighbors=[]
            for bit, nb in zip(bits, nbs):
                p_nb=independent_error_prob[nb]
                if bit == 1:
                    config_prob *= p_nb
                    active_neighbors.append(nb)
                else:
                    config_prob *= (1 - p_nb)
                
            flip_from_neighbors=prob_exactly_one_trigger(active_neighbors, qubit, conditional_nb_flip_prob)
            neighbor_contribution+= config_prob * flip_from_neighbors
        
        one_body_marginals[qubit] = p_i + (1 - p_i) * neighbor_contribution

    if normalize:
        total=sum(one_body_marginals.values())
        one_body_marginals={qubit: val/total for qubit, val in one_body_marginals.items()}
    return one_body_marginals
                
            
    
#-------------------------------


# Error pattern generation
def generate_error_patterns(num_err, independent_error_prob, conditional_nb_flip_prob):
    error_patterns = []
    for _ in range(num_err):
        pattern = np.zeros(n_qubits + 1, dtype=int)
        # Independent flips
        for qubit in range(1, n_qubits + 1):
            if np.random.rand() < independent_error_prob[qubit]:
                pattern[qubit] = 1
        orig_pattern = pattern.copy()
        # Conditional neighbor flips
        for qubit in range(1, n_qubits + 1):
            for neighbor in neighbors[qubit]:
                if orig_pattern[qubit] == 1 and np.random.rand() < conditional_nb_flip_prob.get((qubit, neighbor), 0):
                    if pattern[neighbor] == 0:
                        pattern[neighbor] = 1
        error_patterns.append(pattern[1:])
    return error_patterns


# Main loop over p_values, q_means, and samples
p_values = [0.006]
q_means = [0.1]
n_samples = 56 #Right now as Google cloud has 56 CPUs, we can only run 56 samples in parallel. We can increase this number if we have more CPUs available.
num_patterns = 100000 #Number of error patterns to generate per sample. We will later filter these down to 1000 total (1 zero + 58 basis + 941 sampled) for training data. We can increase this number if we want more variety in the sampled patterns, but it will also increase runtime.

output_dir_errors = "Standard_BP_OSD/90q_BB_p_0.006_q_0.1_std_0.1_data/testing_data"
output_dir_CER = "Standard_BP_OSD/90q_BB_p_0.006_q_0.1_std_0.1_data/correlated_weights"
output_dir_prob = "Standard_BP_OSD/90q_BB_p_0.006_q_0.1_std_0.1_data/assigned_probabilities"
os.makedirs(output_dir_errors, exist_ok=True)
os.makedirs(output_dir_CER, exist_ok=True)
os.makedirs(output_dir_prob, exist_ok=True)



for p_value in p_values:
    for q_mean in q_means:
        for s in range(1, n_samples + 1):
            # Assign probabilities: each undirected pair gets its own q draw
            independent_error_prob, conditional_nb_flip_prob = assign_error_probabilities(p_value, q_mean)
            #save the assigned probabilities for this sample
            prob_file = os.path.join(output_dir_prob, f"assigned_probabilities_p_{p_value}_q_{q_mean}_s_{s}.txt")
            with open(prob_file, "w") as f:
                f.write("Independent Error Probabilities:\n")
                for key, val in independent_error_prob.items():
                    f.write(f"Qubit {key} : {val}\n")
                f.write("\nConditional Neighbor Flip Probabilities:\n")
                for key, val in conditional_nb_flip_prob.items():
                    f.write(f"Pair {key} : {val}\n")
             

            # Compute CER and normalize
            CER_normalized = compute_CER(independent_error_prob, conditional_nb_flip_prob)

            #compute one-qubit marginals that can be used as LLRs for BP
            one_body_marginals = compute_one_qubit_marginals(independent_error_prob, conditional_nb_flip_prob, normalize=True)
            
            #Saving one body marginals first and then two body marginals
            cer_file= os.path.join(output_dir_CER, f"correlated_weights_p_{p_value}_q_{q_mean}_s_{s}.txt")
            with open(cer_file, "w") as f:
                # Write one-body marginals
                for key,val in one_body_marginals.items():
                    f.write(f" {key} : {val}\n")
                f.write("\n")
                # Write two-body marginals
                for key, val in CER_normalized.items():
                    f.write(f"{key} : {val}\n")
           

            # Generate error patterns
            error_patterns = generate_error_patterns(num_patterns, independent_error_prob, conditional_nb_flip_prob)

            # Save errors
            error_file = os.path.join(output_dir_errors, f"test_ballistic_p_{p_value}_q_{q_mean}_s_{s}.txt")
            np.savetxt(error_file, np.array(error_patterns).T, fmt="%d")
            print(f"Completed sample {s} for p={p_value}, q_mean={q_mean}. CER and error patterns saved.")


# Create training data directory sampling from testing data
training_dir = "Standard_BP_OSD/90q_BB_p_0.006_q_0.1_std_0.1_data/training_data"
os.makedirs(training_dir, exist_ok=True)
for p_value in p_values:
    for q_mean in q_means:
        for s in range(1, n_samples + 1):

            filename = f"Standard_BP_OSD/90q_BB_p_0.006_q_0.1_std_0.1_data/testing_data/test_ballistic_p_{p_value}_q_{q_mean}_s_{s}.txt"

            # load (qubits x patterns)
            data = np.loadtxt(filename, dtype=int)

            # compute weight of each error pattern
            weights = np.sum(data, axis=0)

            # keep patterns with weight >= 2
            filtered = data[:, weights >= 2]

            # sample 10000-90-1 random patterns
            indices = np.random.choice(filtered.shape[1], 9909, replace=False)
            sampled_patterns = filtered[:, indices]

            # create weight-1 basis vectors (n_qubits x n_qubits identity)
            basis_errors = np.eye(n_qubits, dtype=int)

            # create zero vector
            zero_pattern = np.zeros((n_qubits, 1), dtype=int)

            # combine all patterns: 1 zero + 90 basis + 9909 sampled = 10000 total
            final_patterns = np.hstack((zero_pattern, basis_errors, sampled_patterns))

            # sanity check
            print(f"(p={p_value}, q={q_mean}, s={s}) -> patterns: {final_patterns.shape[1]}")

            # save training file
            savefile = os.path.join(training_dir, f"train_ballistic_p_{p_value}_q_{q_mean}_s_{s}.txt")
            np.savetxt(savefile, final_patterns, fmt="%d")

            print(f"Saved: {savefile}")

          
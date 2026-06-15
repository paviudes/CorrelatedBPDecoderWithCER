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
import matplotlib.pyplot as plt

#Write BB_code matrix generations here
#Writing the important functions
# Geometry: Ladder grid R x C
R, C = 2, 45 #change to 2,45 for 90 qubits and change file names accordingly
n_qubits = R * C
output_dir_training_errors = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/training_data"
output_dir_testing_errors = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/testing_data"
output_dir_CER = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/correlated_weights"
output_dir_prob = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/assigned_probabilities"
output_dir_plots = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/plots"
os.makedirs(output_dir_training_errors, exist_ok=True)
os.makedirs(output_dir_testing_errors, exist_ok=True)
os.makedirs(output_dir_CER, exist_ok=True)
os.makedirs(output_dir_prob, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)

p_means=[0.010]
p_std=0.01
q_means=[0.001]
q_std=0.01
n_samples=1
num_patterns=100000

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
def assign_error_probabilities(p_value, q_mean,p_std,q_std):
    independent_error_prob = {}
    conditional_nb_flip_prob = {}
    # Pre-sample one q_value per undirected pair so both directions share it
    undirected_pair_q = {}

    for qubit in range(1, n_qubits + 1):
        p_val=np.random.normal(loc=p_value, scale=p_std) # Sample p_i from a normal distribution around p_value with std-dev
        p_val=np.clip(p_val, 0, 1)  # Ensure p_val is in [0,1]
        independent_error_prob[qubit] = p_val
        
        for neighbor in neighbors[qubit]:
            undirected_key = (min(qubit, neighbor), max(qubit, neighbor))
            if undirected_key not in undirected_pair_q:
                q_val = np.random.normal(loc=q_mean, scale=q_std)# Sample q_val from a normal distribution around q_mean with std-dev=scale
                q_val=np.clip(q_val, 0, 1)  # Ensure q_val is in [0,1]
                undirected_pair_q[undirected_key] = q_val
            conditional_nb_flip_prob[(qubit, neighbor)] = undirected_pair_q[undirected_key]

    return  independent_error_prob, conditional_nb_flip_prob


#one qubit marginals:
#each qubit has independent probability of flip p_i and correlated probability of q_ij_1,q_ij_2,q_ij_3,q_ij_4 for each of its three neighbors
#only odd configurations of the four qubits(the qubit is flipped and an odd number of neighbors are flipped) will lead to a flip in the qubit, so we can calculate the marginal probability of flip for each qubit as follows:

def compute_one_qubit_marginals(independent_error_prob, conditional_nb_flip_prob,neighbors):
    one_qubit_marginals = {}
    for qubit in range(1, n_qubits + 1):
        #probabilities of all mechanisms that can flip the qubit
        probs=[independent_error_prob[qubit]]# gets the p_i for the qubit itself
        for nb in sorted(neighbors[qubit]):
            probs.append(conditional_nb_flip_prob[(qubit, nb)])#gets the q_ij for each neighbor of the qubit
        marginal=0.0

    #enumerate over all binary truth table configuration
        for bits in itertools.product([0, 1], repeat=len(probs)):
            #odd number of fired mechanisms means there is a final error on this qubit
            if sum(bits) % 2 == 1:
                prob_config=1.0
                for bit,prob in zip(bits,probs):
                    if bit==1:
                        prob_config*=prob
                    else:
                        prob_config*=(1-prob)
                marginal+=prob_config
        one_qubit_marginals[qubit]=marginal
    #normalize by the sum of all one-qubit marginals to get a distribution
    total=sum(one_qubit_marginals.values())
    print("Total sum of one-qubit marginals before normalization:", total)  
    #for key in one_qubit_marginals:
        #one_qubit_marginals[key]/=total
    #total_after=sum(one_qubit_marginals.values())
    #print("Total sum of one-qubit marginals after normalization:", total_after)

    return one_qubit_marginals          


#two qubit marginals:
#for two qubits i and j, we can calculate the joint probability of flip for both
#there are two cases:
#1) when the edge between two qubits fire with prob q_ij and both qubits flip . But for this case, we have to make sure, the qubits themselves stayed unflipped from all other contribution beside this edge fire
#2) when the edge between two qubits does not fire, but both qubits flip from other contributions. In this case, we have to make sure that the edge between them did not fire

def compute_two_qubit_marginals_all_edge(independent_error_prob, conditional_nb_flip_prob,neighbors):
    two_qubit_marginals={}
    #loop through all directed edges but store each edge only once
    for qubit_i in range(1, n_qubits + 1):
        for qubit_j in sorted(neighbors[qubit_i]):
            #Avoid double counting by only considering pairs (i,j) with i<j
            if qubit_i >qubit_j:
                continue
            #shared edge probability
            q_ij=conditional_nb_flip_prob[(qubit_i, qubit_j)]
            #building probability list for each qubit, excluding the shared edge contribution
            probs_i=[independent_error_prob[qubit_i]]
            for nb in sorted(neighbors[qubit_i]):
                if nb!=qubit_j:
                    probs_i.append(conditional_nb_flip_prob[(qubit_i, nb)])
            #compute the probability of flip for qubit_i without the contribution of the shared edge, so odd parity of the other contributions
            #call it r_i
            r_i=0.0
            for bits in itertools.product([0, 1], repeat=len(probs_i)):
                if sum(bits) % 2 == 1:
                    prob_config=1.0
                    for bit,prob in zip(bits,probs_i):
                        if bit==1:
                            prob_config*=prob
                        else:
                            prob_config*=(1-prob)
                    r_i+=prob_config
            #similarly compute r_j for qubit_j
            probs_j=[independent_error_prob[qubit_j]]
            for nb in sorted(neighbors[qubit_j]):
                if nb!=qubit_i:
                    probs_j.append(conditional_nb_flip_prob[(qubit_j, nb)])
            
            r_j=0.0
            for bits in itertools.product([0, 1], repeat=len(probs_j)):
                if sum(bits) % 2 == 1:
                    prob_config=1.0
                    for bit,prob in zip(bits,probs_j):
                        if bit==1:
                            prob_config*=prob
                        else:
                            prob_config*=(1-prob)
                    r_j+=prob_config

            #Case 1: shared edge did not fire
            #then both qubits must flip from other contributions, so we need odd parity of the other contributions for both qubits and the shared edge did not fire
            case_edge_not_fired=(1-q_ij)*r_i*r_j
            #Case 2: shared edge fired
            #then both qubits flip from this shared edge, so we need even parity of the other contributions for both qubits and the shared edge fired
            case_edge_fired=q_ij*(1-r_i)*(1-r_j)
            two_qubit_marginal=case_edge_not_fired+case_edge_fired
            two_qubit_marginals[(qubit_i, qubit_j)]=two_qubit_marginal
    #normalize by the sum of all two-qubit marginals to get a distribution
    total=sum(two_qubit_marginals.values())
    print("Total sum of two-qubit marginals before normalization:", total)
    #for key in two_qubit_marginals:
        #two_qubit_marginals[key]/=total
    #total_after=sum(two_qubit_marginals.values())
    #print("Total sum of two-qubit marginals after normalization:", total_after)
    return two_qubit_marginals




def plot_one_qubit_marginals(one_marginals, R, C):
    
    #Visualize one-qubit marginals in three ways:
    #1. histogram
    #2. bar plot by qubit index
    #   3. 2 x C grid heatmap
    

    qubits = sorted(one_marginals.keys())
    values = np.array([one_marginals[q] for q in qubits])

    print("One-qubit marginal statistics:")
    print("min  =", np.min(values))
    print("max  =", np.max(values))
    print("mean =", np.mean(values))
    print("std  =", np.std(values))


    # Bar plot by qubit index
    plt.figure(figsize=(12, 5))
    plt.bar(qubits, values)
    plt.xlabel("Qubit index")
    plt.ylabel("One-qubit marginal probability")
    plt.title("One-qubit marginal for each qubit")
    plt.tight_layout()
    
    #save as pdf
    plot_file = os.path.join(output_dir_plots, f"one_qubit_marginals_bar_s_1.pdf")
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, bbox_inches="tight")




def plot_two_qubit_marginals(two_marginals):
    
    #Visualize nearest-neighbor two-qubit marginals.
    

    edges = sorted(two_marginals.keys())
    values = np.array([two_marginals[e] for e in edges])

    print("Two-qubit marginal statistics:")
    print("min  =", np.min(values))
    print("max  =", np.max(values))
    print("mean =", np.mean(values))
    print("std  =", np.std(values))

    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=20, edgecolor="black")
    plt.xlabel("Two-qubit marginal probability")
    plt.ylabel("Number of edges")
    plt.title("Distribution of two-qubit marginals")
    plt.tight_layout()
    plt.show()

    # Bar plot by edge index
    edge_labels = [f"{i}-{j}" for (i, j) in edges]

    plt.figure(figsize=(16, 5))
    plt.bar(range(len(edges)), values)
    plt.xlabel("Nearest-neighbor edge index")
    plt.ylabel("Two-qubit marginal probability")
    plt.title("Two-qubit marginal for each nearest-neighbor edge")
    plt.tight_layout()
    #save as pdf
    plot_file = os.path.join(output_dir_plots, "two _qubit_marginals_bar_s_1.pdf")
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, bbox_inches="tight")

    # Print largest few edges
    sorted_edges = sorted(two_marginals.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 largest two-qubit marginals:")
    for edge, val in sorted_edges[:10]:
        print(edge, ":", val)

    print("\nBottom 10 smallest two-qubit marginals:")
    for edge, val in sorted_edges[-10:]:
        print(edge, ":", val)


def compare_one_and_two_qubit_marginals(one_marginals, two_marginals):
    
    #Useful to check whether two-qubit marginals are about one order
    #of magnitude smaller.
    

    one_values = np.array(list(one_marginals.values()))
    two_values = np.array(list(two_marginals.values()))

    print("\nComparison:")
    print("mean one-qubit marginal =", np.mean(one_values))
    print("mean two-qubit marginal =", np.mean(two_values))
    print("ratio mean(two)/mean(one) =", np.mean(two_values) / np.mean(one_values))



# Error pattern generation
def generate_error_patterns(num_err, independent_error_prob, conditional_nb_flip_prob):
    error_patterns = []
    for _ in range(num_err):
        pattern = np.zeros(n_qubits + 1, dtype=int)
        # Independent flips
        for qubit in range(1, n_qubits + 1):
            if np.random.rand() < independent_error_prob[qubit]:
                pattern[qubit]^= 1
        # Correlated flips from neighbors
        for qubit_i in range(1,n_qubits+1):
            for qubit_j in sorted(neighbors[qubit_i]):
                if qubit_i>qubit_j:
                    continue
                q_ij=conditional_nb_flip_prob[(qubit_i, qubit_j)]
                if np.random.rand() < q_ij:
                    pattern[qubit_i]^=1
                    pattern[qubit_j]^=1
        error_patterns.append(pattern[1:])  # Exclude index 0
    return error_patterns
    
    





for p_value in p_means:
    for q_value in q_means:
        for s in range(1,n_samples+1):
            # Assign probabilities: each undirected pair gets its own q draw
            independent_error_prob, conditional_nb_flip_prob = assign_error_probabilities(p_value, q_value,p_std,q_std)
            #save the assigned probabilities for this sample
            prob_file = os.path.join(output_dir_prob, f"assigned_probabilities_p_{p_value}_q_{q_value}_s_{s}.txt")
            with open(prob_file, "w") as f:
                f.write("Independent Error Probabilities:\n")
                for key, val in independent_error_prob.items():
                    f.write(f"Qubit {key} : {val}\n")
                f.write("\nConditional Neighbor Flip Probabilities:\n")
                for key, val in conditional_nb_flip_prob.items():
                    f.write(f"Pair {key} : {val}\n")

            one_qubit_marginals = compute_one_qubit_marginals(independent_error_prob, conditional_nb_flip_prob,neighbors)
            two_qubit_marginals = compute_two_qubit_marginals_all_edge(independent_error_prob, conditional_nb_flip_prob,neighbors)
            #make plot for one qubit and two qubit marginal spatial distribution for first sample
            if s==1:
                plot_one_qubit_marginals(one_qubit_marginals, R, C)
                plot_two_qubit_marginals(two_qubit_marginals)
                compare_one_and_two_qubit_marginals(one_qubit_marginals, two_qubit_marginals)
            
            
            cer_file = os.path.join(output_dir_CER, f"correlated_weights_p_{p_value}_q_{q_value}_s_{s}.txt")
            with open(cer_file, "w") as f:
                #f.write("One-qubit Marginals:\n")
                for key, val in one_qubit_marginals.items():
                    f.write(f"{key} : {val}\n")
                #f.write("\nTwo-qubit Marginals:\n")
                for key, val in two_qubit_marginals.items():
                    f.write(f"{key} : {val}\n")

            # Generate error patterns
            # Generate error patterns
            train_error_patterns = generate_error_patterns(num_patterns, independent_error_prob, conditional_nb_flip_prob)
            test_error_patterns = generate_error_patterns(num_patterns, independent_error_prob, conditional_nb_flip_prob)

            # Save errors
            train_error_file = os.path.join(output_dir_training_errors, f"train_ballistic_p_{p_value}_q_{q_value}_s_{s}.txt")
            np.savetxt(train_error_file, np.array(train_error_patterns).T, fmt="%d")
            test_error_file = os.path.join(output_dir_testing_errors, f"test_ballistic_p_{p_value}_q_{q_value}_s_{s}.txt")
            np.savetxt(test_error_file, np.array(test_error_patterns).T, fmt="%d")
            print(f"Completed sample {s} for p={p_value}, q_mean={q_value}. CER and error patterns saved.")



""""
# Create training data directory sampling from testing data
training_dir = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/training_data"
os.makedirs(training_dir, exist_ok=True)
for p_value in p_means:
    for q_value in q_means:
        for s in range(1, n_samples + 1):

            filename = f"data/90q_BB_p_0.010_q_0.001_std_0.01_data/testing_data/test_ballistic_p_{p_value}_q_{q_value}_s_{s}.txt"

            # load (qubits x patterns)
            data = np.loadtxt(filename, dtype=int)

            # compute weight of each error pattern
            weights = np.sum(data, axis=0)

            # keep patterns with weight >= 2
            filtered = data[:, weights >= 2]

            # sample 10000-73 random patterns
            indices = np.random.choice(filtered.shape[1], 9927, replace=False)
            sampled_patterns = filtered[:, indices]

            # create weight-1 basis vectors (n_qubits x n_qubits identity)
            basis_errors = np.eye(n_qubits, dtype=int)

            # create zero vector
            zero_pattern = np.zeros((n_qubits, 1), dtype=int)

            # combine all patterns: 1 zero + 72 basis + 9927 sampled = 10000 total
            final_patterns = np.hstack((zero_pattern, basis_errors, sampled_patterns))

            # sanity check
            print(f"(p={p_value}, q={q_value}, s={s}) -> patterns: {final_patterns.shape[1]}")

            # save training file
            savefile = os.path.join(training_dir, f"train_ballistic_p_{p_value}_q_{q_value}_s_{s}.txt")
            np.savetxt(savefile, final_patterns, fmt="%d")

            print(f"Saved: {savefile}")
"""
          

             

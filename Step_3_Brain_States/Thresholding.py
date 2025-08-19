import os
import sys
import time
import json
import numpy as np
import networkx as nx
from numba import njit
from scipy import stats

# -------------------- Load Optimal Alpha & Bootstrap Median --------------------

def load_alpha_and_median(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        print(data["optimal_alpha"], data["bootstrap_median"])
    return data["optimal_alpha"], data["bootstrap_median"]

json_path =f"/projects/illinois/ahs/kch/nakhan2/ACE/HMM_Output/alpha_and_median.json"
optimal_alpha, bootstrap_median = load_alpha_and_median(json_path)


# -------------------- Process Participant --------------------

def process_participant(participant_dir, participant_id, optimal_alpha, bootstrap_median):
    """Processes each participant's data, applies thresholding, and saves results."""
    results = {}
    success = True
    thresholded_correlation_matrices = {}  # Initialize the dictionary to store thresholded matrices

    for mode in ["congruent", "incongruent"]:
        mode_dir = os.path.join(participant_dir, "Correlation_matrices")
        if not os.path.exists(mode_dir):
            print(f"Skipping {participant_id} - {mode}: No correlation matrices found.")
            continue

        correlation_matrices = {}
        results[mode] = {}

        for filename in os.listdir(mode_dir):
            if filename.startswith(f"{participant_id}_{mode}_correlation_matrices"):
                filepath = os.path.join(mode_dir, filename)
                
                # Check if the file exists before loading
                if not os.path.exists(filepath):
                    print(f"Error: Missing file - {filepath}")
                    continue
                
                print(f"Loading file: {filepath}")  # Debugging print
                try:
                    data = np.load(filepath)
                    #print(f"Loaded keys: {data.files}")  # Debugging print

                    for key in data.files:
                        try:
                            tuple_key = tuple(map(int, key.split('_')))
                            matrix = data[key]  # Directly use the matrix without Fisher's r-to-Z transformation
                            correlation_matrices[tuple_key] = matrix
                        except ValueError:
                            continue  # Skip keys that don't convert to integers

                    if not correlation_matrices:
                        print(f"Warning: No valid correlation matrices found in {filepath}. Skipping...")
                        continue

                    thresholded_matrices, pos_corrs, neg_corrs = threshold_functional_connectivity(
                        correlation_matrices, optimal_alpha, bootstrap_median
                    )
                    results[mode]['correlation_matrices'] = correlation_matrices
                    results[mode]['thresholded_matrices'] = thresholded_matrices
                    thresholded_correlation_matrices.update(thresholded_matrices)  # Store thresholded matrices
                    save_thresholded_data(participant_dir, mode, participant_id, thresholded_matrices, pos_corrs, neg_corrs)
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    success = False
                    continue

    return results, success, thresholded_correlation_matrices  # Return the thresholded matrices along with other results

# -------------------- Save Thresholded --------------------

def save_thresholded_data(participant_dir, mode, participant_id, thresholded_matrices, pos_corrs, neg_corrs):
    """Saves thresholded data in .npz format with properly formatted keys."""
    
    new_dir = os.path.join(participant_dir, "Thresholded_matrices", mode)
    os.makedirs(new_dir, exist_ok=True)
    
    print(f"Saving thresholded data to: {new_dir}")  # Debugging print
    
    # Convert tuple keys to string keys
    thresholded_matrices_str_keys = { "_".join(map(str, key)): value for key, value in thresholded_matrices.items() }
    pos_corrs_str_keys = { "_".join(map(str, key)): value for key, value in pos_corrs.items() }
    neg_corrs_str_keys = { "_".join(map(str, key)): value for key, value in neg_corrs.items() }
    
    # Save with proper string keys
    np.savez(os.path.join(new_dir, f"{participant_id}_{mode}_thresholded_matrices.npz"), **thresholded_matrices_str_keys)
    np.savez(os.path.join(new_dir, f"{participant_id}_{mode}_thresholded_pos_corrs.npz"), **pos_corrs_str_keys)
    np.savez(os.path.join(new_dir, f"{participant_id}_{mode}_thresholded_neg_corrs.npz"), **neg_corrs_str_keys)

# -------------------- Apply Threshold --------------------

@njit
def apply_threshold(corr_matrix, optimal_alpha_squared, bootstrap_median):
    """Applies the threshold to a single correlation matrix using numba for acceleration."""
    size = corr_matrix.shape[0]
    thresholded_matrix = np.zeros_like(corr_matrix)
    
    for i in range(size):
        for j in range(size):
            normalized_weight = corr_matrix[i, j] / bootstrap_median
            if normalized_weight ** 2 >= optimal_alpha_squared:
                thresholded_matrix[i, j] = corr_matrix[i, j]
    
    return thresholded_matrix

# -------------------- Threshold Functional Connectivity --------------------

def threshold_functional_connectivity(correlation_matrices, optimal_alpha, bootstrap_median):
    """Applies threshold based on alpha and bootstrap_median to filter correlation matrices."""
    thresholded_correlation_matrices = {}
    thresholded_positive_correlations = {}
    thresholded_negative_correlations = {}
    optimal_alpha_squared = optimal_alpha ** 2

    for key, corr_matrix in correlation_matrices.items():
        thresholded_matrix = apply_threshold(corr_matrix, optimal_alpha_squared, bootstrap_median)
        thresholded_correlation_matrices[key] = thresholded_matrix
        thresholded_positive_correlations[key] = thresholded_matrix[thresholded_matrix > 0]
        thresholded_negative_correlations[key] = thresholded_matrix[thresholded_matrix < 0]

    return thresholded_correlation_matrices, thresholded_positive_correlations, thresholded_negative_correlations

# -------------------- Main Execution --------------------

start_time = time.time()
base_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/HMM_Output"
participants = [p for p in os.listdir(base_dir) if not p.startswith("alpha")]  # Adjust if needed

# Updated file path for the .json



print(f"Processing {len(participants)} participants...")
all_thresholded_matrices = {}

for idx, participant in enumerate(sorted(participants)):
    participant_dir = os.path.join(base_dir, participant)
   
    results, success, thresholded_matrices = process_participant(participant_dir, participant, optimal_alpha, bootstrap_median)
    if success:
        all_thresholded_matrices[participant] = thresholded_matrices

    sys.stdout.write(f"\rProcessed {idx + 1}/{len(participants)} participants ({(idx + 1) / len(participants) * 100:.2f}%)")
    sys.stdout.flush()

print("\nAll participants processed.")

# -------------------- Timing --------------------

total_time_taken = (time.time() - start_time) / 60
print(f"Total processing time: {total_time_taken:.2f} minutes.")

import os
import sys
import time
import json
import numpy as np
import networkx as nx
from numba import njit
from scipy import stats

# -------------------- Load Correlation Matrices --------------------

def load_all_correlation_matrices(base_dir, participants, modes):
    """Loads all correlation matrices with checks for out-of-bounds values."""
    all_matrices = []
    total_participants = len(participants)
    out_of_bounds_count = 0

    for idx, participant in enumerate(participants):
        participant_dir = os.path.join(base_dir, participant)
        for mode in modes:
            mode_dir = os.path.join(participant_dir, "Correlation_matrices")
            
            filepath = os.path.join(mode_dir, f"{participant}_{mode}_correlation_matrices.npy.npz")
            
            if os.path.exists(filepath):
                try:
                    data = np.load(filepath, allow_pickle=True)
                    for key in data.files:
                        matrix = data[key]
                        if matrix.ndim == 2:
                            if np.any(matrix < -1) or np.any(matrix > 1):
                                out_of_bounds_count += 1
                                print(f"Out-of-bounds values found in {filepath}, key {key}. "
                                      f"Min: {np.min(matrix)}, Max: {np.max(matrix)}")
                            all_matrices.append(matrix)
                        else:
                            print(f"Skipping non-2D matrix in {filepath}, key {key}.")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

        sys.stdout.write(f"\rProcessed {idx + 1}/{total_participants} participants "
                         f"({(idx + 1) / total_participants * 100:.2f}%)")
        sys.stdout.flush()

    print(f"\nTotal matrices loaded: {len(all_matrices)}")
    print(f"Number of out-of-bounds matrices: {out_of_bounds_count / max(1, len(all_matrices)) * 100:.2f}%")
    return all_matrices

# -------------------- Optimal Alpha Calculation --------------------

def determine_optimal_alpha(all_matrices):
    print("Converting correlation matrices to NetworkX graphs...")
    windowed_graphs = []
    total_matrices = len(all_matrices)
    for i, matrix in enumerate(all_matrices):
        try:
            windowed_graphs.append(nx.from_numpy_array(matrix))
            # Update progress every 10%
            print(f"Converting matrix {i + 1}/{total_matrices} to graph ({(i + 1) / total_matrices * 100:.2f}%)", end='\r')
        except Exception as e:
            print(f"Skipping matrix {i + 1}/{total_matrices} due to error: {e}")
    print(f"\nGraph conversion completed. Total graphs converted: {len(windowed_graphs)}")
    return aggregated_bootstrapping_and_alpha_threshold(windowed_graphs)

@njit
def test_alpha_numba(threshold_array, alpha):
    """Checks the proportion of edges that exceed the alpha threshold."""
    row_indices, col_indices = np.where(threshold_array >= alpha)
    valid_connections = np.extract(threshold_array >= alpha, threshold_array)
    return np.sum(valid_connections) / valid_connections.size if valid_connections.size > 0 else 0

def test_alpha_numba_wrapper(threshold_array, alpha): 
    result = test_alpha_numba(threshold_array, alpha)
    return result

def aggregated_bootstrapping_and_alpha_threshold(windowed_graphs, num_iterations=10000, num_alphas=100):
    if not windowed_graphs:
        raise ValueError("No valid graphs found for processing.")
    
    print("Starting aggregation of edge weights from all windowed graphs...")
    all_edge_weights = []
    total_graphs = len(windowed_graphs)
    edge_count = 0

    for i, G in enumerate(windowed_graphs):
        graph_edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        all_edge_weights.extend(graph_edge_weights)
        edge_count += len(graph_edge_weights)
        print(f"Aggregating edge weights progress: {i + 1}/{total_graphs} ({(i + 1) / total_graphs * 100:.2f}%)", end='\r')
        sys.stdout.flush()

    all_edge_weights = np.array(all_edge_weights)
    print(f"\nTotal number of edge weights aggregated: {edge_count}")
    print("Aggregation of edge weights completed.")

    print("Starting bootstrapping on aggregated edge weights...")
    bootstrap_weights = np.zeros_like(all_edge_weights)
    for i in range(num_iterations):
        random_indices = np.random.randint(0, len(all_edge_weights), size=len(all_edge_weights))
        bootstrap_sample = all_edge_weights[random_indices]
        bootstrap_weights += bootstrap_sample
        if (i + 1) % 100 == 0:
            print(f"Bootstrapping progress: {i + 1}/{num_iterations} ({(i + 1) / num_iterations * 100:.2f}%)", end='\r')
    bootstrap_median = np.median(bootstrap_weights / num_iterations)
    print("\nBootstrapping completed.")

    if len(all_edge_weights) == 0:
        raise ValueError("No edge weights found for bootstrapping.")
    
    alpha_start = np.percentile(all_edge_weights, 5) / bootstrap_median
    alpha_end = np.percentile(all_edge_weights, 95) / bootstrap_median
    
    # Pre-calculate threshold arrays here (after bootstrap_median is calculated)
    print("Pre-calculating threshold arrays...")
    threshold_arrays = []
    for G in windowed_graphs:
        connectivity_array = np.asarray(nx.to_numpy_array(G))
        threshold_arrays.append((connectivity_array / bootstrap_median) ** 2)
    
    print("Starting golden-section search for optimal alpha...")
    gr = (np.sqrt(5) + 1) / 2
    c = alpha_end - (alpha_end - alpha_start) / gr
    d = alpha_start + (alpha_end - alpha_start) / gr

    iteration_count = 0
    alpha_values = [c, d]
    fc_values = []
    fd_values = []
    while True:
        iteration_count += 1
        # Calculation using pre-calculated arrays
        fc = np.mean([test_alpha_numba_wrapper(arr, c) for arr in threshold_arrays])
        fd = np.mean([test_alpha_numba_wrapper(arr, d) for arr in threshold_arrays])
        alpha_values.append((c + d) / 2)
        fc_values.append(fc)
        fd_values.append(fd)

        if iteration_count > 1:
            relative_threshold = 0.01 * np.std(alpha_values)
            if abs(c - d) < relative_threshold:
                if len(fc_values) > 5:
                    t_stat, p_value = stats.ttest_rel(fc_values[-5:], fd_values[-5:])
                    if p_value > 0.05:
                        break

        print(f"Testing alphas: {c:.4f}, {d:.4f} - Iteration {iteration_count}", end='\r')
        if fc < fd:
            alpha_end = d
            d = c
            c = alpha_end - (alpha_end - alpha_start) / gr
        else:
            alpha_start = c
            c = d
            d = alpha_start + (alpha_end - alpha_start) / gr

    optimal_alpha = np.median(alpha_values)
    print(f"\nOptimal alpha determined: {optimal_alpha:.4f}")
    print("\nTesting completed")
    return optimal_alpha, bootstrap_median

# -------------------- Save Alpha & Median --------------------

def save_alpha_and_median(base_dir, optimal_alpha, bootstrap_median):
    """Saves the optimal alpha and bootstrap median to a JSON file."""
    save_dir = os.path.join(base_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "alpha_and_median.json")
    with open(save_path, 'w') as file:
        json.dump({'optimal_alpha': optimal_alpha, 'bootstrap_median': bootstrap_median}, file)

    print(f"Optimal alpha and bootstrap median saved for: {save_path}")

# -------------------- Main Execution --------------------

start_time = time.time()

# Base directory and participant list
base_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/HMM_Output"
modes = ["congruent", "incongruent"]
participants = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]

print(f"Processing {len(participants)} participants...")



print("Loading correlation matrices...")
all_matrices = load_all_correlation_matrices(base_dir, participants, modes)


print("Determining optimal alpha value...")
optimal_alpha, bootstrap_median = determine_optimal_alpha(all_matrices)

    # Save results for each participant
save_alpha_and_median(base_dir, optimal_alpha, bootstrap_median)

elapsed_time_minutes = (time.time() - start_time) / 60
print(f"\nTotal processing time: {elapsed_time_minutes:.2f} minutes.")

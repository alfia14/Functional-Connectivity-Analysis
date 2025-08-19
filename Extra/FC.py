import os
import sys
import time
import numpy as np
import itertools
import json

# -------------------- Configuration --------------------

base_dir = "/home/alfiap/scratch/Features_Extraction/outputs/Correlation_matrices/"
participants = ['NU148']

# -------------------- Function Definitions --------------------

def process_participant(base_dir, participant_id):
    """Processes a participant's correlation and thresholded matrices."""
    results = {}
    mode = "Congruent"
    
    correlation_matrices = {}
    results[mode] = {}

    filepath1 = os.path.join(base_dir, f'{participant_id}_{mode}_correlation_matrices.npy.npz')
    filepath2 = os.path.join(base_dir, mode, participant_id, f'{participant_id}_{mode}_thresholded_matrices_raw.npz')

    if not os.path.exists(filepath1) or not os.path.exists(filepath2):
        print(f"Skipping {participant_id}: Missing correlation or thresholded matrices.")
        return results

    data = np.load(filepath1)
    for key in data.files:
        try:
            tuple_key = tuple(map(int, key.split('_')))
            correlation_matrices[tuple_key] = data[key]
        except ValueError:
            continue  # Skip invalid keys

    thresholded_matrices = np.load(filepath2)

    results[mode]['correlation_matrices'] = correlation_matrices
    results[mode]['thresholded_matrices'] = thresholded_matrices

    return results

def load_thresholded_matrices(participants):
    """Loads thresholded matrices for all participants."""
    thresholded_correlation_matrices = {}

    for participant in sorted(participants):
        mode = "Congruent"
        mode_dir = os.path.join(base_dir, mode)
        participant_dir = os.path.join(mode_dir, participant)
        filepath = os.path.join(participant_dir, f"{participant}_{mode}_thresholded_matrices_raw.npz")

        if not os.path.exists(filepath):
            print(f"File not found for {participant} in mode {mode}.")
            continue

        try:
            data = np.load(filepath)
            for key in data.files:
                try:
                    tuple_key = tuple(map(int, key.split('_')))
                    thresholded_correlation_matrices[(participant, mode, tuple_key)] = data[key]
                except ValueError:
                    continue
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return thresholded_correlation_matrices

# Load optimal alpha and bootstrap median
def load_alpha_and_median(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["optimal_alpha"], data["bootstrap_median"]

json_path = "/home/alfiap/scratch/Features_Extraction/outputs/Correlation_matrices/alpha_and_median.json"
optimal_alpha, bootstrap_median = load_alpha_and_median(json_path)

# -------------------- Data Processing --------------------

start_time_1 = time.time()
participants_data = {}

for participant in participants:
    results = process_participant(base_dir, participant)
    participants_data[participant] = results

thresholded_correlation_matrices = load_thresholded_matrices(participants_data.keys())

# -------------------- Functional Connectivity --------------------

def calculate_network_connectivity(thresholded_correlation_matrices, networks):
    """Calculates within and between network connectivity."""
    within_network_connectivity = {}
    between_network_connectivity = {}

    networks_indices = {k: [r[0] for r in v] for k, v in networks.items()}
    
    for (participant, mode, window), corr_matrix in sorted(thresholded_correlation_matrices.items()):
        within_network_connectivity[(participant, mode, window)] = []
        between_network_connectivity[(participant, mode, window)] = []

        for network_name, regions in networks.items():
            network_corr_matrix = corr_matrix[np.ix_(networks_indices[network_name], networks_indices[network_name])]
            upper_tri_indices = np.triu_indices_from(network_corr_matrix, k=1)

            for i, j in zip(*upper_tri_indices):
                region1, region2 = regions[i][1], regions[j][1]
                corr_value = network_corr_matrix[i, j]
                within_network_connectivity[(participant, mode, window)].append(f"[{network_name}]: {region1} - {corr_value:.2f} - {region2}")

        for net1, net2 in itertools.combinations(networks.keys(), 2):
            regions1, regions2 = networks[net1], networks[net2]
            between_corr_matrix = corr_matrix[np.ix_(networks_indices[net1], networks_indices[net2])]

            for i, j in itertools.product(range(len(regions1)), range(len(regions2))):
                region1, region2 = regions1[i][1], regions2[j][1]
                corr_value = between_corr_matrix[i, j]
                between_network_connectivity[(participant, mode, window)].append(f"[{net1}, {net2}]: {region1} - {corr_value:.2f} - {region2}")

    return within_network_connectivity, between_network_connectivity

start_time = time.time()
within_network_conn_values, between_network_conn_values = calculate_network_connectivity(thresholded_correlation_matrices, networks)
print(f"Total time taken for calculation: {(time.time() - start_time) / 60:.2f} minutes")

# -------------------- Saving Connectivity Results --------------------

def save_connectivity_results(participant, mode, within_network_conn_values, 
                              between_network_conn_values):
    """Saves connectivity results to files."""
    save_dir = '/home/alfiap/scratch/Features_Extraction/outputs/Network'
    participant_dir = os.path.join(save_dir, participant)
    mode_dir = os.path.join(participant_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)

    try:
        within_conn_file = os.path.join(mode_dir, f"{participant}_{mode}_within_network_conn_raw.npz")
        np.savez(within_conn_file, **{ "_".join(map(str, key)): value for key, value in within_network_conn_values.items() })

        between_conn_file = os.path.join(mode_dir, f"{participant}_{mode}_between_network_conn_raw.npz")
        np.savez(between_conn_file, **{ "_".join(map(str, key)): value for key, value in between_network_conn_values.items() })

        print(f"Successfully saved connectivity results for {participant}, mode: {mode}")
    except Exception as e:
        print(f"Error saving results for {participant}, mode: {mode}. Reason: {str(e)}")

for participant, results in participants_data.items():
    for mode, data in results.items():
        if 'correlation_matrices' in data and 'thresholded_matrices' in data:
            within_conn_values = {key: value for key, value in within_network_conn_values.items() if key[0] == participant and key[1] == mode}
            between_conn_values = {key: value for key, value in between_network_conn_values.items() if key[0] == participant and key[1] == mode}
            save_connectivity_results(participant, mode, within_conn_values, between_conn_values)

# -------------------- Final Timing --------------------

elapsed_time_minutes = (time.time() - start_time_1) / 60
print(f"Total time taken: {elapsed_time_minutes:.2f} minutes")

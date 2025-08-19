import os
import sys
import time
import numpy as np
import itertools
import json
import pandas as pd
from collections import defaultdict

# -------------------- Configuration --------------------

base_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/HMM_Output/"
network_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/Network_Connectivity/"

median_optimal_states = 13

participants = [p for p in os.listdir(base_dir) if p.startswith("ACE")] #adjust if needed

modes = ["congruent", "incongruent"]

# -------------------- Function Definitions --------------------

def process_participant(base_dir, participant_id):
    """Processes a participant's correlation and thresholded matrices."""
    results = {}

    for mode in modes:
        correlation_matrices = {}
        results[mode] = {}

        filepath1 = os.path.join(base_dir, participant_id, "Correlation_matrices", f"{participant_id}_{mode}_correlation_matrices.npy.npz")
        filepath2 = os.path.join(base_dir, participant_id, "Thresholded_matrices", mode, f"{participant_id}_{mode}_thresholded_matrices.npz")

        if not os.path.exists(filepath1) or not os.path.exists(filepath2):
            print(filepath1)
            print(filepath2)
            print(f"Skipping {participant_id} - {mode}: Missing correlation or thresholded matrices.")
            continue

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
        for mode in modes:
            filepath = os.path.join(base_dir, participant, "Thresholded_matrices", mode, f"{participant}_{mode}_thresholded_matrices.npz")

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



# --------------------  Processing --------------------

start_time_1 = time.time()
participants_data = {}

for participant in participants:
    results = process_participant(base_dir, participant)
    participants_data[participant] = results

thresholded_correlation_matrices = load_thresholded_matrices(participants_data.keys())

# -------------------- Define Network Assignments --------------------

# Load labels from the atlas 
labels = [
    '7Networks_LH_Cont_Cing_1-lh',
    '7Networks_LH_Cont_Par_1-lh',
    '7Networks_LH_Cont_PFCl_1-lh',
    '7Networks_LH_Cont_pCun_1-lh',
    '7Networks_LH_Default_Par_1-lh',
    '7Networks_LH_Default_Par_2-lh',
    '7Networks_LH_Default_pCunPCC_1-lh',
    '7Networks_LH_Default_pCunPCC_2-lh',
    '7Networks_LH_Default_PFC_1-lh',
    '7Networks_LH_Default_PFC_2-lh',
    '7Networks_LH_Default_PFC_3-lh',
    '7Networks_LH_Default_PFC_4-lh',
    '7Networks_LH_Default_PFC_5-lh',
    '7Networks_LH_Default_PFC_6-lh',
    '7Networks_LH_Default_PFC_7-lh',
    '7Networks_LH_Default_Temp_1-lh',
    '7Networks_LH_Default_Temp_2-lh',
    '7Networks_LH_DorsAttn_FEF_1-lh',
    '7Networks_LH_DorsAttn_Post_1-lh',
    '7Networks_LH_DorsAttn_Post_2-lh',
    '7Networks_LH_DorsAttn_Post_3-lh',
    '7Networks_LH_DorsAttn_Post_4-lh',
    '7Networks_LH_DorsAttn_Post_5-lh',
    '7Networks_LH_DorsAttn_Post_6-lh',
    '7Networks_LH_DorsAttn_PrCv_1-lh',
    '7Networks_LH_Limbic_OFC_1-lh',
    '7Networks_LH_Limbic_TempPole_1-lh',
    '7Networks_LH_Limbic_TempPole_2-lh',
    '7Networks_LH_SalVentAttn_FrOperIns_1-lh',
    '7Networks_LH_SalVentAttn_FrOperIns_2-lh',
    '7Networks_LH_SalVentAttn_Med_1-lh',
    '7Networks_LH_SalVentAttn_Med_2-lh',
    '7Networks_LH_SalVentAttn_Med_3-lh',
    '7Networks_LH_SalVentAttn_ParOper_1-lh',
    '7Networks_LH_SalVentAttn_PFCl_1-lh',
    '7Networks_LH_SomMot_1-lh',
    '7Networks_LH_SomMot_2-lh',
    '7Networks_LH_SomMot_3-lh',
    '7Networks_LH_SomMot_4-lh',
    '7Networks_LH_SomMot_5-lh',
    '7Networks_LH_SomMot_6-lh',
    '7Networks_LH_Vis_1-lh',
    '7Networks_LH_Vis_2-lh',
    '7Networks_LH_Vis_3-lh',
    '7Networks_LH_Vis_4-lh',
    '7Networks_LH_Vis_5-lh',
    '7Networks_LH_Vis_6-lh',
    '7Networks_LH_Vis_7-lh',
    '7Networks_LH_Vis_8-lh',
    '7Networks_LH_Vis_9-lh',
    '7Networks_RH_Cont_Cing_1-rh',
    '7Networks_RH_Cont_Par_1-rh',
    '7Networks_RH_Cont_Par_2-rh',
    '7Networks_RH_Cont_PFCl_1-rh',
    '7Networks_RH_Cont_PFCl_2-rh',
    '7Networks_RH_Cont_PFCl_3-rh',
    '7Networks_RH_Cont_PFCl_4-rh',
    '7Networks_RH_Cont_PFCmp_1-rh',
    '7Networks_RH_Cont_pCun_1-rh',
    '7Networks_RH_Default_Par_1-rh',
    '7Networks_RH_Default_pCunPCC_1-rh',
    '7Networks_RH_Default_pCunPCC_2-rh',
    '7Networks_RH_Default_PFCdPFCm_1-rh',
    '7Networks_RH_Default_PFCdPFCm_2-rh',
    '7Networks_RH_Default_PFCdPFCm_3-rh',
    '7Networks_RH_Default_PFCv_1-rh',
    '7Networks_RH_Default_PFCv_2-rh',
    '7Networks_RH_Default_Temp_1-rh',
    '7Networks_RH_Default_Temp_2-rh',
    '7Networks_RH_Default_Temp_3-rh',
    '7Networks_RH_DorsAttn_FEF_1-rh',
    '7Networks_RH_DorsAttn_Post_1-rh',
    '7Networks_RH_DorsAttn_Post_2-rh',
    '7Networks_RH_DorsAttn_Post_3-rh',
    '7Networks_RH_DorsAttn_Post_4-rh',
    '7Networks_RH_DorsAttn_Post_5-rh',
    '7Networks_RH_DorsAttn_PrCv_1-rh',
    '7Networks_RH_Limbic_OFC_1-rh',
    '7Networks_RH_Limbic_TempPole_1-rh',
    '7Networks_RH_SalVentAttn_FrOperIns_1-rh',
    '7Networks_RH_SalVentAttn_Med_1-rh',
    '7Networks_RH_SalVentAttn_Med_2-rh',
    '7Networks_RH_SalVentAttn_TempOccPar_1-rh',
    '7Networks_RH_SalVentAttn_TempOccPar_2-rh',
    '7Networks_RH_SomMot_1-rh',
    '7Networks_RH_SomMot_2-rh',
    '7Networks_RH_SomMot_3-rh',
    '7Networks_RH_SomMot_4-rh',
    '7Networks_RH_SomMot_5-rh',
    '7Networks_RH_SomMot_6-rh',
    '7Networks_RH_SomMot_7-rh',
    '7Networks_RH_SomMot_8-rh',
    '7Networks_RH_Vis_1-rh',
    '7Networks_RH_Vis_2-rh',
    '7Networks_RH_Vis_3-rh',
    '7Networks_RH_Vis_4-rh',
    '7Networks_RH_Vis_5-rh',
    '7Networks_RH_Vis_6-rh',
    '7Networks_RH_Vis_7-rh',
    '7Networks_RH_Vis_8-rh'
]
networks = {
    'Visual': [],
    'Somatomotor': [],
    'DorsalAttention': [],
    'VentralAttention': [],
    'Limbic': [],
    'Frontoparietal': [],
    'Default': []
}

def get_network_name(label_name):
    """Extract the network name from the label."""
    if 'Vis' in label_name:
        return 'Visual'
    elif 'SomMot' in label_name:
        return 'Somatomotor'
    elif 'DorsAttn' in label_name:
        return 'DorsalAttention'
    elif 'VentAttn' in label_name:
        return 'VentralAttention'
    elif 'Limbic' in label_name:
        return 'Limbic'
    elif 'Cont' in label_name or 'Frontoparietal' in label_name:
        return 'Frontoparietal'
    elif 'Default' in label_name:
        return 'Default'
    return None

def get_region_name(label_name):
    """Extract the region name from the label name."""
    parts = label_name.split('-')
    network_name = get_network_name(label_name)
    if network_name == 'Visual' or network_name == 'Somatomotor':
        region_info = parts[0].split('_')[2:]
    else:
        region_info = parts[0].split('_')[3:]
    hemisphere = parts[1]
    return '_'.join(region_info) + '-' + hemisphere

for i, label in enumerate(labels):
    network_name = get_network_name(label)
    if network_name:
        networks[network_name].append((i, label))

# -------------------- Functional Connectivity Calculation --------------------

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
            between_corr_matrix = corr_matrix[np.ix_(networks_indices[net1], networks_indices[net2])]

            for i, j in itertools.product(range(len(networks[net1])), range(len(networks[net2]))):
                region1, region2 = networks[net1][i][1], networks[net2][j][1]
                corr_value = between_corr_matrix[i, j]
                between_network_connectivity[(participant, mode, window)].append(f"[{net1}, {net2}]: {region1} - {corr_value:.2f} - {region2}")
               

    return within_network_connectivity, between_network_connectivity

start_time = time.time()
within_network_conn_values, between_network_conn_values = calculate_network_connectivity(thresholded_correlation_matrices, networks)
print(f"Total time taken for calculation: {(time.time() - start_time) / 60:.2f} minutes")

def aggregate_network_connectivity(thresholded_correlation_matrices, networks, median_optimal_states):
    """
    Aggregate network connectivity measures for each state by averaging across windows.
    NOTE: Add back error handling if need to troubleshoot

    Args:
        thresholded_correlation_matrices (dict): Dictionary containing thresholded correlation matrices 
            for each participant, mode, and state-window combination. Keys are tuples
            in the form (participant, mode, (state, window_number)).
        networks (dict): Dictionary mapping network names to lists of region indices and names.
        median_optimal_states (int): The optimal number of states identified.

    Returns:
        tuple: A tuple containing two dictionaries: aggregated_within_network_connectivity and 
            aggregated_between_network_connectivity.
            - aggregated_within_network_connectivity: Average connectivity within each network for each state.
            - aggregated_between_network_connectivity: Average connectivity between different networks for each state.
    """
    print("Starting aggregation of network connectivity...")
    aggregated_within_network_connectivity = {}
    aggregated_between_network_connectivity = {}
    networks_indices = {k: [r[0] for r in v] for k, v in networks.items()}

    participants_processed = set()
    participant_count = len(participants_data)
    current_count = 0
    error_participants = set()  # Use a set to avoid duplicates
    max_message_length = 0

    # Convert median_optimal_states to an integer
    median_optimal_states = 13

    # Initialize the connectivity dictionaries for each state
    for state in range(median_optimal_states):
        aggregated_within_network_connectivity[state] = {network: [] for network in networks.keys()}
        aggregated_between_network_connectivity[state] = {(net1, net2): [] for net1, net2 in itertools.combinations(networks.keys(), 2)}

    # Aggregate connectivity measures for each state
    for (participant, mode, (state, window_number)), corr_matrix in sorted(thresholded_correlation_matrices.items()):
        # Ensure the state key exists in the dictionaries
        if state not in aggregated_within_network_connectivity:
            aggregated_within_network_connectivity[state] = {network: [] for network in networks.keys()}
        if state not in aggregated_between_network_connectivity:
            aggregated_between_network_connectivity[state] = {(net1, net2): [] for net1, net2 in itertools.combinations(networks.keys(), 2)}

        # Within-network connectivity
        for network_name, regions in networks.items():
            network_corr_matrix = corr_matrix[np.ix_(networks_indices[network_name], networks_indices[network_name])]
            upper_tri_indices = np.triu_indices_from(network_corr_matrix, k=1)
            mean_corr = np.mean(network_corr_matrix[upper_tri_indices])
            if not np.isnan(mean_corr):  # Check if mean_corr is not NaN
                aggregated_within_network_connectivity[state][network_name].append(mean_corr)

        # Between-network connectivity
        for net1, net2 in itertools.combinations(networks.keys(), 2):
            regions1 = networks_indices[net1]
            regions2 = networks_indices[net2]
            between_corr_matrix = corr_matrix[np.ix_(regions1, regions2)]
            mean_corr = np.mean(between_corr_matrix)
            if not np.isnan(mean_corr):  # Check if mean_corr is not NaN
                aggregated_between_network_connectivity[state][(net1, net2)].append(mean_corr)
                

        if participant not in participants_processed:
            ec_processed = any((participant, "congruent", (state, wn)) in thresholded_correlation_matrices for wn in range(window_number + 1))
            eo_processed = any((participant, "incongruent", (state, wn)) in thresholded_correlation_matrices for wn in range(window_number + 1))

            if ec_processed and eo_processed:
                current_count += 1
                participants_processed.add(participant)
                elapsed_time_participant = time.time() - start_time
                progress_message = f"Aggregated network connectivity calculated for participant: {participant} ({current_count}/{participant_count})."
                sys.stdout.write('\r' + ' ' * max_message_length + '\r')  # Clear the line
                sys.stdout.write(f"\r{progress_message}")
                sys.stdout.flush()
                max_message_length = max(max_message_length, len(progress_message))
                
         

    # Calculate average connectivity for each state
    for state in range(median_optimal_states):
        for network in aggregated_within_network_connectivity[state]:
            if aggregated_within_network_connectivity[state][network]: 
                aggregated_within_network_connectivity[state][network] = np.mean(aggregated_within_network_connectivity[state][network])
            else:
                aggregated_within_network_connectivity[state][network] = np.nan  # Assign NaN if list is empty

        for net_pair in aggregated_between_network_connectivity[state]:
            if aggregated_between_network_connectivity[state][net_pair]:  # Check if list is not empty
                aggregated_between_network_connectivity[state][net_pair] = np.mean(aggregated_between_network_connectivity[state][net_pair])
            else:
                aggregated_between_network_connectivity[state][net_pair] = np.nan  # Assign NaN if list is empty

    print("\nCompleted aggregation of network connectivity for all participants.")


    return aggregated_within_network_connectivity, aggregated_between_network_connectivity

# Call the function and measure execution time
start_time = time.time()
aggregated_within_conn, aggregated_between_conn = aggregate_network_connectivity(thresholded_correlation_matrices, networks, median_optimal_states)
print(f"Total time taken for aggregation: {(time.time() - start_time) / 60:.2f} minutes")


# -------------------- Saving Connectivity Results --------------------

def save_connectivity_results(participant, mode, within_network_conn_values, between_network_conn_values,  aggregated_within_conn, 
                              aggregated_between_conn):
    """Saves connectivity results to files."""
    participant_dir = os.path.join(network_dir, participant, mode)
    os.makedirs(participant_dir, exist_ok=True)

    try:
        within_conn_file = os.path.join(participant_dir, f"{participant}_{mode}_within_network_conn_raw.npz")
        between_conn_file = os.path.join(participant_dir, f"{participant}_{mode}_between_network_conn_raw.npz")
        aggregated_conn_file = os.path.join(participant_dir, f"{participant}_{mode}_aggregated_conn_raw.npz")

        # Filter dictionary keys that match participant and mode
        within_conn_data = {
            "_".join(map(str, key)): value
            for key, value in within_network_conn_values.items()
            if key[:2] == (participant, mode)  # Ensure participant and mode match
        }
        between_conn_data = {
            "_".join(map(str, key)): value
            for key, value in between_network_conn_values.items()
            if key[:2] == (participant, mode)  # Ensure participant and mode match
        }

        np.savez(within_conn_file, **within_conn_data)
        np.savez(between_conn_file, **between_conn_data)
        np.savez(aggregated_conn_file, within_conn=aggregated_within_conn, between_conn=aggregated_between_conn)

        print(f" Successfully saved connectivity results for {participant}, mode: {mode}")

       
    except Exception as e:
        print(f" Error saving results for {participant}, mode: {mode}. Reason: {str(e)}")

# Save results
for participant, results in participants_data.items():
    for mode in modes:
        save_connectivity_results(participant, mode, within_network_conn_values, between_network_conn_values, aggregated_within_conn,
                                      aggregated_between_conn)

# -------------------- Final Timing --------------------

elapsed_time_minutes = (time.time() - start_time_1) / 60
print(f"Total time taken: {elapsed_time_minutes:.2f} minutes")



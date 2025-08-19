import os
import numpy as np
import time
import sys
# -------------------- Load Processed Data --------------------

def load_thresholded_data(base_dir, participants):
    """Loads thresholded and correlation matrices for each participant."""
    participants_data = {}

    for participant in participants:
        participant_dir = os.path.join(base_dir, participant, "Thresholded_matrices")
        correlation_dir = os.path.join(base_dir, participant, "Correlation_matrices")

        if not os.path.exists(participant_dir):
            print(f" Skipping {participant}: No 'Thresholded_matrices' folder found.")
            continue
        if not os.path.exists(correlation_dir):
            print(f" Skipping {participant}: No 'Correlation_matrices' folder found.")
            continue

        results = {}

        for mode in ["congruent", "incongruent"]:
            mode_thresholded = os.path.join(participant_dir, mode)
            mode_correlation = os.path.join(correlation_dir, f"{participant}_{mode}_correlation_matrices.npy.npz")  # ✅ Updated to match your format

            if not os.path.exists(mode_correlation):
                print(f"⚠️  Missing correlation file: {mode_correlation}")
                continue
            if not os.path.exists(mode_thresholded):
                print(f"⚠️  Missing thresholded folder: {mode_thresholded}")
                continue

            thresholded_file = os.path.join(mode_thresholded, f"{participant}_{mode}_thresholded_matrices.npz")
            if not os.path.exists(thresholded_file):
                print(f"⚠️  Missing thresholded file: {thresholded_file}")
                continue

            try:
                # ✅ Load correlation matrices (from .npy.npz)
                correlation_data = np.load(mode_correlation)
                correlation_matrices = {tuple(map(int, key.split('_'))): correlation_data[key] for key in correlation_data.files}

                # ✅ Load thresholded matrices (from .npz)
                thresholded_data = np.load(thresholded_file)
                thresholded_matrices = {tuple(map(int, key.split('_'))): thresholded_data[key] for key in thresholded_data.files}

                results[mode] = {
                    "correlation_matrices": correlation_matrices,
                    "thresholded_matrices": thresholded_matrices
                }
            except Exception as e:
                print(f"🚨 Error loading data for {participant} - {mode}: {e}")
                continue

        if results:
            participants_data[participant] = results

    return participants_data

# -------------------- Main Execution --------------------

start_time = time.time()
base_dir = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/HMM_Output"
participants = [p for p in os.listdir(base_dir) if p.startswith("NU")]

print(f"Loading processed data for {len(participants)} participants...\n")
participants_data = load_thresholded_data(base_dir, participants)

if not participants_data:
    print(" No valid participants found. Check your files and paths.")
    exit()

# -------------------- EDGE COUNT SUMMARY --------------------

from numba import njit

@njit
def count_edges(matrix):
    """Counts the non-zero entries in the matrix that represent edges."""
    count = 0
    size = matrix.shape[0]
    for i in range(size):
        for j in range(i + 1, size):  # Only count each edge once in an undirected graph
            if matrix[i, j] != 0:
                count += 1
    return count

# Initialize counters
total_unthresholded_edges = 0
total_thresholded_edges = 0
total_edge_counts = 0
thresholded_participants = []
no_modes_processed = []
one_mode_processed = []

print("\nProcessing edge count summary...\n")

for idx, participant in enumerate(sorted(participants_data.keys())):
    data = participants_data[participant]
    processed_modes = [mode for mode, results in data.items() if 'correlation_matrices' in results and 'thresholded_matrices' in results]

    if not processed_modes:
        no_modes_processed.append(participant)
    elif len(processed_modes) == 1:
        one_mode_processed.append((participant, processed_modes[0]))
    else:
        thresholded_participants.append(participant)
        # Loop through each processed mode
        for mode in processed_modes:
            results = data[mode]
            unthresholded_graph_edges = 0
            thresholded_graph_edges = 0
            for key, corr_matrix in results['correlation_matrices'].items():
                unthresholded_graph_edges += count_edges(corr_matrix)
                thresholded_graph_edges += count_edges(results['thresholded_matrices'][key])

            # Update total counts for the participant in each mode
            total_unthresholded_edges += unthresholded_graph_edges
            total_thresholded_edges += thresholded_graph_edges
            total_edge_counts += 1  # This counts the number of mode entries processed, not participants

            sys.stdout.write(f"\rProcessing participant {participant}, Mode {mode}: {idx + 1}/{len(participants_data)}")
            sys.stdout.flush()

print("\n\nAll participants processed.\n")

# Print summary
if no_modes_processed:
    print(" Participants with no modes processed:", ", ".join(no_modes_processed))
if one_mode_processed:
    print("Participants with only one mode processed:")
    for participant, mode in one_mode_processed:
        print(f"  - {participant} (Mode: {mode})")
if thresholded_participants:
    print("Participants with all modes processed and thresholded:", ", ".join(thresholded_participants))

# Calculate and print averages if applicable
if total_edge_counts > 0:
    average_unthresholded = total_unthresholded_edges / total_edge_counts
    average_thresholded = total_thresholded_edges / total_edge_counts
    average_pruned = average_unthresholded - average_thresholded

    print(f"\n Average unthresholded edge count: {average_unthresholded:.2f}")
    print(f" Average thresholded edge count: {average_thresholded:.2f}")
    print(f" Average number of pruned edges: {average_pruned:.2f}")
else:
    print("\n No edge count data available.")

# End timing
end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
print(f"\n Total time taken: {elapsed_time_minutes:.2f} minutes")

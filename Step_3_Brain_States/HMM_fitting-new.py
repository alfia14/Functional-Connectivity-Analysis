import os
import numpy as np
import mne
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import time
import sys
import json
import time
import networkx as nx
from numba import njit
from scipy import stats

base_dir = "/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Orthogonalized_data"

participants = [p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))]

modes = ['congruent', 'incongruent']

total_participants = len(participants) * len(modes)
current_count = 0
start_time = time.time()
#change the number of optimal median states based on dataset (depends on pervious script-see notes)
#NURISH
#median_optimal_state = 13
#ACE_XZ
median_optimal_state = 16

def check_correlation_range(corr_matrix):
    """Check if the correlation matrix values are within the range [-1, 1]."""
    if np.any(corr_matrix < -1) or np.any(corr_matrix > 1):
        return False
    return True

def validate_data(data, context):
    """Validate the input data and ensure it does not contain NaNs or empty slices."""
    if data.size == 0:
        raise ValueError(f"Empty data encountered in {context}.")
    if np.isnan(data).any():
        raise ValueError(f"NaN values encountered in {context}.")
    if np.isinf(data).any():
        raise ValueError(f"Infinite values encountered in {context}.")
    return data

# Uncomment if troubleshooting is needed
# Initialize counters for empty data cases
# total_initial_empty_cases = 0
# total_replaced_empty_cases = 0
# total_remaining_empty_cases = 0

for participant in participants:
    participant_start_time = time.time()  # Record start time for the participant
    for mode in modes:
        input_file = f"/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Orthogonalized_data/{participant}/{mode}/orth.npy"

        base_output_dir = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/HMM_Output'
        output_dir = os.path.join(base_output_dir, participant)
        try:
            if not os.path.exists(input_file):
                print(f"Error: File not found for participant {participant}, mode {mode}: {input_file}")
                continue  # Skip to the next iteration if the file does not exist

            # Load orthogonalized data
            orthogonalized_data = np.load(input_file)
            orthogonalized_data = validate_data(orthogonalized_data, f"orthogonalized data for participant {participant}, mode {mode}")
            
            features = np.mean(orthogonalized_data, axis=2)
            features = validate_data(features, f"mean features for participant {participant}, mode {mode}")
            features = np.ma.masked_invalid(features).filled(np.mean(features, axis=0))
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            #median_optimal_state = np.median(optimal_states[mode])
            
            model = hmm.GaussianHMM(n_components=int(median_optimal_state), n_iter=50, 
                                    covariance_type='full', tol=1e-7, verbose=False, 
                                    params='st', init_params='stmc')
            model.fit(features)
            state_sequence = model.predict(features)
            state_probs = model.predict_proba(features)

            #output_dir = os.path.join(output_dir, participant, mode)

            output_state_sequences = os.path.join(output_dir, "State_sequences")
            output_state_probs = os.path.join(output_dir, "State_probabilities")
            output_correlation_matrices = os.path.join(output_dir, "Correlation_matrices")

            # Ensure subdirectories exist
            os.makedirs(output_state_sequences, exist_ok=True)
            os.makedirs(output_state_probs, exist_ok=True)
            os.makedirs(output_correlation_matrices, exist_ok=True)

            np.save(os.path.join(output_state_sequences, f"{participant}_{mode}_state_sequence.npy"), state_sequence)
            np.save(os.path.join(output_state_probs, f"{participant}_{mode}_state_probs.npy"), state_probs)


            # CALCULATE TEMPORAL FEATURES
            
            # Compute fractional occupancy: fraction of time spent in each state
            fractional_occupancy = np.array([np.sum(state_sequence == i) / len(state_sequence) for i in range(int(median_optimal_state))])

            # Compute transition probabilities
            transition_counts = np.zeros((int(median_optimal_state), int(median_optimal_state)))
            for (i, j) in zip(state_sequence[:-1], state_sequence[1:]):
                transition_counts[i, j] += 1
            transition_probabilities = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)

            # Compute mean lifetime (dwell time) in each state: average time spent in each state before transitioning
            mean_lifetime = np.zeros(int(median_optimal_state))
            for i in range(int(median_optimal_state)):
                # Identify the indices where state changes
                change_indices = np.where(np.diff(state_sequence == i, prepend=False, append=False))[0]
                # Calculate segment lengths by differencing indices of changes; add 1 because diff loses 1
                segment_lengths = np.diff(change_indices) + 1
                # Compute mean segment length for state i
                mean_lifetime[i] = np.mean(segment_lengths) if len(segment_lengths) > 0 else 0

            # Mean Interval Length: average time between consecutive occurrences of each state
            mean_interval_length = np.zeros(int(median_optimal_state))
            for k in range(int(median_optimal_state)):
                # Boolean array where True is the state 'k'
                is_state_k = state_sequence == k
                # Time points where state is 'k'
                time_points_k = np.where(is_state_k)[0]
                # Compute time differences between consecutive occurrences of state 'k'
                intervals_k = np.diff(time_points_k) - 1
                # Compute the mean of these intervals, accounting for the case where state 'k' does not repeat
                mean_interval_length[k] = np.mean(intervals_k) if len(intervals_k) > 0 else 0
                
            np.savez(os.path.join(output_correlation_matrices, f"{participant}_{mode}_temporal_features.npz"), 
                    fractional_occupancy=fractional_occupancy, 
                    transition_probabilities=transition_probabilities,
                    mean_lifetime=mean_lifetime, 
                    mean_interval_length=mean_interval_length)

                    
            # CALCULATE SPATIAL FEATURES (FUNCTIONAL CONNECTIVITY)
            
            def calculate_functional_connectivity(orthogonalized_data, state_sequence, median_optimal_state):
                correlation_matrices = {}
                positive_correlations = {}
                negative_correlations = {}

                initial_empty_case_count = 0  # Counter for initial empty data cases
                replaced_empty_case_count = 0  # Counter for replaced empty data cases

                for state in range(int(median_optimal_state)):
                    state_indices = np.where(state_sequence == state)[0]

                    block_starts = np.where(np.diff(state_indices) > 1)[0] + 1
                    block_starts = np.insert(block_starts, 0, 0)
                    block_ends = np.append(block_starts[1:] - 1, len(state_indices) - 1)
                    state_blocks = zip(state_indices[block_starts], state_indices[block_ends] + 1)

                    for i, (start_index, end_index) in enumerate(state_blocks):
                        state_data = orthogonalized_data[:, :, start_index:end_index]

                        if state_data.size == 0:
                            initial_empty_case_count += 1  # Increment initial empty case count
                            # Replace empty data with the mean of the input data
                            state_data = np.mean(orthogonalized_data, axis=2, keepdims=True) 
                            replaced_empty_case_count += 1

                        state_data = validate_data(state_data, f"state data for state {state}, block {i}, participant {participant}, mode {mode}")

                        # Select the time and sample dimensions for correlation
                        reshaped_data = state_data.swapaxes(1, 2).reshape(102, -1)

                        # Impute NaNs if necessary
                        if np.isnan(state_data).any():
                            reshaped_data = np.nan_to_num(reshaped_data)

                        corr_matrix = np.corrcoef(reshaped_data)
                        if not check_correlation_range(corr_matrix):
                            raise ValueError(f"Correlation values out of range in state {state}, block {i} for participant {participant}, mode {mode}. Check data preprocessing.")

                        correlation_matrices[(state, i)] = corr_matrix

                        upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                        positive_correlations[(state, i)] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] > 0]
                        negative_correlations[(state, i)] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] < 0]

                remaining_empty_case_count = sum(1 for key, value in correlation_matrices.items() if value.size == 0)

                return correlation_matrices, positive_correlations, negative_correlations, initial_empty_case_count, replaced_empty_case_count, remaining_empty_case_count

            # CALCULATE SPATIAL FEATURES (FUNCTIONAL CONNECTIVITY)
            
            correlation_matrices, positive_correlations, negative_correlations, initial_empty_case_count, replaced_empty_case_count, remaining_empty_case_count = calculate_functional_connectivity(
                orthogonalized_data, state_sequence, median_optimal_state)

            correlation_matrices_file = os.path.join(output_dir, f"Correlation_matrices/{participant}_{mode}_correlation_matrices.npy")
            positive_correlations_file = os.path.join(output_dir, f"Correlation_matrices/{participant}_{mode}_positive_correlations.npy")
            negative_correlations_file = os.path.join(output_dir, f"Correlation_matrices/{participant}_{mode}_negative_correlations.npy")
            # Convert tuple keys to strings
            arrays_to_save = {}
            for key, value in correlation_matrices.items():
                key_str = "_".join(map(str, key))
                arrays_to_save[key_str] = value
            np.savez(correlation_matrices_file, **arrays_to_save)
            np.savez(positive_correlations_file, positive_correlations)
            np.savez(negative_correlations_file, negative_correlations)

        except Exception as e:
            print(f"Error processing participant {participant}, mode {mode}: {e}")
        
        current_count += 1
        participant_elapsed_time = (time.time() - participant_start_time) / 60
        total_elapsed_time = (time.time() - start_time) / 60
        avg_time_per_participant = total_elapsed_time / current_count
        #progress_percent = (current_count / total_participants) * 100  # Uncomment if the need to keep track of participant's processing progress arises
        sys.stdout.write(f"\rProcessing {participant} | Mode: {mode} | "
                         f"Participant Progress: {participant_elapsed_time:.2f} min | "
                         #f"Overall Progress: {progress_percent:.2f}% | " # Uncomment if the need to keep track of participant's processing progress arises
                         f"Avg Time/Participant: {avg_time_per_participant:.2f} min")
        sys.stdout.flush()

print("\nAll HMM fittings completed.")
total_time_taken = (time.time() - start_time) / 60
print(f"Total processing time: {total_time_taken:.2f} minutes. Average time per participant: {avg_time_per_participant:.2f} minutes.")


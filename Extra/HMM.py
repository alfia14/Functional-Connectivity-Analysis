import os
import sys
import time
import argparse
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Argument parser for subject_id
parser = argparse.ArgumentParser(description="HMM Fitting for a Single Participant")
parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
args = parser.parse_args()

subject_id = args.subject_id

print(f"Processing participant: {subject_id}")

# Define input and output directories
files_in = '/projects/illinois/ahs/kch/nakhan2/Data/Orthogonalized_data'
files_out = '/projects/illinois/ahs/kch/nakhan2/Data/Correlation_Matrices'

modes = ["congruent", "incongruent"]
optimal_states = {"congruent": 5, "incongruent": 6}

def validate_data(data, context):
    """Check for empty, NaN, or infinite values."""
    if data.size == 0:
        raise ValueError(f"Empty data encountered in {context}.")
    if np.isnan(data).any():
        raise ValueError(f"NaN values encountered in {context}.")
    if np.isinf(data).any():
        raise ValueError(f"Infinite values encountered in {context}.")
    return data

def convert_keys_to_strings(dictionary):
    """Convert dictionary keys to strings for saving in .npz files."""
    return {str(key): value for key, value in dictionary.items()}

def process_hmm(subject, mode, input_dir, output_dir):
    """Loads orthogonalized data, fits HMM, calculates temporal/spatial features, and saves results."""
    file_path = os.path.join(input_dir, "orth.npy")
    
    if os.path.exists(file_path):
        try:
            start_time = time.time()

            # Load orthogonalized data
            orthogonalized_data = np.load(file_path)
            orthogonalized_data = validate_data(orthogonalized_data, f"orthogonalized data for {subject} in {mode}")

            # Feature extraction
            features = np.mean(orthogonalized_data, axis=2)
            features = validate_data(features, f"features for {subject} in {mode}")

            # Normalize features
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            # Prevent overfitting: Reduce components if data is too small
            median_optimal_state = min(optimal_states[mode], features.shape[0] // 10)
            model = hmm.GaussianHMM(n_components=median_optimal_state, n_iter=50, covariance_type='full', tol=1e-7)
            model.fit(features)
            state_sequence = model.predict(features)

            # Compute temporal features
            fractional_occupancy = np.array([np.sum(state_sequence == i) / len(state_sequence) for i in range(median_optimal_state)])
            transition_counts = np.zeros((median_optimal_state, median_optimal_state))
            for (i, j) in zip(state_sequence[:-1], state_sequence[1:]):
                transition_counts[i, j] += 1
            transition_probabilities = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
            mean_lifetime = np.array([np.mean(np.diff(np.where(state_sequence == i)[0])) if np.where(state_sequence == i)[0].size > 1 else 0 for i in range(median_optimal_state)])
            mean_interval_length = np.array([np.mean(np.diff(np.where(state_sequence == k)[0])) if np.where(state_sequence == k)[0].size > 1 else 0 for k in range(median_optimal_state)])

            # Compute spatial features
            correlation_matrices = {}
            positive_correlations = {}
            negative_correlations = {}

            for state in range(median_optimal_state):
                state_indices = np.where(state_sequence == state)[0]
                if len(state_indices) == 0:
                    continue  # Skip if no occurrences of the state

                state_data = orthogonalized_data[:, :, state_indices]
                reshaped_data = state_data.swapaxes(1, 2).reshape(102, -1)

                if np.isnan(state_data).any():
                    reshaped_data = np.nan_to_num(reshaped_data)

                corr_matrix = np.corrcoef(reshaped_data)
                correlation_matrices[state] = corr_matrix

                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                positive_correlations[state] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] > 0]
                negative_correlations[state] = corr_matrix[upper_tri_indices][corr_matrix[upper_tri_indices] < 0]

            # Save results
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{subject}_state_sequence.npy"), state_sequence)

            np.savez(os.path.join(output_dir, f"{subject}_temporal_features.npz"),
                     fractional_occupancy=fractional_occupancy,
                     transition_probabilities=transition_probabilities,
                     mean_lifetime=mean_lifetime,
                     mean_interval_length=mean_interval_length)

            np.savez(os.path.join(output_dir, f"{subject}_correlation_matrices.npz"), **convert_keys_to_strings(correlation_matrices))
            np.savez(os.path.join(output_dir, f"{subject}_positive_correlations.npz"), **convert_keys_to_strings(positive_correlations))
            np.savez(os.path.join(output_dir, f"{subject}_negative_correlations.npz"), **convert_keys_to_strings(negative_correlations))

            elapsed_time = (time.time() - start_time) / 60
            print(f"HMM processing for {subject} in {mode} completed in {elapsed_time:.2f} minutes.")

        except Exception as e:
            print(f"Error processing {subject} in {mode}: {e}")

# Process the subject for both modes
for mode in modes:
    input_dir = os.path.join(files_in, subject_id, mode)
    output_dir = os.path.join(files_out, subject_id, mode)
    os.makedirs(output_dir, exist_ok=True)
    process_hmm(subject_id, mode, input_dir, output_dir)

print(f"Processing complete for participant: {subject_id}")

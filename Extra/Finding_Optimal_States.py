import os
import numpy as np
import mne
import argparse
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time


# Parse command-line argument for subject_id
parser = argparse.ArgumentParser(description="EEG Source Reconstruction")
parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
args = parser.parse_args()

subject_id = args.subject_id

print(f"Processing subject: {subject_id}")

# Define input and output directories
files_in = '/projects/illinois/ahs/kch/nakhan2/Data/Orthogonalized_data'
files_out = '/projects/illinois/ahs/kch/nakhan2/Data/Optimal_States'

modes = ["congruent", "incongruent"]

def determine_optimal_states(orthogonalized_data, subject, mode):
    """Determines the optimal number of states using HMM."""
    reshaped_data = orthogonalized_data.reshape(-1, orthogonalized_data.shape[-1])
    
    pca = PCA(n_components=0.99)
    pca_data = pca.fit_transform(reshaped_data)
    
    scaler = StandardScaler()
    pca_data = scaler.fit_transform(pca_data)
    
    participant_start_time = time.time()
    state_numbers = range(3, 17)
    
    results = []
    
    for n_states in state_numbers:
        state_start_time = time.time()
        print(f"Processing state: {n_states} | Subject: {subject} | Mode: {mode}")
        
        model = hmm.GaussianHMM(n_components=n_states, n_iter=50, covariance_type='full', tol=1e-7, verbose=False)
        model.fit(pca_data)
        
        log_likelihood = model.score(pca_data)
        n_params = n_states * (2 * pca_data.shape[1] - 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(pca_data.shape[0]) * n_params - 2 * log_likelihood
        
        results.append((n_states, aic, bic))
        state_elapsed_time = (time.time() - state_start_time) / 60
        print(f"Time taken for state {n_states}: {state_elapsed_time:.2f} minutes")
    
    optimal_states = min(results, key=lambda x: x[2])[0]  # Select based on BIC
    
    with open(f"aic_bic_{subject}_{mode}.txt", "w") as f:
        for n_states, aic, bic in results:
            f.write(f"{n_states}\t{aic}\t{bic}\n")
        f.write(f"\nOptimal state: {optimal_states}\n")
    
    print(f"Optimal number of states for {subject} in {mode}: {optimal_states}")
    return optimal_states

def process_participant(subject, mode, dir_in, dir_out):
    orthogonalized_file = os.path.join(dir_in, "orth.npy")
    
    if os.path.exists(orthogonalized_file):
        try:
            orthogonalized_data = np.load(orthogonalized_file)
            print(f"Loaded orthogonalized data for {subject} in mode {mode}")
            
            optimal_states = determine_optimal_states(orthogonalized_data, subject, mode)
            return optimal_states
        except Exception as e:
            print(f"Error processing {subject} in {mode}: {e}")
    else:
        print(f"File not found: {orthogonalized_file}")
    
    return None

# Process the given subject for both modes
for mode in modes:
    input_dir = os.path.join(files_in, subject_id, mode)
    output_dir = os.path.join(files_out, subject_id, mode)
    os.makedirs(output_dir, exist_ok=True)
    process_participant(subject_id, mode, input_dir, output_dir)

print(f"Processing complete for subject: {subject_id}")

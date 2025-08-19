import os
import sys
import numpy as np
import mne
from mne_connectivity import symmetric_orth
from scipy.signal import hilbert, resample, butter, lfilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import time
import os.path as op
import argparse

def log_message(message):
    """Logs message immediately to stdout and flushes."""
    print(message)
    sys.stdout.flush()

def downsample_with_filtering(data, original_fs, target_fs):
    """Downsamples data with an anti-aliasing filter."""
    log_message("Starting downsampling with filtering...")
    nyq_rate = original_fs / 2.0
    cutoff_freq = target_fs / 2.0
    normalized_cutoff = cutoff_freq / nyq_rate
    b, a = butter(4, normalized_cutoff, btype='low')
    filtered_data = lfilter(b, a, data, axis=2)
    
    duration = data.shape[2] / original_fs
    new_num_samples = int(duration * target_fs)
    downsampled_data = resample(filtered_data, new_num_samples, axis=2)
    
    log_message("Downsampling completed.")
    return downsampled_data

def determine_optimal_states(orthogonalized_data, subject, mode, files_out):
    """Determines the optimal number of states using AIC and BIC criteria."""
    log_message(f"Starting determine_optimal_states function for subject {subject}, mode {mode}")

    log_message("Step 1 - Computing mean features")
    features = np.mean(orthogonalized_data, axis=2)
    features = np.ma.masked_invalid(features).filled(0)

    log_message("Step 2 - Reshaping data for PCA")
    reshaped_data = orthogonalized_data.reshape(orthogonalized_data.shape[0], -1).T

    log_message("Step 3 - Applying PCA")
    pca = PCA(n_components=0.99)
    pca_data = pca.fit_transform(reshaped_data)
   
    log_message("Step 4 - Standardizing PCA data")
    scaler = StandardScaler()
    pca_data = scaler.fit_transform(pca_data)

    log_message("Step 5 - Initializing state range")
    state_numbers = range(3, 17)

    os.makedirs(files_out, exist_ok=True)
    aic_bic_file = op.join(files_out, f"aic_bic_{subject}_{mode}.txt")
    
    log_message("Step 6 - Iterating through different numbers of states")
    for n_states in state_numbers:
        state_start_time = time.time()
        log_message(f"Processing state: {n_states} | Subject: {subject} | Mode: {mode}")
        
        model = hmm.GaussianHMM(n_components=n_states, n_iter=50, covariance_type='full', tol=1e-7, verbose=False)
        model.fit(pca_data)
        log_likelihood = model.score(pca_data)
        n_params = n_states * (2 * pca_data.shape[1] - 1)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(pca_data.shape[0]) * n_params - 2 * log_likelihood

        state_elapsed_time = (time.time() - state_start_time) / 60
        log_message(f"Time taken for state {n_states}: {state_elapsed_time:.2f} minutes")

        with open(aic_bic_file, "a") as f:
            f.write(f"{n_states}\t{aic}\t{bic}\n")

    log_message("Step 7 - Finding optimal number of states")
    with open(aic_bic_file, "r") as f:
        lines = f.readlines()
        state_numbers = []
        aics = []
        bics = []
        for line in lines:
            parts = line.split()
            state_numbers.append(int(parts[0]))
            aics.append(float(parts[1]))
            bics.append(float(parts[2]))
        min_aic_index = np.argmin(aics)
        min_bic_index = np.argmin(bics)
        optimal_state_aic = state_numbers[min_aic_index]
        optimal_state_bic = state_numbers[min_bic_index]
        optimal_states = int((optimal_state_aic + optimal_state_bic) / 2)

    log_message("Step 8 - Writing results to file")
    optimal_state_file = op.join(files_out, f"optimal_states_{subject}_{mode}.txt")
    with open(optimal_state_file, "a") as f:
        f.write(f"\nOptimal state (AIC): {optimal_state_aic}\n")
        f.write(f"Optimal state (BIC): {optimal_state_bic}\n")
        f.write(f"Optimal state (Average): {optimal_states}\n")

    log_message(f"Optimal number of states for {subject} ({mode}): {optimal_states}")
    return optimal_states

def main():
    parser = argparse.ArgumentParser(description="Process EEG data for a given subject.")
    parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
    args = parser.parse_args()
    subject = args.subject_id

    files_out = '/projects/illinois/ahs/kch/nakhan2/ACE/Optimal_States'
    files_in = '/projects/illinois/ahs/kch/nakhan2/ACE/Orthogonalized_data'

    modes = ['congruent', 'incongruent']

    for mode in modes:
        input_file = os.path.join(files_in, subject, mode, f"orth.npy")
        
        if os.path.exists(input_file):
            log_message(f"Loading previously generated orthogonalized data for {subject}, mode {mode}")
            orthogonalized_data = np.load(input_file)
            determine_optimal_states(orthogonalized_data, subject, mode, files_out)
        else:
            log_message(f"Error: Orthogonalized data not found for {subject}, mode {mode} at {input_file}")

if __name__ == "__main__":
    log_message("Starting EEG processing script...")
    main()
    log_message("EEG processing completed.")

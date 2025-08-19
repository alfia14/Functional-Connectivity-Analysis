import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mne_icalabel import label_components
from mne.preprocessing import ICA
import sys

# Check if a subject ID was provided as argument
if len(sys.argv) != 2:
    print("Usage: python script_name.py <subject_id>")
    sys.exit(1)

# Get subject ID from command line argument
subject_id_arg = sys.argv[1]

# Paths
input_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Post/ICA/BD_Interpolated'
output_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Post/ICA'
exclude_idx_path = f"{output_path}/ICA_ExcludeIdx"
timeseries_path = f"{output_path}/Timeseries"
topographies_path = f"{output_path}/Topographies"
labels_path = f"{output_path}/ICA_Labels"

# Ensure directories exist
os.makedirs(exclude_idx_path, exist_ok=True)
os.makedirs(timeseries_path, exist_ok=True)
os.makedirs(topographies_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

average_components = 20

def process_subject(subject_filename):
    print(f"Processing {subject_filename}")
    subject_id = subject_filename.split('_')[0]
    
    # Check if this is the subject we're looking for
    if subject_id != subject_id_arg:
        return
    
    # Load EEG data
    EEG = mne.io.read_raw_fif(f"{input_path}/{subject_filename}", preload=True)
    EEG.set_eeg_reference('average')
    exclude_channels = ['VEOG', 'HEOG', 'M1', 'M2', 'F11', 'F12', 'FT11', 'FT12', 'CB1', 'CB2']
    data = EEG.get_data().T
    explained_variance_threshold = 0.99
    
    # Perform PCA
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_actual = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
    n_components = max(n_components_actual, average_components)
    
    # Save the number of ICA components used
    df = pd.DataFrame({'n_components': [n_components]})
    df.to_csv(f"{exclude_idx_path}/{subject_id}_n_components.csv", index=False)
    
    # Perform ICA
    ica = ICA(n_components=n_components, max_iter="auto", method="infomax",
              random_state=97, fit_params=dict(extended=True))
    picks_eeg = mne.pick_types(EEG.info, meg=False, eeg=True, eog=False,
                               stim=False, emg=False, exclude=exclude_channels)
    ica.fit(EEG, picks=picks_eeg, decim=3)
    
    # Plot ICA time series and save it
    fig = ica.plot_sources(EEG, show_scrollbars=False, show=False)
    fig.savefig(f"{timeseries_path}/{subject_id}_ica_timeseries.png")
    plt.close(fig)
    
    # Plot ICA components
    figs = ica.plot_components(show=False)  # Returns a LIST of figures

    # If figs is a list (new MNE versions), save only the first figure
    if isinstance(figs, list):
        fig = figs[0]  # Take the first figure (the big grid layout)
    else:
        fig = figs  # If older version and only one figure, use it directly

    # Save the figure
    fig.savefig(f"{topographies_path}/{subject_id}_ica_components.png")
    plt.close(fig)

    
    # Label ICA components
    ic_labels = label_components(EEG, ica, method='iclabel')
    component_labels = ic_labels["labels"]
    component_probabilities = ic_labels["y_pred_proba"]
    
    # Save labels and probabilities
    df = pd.DataFrame({'Labels': component_labels, 'Probabilities': component_probabilities})
    df.to_csv(f"{labels_path}/{subject_id}_probs.csv", index=False)
    
    # Determine components to exclude
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(component_labels, component_probabilities))
                  if label not in ["brain", "other"] or prob < 0.70]
    
    # Save the exclude index
    df = pd.DataFrame({'Exclude_Idx': exclude_idx})
    df.to_csv(f"{exclude_idx_path}/{subject_id}_exclude_idx.csv", index=False)
    
    print(f"ICA processing complete for subject {subject_id}. Plots and exclusion indices saved.")

# List all .fif files
subject_files = [f for f in os.listdir(input_path) if f.endswith('.fif')]
print(f"Found {len(subject_files)} .fif files")

# Check if the requested subject ID exists in the files
matching_files = [f for f in subject_files if f.split('_')[0] == subject_id_arg]

if not matching_files:
    print(f"No files found for subject ID: {subject_id_arg}")
    print(f"Available subject IDs: {[f.split('_')[0] for f in subject_files]}")
    sys.exit(1)

# Process only the matching files
for subject_file in matching_files:
    process_subject(subject_file)
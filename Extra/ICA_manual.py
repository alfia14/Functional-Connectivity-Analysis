import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from mne_icalabel import label_components
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA
import os
import pandas as pd

input_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/ICA/BD_Interpolated'
output_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/ICA'

average_components = 20

# Ensure the output directories exist
os.makedirs(f"{output_path}/Timeseries", exist_ok=True)
os.makedirs(f"{output_path}/Topographies", exist_ok=True)
os.makedirs(f"{output_path}/ICA_Labels", exist_ok=True)
os.makedirs(f"{output_path}/Final", exist_ok=True)

def process_subject(subject):
    print(f"Processing {subject}")

    subject_id = subject.split('_')[0]

    # Load data
    EEG = mne.io.read_raw_fif(f"{input_path}/{subject}", preload=True)

    EEG.set_eeg_reference('average')
    exclude_channels = ['VEOG', 'HEOG', 'M1', 'M2']

    data = EEG.get_data().T
    explained_variance_threshold = 0.99

    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that reach the threshold
    n_components_actual = np.argmax(cumulative_variance >= explained_variance_threshold) + 1

    n_components = max(n_components_actual, average_components)
    
    # Save the number of ICA components used
    df = pd.DataFrame({'n_components': [n_components]})
    df.to_csv(f"{exclude_idx_path}/{subject_id}_n_components.csv", index=False)


    ica = ICA(n_components=n_components, max_iter="auto", method="infomax",
              random_state=97, fit_params=dict(extended=True))

    picks_eeg = mne.pick_types(EEG.info, meg=False, eeg=True, eog=False,
                               stim=False, emg=False, exclude=exclude_channels)

    ica.fit(EEG, picks=picks_eeg, decim=3)

    montage_path = '/projects/illinois/ahs/kch/nakhan2/scripts/montage/montage.sfp'
    montage = mne.channels.read_custom_montage(montage_path)
    EEG.set_montage(montage)

    # Plot ICA time series
    ica.plot_sources(EEG, show_scrollbars=False, show=False)
    plt.savefig(f"{output_path}/Timeseries/{subject_id}_ica_timeseries.png")
    plt.close()

    # Plot and save ICA component topographies
    fig = ica.plot_components(show=False)  # Returns a single MNEFigure object
    fig.savefig(f"{output_path}/Topographies/{subject_id}_ica_components.png")
    plt.close(fig)

    # Label ICA components
    ic_labels = label_components(EEG, ica, method='iclabel')
    component_labels = ic_labels["labels"]
    component_probabilities = ic_labels["y_pred_proba"]

    # Save labels and probabilities
    df = pd.DataFrame({'Labels': component_labels, 'Probabilities': component_probabilities})
    df.to_csv(f"{output_path}/ICA_Labels/{subject_id}_probs.csv", index=False)

    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(component_labels, component_probabilities))
                   if label not in ["brain", "other"] or prob < 0.70]

    # Apply ICA
    ica.apply(EEG, exclude=exclude_idx)

    # Save ICA-processed data
    EEG.save(f"{output_path}/Final/{subject_id}_ICA.fif", overwrite=True)


# List all files in the input directory
subject_list = [f for f in os.listdir(input_path) if f.endswith('.fif')]

# Print out the list to verify files to process
print("Files to process:", subject_list)
print(len(subject_list))

for subject in subject_list:
    process_subject(subject)

print("Processing complete.")

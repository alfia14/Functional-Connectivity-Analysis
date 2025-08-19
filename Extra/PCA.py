import mne
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt

# Define input and output paths
input_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/Bad_Channels_Marked/'
output_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/PCA/'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# List all subjects with .fif files
subjects = [f for f in os.listdir(input_path) if f.endswith('.fif')]

# Define the explained variance threshold
explained_variance_threshold = 0.99
components_list = []

# Iterate over all participants' EEG data
for subject in subjects:
    EEG = mne.io.read_raw_fif(os.path.join(input_path, subject), preload=True)
    picks_good = mne.pick_types(EEG.info, meg=False, eeg=True, exclude='bads')

    original_EEG = EEG.copy().pick(picks_good)
    original_EEG.set_eeg_reference('average')

    # Drop channels if they exist
    channels_to_drop = ['M1', 'M2', 'VEOG', 'HEOG']
    existing_channels = [ch for ch in channels_to_drop if ch in original_EEG.ch_names]
    original_EEG.drop_channels(existing_channels)

    # Get EEG data and transpose
    data = original_EEG.get_data().T  

    # Perform PCA
    pca = PCA(n_components=0.99)
    pca.fit(data)

    # Plot PCA explained variance
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Explained Variance by PCA Components: {subject}')
    plt.grid(True)

    # Save figure with correct path handling
    output_file = os.path.join( f"{output_path}/{subject}_PCA_variance.png")
    #plt.savefig(output_file)
    plt.close()

    # Compute the cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that reach the threshold
    num_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1

    # Append to the list
    components_list.append(num_components)

# Compute the average number of components across all participants
average_components = np.mean(components_list)
print(f"Average number of components selected: {average_components:.2f}")
print(components_list)
print(cumulative_variance)

import mne
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mne_icalabel import label_components
from mne.preprocessing import ICA

# Paths
input_path = '/projects/illinois/ahs/kch/nakhan2/ACE/ICA/Bad_Channels_Marked'
output_path = '/projects/illinois/ahs/kch/nakhan2/ACE/ICA'
exclude_idx_path = f"{output_path}/ICA_ExcludeIdx"
timeseries_path = f"{output_path}/Timeseries"
topographies_path = f"{output_path}/Topographies"
labels_path = f"{output_path}/ICA_Labels"
cleaned_path = f"{output_path}/Final"

# Ensure directories exist
os.makedirs(exclude_idx_path, exist_ok=True)
os.makedirs(timeseries_path, exist_ok=True)
os.makedirs(topographies_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)
os.makedirs(cleaned_path, exist_ok=True)

average_components = 20

def process_subject(subject):
    print(f"Processing {subject}")
    subject_id = subject.split('_')[0]
    
    # 1. Load EEG data (bad channels are already in EEG.info['bads'])
    EEG = mne.io.read_raw_fif(os.path.join(input_path, subject), preload=True)
    
    # 2. Set reference
    EEG.set_eeg_reference('average')
    
    # 3. Define non-brain channels to exclude
    non_brain_channels = ['FT7', 'FT8', 'CB1', 'CB2', 'TP7', 'TP8', 'VEO', 'HEO']
    
    # 4. Combine bad channels and non-brain channels
    all_exclude = list(set(non_brain_channels) | set(EEG.info['bads']))
    
    # 5. PCA for component determination
    data = EEG.get_data().T
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_actual = np.argmax(cumulative_variance >= 0.99) + 1
    n_components = max(n_components_actual, average_components)
    
    # Save component count
    pd.DataFrame({'n_components': [n_components]}).to_csv(
        f"{exclude_idx_path}/{subject_id}_n_components.csv", index=False
    )
    
    # 6. Run ICA (excluding all_exclude channels)
    ica = ICA(n_components=n_components, max_iter="auto", method="infomax",
              random_state=97, fit_params=dict(extended=True))
    
    picks_eeg = mne.pick_types(EEG.info, eeg=True, exclude=all_exclude)
    ica.fit(EEG, picks=picks_eeg, decim=3)
    
    # 7. Plot components
    fig = ica.plot_sources(EEG, show_scrollbars=False, show=False)
    fig.savefig(f"{timeseries_path}/{subject_id}_ica_timeseries.png")
    plt.close(fig)
    
    comp_fig = ica.plot_components(show=False)
    if isinstance(comp_fig, list):  # Handle MNE version differences
        comp_fig = comp_fig[0]
    comp_fig.savefig(f"{topographies_path}/{subject_id}_ica_components.png")
    plt.close(comp_fig)
    
    # 8. Label components
    ic_labels = label_components(EEG, ica, method='iclabel')
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]
    
    pd.DataFrame({'Labels': labels, 'Probabilities': probs}).to_csv(
        f"{labels_path}/{subject_id}_probs.csv", index=False
    )
    
    # 9. Identify components to exclude
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(component_labels, component_probabilities))
                       if label not in ["brain", "other"] or prob < 0.70]
    
    # 10. Apply ICA and interpolate AFTER
    raw_clean = ica.apply(EEG.copy(), exclude=exclude_idx)
    raw_clean.interpolate_bads(reset_bads=True)  
    
    # 11. Save cleaned data
    raw_clean.save(f"{cleaned_path}/{subject_id}_clean.fif", overwrite=True)
    
    # 12. Save exclude indices
    pd.DataFrame({'Exclude_Idx': exclude_idx}).to_csv(
        f"{exclude_idx_path}/{subject_id}_exclude_idx.csv", index=False
    )

# Process subjects
subject_list = [f for f in os.listdir(input_path) if f.endswith('.fif')]
print(f"Files to process: {subject_list}")

for subject in subject_list:
    process_subject(subject)

print("Processing complete. Cleaned data saved.")

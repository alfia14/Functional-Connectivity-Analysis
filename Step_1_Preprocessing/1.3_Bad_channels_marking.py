from sklearn.decomposition import PCA
import numpy as np
import mne
import os
import pandas as pd
import shutil  # Added for copying files

# Define input and output paths
input_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XW/Filtered/'
output_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XW/Bad_Channels_Marked/'
csv_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XW/bad_channels.csv'  # Path to bad channels CSV file

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

# Check if input directory exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Directory {input_path} not found. Please check the path.")

# Read bad channels from CSV into a dictionary
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file {csv_path} not found. Please check the path.")

bad_channels_df = pd.read_csv(csv_path)
bad_channels_dict = dict(zip(
    bad_channels_df['Subject'],
    bad_channels_df['Bad Channels'].apply(
        lambda x: [ch.strip() for ch in x.strip("[]").replace("'", "").split(', ') if ch.strip()] 
        if isinstance(x, str) and x.strip() else []
    )
))

# Get list of all .fif files
subjects = [f for f in os.listdir(input_path) if f.endswith('.fif')]

# Iterate over all participants' EEG data
for subject in subjects:
    try:
        subject_id = subject.split('_')[0]
        raw_file_path = os.path.join(input_path, subject)
        output_file_path = os.path.join(output_path, f"{subject_id}_badchannels.fif")

        # Load EEG data
        EEG = mne.io.read_raw_fif(raw_file_path, preload=True)
        EEG = EEG.drop_channels(['FT7', 'FT8', 'CB1', 'CB2', 'TP7', 'TP8', 'VEO', 'HEO')])

        # Get bad channels from dictionary and filter only valid ones
        bad_txt = bad_channels_dict.get(subject_id, [])
        valid_bad_channels = [ch for ch in bad_txt if ch in EEG.ch_names]

        if valid_bad_channels:
            EEG.info['bads'].extend(valid_bad_channels)
            print(f"Marked bad channels for {subject}: {EEG.info['bads']}")
            EEG.save(output_file_path, overwrite=True)  # Save modified file

        else:
            print(f"No valid bad channels for {subject}. Copying file unchanged.")
            shutil.copy(raw_file_path, output_file_path)  # Directly copy the file

        print(f"Saved: {output_file_path}")

    except Exception as e:
        print(f"Error processing {subject}: {e}")

import mne
import os
import pandas as pd

# Define input files path
input_files_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XW/Filtered'

# Check if the directory exists
if not os.path.exists(input_files_path):
    raise FileNotFoundError(f"Directory {input_files_path} not found. Please check the path.")

# List all .fif files in the input directory
filtered_files = [f for f in os.listdir(input_files_path) if f.endswith('.fif')]

# Initialize an empty dictionary to store bad channels for each participant
bad_channels_dict = {}

# Define the names of the channels to exclude
exclude_channels = ['VEO', 'HEO', 'M1', 'M2']

# Loop through each participant's raw data
for raw_file in filtered_files:
    try:
        # Extract the subject name from the file name
        subject_name = os.path.basename(raw_file).split('_')[0]
        raw_path = os.path.join(input_files_path, raw_file)

        # Load the raw data
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        #black
        raw = raw.drop_channels([ 'FT7', 'FT8', 'CB1', 'CB2','TP7', 'TP8'])
        #blue caps
        #raw = raw.drop_channels([ 'F11', 'F12', 'FT11', 'FT12', 'CB1', 'CB2'])
        # Find all channels that are not in the exclusion list
        picks = mne.pick_channels(raw.info['ch_names'], include=raw.info['ch_names'], exclude=exclude_channels)

        # Run autoreject to find bad channels for this participant
        reject_log = mne.preprocessing.find_bad_channels_lof(raw, picks=picks)

        # Store the bad channels in the dictionary
        bad_channels_dict[subject_name] = reject_log

        print(f"Processed {subject_name}: Bad channels -> {reject_log}")

    except Exception as e:
        print(f"Error processing {raw_file}: {e}")

# Convert the dictionary to a DataFrame
bad_channels_df = pd.DataFrame(list(bad_channels_dict.items()), columns=['Subject', 'Bad Channels'])

# Save the DataFrame to a CSV file
csv_output_path = "/projects/illinois/ahs/kch/nakhan2/ACE_XW/bad_channels.csv"
bad_channels_df.to_csv(csv_output_path, index=False)

print(f"CSV file saved: {csv_output_path}")

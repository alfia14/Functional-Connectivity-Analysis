# Import necessary libraries
import mne
import numpy as np
import os

# Define input and output file paths
input_files_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/EDF_Files'
output_files_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Filtered'

# Ensure the output directory exists
os.makedirs(output_files_path, exist_ok=True)

# List all .edf files in the input directory
edf_files = [f for f in os.listdir(input_files_path) if f.endswith('.edf')]

# Print out the list to verify files to process
print("Files to process:", edf_files)
print("Total number of files:", len(edf_files))

# Define montageblack path
montage_path = '/projects/illinois/ahs/kch/nakhan2/scripts/montage/montageblack.sfp'
montageblack = mne.channels.read_custom_montage(montage_path)

# Channels to drop
channels_to_drop = ['EKG', 'EMG', 'TRIGGER', 'Status']

# Loop over each EDF file
for edf_file in edf_files:
    edf_path = os.path.join(input_files_path, edf_file)

    # Load raw EDF data
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Drop specific channels
    raw.drop_channels(channels_to_drop)

    # Set montageblack (electrode locations)
    raw.set_montage(montageblack)

    # Apply filtering (band-pass filter between 1-50 Hz)
    raw.filter(l_freq=1, h_freq=50)

    # Apply notch filter at 60Hz
    raw.notch_filter(freqs=60, notch_widths=1, fir_design='firwin')

    # Define output file path
    output_path = os.path.join(output_files_path, edf_file.replace('.edf', '_filtered_raw.fif'))

    # Save the filtered raw data
    raw.save(output_path, overwrite=True)

    print(f"Processed and saved: {output_path}")

print("Processing complete.")

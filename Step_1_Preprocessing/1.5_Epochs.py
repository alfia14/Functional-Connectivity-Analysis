# Epochs Generation
import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import zscore


#Filepaths

input_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/ICA/Final'
output_path = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Epochs'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)


# Function to process and save events for each participant
def process_subject(participant_id):

    # Reading a text file into a DataFrame (assuming space, comma, or tab-separated values)
    file_path = f'/projects/illinois/ahs/kch/nakhan2/ACE_XZ/EventLists/{participant_id}_EL.txt'

    # If the file is tab-separated, use this instead
    df = pd.read_csv(file_path, sep='\t',on_bad_lines='skip', comment = '#')
    df = df.drop(df.columns[[6,7,8,9]], axis=1)


    EEG = mne.io.read_raw_fif(f'{input_path}/{participant_id}_ICA.fif', preload=True)
    sfreq = EEG.info['sfreq']
    print(f"Sampling frequency for {participant_id} is {sfreq}")
    


    new_column_names = ['item', 'bepoch', 'ecode', 'label','onset', 'duration']  # Replace with your actual column names
    df.columns = new_column_names

    events = np.array([[int(onset * sfreq), 0, ecode] for onset, ecode in zip(df['onset'], df['ecode'])])

    # Define epoching parameters
    #blue cap event list Flanker
    #for NURISH
    event_id = dict(congruent_left=14, congruent_right=26, incongruent_left=116, incongruent_right=128)
    #this is for ACE_XZ-black caps
    #event_id = dict(congruent_left=14, congruent_left=16, congruent_left=18, congruent_right=26, congruent_right=28, congruent_right=24, incongruent_left=116, incongruent_left=118, incongruent_left=114,incongruent_right=128,incongruent_right=124, incongruent_right=126 )
    tmin = -0.2  # Start of each epoch (200ms before the event)
    tmax = 1.2   # End of each epoch (1200ms after the event)

    # Create epochs
    epochs = mne.Epochs(EEG, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)


    # Define Z-score threshold for epoch rejection
    zscore_threshold = 6
    to_drop = []

    # Create a temporary array to store Z-scored data for each epoch
    temp_data = np.zeros_like(epochs._data)

    # Loop through each epoch and apply Z-score normalization across channels
    for i in range(len(epochs)):
        temp_data[i] = zscore(epochs._data[i], axis=1)

        # If any value in this epoch exceeds the Z-score threshold, mark it for rejection
        if np.any(np.abs(temp_data[i]) > zscore_threshold):
            to_drop.append(i)

    # Drop the marked epochs
    epochs.drop(to_drop)

    # Save epochs if needed
    output_epochs_file = f'/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Epochs/{participant_id}_epochs-epo.fif'
    epochs.save(output_epochs_file, overwrite=True)
    print(f"Saved epochs for {participant_id}")

    # Output the result
    print(f"Epochs dropped: {len(to_drop)}")

# List all files in the input directory
subject_list = [f for f in os.listdir(input_path) if f.endswith('.fif')]

# Print out the list to verify files to process
print("Files to process:", subject_list)
print(len(subject_list))

for subject in subject_list:
    subject_id = subject.split('_')[0]

    process_subject(subject_id)

print("Processing complete.")

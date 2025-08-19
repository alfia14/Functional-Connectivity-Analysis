import os
import mne
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for headless environments 

# Define input and output directories
input_folder = "/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Filtered"  # Change this path
output_folder = "/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Plots_Channels_Inspection"  # Change this path
plt.rcParams['figure.figsize'] = [20, 8] 

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get all EEG files in the input folder
eeg_files = [f for f in os.listdir(input_folder) if f.endswith(".fif")]

if not eeg_files:
    print("No EEG files found in the input folder. Please check the path.")
    exit()

print(f"Found {len(eeg_files)} EEG files. Processing...")

def chunk_list(lst, n):
 
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_channels_in_chunks(subject_file):
    """Function to plot EEG channels in groups of 10 and save each plot."""
    subject_id = subject_file.split(".")[0]
    file_path = os.path.join(input_folder, subject_file)

    # Load EEG file
    EEG = mne.io.read_raw_fif(file_path, preload=True)

    custom_montage = mne.channels.read_custom_montage('/projects/illinois/ahs/kch/nakhan2/scripts/montage/montageblack.sfp')
    EEG.set_montage(custom_montage)


    # Channels to exclude explicitly
    exclude_channels = {'FT7', 'FT8', 'CB1', 'CB2', 'T7', 'T8','TP7', 'TP8', 'VEO', 'HEO'}

    # Exclude channels that start with 'T' or are in the exclude list
    all_channels = [
        ch for ch in EEG.ch_names 
        if not ch.startswith('T') and ch not in exclude_channels
    ]


    all_channels = EEG.ch_names
    for idx, chunk in enumerate(chunk_list(all_channels, 10)):
        fig = EEG.copy().pick_channels(chunk).plot(
            n_channels=len(chunk),
            duration=20,
            scalings="50e-6",
            title=f"{subject_id} - Channels {idx*10 + 1} to {idx*10 + len(chunk)}",
            show=False
        )

        output_file = os.path.join(output_folder, f"{subject_id}_chunk_{idx+1}.png")
        fig.savefig(output_file, dpi=300)
        print(f"Saved plot: {output_file}")

# Process all EEG files
for eeg_file in eeg_files:
    try:
        plot_channels_in_chunks(eeg_file)
    except Exception as e:
        print(f"Error processing {eeg_file}: {e}")

print("All EEG plots have been saved successfully!")

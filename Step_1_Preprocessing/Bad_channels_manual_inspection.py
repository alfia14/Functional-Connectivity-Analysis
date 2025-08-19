import os
import mne
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for headless environments 

# Define input and output directories
input_folder = "/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Filtered"  # Change this path
output_folder = "/projects/illinois/ahs/kch/nakhan2/ACE_XZ/EEG_Plots"  # Change this path

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get all EEG files in the input folder
eeg_files = [f for f in os.listdir(input_folder) if f.endswith(".fif")]

if not eeg_files:
    print("No EEG files found in the input folder. Please check the path.")
    exit()

print(f"Found {len(eeg_files)} EEG files. Processing...")

def plot_all_channels(subject_file):
    """Function to plot all EEG raw data channels and save the plot."""
    subject_id = subject_file.split(".")[0]  # Extract subject ID without file extension
    file_path = os.path.join(input_folder, subject_file)

    # Load EEG file
    EEG = mne.io.read_raw_fif(file_path, preload=True)

    # Set montage (if available)
    if EEG.info.get('dig') is None:
        EEG.set_montage("standard_1020")  # Default electrode locations

    # Get all channel names
    all_channels = EEG.ch_names

    # Plot all channels
    fig = EEG.plot(
        n_channels=len(EEG.ch_names),  # Show all channels
        duration=10,  # Display 10 seconds per window
        scalings="auto",  # Auto scale signals
        title=f"EEG Raw ACE_XZ - {subject_id} (All Channels)",
        show=False
    )

    # Save the plot
    output_file = os.path.join(output_folder, f"{subject_id}_all_channels.png")
    fig.savefig(output_file, dpi=300)  # Save as high-resolution PNG
    print(f"Saved plot: {output_file}")

# Process all EEG files
for eeg_file in eeg_files:
    try:
        plot_all_channels(eeg_file)
    except Exception as e:
        print(f"Error processing {eeg_file}: {e}")

print("All EEG plots have been saved successfully!")

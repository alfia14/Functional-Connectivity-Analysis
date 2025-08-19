import os
import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend

# === USER CONFIGURATION ===
subject_id = "ACE118"  # ← Just change this
channels_to_plot = [ 'FPZ', 'FZ', 'FP1', 'FP2', 'AF3', 'AF4',
    'F3', 'F4', 'F5', 'F6', 'F7', 'F8',  'CZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
    'P7', 'P8', 'OZ', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8']
input_dir = "/projects/illinois/ahs/kch/nakhan2/ACE_XW/Filtered"
output_dir = "/projects/illinois/ahs/kch/nakhan2/ACE_XW/Bad_Channels_specific_channels"
duration = 30
scaling_uv = 'auto'

'''
frontal_channels = [
    'FPZ', 'FZ', 'FP1', 'FP2', 'AF3', 'AF4',
    'F3', 'F4', 'F5', 'F6', 'F7', 'F8'
]

central_channels = [
    'CZ', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'

]

parietal_channels = [
    'PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
    'P7', 'P8'
]

occipital_channels = [
    'OZ', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8'
]

'''
# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Locate the subject file ===
subject_file = next((f for f in os.listdir(input_dir)
                     if f.startswith(subject_id) and f.endswith(".fif")), None)

if not subject_file:
    print(f"No file found for subject ID {subject_id}")
    exit()

input_path = os.path.join(input_dir, subject_file)
output_path = os.path.join(output_dir, f"{subject_id}_selected_channels.png")

# === Load and Plot ===
try:
    raw = mne.io.read_raw_fif(input_path, preload=True)

    available_channels = [ch for ch in channels_to_plot if ch in raw.ch_names]
    if not available_channels:
        print(f"⚠️ None of the specified channels found for {subject_id}.")
        exit()

    raw_copy = raw.copy().pick_channels(available_channels)

    # === Signal range diagnostics ===
    data, _ = raw_copy.get_data(return_times=True)
    print(f"📊 Signal Stats for {subject_id} - Selected Channels:")
    print(f"    Min: {np.min(data) * 1e6:.2f} µV")
    print(f"    Max: {np.max(data) * 1e6:.2f} µV")
    print(f"    Std Dev: {np.std(data) * 1e6:.2f} µV")

    # === Plot and save ===
    fig = raw_copy.plot(
        n_channels=len(available_channels),
        duration=duration,
        scalings={'eeg': scaling_uv},
        title=f"{subject_id} - Selected EEG Channels",
        show=False
    )
    fig.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to: {output_path}")

except Exception as e:
    print(f"❌ Error processing {subject_id}: {e}")

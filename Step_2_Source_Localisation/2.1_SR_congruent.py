import os
import gc
import argparse
import numpy as np
import mne
import os.path as op
from scipy.stats import zscore
from scipy import signal
from sklearn.decomposition import PCA
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt

# Parse command-line argument for subject_id
parser = argparse.ArgumentParser(description="EEG Source Reconstruction")
parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
args = parser.parse_args()
subject_id = args.subject_id

print(f"Processing subject: {subject_id}")

# Set fsaverage download path
custom_subjects_dir = "/projects/illinois/ahs/kch/nakhan2/mne_data"
os.makedirs(custom_subjects_dir, exist_ok=True)
mne.set_config("SUBJECTS_DIR", custom_subjects_dir)

# Define fsaverage paths
subjects_dir = custom_subjects_dir
subject = "fsaverage"
trans = "fsaverage"
fs_dir = f'{custom_subjects_dir}/{subject}'
#source space- has info about cortical surfaces
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
# boundary element model- geometric info about how the cortical activity propagates through the scalp
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

# Define input and output directories
input_path = "/projects/illinois/ahs/kch/nakhan2/ACE_XW/Epochs/"
files_out = "/projects/illinois/ahs/kch/nakhan2/ACE_XW/Source_Localised_Data/"
os.makedirs(files_out, exist_ok=True)

# Processing modes
modes = ['congruent']

# Verify input file exists
input_file = f"{input_path}/{subject_id}_epochs-epo.fif"
if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found.")
    exit(1)

# Process EEG data for each mode
for mode in modes:
    output_path = os.path.join(files_out, mode)
    os.makedirs(output_path, exist_ok=True)

    # Load epochs
    epochs = mne.read_epochs(input_file)

    # Drop unnecessary channels
    channels_to_drop = ['HEO', 'VEO', 'M1', 'M2']
    try:
        epochs.drop_channels(channels_to_drop)
    except Exception as e:
        print(f"Warning: Could not drop channels: {e}")

    # Load custom montage
    #need to change for diff studies likie ACE_XW or where a different cap was used 
    montage_path = "/projects/illinois/ahs/kch/nakhan2/scripts/montage/montageblack.sfp"
    montage = mne.channels.read_custom_montage(montage_path)
    epochs.set_montage(montage)

    # Split epochs by mode
    epochs_left = epochs[f"{mode}_left"]
    epochs_right = epochs[f"{mode}_right"]
    epochs = mne.concatenate_epochs([epochs_left, epochs_right])

    # Apply EEG referencing
    epochs.set_eeg_reference(projection=True)
    epochs.apply_proj()

    # Compute forward solution
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
    )

    # Adjust EEG channel selection
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=True, stim=False)

    # Compute noise covariance
    noise_cov = mne.compute_covariance(
        epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
    )

    # Visualize covariance
    fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)

    # Convert forward solution
    mne.convert_forward_solution(fwd, surf_ori=True, copy=False)

    # Extract source space info
    lh = fwd["src"][0]
    dip_pos = lh["rr"][lh["vertno"]]
    dip_ori = lh["nn"][lh["vertno"]]
    dip_len = len(dip_pos)
    dip_times = [0]
    actual_amp = np.ones(dip_len)
    actual_gof = np.ones(dip_len)

    # Create Dipole instance
    dipoles = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori, actual_gof)

    # Save forward solution
    fwd_solution_path = os.path.join(output_path, "Forward_Solution")
    os.makedirs(fwd_solution_path, exist_ok=True)

    mne.write_forward_solution(
        f"{fwd_solution_path}/{subject_id}_forwardsolution_MRItemplate.fif",
        fwd,
        overwrite=True,
    )

    # Compute inverse operator
    inv = make_inverse_operator(
        epochs.info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8, verbose=True
    )

    # Compute eLORETA inverse solution
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stcs = apply_inverse_epochs(
        epochs, inv, lambda2, "eLORETA", verbose=True, pick_ori=None
    )

    # Save inverse solutions
    print("Saving inverse solutions...")

    subject_dir = os.path.join(output_path, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    for idx, stc in enumerate(stcs):
        inverse_solution_file = f"{subject_dir}/{subject_id}_inversesolution_epoch{idx}.fif"
        stc.save(inverse_solution_file, overwrite=True)

    # Cleanup memory
    del epochs, stcs
    gc.collect()

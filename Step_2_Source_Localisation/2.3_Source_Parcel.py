import mne
import os
import glob
import numpy as np
import argparse
from nilearn import datasets
import os.path as op

# Parse command-line argument for subject_id
parser = argparse.ArgumentParser(description="EEG Source Reconstruction")
parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
args = parser.parse_args()
subject_id = args.subject_id

print(f"Processing subject: {subject_id}")

mne.datasets.fetch_fsaverage(verbose=True)
print(mne.get_config('MNE_DATA'))
src = mne.setup_source_space('fsaverage', spacing='ico5', add_dist=False)
mne.write_source_spaces('fsaverage-ico-5-src.fif', src, overwrite= True)

# Fetch fsaverage and Schaefer atlas
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# Set paths to the source space and BEM model
src_file = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(src_file)

# Define input and output directories
files_in = '/projects/illinois/ahs/kch/nakhan2/ACE_RZ/Source_Localised_Data'
files_out = '/projects/illinois/ahs/kch/nakhan2/ACE_RZ/TimeCourses'

os.makedirs(files_out, exist_ok=True)

# Modes to process
modes = ['congruent', 'incongruent']

# Load the Schaefer atlas
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

# Process only the provided subject_id
for mode in modes:
    print(f"Processing: {subject_id}, {mode}")

    # Define directory paths for the current subject and mode
    directory = op.join(files_in, mode, subject_id)
    stc_files_lh = glob.glob(op.join(directory, '*lh.stc'))
    stc_files_rh = glob.glob(op.join(directory, '*rh.stc'))

    if not stc_files_lh or not stc_files_rh:
        print(f"No STC files found for subject {subject_id} in mode {mode}")
        continue

    # Load source estimates for left and right hemispheres
    stcs_lh = [mne.read_source_estimate(stc_file, subject='fsaverage') for stc_file in stc_files_lh]
    stcs_rh = [mne.read_source_estimate(stc_file, subject='fsaverage') for stc_file in stc_files_rh]

    # Load labels for left and right hemispheres from the Schaefer atlas
    labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=subjects_dir)

    # Initialize list to store label time courses
    label_time_courses = []

    # Extract label time courses for both hemispheres
    for idx, (stc_lh, stc_rh) in enumerate(zip(stcs_lh, stcs_rh)):
        try:
            label_tc_lh = stc_lh.extract_label_time_course(labels, src=src, mode='mean_flip')
            label_tc_rh = stc_rh.extract_label_time_course(labels, src=src, mode='mean_flip')
            # Combine left and right hemisphere time courses
            label_time_courses.extend([label_tc_lh, label_tc_rh])
        except Exception as e:
            print(f"Error extracting label time courses for iteration {idx}: {e}")

    if label_time_courses:
        # Combine all label time courses and save
        label_time_courses_np = np.array(label_time_courses)
        label_time_courses_file = op.join(files_out, mode, f"{subject_id}_label_time_courses.npy")
        os.makedirs(op.dirname(label_time_courses_file), exist_ok=True)
        np.save(label_time_courses_file, label_time_courses)

        print(f"Saved label time courses for {subject_id} in {mode} mode.")

    del stcs_lh, stcs_rh, labels, label_tc_lh, label_tc_rh
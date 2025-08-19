import mne
import os
from nilearn import datasets

# Define the correct subjects directory
subjects_dir = "/projects/illinois/ahs/kch/nakhan2/"
os.environ["SUBJECTS_DIR"] = subjects_dir  # Ensure MNE uses this path

# Download fsaverage into the specified directory
fs_dir = mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

print(f"fsaverage downloaded to: {fs_dir}")


# Set the nilearn dataset directory
nilearn_data_dir = "/projects/illinois/ahs/kch/nakhan2/nilearn"
os.environ["NILEARN_DATA"] = nilearn_data_dir  # Force nilearn to use this directory

# Fetch Schaefer atlas
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, data_dir=nilearn_data_dir)

print(f"Schaefer atlas downloaded to: {schaefer_atlas.maps}")

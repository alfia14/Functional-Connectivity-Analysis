import os
import mne

# Define custom directory for fsaverage
custom_subjects_dir = "/projects/illinois/ahs/kch/nakhan2/mne_data"

# Ensure the directory exists
os.makedirs(custom_subjects_dir, exist_ok=True)

# Set MNE SUBJECTS_DIR to the custom path
mne.set_config("SUBJECTS_DIR", custom_subjects_dir)

# Download fsaverage data
mne.datasets.fetch_fsaverage(subjects_dir=custom_subjects_dir, verbose=True)

print(f"fsaverage dataset downloaded to: {custom_subjects_dir}")

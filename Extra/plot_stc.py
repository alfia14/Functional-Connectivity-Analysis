import mne
import numpy as np
import os

# --- USER INPUTS ---
subject_prefix = "NU"
subject_start = 101
subject_end = 148
base_dir = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort1/Source_Localised_Data/congruent"
n_epochs = 200  # Set to your maximum possible number of epochs
subject_fsaverage = 'fsaverage'
output_group_stc_stem = os.path.join(base_dir, "group_average_inversesolution")

# Collect per-subject means in this list
subject_mean_stcs = []

for subj_num in range(subject_start, subject_end + 1):
    subject_id = f"{subject_prefix}{subj_num}"
    subject_dir = os.path.join(base_dir, subject_id)
    print(subject_dir)
    if not os.path.exists(subject_dir):
        print(f"Subject directory missing for {subject_id}, skipping.")
        continue

    stcs = []
    for idx in range(n_epochs):
        fname_stem = os.path.join(subject_dir, f"{subject_id}_inversesolution_epoch{idx}.fif")
        lh_file = fname_stem + "-lh.stc"
        rh_file = fname_stem + "-rh.stc"
        if not (os.path.exists(lh_file) and os.path.exists(rh_file)):
            continue  # skip missing epochs
        try:
            stc = mne.read_source_estimate(fname_stem, subject=subject_fsaverage)
            stcs.append(stc)
        except Exception as e:
            print(f"Error loading {fname_stem}: {e}")

    if len(stcs) == 0:
        print(f"No valid STCs found for {subject_id}, skipping.")
        continue

    # Average across epochs for this subject
    data = np.stack([stc.data for stc in stcs], axis=0)
    mean_data = np.mean(data, axis=0)
    mean_stc = mne.SourceEstimate(
        mean_data,
        vertices=stcs[0].vertices,
        tmin=stcs[0].tmin,
        tstep=stcs[0].tstep,
        subject=stcs[0].subject
    )
    # Save the per-subject averaged STC
    avg_stc_stem = os.path.join(subject_dir, f"{subject_id}_inversesolution_average")
    mean_stc.save(avg_stc_stem, overwrite=True)
    print(f"Saved average STC for {subject_id} to {avg_stc_stem}-lh.stc and -rh.stc")
    print(f"{subject_id} average STC data shape: {mean_stc.data.shape}")
    subject_mean_stcs.append(mean_stc)
    print(f"Added average for {subject_id} ({len(stcs)} epochs) to group list.")

# --- GROUP AVERAGE ACROSS ALL SUBJECTS ---
if len(subject_mean_stcs) == 0:
    raise RuntimeError("No subject averages found. Check your file paths.")

# Print all shapes for debugging
print("Shapes of all subject mean STCs:")
for stc in subject_mean_stcs:
    print(stc.data.shape)

# Proceed only if all shapes match
shapes = [stc.data.shape for stc in subject_mean_stcs]
from collections import Counter
most_common_shape = Counter(shapes).most_common(1)[0][0]
subject_mean_stcs_filtered = [stc for stc in subject_mean_stcs if stc.data.shape == most_common_shape]
print(f"Using {len(subject_mean_stcs_filtered)} subjects with shape {most_common_shape}")

group_data = np.stack([stc.data for stc in subject_mean_stcs_filtered], axis=0)
group_mean_data = np.mean(group_data, axis=0)

group_mean_stc = mne.SourceEstimate(
    group_mean_data,
    vertices=subject_mean_stcs_filtered[0].vertices,
    tmin=subject_mean_stcs_filtered[0].tmin,
    tstep=subject_mean_stcs_filtered[0].tstep,
    subject=subject_mean_stcs_filtered[0].subject
)
group_mean_stc.save(output_group_stc_stem, overwrite=True)
print(f"Saved group-averaged STC to {output_group_stc_stem}-lh.stc and -rh.stc")

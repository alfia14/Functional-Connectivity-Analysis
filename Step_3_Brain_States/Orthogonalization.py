import os
import numpy as np
import mne
import argparse
from mne_connectivity import symmetric_orth
from scipy.signal import hilbert


# Parse command-line argument for subject_id
parser = argparse.ArgumentParser(description="EEG Source Reconstruction")
parser.add_argument("--subject_id", type=str, required=True, help="Participant ID")
args = parser.parse_args()

subject_id = args.subject_id

print(f"Processing subject: {subject_id}")

# Define input and output directories
files_in = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/TimeCourses'
files_out = '/projects/illinois/ahs/kch/nakhan2/ACE_XZ/Orthogonalized_data'

modes = ["congruent", "incongruent"]

def apply_orthogonalization(data):
    """Applies orthogonalization to the given data."""
    analytic_signal = hilbert(data, axis=2)
    amplitude_envelope = np.abs(analytic_signal)
    orthogonalized_data = symmetric_orth(amplitude_envelope)
    orthogonalized_data = orthogonalized_data.reshape(amplitude_envelope.shape)
    return orthogonalized_data

def process_participant(subject, mode, dir_in, dir_out):
    label_time_courses_file = os.path.join(dir_in, f"{subject}_label_time_courses.npy")
    
    if os.path.exists(label_time_courses_file):
        try:
            label_time_courses = np.load(label_time_courses_file)
            print(f"Loaded data for {subject} in mode {mode}")
            
            orthogonalized_data = apply_orthogonalization(label_time_courses)
            
            output_file_path = os.path.join(dir_out, "orth.npy")
            np.save(output_file_path, orthogonalized_data)
            print(f"File saved successfully for participant {subject}, mode {mode} at {output_file_path}")
            
            return orthogonalized_data
        except Exception as e:
            print(f"Error processing {subject} in {mode}: {e}")
    else:
        print(f"File not found: {label_time_courses_file}")
    
    return None

# Process the given subject for both modes
for mode in modes:
    input_dir = os.path.join(files_in, mode)
    output_dir = os.path.join(files_out, subject_id, mode)
    os.makedirs(output_dir, exist_ok=True)
    process_participant(subject_id, mode, input_dir, output_dir)

print(f"Processing complete for subject: {subject_id}")

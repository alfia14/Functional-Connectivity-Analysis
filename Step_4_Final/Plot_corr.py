import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Configuration --------------------
base_dir = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort1/HMM_Output/"
participant_id = "NU103"  
mode = "incongruent"
state = 7
block = 0

# -------------------- File Paths --------------------
original_path = os.path.join(
    base_dir, participant_id, "Correlation_matrices", 
    f"{participant_id}_{mode}_correlation_matrices.npy.npz"
)

# -------------------- Load Matrix --------------------
def load_matrix(npz_path, state, block):
    key = f"{state}_{block}"
    with np.load(npz_path) as data:
        if key not in data:
            available_keys = list(data.keys())
            raise KeyError(f"Key {key} not found. Available keys: {available_keys[:10]}{'...' if len(available_keys)>10 else ''}")
        return data[key]

original_matrix = load_matrix(original_path, state, block)

# -------------------- Plot --------------------
plt.figure(figsize=(10, 8))  # Adjusted for single plot

# Use symmetric color scale centered at 0
vmin, vmax = -0.3, 0.3  # Adjust based on your data range
im = plt.imshow(original_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)

# Add title and labels
plt.title(f'Functional Connectivity: {participant_id} - {mode}\nState {state}, Block {block}', fontsize=14)
plt.xlabel('Brain Regions', fontsize=12)
plt.ylabel('Brain Regions', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, shrink=0.8)
cbar.set_label('Correlation', fontsize=12)

# -------------------- Save Figure --------------------
output_dir = os.path.join(base_dir, participant_id, "Plots")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{participant_id}_{mode}_state{state}_block{block}_correlation.png")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Correlation matrix plot saved to: {output_path}")

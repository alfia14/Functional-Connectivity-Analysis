import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Configuration --------------------
participant = "NU103"
mode = "incongruent"
state = 7
block = 0

base_dir = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort1/HMM_Output/"
correlation_path = os.path.join(base_dir, participant, "Correlation_matrices", 
                               f"{participant}_{mode}_correlation_matrices.npy.npz")
output_dir = os.path.join(base_dir, participant, "Plots")
os.makedirs(output_dir, exist_ok=True)

# -------------------- Load Correlation Matrix --------------------
def load_matrix(npz_path, state, block):
    key = f"{state}_{block}"
    with np.load(npz_path) as data:
        return data[key]

corr_matrix = load_matrix(correlation_path, state, block)

# -------------------- Network Labels --------------------
labels = [
    '7Networks_LH_Cont_Cing_1-lh', '7Networks_LH_Cont_Par_1-lh', '7Networks_LH_Cont_PFCl_1-lh',
    '7Networks_LH_Cont_pCun_1-lh', '7Networks_LH_Default_Par_1-lh', '7Networks_LH_Default_Par_2-lh',
    '7Networks_LH_Default_pCunPCC_1-lh', '7Networks_LH_Default_pCunPCC_2-lh', '7Networks_LH_Default_PFC_1-lh',
    '7Networks_LH_Default_PFC_2-lh', '7Networks_LH_Default_PFC_3-lh', '7Networks_LH_Default_PFC_4-lh',
    '7Networks_LH_Default_PFC_5-lh', '7Networks_LH_Default_PFC_6-lh', '7Networks_LH_Default_PFC_7-lh',
    '7Networks_LH_Default_Temp_1-lh', '7Networks_LH_Default_Temp_2-lh', '7Networks_LH_DorsAttn_FEF_1-lh',
    '7Networks_LH_DorsAttn_Post_1-lh', '7Networks_LH_DorsAttn_Post_2-lh', '7Networks_LH_DorsAttn_Post_3-lh',
    '7Networks_LH_DorsAttn_Post_4-lh', '7Networks_LH_DorsAttn_Post_5-lh', '7Networks_LH_DorsAttn_Post_6-lh',
    '7Networks_LH_DorsAttn_PrCv_1-lh', '7Networks_LH_Limbic_OFC_1-lh', '7Networks_LH_Limbic_TempPole_1-lh',
    '7Networks_LH_Limbic_TempPole_2-lh', '7Networks_LH_SalVentAttn_FrOperIns_1-lh', '7Networks_LH_SalVentAttn_FrOperIns_2-lh',
    '7Networks_LH_SalVentAttn_Med_1-lh', '7Networks_LH_SalVentAttn_Med_2-lh', '7Networks_LH_SalVentAttn_Med_3-lh',
    '7Networks_LH_SalVentAttn_ParOper_1-lh', '7Networks_LH_SalVentAttn_PFCl_1-lh', '7Networks_LH_SomMot_1-lh',
    '7Networks_LH_SomMot_2-lh', '7Networks_LH_SomMot_3-lh', '7Networks_LH_SomMot_4-lh', '7Networks_LH_SomMot_5-lh',
    '7Networks_LH_SomMot_6-lh', '7Networks_LH_Vis_1-lh', '7Networks_LH_Vis_2-lh', '7Networks_LH_Vis_3-lh',
    '7Networks_LH_Vis_4-lh', '7Networks_LH_Vis_5-lh', '7Networks_LH_Vis_6-lh', '7Networks_LH_Vis_7-lh',
    '7Networks_LH_Vis_8-lh', '7Networks_LH_Vis_9-lh', '7Networks_RH_Cont_Cing_1-rh', '7Networks_RH_Cont_Par_1-rh',
    '7Networks_RH_Cont_Par_2-rh', '7Networks_RH_Cont_PFCl_1-rh', '7Networks_RH_Cont_PFCl_2-rh', '7Networks_RH_Cont_PFCl_3-rh',
    '7Networks_RH_Cont_PFCl_4-rh', '7Networks_RH_Cont_PFCmp_1-rh', '7Networks_RH_Cont_pCun_1-rh', '7Networks_RH_Default_Par_1-rh',
    '7Networks_RH_Default_pCunPCC_1-rh', '7Networks_RH_Default_pCunPCC_2-rh', '7Networks_RH_Default_PFCdPFCm_1-rh',
    '7Networks_RH_Default_PFCdPFCm_2-rh', '7Networks_RH_Default_PFCdPFCm_3-rh', '7Networks_RH_Default_PFCv_1-rh',
    '7Networks_RH_Default_PFCv_2-rh', '7Networks_RH_Default_Temp_1-rh', '7Networks_RH_Default_Temp_2-rh',
    '7Networks_RH_Default_Temp_3-rh', '7Networks_RH_DorsAttn_FEF_1-rh', '7Networks_RH_DorsAttn_Post_1-rh',
    '7Networks_RH_DorsAttn_Post_2-rh', '7Networks_RH_DorsAttn_Post_3-rh', '7Networks_RH_DorsAttn_Post_4-rh',
    '7Networks_RH_DorsAttn_Post_5-rh', '7Networks_RH_DorsAttn_PrCv_1-rh', '7Networks_RH_Limbic_OFC_1-rh',
    '7Networks_RH_Limbic_TempPole_1-rh', '7Networks_RH_SalVentAttn_FrOperIns_1-rh', '7Networks_RH_SalVentAttn_Med_1-rh',
    '7Networks_RH_SalVentAttn_Med_2-rh', '7Networks_RH_SalVentAttn_TempOccPar_1-rh', '7Networks_RH_SalVentAttn_TempOccPar_2-rh',
    '7Networks_RH_SomMot_1-rh', '7Networks_RH_SomMot_2-rh', '7Networks_RH_SomMot_3-rh', '7Networks_RH_SomMot_4-rh',
    '7Networks_RH_SomMot_5-rh', '7Networks_RH_SomMot_6-rh', '7Networks_RH_SomMot_7-rh', '7Networks_RH_SomMot_8-rh',
    '7Networks_RH_Vis_1-rh', '7Networks_RH_Vis_2-rh', '7Networks_RH_Vis_3-rh', '7Networks_RH_Vis_4-rh',
    '7Networks_RH_Vis_5-rh', '7Networks_RH_Vis_6-rh', '7Networks_RH_Vis_7-rh', '7Networks_RH_Vis_8-rh'
]

# -------------------- Map Regions to Networks --------------------
network_mapping = {
    'Vis': 'Visual',
    'SomMot': 'Somatomotor', 
    'DorsAttn': 'DorsalAttention',
    'SalVentAttn': 'VentralAttention',
    'Limbic': 'Limbic',
    'Cont': 'Frontoparietal',
    'Default': 'Default'
}

network_assignments = []
for label in labels:
    for key, network in network_mapping.items():
        if key in label:
            network_assignments.append(network)
            break
# -------------------- Create Masks Without Reordering --------------------
# Create masks directly on the original correlation matrix
mask_wn = np.zeros_like(corr_matrix, dtype=bool)
mask_bn = np.ones_like(corr_matrix, dtype=bool)

# Create a network index array
network_indices = np.array([network_assignments])

# Compare network assignments for all region pairs
for i, net_i in enumerate(network_assignments):
    for j, net_j in enumerate(network_assignments):
        if net_i == net_j:
            mask_wn[i, j] = True
            mask_bn[i, j] = False

# Create masked matrices
wn_matrix = np.full_like(corr_matrix, np.nan)
bn_matrix = np.full_like(corr_matrix, np.nan)
wn_matrix[mask_wn] = corr_matrix[mask_wn]
bn_matrix[mask_bn] = corr_matrix[mask_bn]

# -------------------- Plotting --------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Color scale - use original matrix for scaling
#vmax = np.nanpercentile(np.abs(corr_matrix), 95)
#vmin = -vmax

vmin, vmax = -0.3, 0.3

# WN FC Plot (within-network only)
sns.heatmap(wn_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=ax1,
            cbar_kws={'label': 'Correlation'}, square=True)
ax1.set_title('WN FC (Within-Network)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Regions')
ax1.set_ylabel('Regions')

# BN FC Plot (between-network only)
sns.heatmap(bn_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=ax2,
            cbar_kws={'label': 'Correlation'}, square=True)
ax2.set_title('BN FC (Between-Network)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Regions')
ax2.set_ylabel('Regions')

plt.tight_layout()

# Save plot
output_path = os.path.join(output_dir, f"{participant}_{mode}_state{state}_block{block}_WN_BN_FC_original_order.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved plot to: {output_path}")

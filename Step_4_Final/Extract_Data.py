import os
import numpy as np
import pandas as pd
import re
import ast
from collections import defaultdict

# -------------------- Config --------------------

network_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/Network_Connectivity/"
summary_file = os.path.join(network_dir, "combined_connectivity_summary.csv")
summary_excel_file = os.path.join(network_dir, "combined_connectivity_summary.xlsx")
summary_avg_file = os.path.join(network_dir, "combined_connectivity_summary_avg_across_states.csv")
summary_avg_excel_file = os.path.join(network_dir, "combined_connectivity_summary_avg_across_states.xlsx")

column_replacements = {
    'Default': 'DMN',
    'DorsalAttention': 'DAN',
    'Limbic': 'LIN',
    'Somatomotor': 'SMN',
    'VentralAttention': 'VAN',
    'Visual': 'VIN',
    'Frontoparietal': 'FPN',
    'Default-DorsalAttention': 'DAN-DMN',
    'Default-Limbic': 'LIN-DMN',
    'Default-Somatomotor':  'SMN-DMN',
    'Default-VentralAttention': 'VAN-DMN',
    'Default-Visual': 'VIN-DMN',
    'DorsalAttention-Limbic': 'DAN-LIN',
    'DorsalAttention-Somatomotor':  'SMN-DAN',
    'DorsalAttention-VentralAttention': 'DAN-VAN',
    'DorsalAttention-Visual': 'VIN-DAN',
    'Limbic-Somatomotor': 'SMN-LIN',
    'Limbic-VentralAttention': 'VAN-LIN',
    'Limbic-Visual': 'VIN-LIN',
    'Somatomotor-VentralAttention': 'SMN-VAN',
    'Somatomotor-Visual': 'VIN-SMN',
    'VentralAttention-Visual': 'VIN-VAN',
    'Default-Frontoparietal': 'FPN-DMN',
    'DorsalAttention-Frontoparietal': 'DAN-FPN',
    'Frontoparietal-Limbic': 'LIN-FPN',
    'Frontoparietal-Somatomotor': 'SMN-FPN',
    'Frontoparietal-VentralAttention': 'VAN-FPN',
    'Frontoparietal-Visual': 'VIN-FPN',
    
    
}

# -------------------- Load Raw Connectivity Data --------------------

def load_raw_connectivity_data(network_dir):
    within_data = defaultdict(list)
    between_data = defaultdict(list)

    for participant in os.listdir(network_dir):
        part_path = os.path.join(network_dir, participant)
        if not os.path.isdir(part_path):
            continue

        for mode in ["congruent", "incongruent"]:
            mode_path = os.path.join(part_path, mode)
            within_file = os.path.join(mode_path, f"{participant}_{mode}_within_network_conn_raw.npz")
            between_file = os.path.join(mode_path, f"{participant}_{mode}_between_network_conn_raw.npz")

            if os.path.exists(within_file):
                with np.load(within_file, allow_pickle=True) as data:
                    for key in data.files:
                        try:
                            window = ast.literal_eval(key.split('_', 2)[2])
                            within_data[(participant, mode, window)] = data[key].tolist()
                        except Exception as e:
                            print(f"[ERROR] Failed to parse key '{key}' in within-file: {e}")

            if os.path.exists(between_file):
                with np.load(between_file, allow_pickle=True) as data:
                    for key in data.files:
                        try:
                            window = ast.literal_eval(key.split('_', 2)[2])
                            between_data[(participant, mode, window)] = data[key].tolist()
                        except Exception as e:
                            print(f"[ERROR] Failed to parse key '{key}' in between-file: {e}")

    return within_data, between_data

# -------------------- DataFrame Construction --------------------

def extract_avg_connectivity(conn_dict, is_between=False):
    all_networks = set()
    processed_rows = []

    for (participant, mode, window), conn_list in conn_dict.items():
        row = {'participant_id': participant, 'mode': mode, 'state': window[0]}
        temp_values = defaultdict(list)

        for entry in conn_list:
            net_match = re.match(r"\[([^\]]+)\]: .* - ([\d\.\-]+) - .*", entry)
            if net_match:
                net_name = net_match.group(1)
                conn_val = abs(float(net_match.group(2)))  # Take the absolute value of the correlation

                if is_between:
                    net_pair = '-'.join(sorted(net_name.replace(" ", "").split(',')))
                    all_networks.add(net_pair)
                    temp_values[net_pair].append(conn_val)
                else:
                    all_networks.add(net_name)
                    temp_values[net_name].append(conn_val)

        for net in all_networks:
            row[net] = round(sum(temp_values[net]) / len(temp_values[net]), 5) if temp_values[net] else None

        processed_rows.append(row)

    return processed_rows, sorted(all_networks)

# -------------------- Run --------------------

within_data, between_data = load_raw_connectivity_data(network_dir)
within_rows, within_columns = extract_avg_connectivity(within_data, is_between=False)
between_rows, between_columns = extract_avg_connectivity(between_data, is_between=True)

within_df = pd.DataFrame(within_rows)
within_df = within_df[['participant_id', 'mode', 'state'] + within_columns]

between_df = pd.DataFrame(between_rows)
between_df = between_df[['participant_id', 'mode', 'state'] + between_columns]

combined_df = pd.merge(within_df, between_df, on=['participant_id', 'mode', 'state'], how='outer')
#combined_df.to_csv(summary_file, index=False)
#combined_df.to_excel(summary_excel_file, index=False)
print("[INFO] Combined connectivity DataFrame saved at:", summary_file)

# -------------------- Average Across States --------------------

avg_df = combined_df.groupby(['participant_id', 'mode','state']).mean(numeric_only=True).reset_index()
avg_df = avg_df.round(5)
avg_df.rename(columns=column_replacements, inplace=True)
avg_df.to_csv(summary_avg_file, index=False)
avg_df.to_excel(summary_avg_excel_file, index=False)
print("[INFO] Averaged connectivity DataFrame saved at:", summary_avg_file)

# -------------------- Save Separate Excel Files for Each Mode --------------------

for mode in avg_df['mode'].unique():
    mode_df = avg_df[avg_df['mode'] == mode].copy()
    mode_excel_file = os.path.join(network_dir, f"avg_connectivity_{mode}.xlsx")
    mode_df.to_excel(mode_excel_file, index=False)
    print(f"[INFO] Saved file for mode '{mode}': {mode_excel_file}")

'''
# -------------------- Compute FC Interference --------------------

# Pivot the dataframe to have one row per participant, with 'congruent' and 'incongruent' as columns
pivot_df = avg_df.pivot(index='participant_id', columns='mode')

# Flatten multi-level column names
pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
pivot_df = pivot_df.reset_index()

# Calculate FC Interference for each network (Incongruent - Congruent)
fc_cols = [col for col in pivot_df.columns if col.endswith('_incongruent') and not col.startswith(('participant_id', 'state'))]
interference_data = {}

for col in fc_cols:
    base_name = col.replace('_incongruent', '')
    cong_col = f"{base_name}_congruent"
    inter_col = f"{base_name}_interference"
    if cong_col in pivot_df.columns:
        pivot_df[inter_col] = pivot_df[col] - pivot_df[cong_col]
        interference_data[inter_col] = pivot_df[inter_col]

# Save to CSV
fc_interference_file = os.path.join(network_dir, "fc_interference_by_network.csv")
pivot_df.to_csv(fc_interference_file, index=False)
print(f"[INFO] Functional Connectivity Interference saved at: {fc_interference_file}")
'''
import os
import numpy as np

base_dir = "/projects/illinois/ahs/kch/nakhan2/ACE/Optimal_States/"
participants = [f for f in os.listdir(base_dir) if f.startswith('optimal')]

modes = ['congruent', 'incongruent']
optimal_states = {mode: [] for mode in modes}  # Store optimal states for each mode


for participant in participants:
   
   
    for mode in modes:
        file_path = os.path.join(base_dir, participant)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                found = False
                for line in f:
                    if line.startswith("Optimal state (Average):"):
                        try:
                            # Extract the integer value after colon, strip spaces
                            value_str = line.split(":")[1].strip()
                            optimal_state = int(value_str)
                            optimal_states[mode].append(optimal_state)
                            found = True
                            break
                        except (IndexError, ValueError):
                            print(f"Failed to parse optimal state in file: {file_path}")
                if not found:
                    print(f"'Optimal state (Average)' not found in file: {file_path}")
        else:
            print(f"Missing file for participant {participant}, mode {mode}")

# Compute and print median optimal state for each mode
for mode, states in optimal_states.items():
    if not states:
        print(f"No data found for mode '{mode}'")
        continue
    median_state = np.median(states)
    print(f"Median Optimal State ({mode}): {median_state}")

    # Save median to file
    output_file = os.path.join(base_dir, f"median_optimal_state_{mode}.txt")
    with open(output_file, 'w') as out_f:
        out_f.write(f"Median Optimal State ({mode}): {median_state}\n")

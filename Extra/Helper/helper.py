import numpy as np

file_path = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/Network_Connectivity/NU151W/congruent/NU151W_congruent_aggregated_conn_raw.npz"

def print_npz_contents(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(f"Loaded NPZ file: {file_path}")
    print("Keys in the file:", data.files, "\n")
    
    for key in data.files:
        arr = data[key]
        print(f"=== Key: '{key}' ===")
        print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
        
        # If the array is small or 1D, print the entire data
        # Otherwise, just show a short preview to avoid huge output
        if arr.ndim == 1 and arr.size < 50:
            print("Data:", arr)
        elif arr.size < 50:
            print("Data:\n", arr)
        else:
            # Print only the first 10 elements if 1D
            # or the first row if 2D (or if it's bigger dimension, we show first row)
            if arr.ndim == 1:
                print("Data (first 10 elements):", arr[:10], "...")
            elif arr.ndim == 2:
                print("Data (first row):", arr[0, :], "...")
            else:
                print("Data preview:", arr.flatten()[:10], "...")

        print()  # blank line for readability

if __name__ == "__main__":
    print_npz_contents(file_path)

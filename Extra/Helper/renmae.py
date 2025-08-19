import os

# Define the folder path
folder_path = "/projects/illinois/ahs/kch/nakhan2/NURISH_Cohort2/EventLists/"

# List all files in the directory
files = os.listdir(folder_path)

# Loop through each file and rename if necessary
for file in files:
    if file.endswith("_EL.txt") and not file.endswith("W_EL.txt"):  # Avoid renaming twice
        old_path = os.path.join(folder_path, file)
        
        # Insert "W" before "_EL.txt"
        new_file = file.replace("_EL.txt", "W_EL.txt")
        new_path = os.path.join(folder_path, new_file)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} → {new_file}")

print("File renaming complete.")

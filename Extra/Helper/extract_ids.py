import os

folder = '/projects/illinois/ahs/kch/nakhan2/ACE/EDF_Files'
output_file = '/projects/illinois/ahs/kch/nakhan2/ACE/ids.txt'

def extract_ids(folder):
    ids = set()  # Using a set to ensure uniqueness
    for filename in os.listdir(folder):
        if filename.endswith('.edf'):
            filename_wd = os.path.splitext(filename)[0]
            id = filename_wd.split('_')[0]
            ids.add(id)  # Add to set to avoid duplicates
    return sorted(ids)  # Sort the IDs before returning

ids = extract_ids(folder)

# Save sorted IDs to a text file
with open(output_file, 'w') as f:
    for id in ids:
        f.write(id + '\n')

print(f"Sorted IDs saved to {output_file}")
 
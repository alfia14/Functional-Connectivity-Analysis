import os

def count_files_in_folder(folder_path):
    count = 0
    count_folder = 0
    for entry in os.listdir(folder_path):
        count_folder +=1
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            count += 1
    return count,count_folder

# Example usage
folder_path = '/projects/illinois/ahs/kch/nakhan2/NURISH_Post/Network_Connectivity'
num_files = count_files_in_folder(folder_path)
print(f'Total number of files: {num_files[0], num_files[1]}')

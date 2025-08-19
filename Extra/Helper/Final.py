import pandas as pd
import os

# Load the CSV file
def process_data(file_path, save_directory):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Group by the specified columns and calculate the mean of 'Value'
    result = df.groupby(["ID", "Mode", "State", "Window", "Connectivity_Type"])["Value"].mean().reset_index()
    
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Define the save path
    save_path = os.path.join(save_directory, "processed_data.csv")
    
    # Save the result to a new CSV file
    result.to_csv(save_path, index=False)
    
    return result

# Example usage
file_path = "/projects/illinois/ahs/kch/nakhan2/Data/final_merged_data.csv"  # Replace with your actual file path
save_directory = "/projects/illinois/ahs/kch/nakhan2/Data"  # Directory where processed data will be saved

processed_df = process_data(file_path, save_directory)

# Display the processed data
import ace_tools as tools
tools.display_dataframe_to_user(name="Processed Data", dataframe=processed_df)

import glob
import json
import pandas as pd
import os

# Define the path to the folder containing the JSON files
folder_path = 'metrics/'

# Use glob to find all the JSON files in the folder
json_files = glob.glob(os.path.join(folder_path, '*.json'))

# Initialize an empty list to store dataframes for each file
dataframes = []

# Loop through each JSON file
for file in json_files:
    # Open and load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)
    
    # Extract the dataset names (keys) and accuracy values
    # Remove everything before '/' in the dataset names
    extracted_data = {key.split('/')[1]: value for key, value in data.items()}
    
    # Add the filename as a column
    extracted_data['filename'] = os.path.basename(file)
    
    # Append the dataframe to the list
    dataframes.append(extracted_data)

# Concatenate all the individual dataframes into a single dataframe
df = pd.DataFrame(dataframes)
print(df.to_string())

df['Param Ratio'] = 1 - df['filename'].str.extract(r'_(\d+\.\d+)').astype(float)
df['Method'] = df['filename'].apply(lambda x: 'LLM Pruner + Finetune' if 'train' in x else 'LLM Pruner')
print(df.to_string())
df = df[['Method', 'Param Ratio', 'nq_open', 'mmlu', 'boolq', 'piqa', 'openbookqa']]
print(df.to_string())

import os
import pandas as pd

def merge_csv_files(input_folder, output_file):
    """
    Merges all CSV files in the input folder into a single DataFrame based on 'observation_date'.
    The combined DataFrame is saved as a CSV file in the output location.

    Args:
    - input_folder (str): Path to the folder containing raw CSV files.
    - output_file (str): Path to save the combined CSV file.
    """
    # List all CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # Initialize an empty list to store DataFrames
    data_frames = []

    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        # Read each CSV file
        df = pd.read_csv(file_path)
        # Rename the value column to the file name (without extension) for clarity
        value_column = df.columns[1]  # Assuming second column contains values
        df.rename(columns={value_column: csv_file.replace('.csv', '')}, inplace=True)
        data_frames.append(df)

    # Merge all DataFrames on 'observation_date'
    combined_df = data_frames[0]
    for df in data_frames[1:]:
        combined_df = pd.merge(combined_df, df, on='observation_date', how='outer')

    # Sort by observation_date
    combined_df['observation_date'] = pd.to_datetime(combined_df['observation_date'])
    combined_df.sort_values(by='observation_date', inplace=True)

    # Save the combined DataFrame to a CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

def preprocess_data(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Convert observation_date to datetime
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # Drop specified columns
    columns_to_drop = ['SPPOPGROWUSA', 'FIXHAI', 'FEDFUNDS', 'UNRATE', 'CPIAUCSL']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    print(f"Dropped columns: {columns_to_drop}")
    
    # Interpolate missing values for the MORTGAGE30US column
    if 'MORTGAGE30US' in df.columns:
        # Set observation_date as index for time-based interpolation
        df.set_index('observation_date', inplace=True)
        df['MORTGAGE30US'] = df['MORTGAGE30US'].interpolate(method='time')
        # Reset the index after interpolation
        df.reset_index(inplace=True)

    # Filter to keep only the first row of every month
    df = df[df['observation_date'].dt.is_month_start]
    print("Filtered to keep only the first row of every month.")
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Define input and output paths
    raw_data_folder = 'data/raw/'
    combined_data_file = 'data/processed/combined_data.csv'
    
    # Call the function to merge CSV files
    merge_csv_files(raw_data_folder, combined_data_file)

    # Input and output paths
    input_file = 'data/processed/combined_data.csv'
    output_file = 'data/processed/cleaned_data.csv'
    
    # Run preprocessing
    preprocess_data(input_file, output_file)

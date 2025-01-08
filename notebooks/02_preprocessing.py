import pandas as pd

def preprocess_data(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file)
    
    # Convert observation_date to datetime
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    
    # Drop specified columns
    columns_to_drop = ['SPPOPGROWUSA', 'FIXHAI', 'CES0500000003']
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



# Run preprocessing
preprocess_data(input_file, output_file)


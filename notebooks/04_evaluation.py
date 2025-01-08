import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# File path to the processed data
data_file = 'data/processed/combined_data.csv'

# Load the dataset
df = pd.read_csv(data_file)

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Plot the distribution of the target variable
    plt.figure(figsize=(8, 5))
    sns.histplot(data['CSUSHPISA'], kde=True, bins=30)
    plt.title("Distribution of Home Price Index")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

    # Time series trend of the target variable
    if 'observation_date' in data.columns:
        data['Date'] = pd.to_datetime(data['observation_date'])
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x='Date', y='Home_Price_Index', label='Home Price Index')
        plt.title("Trend of Home Price Index Over Time")
        plt.xlabel("Year")
        plt.ylabel("Home Price Index")
        plt.legend()
        plt.show()

# Perform EDA
perform_eda(df)

# Feature Engineering
def feature_engineering(data):
    # Handle missing values (example: fill with median or drop)
    data.fillna(data.median(), inplace=True)

    # Create lagged features for potential leading indicators
    lag_columns = ['Interest_Rate', 'Unemployment_Rate', 'Inflation_Rate']
    for col in lag_columns:
        if col in data.columns:
            data[f'{col}_Lag1'] = data[col].shift(1)
            data[f'{col}_Lag2'] = data[col].shift(2)

    # Create rate of change features for key variables
    rate_columns = ['Interest_Rate', 'Inflation_Rate']
    for col in rate_columns:
        if col in data.columns:
            data[f'{col}_Rate_Change'] = data[col].pct_change()

    # Standardize numerical features
    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'Home_Price_Index']  # Exclude target variable
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Drop rows with NaN values after feature engineering
    data.dropna(inplace=True)

    return data

# Apply feature engineering
#df = feature_engineering(df)

# Save the feature-engineered dataset
output_file = 'data/processed/feature_data.csv'
df.to_csv(output_file, index=False)
print(f"Feature-engineered dataset saved to {output_file}")

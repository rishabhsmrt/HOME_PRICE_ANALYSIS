import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the cleaned data
file_path = 'data/processed/cleaned_data.csv'  # Replace with the correct path to your file
output_folder_scatter = 'reports/figures/scatter_plots/'  # Folder to save scatter plots
output_folder_time_series = 'reports/figures/time_series/'  # Folder to save time-series plots
df = pd.read_csv(file_path)

# Ensure observation_date is datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Ensure the output folders exist
os.makedirs(output_folder_scatter, exist_ok=True)
os.makedirs(output_folder_time_series, exist_ok=True)

def create_scatter_plot(x_col, y_col, save_path):
    """
    Create and save a scatter plot with a trend line.
    :param x_col: Column name for the x-axis
    :param y_col: Column name for the y-axis
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title(f"Scatter Plot: {x_col} vs. {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to: {save_path}")

def create_time_series_plot(x_col, y_col, save_path):
    """
    Create and save a time-series plot for a single column against observation_date.
    :param x_col: The x-axis column (usually observation_date)
    :param y_col: The column to plot against x_col
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], label=y_col, color='blue', alpha=0.8)
    plt.plot(df[x_col], df['CSUSHPISA'], label='CSUSHPISA', color='red', alpha=0.8)
    plt.title(f"Time-Series Plot: {y_col} and CSUSHPISA over Time")
    plt.xlabel(x_col)
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time-series plot saved to: {save_path}")

# Generate time-series plots for CSUSHPISA and each other column
x_column = 'observation_date'
target_column = 'CSUSHPISA'
other_columns = [col for col in df.columns if col not in ['observation_date', target_column]]

for column in other_columns:
    save_path_time_series = os.path.join(output_folder_time_series, f"time_series_plot_{target_column}_vs_{column}.png")
    create_time_series_plot(x_column, column, save_path_time_series)

for column in other_columns:
    save_path_scatter = os.path.join(output_folder_scatter, f"scatter_plot_{target_column}_vs_{column}.png")
    create_scatter_plot(column, target_column, save_path_scatter)

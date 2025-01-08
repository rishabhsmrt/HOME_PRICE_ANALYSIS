import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import os

# Load the cleaned data
file_path = 'data/processed/cleaned_data.csv'  # Replace with the correct path to your file
df = pd.read_csv(file_path)

df = df.dropna()

# Ensure observation_date is datetime and drop it for modeling
df['observation_date'] = pd.to_datetime(df['observation_date'])
df.drop(columns=['observation_date'], inplace=True)

# Handle missing values if any (ensure dataset is clean)
df.fillna(method='ffill', inplace=True)

# Define target variable and features
target = 'CSUSHPISA'
X = df.drop(columns=[target])
y = df[target]

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define a function for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Fit the model, evaluate its performance, and return metrics.
    """
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return r2, rmse

# Baseline Model: Linear Regression
baseline_model = LinearRegression()
baseline_r2, baseline_rmse = evaluate_model(baseline_model, X_train, y_train, X_test, y_test)

# Advanced Model: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_r2, rf_rmse = evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Advanced Model: XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_r2, xgb_rmse = evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

# Cross-validation for Linear Regression
cv_scores = cross_val_score(baseline_model, X, y, scoring='r2', cv=5)
print(f"Cross-Validation R² (Linear Regression): {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")

# Save results to a file
output_folder = 'reports/metrics/'
os.makedirs(output_folder, exist_ok=True)
results_path = os.path.join(output_folder, 'model_metrics.txt')

with open(results_path, 'w') as f:
    f.write("Model Evaluation Metrics\n")
    f.write(f"Baseline Model (Linear Regression): R²={baseline_r2:.4f}, RMSE={baseline_rmse:.4f}\n")
    f.write(f"Random Forest: R²={rf_r2:.4f}, RMSE={rf_rmse:.4f}\n")
    f.write(f"XGBoost: R²={xgb_r2:.4f}, RMSE={xgb_rmse:.4f}\n")
    f.write(f"Cross-Validation R² (Linear Regression): {cv_scores.tolist()}\n")
    f.write(f"Mean CV R²: {cv_scores.mean():.4f}\n")

print(f"Model evaluation metrics saved to: {results_path}")

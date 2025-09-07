# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load the dataset (example path; replace with actual relative path if needed)
file_path = Path("example.csv")
data = pd.read_csv(file_path)

print(data.head())

# Prepare the data for Lasso regression
# Prepare features and labels (check your column names if different)
X = data.drop(columns=['Wavelength', 'Class'])
y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Lasso regression with a smaller alpha
lasso = Lasso(alpha=0.01)  # Adjust alpha as needed
lasso.fit(X_scaled, y)

# Extract coefficients
coef = lasso.coef_

# Select features with non-zero coefficients
selected_features = X.columns[coef != 0]

# Build new DataFrame with selected features
selected_data = data[['Wavelength'] + list(selected_features) + ['Class']]

# Ensure output directory exists
output_dir = Path("outputs/example")
output_dir.mkdir(parents=True, exist_ok=True)

# Save filtered dataset to CSV
output_file_path = output_dir / "Groundnut_mst_filtered.csv"
selected_data.to_csv(output_file_path, index=False)

print(f"Filtered data saved to {output_file_path}")

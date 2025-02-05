import numpy as np
import pandas as pd

file_path = "train.csv"
df = pd.read_csv(file_path)

# Define the number of data points to generate
num_samples = 100000

# Extract statistical properties of the existing dataset
means = df.mean()
stds = df.std()

# Generate new synthetic dataset following the same distribution
new_data = {col: np.random.normal(means[col], stds[col], num_samples).astype(int) if df[col].dtype == 'int64' else np.random.normal(means[col], stds[col], num_samples) for col in df.columns[:-1]}

# Ensure categorical columns have valid values
new_data["sex"] = np.random.choice([0, 1], num_samples)
new_data["cp"] = np.random.choice(df["cp"].unique(), num_samples)
new_data["fbs"] = np.random.choice([0, 1], num_samples)
new_data["restecg"] = np.random.choice(df["restecg"].unique(), num_samples)
new_data["exang"] = np.random.choice([0, 1], num_samples)
new_data["slope"] = np.random.choice(df["slope"].unique(), num_samples)
new_data["ca"] = np.random.choice(df["ca"].unique(), num_samples)
new_data["thal"] = np.random.choice(df["thal"].unique(), num_samples)

# Generate target values based on intensity, scaled between 0 to 5
target_values = np.sum([new_data[col] for col in df.columns[:-1]], axis=0)
normalized_target = (target_values - np.min(target_values)) / (np.max(target_values) - np.min(target_values)) * 5
new_data["target"] = normalized_target.round().astype(int)

# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# Ensure the distribution is balanced
new_df["target"] = np.clip(new_df["target"], 0, 5)

output_file_path = "extended_dataset.csv"

# Save the dataset as a CSV file
new_df.to_csv(output_file_path, index=False)

# Provide the download link
output_file_path

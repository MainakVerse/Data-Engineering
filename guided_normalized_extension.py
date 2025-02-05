import numpy as np
import pandas as pd

# Load the dataset
file_path = "diabetes.csv"
df = pd.read_csv(file_path)

# Display basic info about the dataset
df.info(), df.head()

# Identify the pattern in numeric columns
stats = df.describe()

# Generate synthetic data following the identified pattern
num_new_samples = 100000 - len(df)
new_data = {
    "Pregnancies": np.random.randint(stats.loc["min", "Pregnancies"], stats.loc["max", "Pregnancies"] + 1, num_new_samples),
    "Glucose": np.random.randint(stats.loc["min", "Glucose"], stats.loc["max", "Glucose"] + 1, num_new_samples),
    "BloodPressure": np.random.randint(stats.loc["min", "BloodPressure"], stats.loc["max", "BloodPressure"] + 1, num_new_samples),
    "SkinThickness": np.random.randint(stats.loc["min", "SkinThickness"], stats.loc["max", "SkinThickness"] + 1, num_new_samples),
    "Insulin": np.random.randint(stats.loc["min", "Insulin"], stats.loc["max", "Insulin"] + 1, num_new_samples),
    "BMI": np.random.uniform(stats.loc["min", "BMI"], stats.loc["max", "BMI"], num_new_samples),
    "HbA1c_level": np.random.uniform(stats.loc["min", "HbA1c_level"], stats.loc["max", "HbA1c_level"], num_new_samples),
    "DiabetesPedigreeFunction": np.random.uniform(stats.loc["min", "DiabetesPedigreeFunction"], stats.loc["max", "DiabetesPedigreeFunction"], num_new_samples),
    "Age": np.random.randint(stats.loc["min", "Age"], stats.loc["max", "Age"] + 1, num_new_samples),
}

# Create new DataFrame
df_extended = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

# Normalize HbA1c_level and DiabetesPedigreeFunction to a 0-1 range
df_extended['HbA1c_level_norm'] = (df_extended['HbA1c_level'] - df_extended['HbA1c_level'].min()) / (df_extended['HbA1c_level'].max() - df_extended['HbA1c_level'].min())
df_extended['DiabetesPedigreeFunction_norm'] = (df_extended['DiabetesPedigreeFunction'] - df_extended['DiabetesPedigreeFunction'].min()) / (df_extended['DiabetesPedigreeFunction'].max() - df_extended['DiabetesPedigreeFunction'].min())

# Combine the normalized values to create a score
df_extended['combined_score'] = (df_extended['HbA1c_level_norm'] + df_extended['DiabetesPedigreeFunction_norm']) / 2

# Map the combined score to the range 0-5
df_extended['Outcome'] = np.floor(df_extended['combined_score'] * 6).clip(0, 5)

# Drop the intermediate columns used for calculation
df_extended.drop(columns=['HbA1c_level_norm', 'DiabetesPedigreeFunction_norm', 'combined_score'], inplace=True)

# Save the new dataset
extended_file_path = "diabetes_extended.csv"
df_extended.to_csv(extended_file_path, index=False)

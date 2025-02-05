import pandas as pd
import numpy as np
from sklearn.utils import resample

def generate_synthetic_data(df, target_size=100000):
    current_size = len(df)
    additional_size = target_size - current_size
    
    # Resampling original data with replacement to get a base
    synthetic_data = resample(df, replace=True, n_samples=additional_size, random_state=42)
    
    # Introduce slight variations based on original probability distributions
    for col in df.columns:
        if col != 'UnderRisk':
            prob = df[col].mean()
            prob = np.clip(prob, 0, 1)  # Ensure probability stays in valid range
            synthetic_data[col] = np.random.binomial(1, prob, size=additional_size)
    
    # Ensure UnderRisk is proportional to specific columns
    weight_columns = ['HighBP', 'Obese', 'Metabolic_syndrome', 'Use_of_stimulant_drugs', 'Family_history']
    weight_sum = synthetic_data[weight_columns].sum(axis=1)
    
    # Define UnderRisk based on weighted sum probability
    synthetic_data['UnderRisk'] = np.where(weight_sum >= 3, 1, 0)
    
    # Merge original and synthetic data
    extended_df = pd.concat([df, synthetic_data], ignore_index=True)
    return extended_df

# Load original dataset
file_path = "Train.csv"
df = pd.read_csv(file_path)

# Fill NaN values in dataset (if any)
df.fillna(0, inplace=True)

# Generate extended dataset
extended_df = generate_synthetic_data(df, target_size=100000)

# Save to new CSV
output_file_path = "Extended_Train.csv"
extended_df.to_csv(output_file_path, index=False)

print(f"Extended dataset saved to: {output_file_path}")

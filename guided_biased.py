import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
n = 100000

# Define the 20 parameters and their probabilities (p) for Bernoulli distribution
parameters = {
    "Latitude": 0.6,
    "Altitude": 0.4,
    "Proximity_to_Water_Bodies": 0.7,
    "Prevailing_Winds": 0.5,
    "Topography": 0.3,
    "Ocean_Currents": 0.4,
    "Atmospheric_Pressure": 0.6,
    "Humidity": 0.8,
    "Temperature": 0.7,
    "Intertropical_Convergence_Zone": 0.2,
    "Monsoon_Winds": 0.5,
    "Jet_Streams": 0.4,
    "El_Nino_La_Nina": 0.3,
    "Vegetation_Cover": 0.6,
    "Urbanization": 0.5,
    "Air_Pollution": 0.4,
    "Cyclones_and_Storms": 0.3,
    "Seasonal_Changes": 0.7,
    "Distance_from_Equator": 0.5,
    "Climate_Change": 0.6
}

# Generate Bernoulli-distributed data for each parameter
data = {param: np.random.binomial(1, p, n) for param, p in parameters.items()}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("rainfall_dataset.csv", index=False)

print("Dataset generated and saved as 'rainfall_dataset.csv'.")

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('Train.csv')

# Prepare the data for modeling UnderRisk
X = df.drop(columns=["UnderRisk"])
y = df["UnderRisk"]

# Train a Decision Tree Classifier to model dependencies
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# Generate new data maintaining feature distributions
num_new_samples = 100000 - len(df)
new_data = pd.DataFrame()

for col in X.columns:
    new_data[col] = np.random.choice(df[col].values, num_new_samples, replace=True)

# Predict UnderRisk for generated data
new_data["UnderRisk"] = model.predict(new_data)

# Combine with the original dataset
extended_df = pd.concat([df, new_data], ignore_index=True)

# Save the extended dataset
extended_file_path = "Extended_Train.csv"
extended_df.to_csv(extended_file_path, index=False)

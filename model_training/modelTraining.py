import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create model_training directory if it doesn't exist
model_training_dir = "model_training"
os.makedirs(model_training_dir, exist_ok=True)

# Change to the model_training directory
os.chdir(model_training_dir)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(0, 10, n_samples),
        'monthly_charges': np.random.uniform(20, 200, n_samples).round(2),
        'total_charges': np.random.uniform(0, 8000, n_samples).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% churn rate
    }
    df = pd.DataFrame(data)
    df['total_charges'] = (df['monthly_charges'] * df['tenure']).round(2)
    return df

# Generate and save synthetic data
data = generate_synthetic_data()
data.to_csv("customer_data.csv", index=False)
print("Synthetic customer_data.csv generated successfully in the model_training directory.")

# Load data
try:
    data = pd.read_csv("customer_data.csv")
except FileNotFoundError:
    print("Error: customer_data.csv not found. Please ensure the file is in the correct location.")
    exit(1)

# Preprocess data (e.g., encode categorical features)
data = pd.get_dummies(data, columns=["contract_type"])

# Split data
X = data.drop("churn", axis=1)
y = data["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("Model trained and saved successfully in the model_training directory.")

# Save feature names
feature_names = X.columns.tolist()
with open("feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))
print("Feature names saved successfully in the model_training directory.")
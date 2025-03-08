import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load warehouse data (replace 'warehouse_data.csv' with your actual file)
df = pd.read_csv("/home/richa/entrepot_optimise/custom_data/warehouse_data.csv")

# Select relevant features
X = df[['volume', 'past_sales', 'storage_days']]
y = df['profit']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Save the trained model
joblib.dump(model, "profit_model.pkl")
print("Model saved as 'profit_model.pkl'")

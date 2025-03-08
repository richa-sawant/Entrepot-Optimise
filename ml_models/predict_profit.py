import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("profit_model.pkl")

# Example input with correct column names
example_product = pd.DataFrame([[50, 200, 10]], columns=["volume", "past_sales", "storage_days"])

# Predict profit
predicted_profit = model.predict(example_product)
print(f"Predicted Profit: {predicted_profit[0]:.2f}")

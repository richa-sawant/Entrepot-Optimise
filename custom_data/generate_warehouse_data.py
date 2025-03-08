import pandas as pd
import numpy as np

# Define the number of products
num_products = 1000  

# Generate random product names
product_names = [f"Product_{i}" for i in range(1, num_products + 1)]

# Generate random values for each feature
np.random.seed(42)  # For reproducibility
volume = np.random.randint(1, 100, num_products)  # Product volume in cubic meters
past_sales = np.random.randint(50, 1000, num_products)  # Sales history (demand)
storage_days = np.random.randint(1, 60, num_products)  # Days stored in warehouse
profit = np.round(volume * np.random.uniform(2, 10, num_products), 2)  # Profit (based on volume)

# Create DataFrame
df = pd.DataFrame({
    "product_name": product_names,
    "volume": volume,
    "past_sales": past_sales,
    "storage_days": storage_days,
    "profit": profit
})

# Save to CSV
df.to_csv("warehouse_data.csv", index=False)

print("✅ Custom warehouse dataset generated: warehouse_data.csv")

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

# Load warehouse data
df = pd.read_csv("/home/richa/entrepot_optimise/custom_data/warehouse_data.csv")

# Select features for clustering
X = df[['volume', 'past_sales', 'storage_days']].values  # Convert to numpy array

# Apply GMM Clustering (3 clusters)
gmm = GaussianMixture(n_components=3, random_state=42)
df['category'] = gmm.fit_predict(X)

# Map categories to meaningful names
category_map = {0: "Low Demand", 1: "Medium Demand", 2: "High Demand"}
df['category_name'] = df['category'].map(category_map)

# Save the clustered data
df.to_csv("categorized_warehouse_data.csv", index=False)

# Save the trained GMM model
joblib.dump(gmm, "gmm_model.pkl")

print("Product Categorization Completed!")
print(df[['product_name', 'category_name']].head())  # Display first few categorized products

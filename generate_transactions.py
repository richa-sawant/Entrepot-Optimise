import pandas as pd
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Define product categories (500 products instead of 1000)
products = ["Product_" + str(i) for i in range(1, 501)]  

# Define some frequently bought-together products
frequent_pairs = [
    ("Product_129", "Product123"), ("Product_5", "Product_6"), ("Product_182", "Product_229"),
    ("Product_23", "Product_24"), ("Product_82", "Product_84"), ("Product_10", "Product_12")
]

# Generate random transactions (2000 orders)
transactions = []
for _ in range(2000):
    num_items = random.randint(4, 10)  # Increase number of items per transaction
    transaction = random.sample(products, num_items - 2)  # Random selection
    transaction += list(random.choice(frequent_pairs))  # Add a common pair
    transactions.append(transaction)

# Convert transactions into a DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_array, columns=te.columns_)

# Apply Apriori with optimized min_support
frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)  

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Save results
rules.to_csv("frequent_itemsets.csv", index=False)
print("Frequent itemsets saved in 'frequent_itemsets.csv'.")

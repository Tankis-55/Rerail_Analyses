import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.frequent_patterns import fpgrowth, association_rules
import gc #for free up of memory

# Load cleaned dataset with optimized data types
dtype_dict = {"CustomerID": "int32", "StockCode": "category", "Quantity": "int16"}
df = pd.read_csv("clean_online_retail.csv", dtype=dtype_dict)
print(df["StockCode"].unique()) 

min_transactions = 5
valid_items = df["StockCode"].value_counts()
valid_items = valid_items[valid_items >= min_transactions].index
df = df[df["StockCode"].isin(valid_items)]

# 500-Top items
top_items = df["StockCode"].value_counts().nlargest(500).index
df = df[df["StockCode"].isin(top_items)]

# Create a pivot table (CustomerID × StockCode)
basket = df.pivot_table(index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", observed=False).fillna(0)

# Convert to binary format (0 or 1) to bool
basket = (basket > 0)

# free up memory from DataFrame
del df
gc.collect()

# Apply fpgrowth algorithm to find frequent itemsets
frequent_itemsets = fpgrowth(basket, min_support=0.03, use_colnames=True)
if frequent_itemsets.empty:
    frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Save rules
rules.to_csv("association_rules.csv", index=False)
print("✅ Association rules generated and saved!")

if rules.empty:
    print("⚠️ Warning: No association rules found! Try lowering min_support.")

# Function to recommend products
def recommend_products(product_id, rules, top_n=5):
    """Recommends products based on association rules."""
    recommendations = rules[rules["antecedents"].apply(lambda x: product_id in x)]
    recommendations = recommendations.sort_values("lift", ascending=False).head(top_n)
    return recommendations[["antecedents", "consequents", "lift"]]

# Example: Recommend products for a specific item
product_recommendations = recommend_products("84879", rules)
print(product_recommendations)
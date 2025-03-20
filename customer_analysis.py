import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("clean_online_retail.csv")

# Remove negative quantities (returns)
df = df[df["Quantity"] > 0]

# Calculate total spending per customer
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
customer_spending = df.groupby("CustomerID")["TotalPrice"].sum().reset_index()

# Calculate number of unique purchases per customer
customer_orders = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
customer_orders.columns = ["CustomerID", "OrderCount"]

# Merge both DataFrames
customer_data = pd.merge(customer_spending, customer_orders, on="CustomerID")
filtered_customer_data = customer_data[customer_data["TotalPrice"] > 0]  # Only positive spending

# Filter out extreme outliers (above 99th percentile)
threshold = customer_data["TotalPrice"].quantile(0.99)
filtered_data = filtered_customer_data[filtered_customer_data["TotalPrice"] <= threshold]

# Plot spending distribution
plt.figure(figsize=(12, 6))
sns.histplot(filtered_data["TotalPrice"], bins=50, kde=True, color="royalblue")
plt.title("Distribution of Customer Spending (Filtered)")
plt.xlabel("Total Spending (£)")
plt.ylabel("Number of Customers")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("customer_spending.svg", dpi=300, bbox_inches="tight")
plt.show()

# Save customer analysis data
customer_data.to_csv("customer_analysis.csv", index=False)

print("✅ Customer analysis completed and saved!")
import pandas as pd

# Load dataset
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Remove rows with missing CustomerID
df.dropna(subset=["CustomerID"], inplace=True)

# Convert InvoiceDate to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove negative and zero quantities
df = df[df["Quantity"] > 0]

# Save cleaned dataset
df.to_csv("clean_online_retail.csv", index=False)

print("✅ Data cleaned and saved successfully!")
print(df.head())
print(df.tail())
print(df.info())
print(df.isnull().sum())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load cleaned dataset
df = pd.read_csv("clean_online_retail.csv")

# Convert InvoiceDate to datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M")
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
monthly_sales = df.groupby("Month")["TotalPrice"].sum()
monthly_sales = monthly_sales.replace([np.inf, -np.inf], np.nan).dropna()

# Decompose sales time series
decomposition = seasonal_decompose(monthly_sales, model="multiplicative", period=6)

# Convert PeriodIndex to Timestamp for plotting
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot seasonal decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(monthly_sales.index, monthly_sales, label="Original", color="blue")
plt.legend(loc="upper left")
plt.subplot(412)
plt.plot(decomposition.trend.index.to_timestamp(), decomposition.trend, label="Trend", color="green")
plt.legend(loc="upper left")
plt.subplot(413)
plt.plot(decomposition.seasonal.index.to_timestamp(), decomposition.seasonal, label="Seasonality", color="orange")
plt.legend(loc="upper left")
plt.subplot(414)
plt.plot(decomposition.resid.index.to_timestamp(), decomposition.resid, label="Residual", color="red")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("seasonal decomposition.svg", dpi=300, bbox_inches="tight")


# Plot sales trends
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Total Price")
plt.grid(True)
plt.savefig("sales trends.svg", dpi=300, bbox_inches="tight")

# Find top-selling products
top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

# Plot top products
plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.index, y=top_products.values, palette="coolwarm")
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Product Name")
plt.ylabel("Total  Quantity Sold")
plt.xticks(rotation=45)
plt.savefig("top products.svg", dpi=300, bbox_inches="tight")

# Forecasting sales using ARIMA
train_data = monthly_sales[:-7]
test_data = monthly_sales[-6:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5,1,0))
arima_fit = arima_model.fit()

# Forecast next 6 months
arima_forecast = arima_fit.forecast(steps=6)

# Plot ARIMA forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales, label="Actual Sales")
plt.plot(test_data.index, arima_forecast, label="ARIMA Forecast", linestyle="dashed")
plt.title("Sales Forecast with ARIMA")
plt.xlabel("Month")
plt.ylabel("Total Price")
plt.legend()
plt.grid(True)
plt.savefig("ARIMA forecast.svg", dpi=300, bbox_inches="tight")

# Fit SARIMA model
sarima_model = SARIMAX(train_data, 
                       order=(1,1,1), 
                       seasonal_order=(1,1,1,12),
                       enforce_stationarity=False)
sarima_fit = sarima_model.fit()

# Forecast with SARIMAX
sarimax_forecast = sarima_fit.forecast(steps=6)

# Plot SARIMAX forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales, label="Actual Sales")
plt.plot(test_data.index, sarimax_forecast, label="SARIMAX Forecast", linestyle="dashed")
plt.title("Sales Forecast with SARIMAX")
plt.xlabel("Month")
plt.ylabel("Total Price")
plt.legend()
plt.grid(True)
plt.savefig("SARIMAX forecast.svg", dpi=300, bbox_inches="tight")

# Calculate error metrics ARIMA
arima_mae = mean_absolute_error(test_data, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
arima_mape = np.mean(np.abs((test_data - arima_forecast) / test_data)) * 100
arima_accuracy = 100 - arima_mape

print("\n📊 Model Accuracy Metrics - ARIMA:")
print(f"✅ Mean Absolute Error (MAE): {arima_mae:.2f}")  
print(f"✅ Root Mean Squared Error (RMSE): {arima_rmse:.2f}") 
print(f"✅ Mean Absolute Percentage Error (MAPE): {arima_mape:.2f}%")  
print(f"🎯 Forecast Accuracy: {arima_accuracy:.2f}%") 

# Calculate error metrics SARIMAX
sarimax_mae = mean_absolute_error(test_data, sarimax_forecast)
sarimax_rmse = np.sqrt(mean_squared_error(test_data, sarimax_forecast))
sarimax_mape = np.mean(np.abs((test_data - sarimax_forecast) / test_data)) * 100
sarimax_accuracy = 100 - sarimax_mape

print("\n📊 Model Accuracy Metrics - SARIMAX:")
print(f"✅ Mean Absolute Error (MAE): {sarimax_mae:.2f}")  
print(f"✅ Root Mean Squared Error (RMSE): {sarimax_rmse:.2f}") 
print(f"✅ Mean Absolute Percentage Error (MAPE): {sarimax_mape:.2f}%")  
print(f"🎯 Forecast Accuracy: {sarimax_accuracy:.2f}%") 

print("\n🚀✅ Sales trends analyzed and forecasts generated successfully!")
# Importing necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Step 1: Download Apple stock data
symbol = "AAPL"
data = yf.download(symbol, start="2010-01-01", end="2025-01-01")

# Check if 'Adj Close' column exists; if not, use 'Close'
if "Adj Close" not in data.columns:
    data["Adj Close"] = data["Close"]

# Calculate log returns
data["Log_Returns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))

# Dropping NaN values (first row due to shift)
log_returns = data["Log_Returns"].dropna()

# Step 2: Train-test split (80% train, 20% test)
train_size = int(len(log_returns) * 0.8)
train_returns = log_returns[:train_size]
test_returns = log_returns[train_size:]

# Step 3: Fit GARCH(1, 1) model to training data
model = arch_model(train_returns, vol="Garch", p=1, q=1)
garch_fit = model.fit(disp="off")

# Display the GARCH model summary
print(garch_fit.summary())

# Step 4: Rolling forecast for the test set
rolling_forecasted_variance = []
rolling_forecasted_mean = []

# Start from training data and dynamically forecast one step at a time
for i in range(len(test_returns)):
    # Define the rolling data: train + test up to the current point
    current_data = pd.concat([train_returns, test_returns[:i]])
    
    # Re-fit the model on the current data
    rolling_model = arch_model(current_data, vol="Garch", p=1, q=1)
    rolling_fit = rolling_model.fit(disp="off")
    
    # Forecast one step ahead
    forecast = rolling_fit.forecast(horizon=1)
    
    # Save the forecasted mean and variance
    rolling_forecasted_variance.append(forecast.variance.values[-1, 0])
    rolling_forecasted_mean.append(forecast.mean.values[-1, 0])

# Convert forecasts to Series with appropriate index
forecasted_variance = pd.Series(rolling_forecasted_variance, index=test_returns.index)
forecasted_mean = pd.Series(rolling_forecasted_mean, index=test_returns.index)

# Actual variance as a proxy from test data
test_volatility = test_returns ** 2  # Squared returns represent variance

# Step 5: Plot actual vs forecasted volatility
plt.figure(figsize=(12, 6))
plt.plot(test_volatility, label="Actual Variance (Test Data)", color="blue")
plt.plot(forecasted_variance, label="Forecasted Variance (GARCH)", color="orange")
plt.legend()
plt.title("Actual vs. Forecasted Volatility (Test Set)")
plt.xlabel("Date")
plt.ylabel("Variance")
plt.show()

# Print some forecast results
print("\nFirst 5-Day Forecast on Test Data:")
print("Day | Forecasted Mean | Forecasted Variance | Actual Variance")
for i, date in enumerate(test_returns.index[:5]):
    print(f"{date.date()} | {forecasted_mean.iloc[i]:>15.6f} | {forecasted_variance.iloc[i]:>18.6f} | {test_volatility.iloc[i]:>15.6f}")

# Install required packages
# pip install yfinance arch pandas matplotlib numpy

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

# Step 2: Fit GARCH(1, 1) model
model = arch_model(log_returns, vol="Garch", p=1, q=1)
garch_fit = model.fit(disp="off")

# Display the GARCH model summary
print(garch_fit.summary())

# Step 3: Plot Log Returns and Conditional Volatility
fitted_volatility = garch_fit.conditional_volatility

plt.figure(figsize=(12, 6))
plt.plot(log_returns, label="Log Returns", color="blue")
plt.plot(fitted_volatility, label="Conditional Volatility (GARCH)", color="red")
plt.legend(loc="upper right")
plt.title("Log Returns and Fitted GARCH Volatility")
plt.show()

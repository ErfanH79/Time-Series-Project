import yfinance as yf
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download data
gs = yf.download("AAPL", start="2013-10-01", end="2024-10-30")

# Preprocess data
dataset_ex_df = gs.copy()
dataset_ex_df = dataset_ex_df.reset_index()
dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
dataset_ex_df.set_index('Date', inplace=True)

# Extract the 'Close' column as a Series
close_prices = dataset_ex_df['Close']
decomposition = seasonal_decompose(close_prices, model='multiplicative', period=365)

residual = decomposition.resid
residual = residual.dropna()

# Auto ARIMA to select optimal ARIMA parameters
model = auto_arima(residual, seasonal=False, trace=True)
print(model.summary())

# Define the ARIMA model
def arima_forecast(history):
    # Ensure history is 1D
    history = np.ravel(history)  # Flatten to 1D
    model = ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    return yhat

# Split data into train and test sets
train_size = int(len(dataset_ex_df) * 0.8)
train = close_prices.iloc[:train_size]  # Series
test = close_prices.iloc[train_size:]  # Series

# Walk-forward validation
history = train.values  # Convert to NumPy array
predictions = []

for t in range(len(test)):
    yhat = arima_forecast(history)  # Forecast the next value
    predictions.append(yhat)
    obs = test.iloc[t]  # Get the actual value
    history = np.append(history, obs)  # Append as a 1D array
    print(len(test)-(t+1))

# Calculate RMSE
mse_arima = mean_squared_error(test, predictions)
rmse_arima = np.sqrt(mse_arima)
print(f'ARIMA Model - MSE: {mse_arima}, RMSE: {rmse_arima}')


# Plot results
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(close_prices.index, close_prices, label='Actual')
plt.plot(test.index, predictions, color='red', linestyle='dotted', label='Predicted')
plt.title('ARIMA Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

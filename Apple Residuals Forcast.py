import yfinance as yf
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Download data
gs = yf.download("AAPL", start="2013-10-01", end="2024-10-30")

# Preprocess data
dataset_ex_df = gs.copy()
dataset_ex_df = dataset_ex_df.reset_index()
dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
dataset_ex_df.set_index('Date', inplace=True)

# Extract the 'Close' column as a Series
close_prices = dataset_ex_df['Close']
decomposition = seasonal_decompose(close_prices, model='additive', period=365)
residual = decomposition.resid
residual = residual.dropna()

# Create ACF and PACF plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot the ACF
plot_acf(residual, ax=ax[0], lags=49)
plt.title('Autocorrelation Function (ACF)')

# Plot the PACF
plot_pacf(residual, ax=ax[1], lags=20, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')


# Auto ARIMA to select optimal ARIMA parameters
model = auto_arima(residual, seasonal=False, trace=True)
print(model.summary())


def arima_forecast(history):
    # Ensure history is 1D
    history = np.ravel(history)  # Flatten to 1D
    model = ARIMA(history, order=(10, 0, 0))
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]
    return yhat

# Split data into train and test sets
train_size = int(len(residual) * 0.9)
train = residual.iloc[:train_size]  # Series
test = residual.iloc[train_size:]  # Series


# Walk-forward validation
history = train.values  # Convert to NumPy array
predictions = []


s = 1
A = []
for t in range(len(test)):
    yhat = arima_forecast(history)  # Forecast the next value
    predictions.append(yhat)
    if s<=len(test):
        history = np.append(history, yhat)  # Append as a 1D array
        A = np.append(A,test.iloc[t]) # Get the actual value
        s+=1
    else:
        history = history[:-len(test)]
        for i in A:
            history = np.append(history ,i)
        s = 1
        A =[]
    print(len(test)-(t+1))

# Calculate RMSE
print(len(test),len(predictions))
mse_arima = mean_squared_error(test, predictions)
rmse_arima = np.sqrt(mse_arima)
print(f'ARIMA Model - MSE: {mse_arima}, RMSE: {rmse_arima}')

# Plot results
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(residual.index, residual, color = "blue", label='Actual')
plt.plot(test.index, predictions, color='red',label='Predicted')
plt.title('ARIMA Predictions vs Actual Values')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
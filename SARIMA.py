

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Download Apple stock data
ticker = "NVDA"
data = yf.download(ticker, period="5y", interval="1d")
df = data['Close'].reset_index()
df.columns = ['Date', 'Price']

# Set date as index
df.set_index('Date', inplace=True)

# Check stationarity with Augmented Dickey-Fuller test
def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

test_stationarity(df['Price'])

# If not stationary, apply differencing (usually needed for stock prices)
df['Price_diff'] = df['Price'].diff().dropna()
test_stationarity(df['Price_diff'])

# Auto ARIMA to find optimal parameters (caution: can be slow)
auto_model = auto_arima(df['Price'], seasonal=True, m=5,  # 5 trading days per week
                        suppress_warnings=True, stepwise=True,
                        trace=True, error_action='ignore')

print(auto_model.summary())

# Manual SARIMA configuration (use auto_arima suggestions as starting point)
order = (2, 1, 2)          # (p, d, q)
seasonal_order = (1, 1, 1, 5)  # (P, D, Q, S)

# Fit SARIMA model
model = SARIMAX(df['Price'],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)
print(results.summary())

# Model diagnostics
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# Forecast next 30 days
forecast_steps = 30
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(df.index[-1], periods=forecast_steps+1, freq='B')[1:]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df['Price'], label='Historical Price')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='orange')
plt.fill_between(forecast_index,
                 forecast.conf_int()['lower Price'],
                 forecast.conf_int()['upper Price'],
                 color='orange', alpha=0.2)

plt.title(f'{ticker} Stock Price SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print forecast values
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Forecast': forecast.predicted_mean,
    'Lower CI': forecast.conf_int()['lower Price'],
    'Upper CI': forecast.conf_int()['upper Price']
})
print("\n30-Day Forecast:")
print(forecast_df)
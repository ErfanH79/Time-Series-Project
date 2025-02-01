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



from prophet import Prophet

# Prepare the data for Prophet
prophet_data = gs['Close'].reset_index()
prophet_data.columns = ['ds', 'y']

# Split the data
train_prophet = prophet_data[prophet_data['ds'] <= '2023-09-30']
test_prophet = prophet_data[prophet_data['ds'] > '2022-09-30']

# Fit the Prophet model
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(train_prophet)

# Make future dataframe for predictions
future_dates = prophet_model.make_future_dataframe(periods=len(test_prophet))
forecast = prophet_model.predict(future_dates)

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))
prophet_model.plot(forecast, ax=ax)
ax.plot(test_prophet['ds'], test_prophet['y'], label='Actual Prices', color='red', linewidth=2)
plt.title('Facebook Prophet Model Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# Evaluating the model
mse_prophet = mean_squared_error(gs['Close'], forecast['trend_upper'])
rmse_prophet = mse_prophet ** 0.5
print(f'Prophet Model - MSE: {mse_prophet}, RMSE: {rmse_prophet}')
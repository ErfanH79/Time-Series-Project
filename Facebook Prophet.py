

import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

# Download Apple stock data
ticker = "NVDA"
data = yf.download(ticker, period="5y")  # Last 5 years of data
df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']  # Rename for Prophet

# Initialize and fit Prophet model
model = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(df)

# Create future dates (1 year of business days)
future = model.make_future_dataframe(periods=252, freq='B')

# Generate forecasts
forecast = model.predict(future)

# Plot results
fig1 = model.plot(forecast)
plt.title(f"{ticker} Stock Price Forecast (Prophet)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()

# Interactive Plotly plots (uncomment to use)
# plot_plotly(model, forecast)
# plot_components_plotly(model, forecast)

# Show final forecasted values
print("\nForecasted Closing Prices:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Optional: Save forecast to CSV
# forecast.to_csv(f"{ticker}_prophet_forecast.csv", index=False)
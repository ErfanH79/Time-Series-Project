from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Download data
gs = yf.download("NVDA", start="2013-10-01", end="2024-09-30")



# Preprocess data
dataset_ex_df = gs.copy()
dataset_ex_df = dataset_ex_df.reset_index()
dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
dataset_ex_df.set_index('Date', inplace=True)

# Split data into train and test sets
X = dataset_ex_df.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(gs['Close'].values.reshape(-1, 1))


# Prepare the data for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 60
X_train, y_train = create_dataset(scaled_data[:len(train)], time_step)
X_test, y_test = create_dataset(scaled_data[len(train):], time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(100, return_sequences=False))
lstm_model.add(Dense(250))
lstm_model.add(Dense(1))


# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predicting and inverse scaling
train_predict = lstm_model.predict(X_train)
test_predict = lstm_model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(gs.index, gs['Close'], label='Actual Prices')
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.plot(gs.index, train_predict_plot, label='LSTM Training Prediction')
plt.plot(gs.index, test_predict_plot, label='LSTM Testing Prediction')
plt.title('LSTM Model Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# Evaluating the model
mse_lstm = mean_squared_error(gs['Close'][len(train_predict) + (time_step * 2) + 2:], test_predict)
rmse_lstm = np.sqrt(mse_lstm)
print(f'LSTM Model - MSE: {mse_lstm}, RMSE: {rmse_lstm}')
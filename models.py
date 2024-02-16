import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.preprocessing import MinMaxScaler

### BILSTM

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input



def LSTM(train, test, ):
    # Define parameters
    look_back = 30  # Number of past days to consider (?)
    # The "look back" refers to the number of time steps (or days, in the context of stock prices) 
    # that the model uses to make predictions.

    n_stocks = len(train.columns)

    forecast_horizon = len(test)

    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train)

    # Create sequences for input and output
    def create_sequences(data, look_back, forecast_horizon):
        X, y = [], []
        for i in range(len(data) - look_back - forecast_horizon):
            X.append(data[i:(i + look_back)])
            y.append(data[(i + look_back):(i + look_back + forecast_horizon)])
        return np.array(X), np.array(y)

    # Create input and output sequences
    X, y = create_sequences(scaled_data, look_back, forecast_horizon)


    # Define the model
    inputs = Input(shape=(look_back, n_stocks))
    lstm_out = LSTM(50)(inputs)
    ### add layers
    outputs = []
    for _ in range(n_stocks):
        outputs.append(Dense(forecast_horizon)(lstm_out))

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, [y[:, :, i] for i in range(n_stocks)], epochs=100, batch_size=32)

    # Predict the next period's stock prices iteratively
    scaled_test_data = scaler.transform(test_data[-look_back:, :])  # Use the last look_back days as initial input
    scaled_test_data = scaled_test_data.reshape((1, look_back, n_stocks))  # Reshape for LSTM input

    predictions = []
    for _ in range(num_predictions):
        pred = model.predict(scaled_test_data)
        predictions.append(pred)
        # Update scaled_test_data for the next prediction
        scaled_test_data = np.concatenate((scaled_test_data[:, forecast_horizon:, :], pred), axis=1)

    # Inverse transform the predictions to get actual prices
    predicted_prices = []
    for pred in predictions:
        pred_prices = np.zeros((forecast_horizon, n_stocks))
        for i in range(n_stocks):
            pred_prices[:, i] = scaler.inverse_transform(pred[i])
        predicted_prices.append(pred_prices)
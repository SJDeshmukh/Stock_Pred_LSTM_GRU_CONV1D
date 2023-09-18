# # Import necessary libraries and modules
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import yfinance as yf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

# # Define the stock symbol and time period
# stock_symbol = "GOOG"
# start_date = "2014-07-01"
# end_date = "2023-09-06"

# # Fetch historical data using yfinance
# data = yf.download(stock_symbol, start=start_date, end=end_date)

# # Remove NaN values
# data.dropna(inplace=True)

# # Create lagged features
# num_lags = 1  # Number of lagged values
# for i in range(1, num_lags + 1):
#     data[f"Lag_{i}"] = data["Close"].shift(i)

# # Remove rows with NaN values
# data.dropna(inplace=True)

# # Prepare data
# X = data.iloc[:, -num_lags:].values
# y = data["Close"].values

# # Normalize the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_scaled = scaler.fit_transform(X)
# y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# # Reshape the input data to have three dimensions (batch_size, num_lags, features)
# X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# # Define the input shape for each branch
# input_shape = (X_reshaped.shape[1], 1)  # Assuming one lagged feature per time step

# # Define the branches of the hybrid model
# lstm_input = Input(shape=input_shape)
# gru_input = Input(shape=input_shape)
# cnn_input = Input(shape=input_shape)

# lstm_branch = LSTM(64, activation='relu')(lstm_input)
# gru_branch = GRU(64, activation='relu')(gru_input)
# cnn_branch = Conv1D(64, kernel_size=3, padding='same', activation='relu')(cnn_input)

# # Modify MaxPooling1D to avoid dimension issues
# cnn_branch = MaxPooling1D(pool_size=1)(cnn_branch)  # Pool size adjusted
# cnn_branch = Flatten()(cnn_branch)

# # Merge the branches
# merged = concatenate([lstm_branch, gru_branch, cnn_branch])

# # Add fully connected layers
# merged = Dense(128, activation='relu')(merged)
# output_layer = Dense(1)(merged)  # Output layer for regression

# # Create the hybrid model
# model = Model(inputs=[lstm_input, gru_input, cnn_input], outputs=output_layer)

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')  # Use appropriate loss for your task

# # Train the model
# model.fit([X_reshaped, X_reshaped, X_reshaped], y_scaled, epochs=50, batch_size=64)  # Adjust epochs and batch_size as needed

# # Predict the entire dataset
# y_pred_scaled = model.predict([X_reshaped, X_reshaped, X_reshaped])
# y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

# # Calculate evaluation metrics on the entire dataset
# mae = mean_absolute_error(y, y_pred)
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# r2 = r2_score(y, y_pred)

# # Print evaluation results
# print(f"Mean Absolute Error (MAE): {mae:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
# print(f"R-squared (R2) Score: {r2:.4f}")
# mse = mean_squared_error(y, y_pred)
# print(f"Mean squared error (MSE) Score: {mse:.4f}")

# # Plot actual vs. predicted prices for the entire dataset
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, y, label="Actual Prices")
# plt.plot(data.index, y_pred, label="Predicted Prices", linestyle="--")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.title("Actual vs. Predicted Stock Prices (Entire Dataset)")
# plt.legend()
# plt.grid()
# plt.show()
# target_date_str = "2023-08-18"
# Import necessary libraries and modules
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Define the stock symbol and time period
stock_symbol = "GOOG"
start_date = "2014-07-01"
end_date = "2023-09-06"

# Fetch historical data using yfinance
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Remove NaN values
data.dropna(inplace=True)

# Create lagged features
num_lags = 1  # Number of lagged values
for i in range(1, num_lags + 1):
    data[f"Lag_{i}"] = data["Close"].shift(i)

# Remove rows with NaN values
data.dropna(inplace=True)

# Prepare data
X = data.iloc[:, -num_lags:].values
y = data["Close"].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Reshape the input data to have three dimensions (batch_size, num_lags, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Define the input shape for each branch
input_shape = (X_reshaped.shape[1], 1)  # Assuming one lagged feature per time step

# Define the branches of the hybrid model
lstm_input = Input(shape=input_shape)
gru_input = Input(shape=input_shape)
cnn_input = Input(shape=input_shape)

lstm_branch = LSTM(64, activation='relu')(lstm_input)
gru_branch = GRU(64, activation='relu')(gru_input)
cnn_branch = Conv1D(64, kernel_size=3, padding='same', activation='relu')(cnn_input)

# Modify MaxPooling1D to avoid dimension issues
cnn_branch = MaxPooling1D(pool_size=1)(cnn_branch)  # Pool size adjusted
cnn_branch = Flatten()(cnn_branch)

# Merge the branches
merged = concatenate([lstm_branch, gru_branch, cnn_branch])

# Add fully connected layers
merged = Dense(128, activation='relu')(merged)
output_layer = Dense(1)(merged)  # Output layer for regression

# Create the hybrid model
model = Model(inputs=[lstm_input, gru_input, cnn_input], outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Use appropriate loss for your task

# Train the model
model.fit([X_reshaped, X_reshaped, X_reshaped], y_scaled, epochs=200, batch_size=32)  # Adjust epochs and batch_size as needed

# Predict the entire dataset
y_pred_scaled = model.predict([X_reshaped, X_reshaped, X_reshaped])
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

# Calculate evaluation metrics on the entire dataset
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")
mse = mean_squared_error(y, y_pred)
print(f"Mean squared error (MSE) Score: {mse:.4f}")

# Plot actual vs. predicted prices for the entire dataset
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label="Actual Prices")
plt.plot(data.index, y_pred, label="Predicted Prices", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs. Predicted Stock Prices (Entire Dataset)")
plt.legend()
plt.grid()
plt.show()

# Continue with target date prediction or other analyses if needed
target_date_str = "2023-08-18"
target_date = pd.to_datetime(target_date_str)

# Convert target_date to the same timezone as your data's index
target_date = target_date.tz_localize(data.index.tz)

# Find the nearest available date in your dataset
nearest_date = None
for date in data.index:
    if date >= target_date:
        nearest_date = date
        break

if nearest_date is None:
    print("No available data for prediction on or after the specified date.")
else:
    # Find the index of the nearest date in your data
    target_index = data.index.get_loc(nearest_date)

    # Extract the lagged features for the target date
    target_X = X_scaled[target_index - num_lags + 1: target_index + 1, :]

    if target_X.shape[0] < num_lags:
        print("Not enough data for prediction on the specified date.")
    else:
        # Ensure that target_X has the same shape as X_train (number of lagged features)
        if target_X.shape[1] != X_reshaped.shape[1]:
            print(f"Number of lagged features in target_X ({target_X.shape[1]}) is different from X_reshaped ({X_reshaped.shape[1]}).")
        else:
            # Predict the stock price for the target date using your trained model
            target_X = target_X.reshape(1, num_lags, X_reshaped.shape[2])
            target_pred_scaled = model.predict([target_X, target_X, target_X])
            target_pred = scaler.inverse_transform(target_pred_scaled.reshape(-1, 1)).flatten()

            # Get the actual stock price for the target date
            target_actual = data["Close"].iloc[target_index]

            print(f"Date: {nearest_date}")
            print(f"Actual stock price: {target_actual:.2f}")
            print(f"Predicted stock price: {target_pred[0]:.2f}")
            
            print('=' * 50)

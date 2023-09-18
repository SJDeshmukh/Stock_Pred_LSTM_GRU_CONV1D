# Import necessary libraries and modules
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor for neural networks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator, RegressorMixin

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

# Split the data into training and testing sets
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Define a custom LSTM regressor
class LSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, y):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model.fit(X, y, epochs=200, batch_size=32)

    def predict(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X).flatten()

# Define a list of hybrid model pairs to test
hybrid_model_pairs = [
    #     ('XGBRegression + Linear', [
    #     ('lr', LinearRegression()),
    #     ('xgb', XGBRegressor(random_state=42))
    # ]),
    # # Feature Stacking + Support Vector Machines (SVM)
    # ('Feature Stacking + SVM', [
    #     ('lr', LinearRegression()),
    #     ('svr', SVR())
    # ]),

    # # Ensemble of Neural Networks + Gradient Boosting
    # ('Ensemble of Neural Networks + Gradient Boosting', [
    #     ('nn', MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)),  # Replace with your neural network model
    #     ('gb', GradientBoostingRegressor(random_state=42))
    # ]),
    

    # LSTM (Deep Learning) + Reinforcement Learning (RL) + Linear Regression (lr)
    ('LSTM + Reinforcement Learning + Linear Regression', [
        ('xgb', XGBRegressor(random_state=42)),
        ('lr', LinearRegression()),
        ('lstm', LSTMRegressor(input_shape=(X_train.shape[1], 1)))
    ])
]

# Iterate through hybrid model pairs
for model_name, models in hybrid_model_pairs:
    # Build the ensemble model without AgglomerativeClustering
    ensemble_model = StackingRegressor(estimators=models, final_estimator=LinearRegression())

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Predict the stock prices on the test set
    y_pred_scaled = ensemble_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Inverse transform the scaled actual prices
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Print model name
    print(f"Hybrid Model Pair: {model_name}")

    # Plot the actual and predicted stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten(), label="Actual Prices")
    plt.plot(data.index[train_size:], y_pred, label="Predicted Prices", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Actual vs. Predicted Stock Prices ({model_name})")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    # Print numerical results
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    mse = mean_squared_error(y_test, y_pred_scaled)
    print(f"Mean squared error (MSE) Score: {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

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
            if target_X.shape[1] != X_train.shape[1]:
                print(f"Number of lagged features in target_X ({target_X.shape[1]}) is different from X_train ({X_train.shape[1]}).")
            else:
                # Predict the stock price for the target date
                target_pred_scaled = ensemble_model.predict(target_X.reshape(1, num_lags))
                target_pred = scaler.inverse_transform(target_pred_scaled.reshape(-1, 1)).flatten()

                # Get the actual stock price for the target date
                target_actual = data["Close"].iloc[target_index]

                print(f"Date: {nearest_date}")
                print(f"Actual stock price: {target_actual:.2f}")
                print(f"Predicted stock price: {target_pred[0]:.2f}")
                
                print('=' * 50)
# train.py: Cryptocurrency Price Prediction
# This script implements a comprehensive pipeline for forecasting Bitcoin prices using CoinGecko API data,
# Prophet and LSTM models, and interactive Plotly visualizations, optimized for financial stakeholders.

# Importing core libraries for data acquisition, forecasting, and visualization
import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Setting up a professional visualization theme
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.family'] = 'Arial'

# --- 1. Data Acquisition ---
# Fetch Bitcoin price data from CoinGecko
cg = CoinGeckoAPI()
prices = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=365)
df = pd.DataFrame({
    'Date': [pd.to_datetime(price[0], unit='ms') for price in prices['prices']],
    'Close': [price[1] for price in prices['prices']]
})

# Display dataset profile
print("Dataset Profile:")
print(f"Records: {df.shape[0]}, Features: {df.shape[1]}")
print("\nSample Data:")
print(df.head())

# --- 2. Data Preprocessing ---
# Prepare data for Prophet
prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Prepare data for LSTM
def create_lstm_data(data, time_steps=10):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])
    return np.array(X), np.array(y), scaler

time_steps = 10
X, y, scaler = create_lstm_data(df, time_steps)
X_train, X_test, y_train, y_test = X[:-30], X[-30:], y[:-30], y[-30:]

# --- 3. Price Forecasting ---
# Prophet Model
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)

# LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict with LSTM
lstm_pred = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_scaled = scaler.inverse_transform(y_test)

# Evaluate models
prophet_pred = prophet_forecast.tail(30)['yhat'].values
print("\nProphet Performance:")
print(f"RMSE: {mean_squared_error(y_test_scaled, prophet_pred, squared=False):.2f}")
print(f"MAE: {mean_absolute_error(y_test_scaled, prophet_pred):.2f}")
print("\nLSTM Performance:")
print(f"RMSE: {mean_squared_error(y_test_scaled, lstm_pred, squared=False):.2f}")
print(f"MAE: {mean_absolute_error(y_test_scaled, lstm_pred):.2f}")

# --- 4. Visualizations ---
# Price Forecast
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Price'))
fig1.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Prophet Forecast'))
fig1.update_layout(title='Bitcoin Price Forecast vs. Actual', title_x=0.5, height=500,
                   xaxis_title='Date', yaxis_title='Price (USD)')
fig1.write_html('price_forecast.html')

# Forecast Error Distribution
errors = y_test_scaled.flatten() - lstm_pred.flatten()
fig2 = px.histogram(x=errors, nbins=30, title='LSTM Forecast Error Distribution',
                    labels={'x': 'Error (USD)', 'y': 'Count'})
fig2.update_layout(height=500, title_x=0.5)
fig2.write_html('error_distribution.html')

# --- 5. Financial Insights ---
print("\nFinancial Insights:")
print("1. Dataset: Daily Bitcoin prices from CoinGecko (1-year historical data).")
print("2. Forecast Accuracy: Prophet captures seasonal trends; LSTM models complex patterns.")
print("3. Applications: Inform trading strategies and investment decisions.")
print("4. Future Work: Incorporate market sentiment or trading volume for enhanced accuracy.")

# --- 6. Output Preservation ---
# Save processed data and models
df.to_csv('processed_bitcoin_data.csv', index=False)
lstm_model.save('lstm_bitcoin_model.keras')
print("\nOutputs saved: 'processed_bitcoin_data.csv', 'lstm_bitcoin_model.keras', 'price_forecast.html', 'error_distribution.html'.")
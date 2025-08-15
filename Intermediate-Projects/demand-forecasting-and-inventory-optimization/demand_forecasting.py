# demand_forecasting.py: End-to-End Demand Forecasting and Inventory Optimization
# This script implements a comprehensive pipeline for forecasting demand and optimizing inventory
# using the demand_inventory.csv dataset, Prophet and XGBoost models, interactive Plotly visualizations,
# and a Streamlit interface for real-time analysis, optimized for supply chain stakeholders.

# Importing core libraries for forecasting, optimization, visualization, and deployment
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from xgboost import XGBRegressor
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
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('demand_inventory.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# --- 2. Data Preprocessing ---
def preprocess_data(df):
    """Extract features for forecasting."""
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Lag1'] = df['Demand'].shift(1).fillna(df['Demand'].mean())
    df['RollingMean7'] = df['Demand'].rolling(window=7).mean().fillna(df['Demand'].mean())
    return df

# --- 3. Demand Forecasting ---
def train_prophet(df):
    """Train Prophet model for time-series forecasting."""
    prophet_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    return model

def train_xgboost(df):
    """Train XGBoost model for feature-based forecasting."""
    X = df[['Day', 'Month', 'Weekday', 'Lag1', 'RollingMean7']]
    y = df['Demand']
    X_train, X_test, y_train, y_test = X[:-30], X[-30:], y[:-30], y[-30:]
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# --- 4. Inventory Optimization ---
def calculate_inventory_metrics(avg_demand, std_demand, holding_cost=1.0, ordering_cost=50.0, lead_time=7):
    """Calculate EOQ, Reorder Point, and Safety Stock."""
    eoq = np.sqrt((2 * avg_demand * ordering_cost) / holding_cost)
    safety_stock = 1.96 * std_demand * np.sqrt(lead_time)
    reorder_point = avg_demand * lead_time + safety_stock
    total_cost = (ordering_cost * avg_demand / eoq) + (holding_cost * (eoq / 2 + safety_stock))
    return eoq, reorder_point, safety_stock, total_cost

# --- 5. Streamlit Interface ---
def main():
    st.title('Demand Forecasting and Inventory Optimization')
    st.markdown('Forecast future demand and optimize inventory levels for product P1 using historical data.')

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    # Display dataset profile
    st.subheader('Dataset Profile')
    st.write(f"Records: {df.shape[0]}, Features: {df.shape[1]}")
    st.write(df.head())

    # Historical demand and inventory
    st.subheader('Historical Demand and Inventory')
    fig1 = px.line(df, x='Date', y=['Demand', 'Inventory'], title='Historical Demand and Inventory Levels',
                   labels={'value': 'Value', 'Date': 'Date'})
    fig1.update_layout(height=500, title_x=0.5)
    st.plotly_chart(fig1)

    # Train models
    prophet_model = train_prophet(df)
    xgb_model, X_test, y_test = train_xgboost(df)

    # Forecast demand
    st.subheader('Demand Forecast')
    days_ahead = st.slider('Select Forecast Horizon (Days):', 1, 30, 7)
    future = prophet_model.make_future_dataframe(periods=days_ahead)
    forecast = prophet_model.predict(future)
    prophet_pred = forecast.tail(days_ahead)['yhat'].values

    # Evaluate Prophet
    prophet_rmse = mean_squared_error(y_test, prophet_pred[:len(y_test)], squared=False) if len(y_test) >= days_ahead else np.nan
    prophet_mae = mean_absolute_error(y_test, prophet_pred[:len(y_test)]) if len(y_test) >= days_ahead else np.nan

    # Evaluate XGBoost
    xgb_pred = xgb_model.predict(X_test[:days_ahead])
    xgb_rmse = mean_squared_error(y_test[:len(xgb_pred)], xgb_pred, squared=False)
    xgb_mae = mean_absolute_error(y_test[:len(xgb_pred)], xgb_pred)

    st.write(f"**Prophet Performance**: RMSE = {prophet_rmse:.2f}, MAE = {prophet_mae:.2f}")
    st.write(f"**XGBoost Performance**: RMSE = {xgb_rmse:.2f}, MAE = {xgb_mae:.2f}")

    # Visualize forecast
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Demand'], mode='lines', name='Actual Demand'))
    fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prophet Forecast'))
    fig2.update_layout(title=f'Demand Forecast for Next {days_ahead} Days', title_x=0.5, height=500,
                       xaxis_title='Date', yaxis_title='Demand')
    st.plotly_chart(fig2)

    # Inventory optimization
    st.subheader('Inventory Optimization')
    holding_cost = st.number_input('Holding Cost per Unit per Day:', min_value=0.1, value=1.0)
    ordering_cost = st.number_input('Ordering Cost per Order:', min_value=1.0, value=50.0)
    lead_time = st.number_input('Lead Time (Days):', min_value=1, value=7)
    avg_demand = forecast.tail(days_ahead)['yhat'].mean()
    std_demand = df['Demand'].std()

    eoq, reorder_point, safety_stock, total_cost = calculate_inventory_metrics(avg_demand, std_demand, holding_cost, ordering_cost, lead_time)
    st.write(f"**Optimal Order Quantity (EOQ)**: {eoq:.2f}")
    st.write(f"**Reorder Point**: {reorder_point:.2f}")
    st.write(f"**Safety Stock**: {safety_stock:.2f}")
    st.write(f"**Total Cost**: {total_cost:.2f}")

    # Supply chain insights
    st.subheader('Supply Chain Insights')
    st.write("1. Dataset: Daily demand and inventory for product P1 (June-August 2023).")
    st.write("2. Forecast Accuracy: Prophet and XGBoost provide reliable demand predictions.")
    st.write("3. Inventory Strategy: EOQ and safety stock ensure balanced inventory management.")
    st.write("4. Applications: Optimize supply chain and reduce costs in retail/manufacturing.")
    st.write("5. Future Work: Incorporate external factors (e.g., promotions, holidays).")

    # Save outputs
    df.to_csv('processed_demand_inventory.csv', index=False)
    xgb_model.save_model('xgb_demand_model.json')
    fig1.write_html('historical_demand_inventory.html')
    fig2.write_html('demand_forecast.html')
    st.write("\nOutputs saved: 'processed_demand_inventory.csv', 'xgb_demand_model.json', 'historical_demand_inventory.html', 'demand_forecast.html'.")

if __name__ == '__main__':
    main()
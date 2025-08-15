# app.py: Real-Time Demand Forecasting and Inventory Optimization with Streamlit
# This script deploys a demand forecasting and inventory optimization pipeline using a Streamlit interface,
# with interactive Plotly visualizations for supply chain analysis.

# Importing core libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from xgboost import XGBRegressor
import os

# --- 1. Model and Data Setup ---
# Load data and model
df = pd.read_csv('processed_demand_inventory.csv')
df['Date'] = pd.to_datetime(df['Date'])
xgb_model = XGBRegressor()
xgb_model.load_model('xgb_demand_model.json')

# --- 2. Streamlit Interface ---
st.title('Real-Time Demand Forecasting and Inventory Optimization')
st.markdown('View historical demand, forecast future demand, and optimize inventory levels.')

# Display historical data
st.subheader('Historical Demand and Inventory')
fig1 = px.line(df, x='Date', y=['Demand', 'Inventory'], title='Historical Demand and Inventory Levels',
               labels={'value': 'Value', 'Date': 'Date'})
fig1.update_layout(height=500, title_x=0.5)
st.plotly_chart(fig1)

# Forecast future demand
st.subheader('Demand Forecast')
days_ahead = st.slider('Select Forecast Horizon (Days):', 1, 30, 7)
prophet_df = df[['Date', 'Demand']].rename(columns={'Date': 'ds', 'Demand': 'y'})
prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=days_ahead)
forecast = prophet_model.predict(future)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['Date'], y=df['Demand'], mode='lines', name='Actual Demand'))
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
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

eoq = np.sqrt((2 * avg_demand * ordering_cost) / holding_cost)
safety_stock = 1.96 * std_demand * np.sqrt(lead_time)
reorder_point = avg_demand * lead_time + safety_stock
total_cost = (ordering_cost * avg_demand / eoq) + (holding_cost * (eoq / 2 + safety_stock))

st.write(f"**Optimal Order Quantity (EOQ)**: {eoq:.2f}")
st.write(f"**Reorder Point**: {reorder_point:.2f}")
st.write(f"**Safety Stock**: {safety_stock:.2f}")
st.write(f"**Total Cost**: {total_cost:.2f}")

# --- 3. Supply Chain Insights ---
st.subheader('Supply Chain Insights')
st.write("1. Forecast Accuracy: Prophet provides reliable demand predictions for the next 1-30 days.")
st.write("2. Inventory Strategy: EOQ and safety stock optimize stock levels while minimizing costs.")
st.write("3. Applications: Streamline supply chain operations in retail/manufacturing.")
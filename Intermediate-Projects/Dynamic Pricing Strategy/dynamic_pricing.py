import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pio.templates.default = "plotly_white"

def load_data(data_path):
    """
    Load the ride-sharing dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 
                          'Expected_Ride_Duration', 'Historical_Cost_of_Ride']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, outliers, and encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset with adjusted ride cost.
    """
    try:
        # Handle missing values
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
        df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

        # Handle outliers using IQR
        for feature in numeric_features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[feature] = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound),
                                   df[feature].mean(), df[feature])

        # Encode Vehicle_Type
        df['Vehicle_Type'] = df['Vehicle_Type'].map({'Premium': 1, 'Economy': 0})

        # Calculate demand-supply ratio
        df['Demand_Supply_Ratio'] = df['Number_of_Riders'] / df['Number_of_Drivers']

        # Define thresholds for price adjustment
        demand_threshold_high = 1.2
        demand_threshold_low = 0.8
        supply_threshold_high = 0.8
        supply_threshold_low = 1.2

        # Adjust prices based on demand-supply dynamics
        df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
            np.where(df['Demand_Supply_Ratio'] > demand_threshold_high, 1.2, 
                     np.where(df['Demand_Supply_Ratio'] < demand_threshold_low, 0.8, 1.0)) *
            np.where(df['Number_of_Drivers'] < supply_threshold_high * df['Number_of_Drivers'].mean(), 1.1, 
                     np.where(df['Number_of_Drivers'] > supply_threshold_low * df['Number_of_Drivers'].mean(), 0.9, 1.0))
        )

        logging.info("Data preprocessing completed")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        raise

def perform_eda(df, output_dir):
    """
    Perform exploratory data analysis and save visualizations.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Ride cost distribution
        fig = px.histogram(df, x='Historical_Cost_of_Ride', nbins=30, title='Distribution of Ride Costs')
        fig.update_xaxes(title_text='Historical Ride Cost')
        fig.update_yaxes(title_text='Frequency')
        fig.write(os.path.join(output_dir, 'ride_cost_distribution.png'))
        logging.info("Saved ride cost distribution plot")

        # Demand-supply ratio vs. ride cost
        fig = px.scatter(df, x='Demand_Supply_Ratio', y='Historical_Cost_of_Ride', color='Vehicle_Type',
                         title='Ride Cost vs. Demand-Supply Ratio')
        fig.update_xaxes(title_text='Demand-Supply Ratio')
        fig.update_yaxes(title_text='Historical Ride Cost')
        fig.write(os.path.join(output_dir, 'demand_supply_ratio.png'))
        logging.info("Saved demand-supply ratio plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def train_model(df):
    """
    Train a Random Forest Regressor to predict ride prices.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
    
    Returns:
        RandomForestRegressor: Trained model.
    """
    try:
        X = df[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']]
        y = df['adjusted_ride_cost'].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully")
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def predict_price(model, number_of_riders, number_of_drivers, vehicle_type, expected_ride_duration):
    """
    Predict ride price based on input features.
    
    Args:
        model: Trained RandomForestRegressor model.
        number_of_riders (int): Number of riders.
        number_of_drivers (int): Number of drivers.
        vehicle_type (str): Vehicle type ('Premium' or 'Economy').
        expected_ride_duration (int): Expected ride duration in minutes.
    
    Returns:
        float: Predicted ride price.
    """
    try:
        vehicle_type_numeric = {'Premium': 1, 'Economy': 0}.get(vehicle_type)
        if vehicle_type_numeric is None:
            raise ValueError(f"Invalid vehicle type: {vehicle_type}")
        
        input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, expected_ride_duration]])
        predicted_price = model.predict(input_data)[0]
        logging.info(f"Predicted price for {number_of_riders} riders, {number_of_drivers} drivers, "
                     f"{vehicle_type}, {expected_ride_duration} min: ${predicted_price:.2f}")
        return predicted_price
    except Exception as e:
        logging.error(f"Error predicting price: {str(e)}")
        raise

def visualize_predictions(model, X_test, y_test, output_dir):
    """
    Visualize actual vs. predicted ride prices.
    
    Args:
        model: Trained RandomForestRegressor model.
        X_test: Test features.
        y_test: Actual test prices.
        output_dir (str): Directory to save plot.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        y_pred = model.predict(X_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Actual vs Predicted'))
        fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                                 mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Actual vs Predicted Ride Prices',
                          xaxis_title='Actual Prices', yaxis_title='Predicted Prices')
        fig.write(os.path.join(output_dir, 'actual_vs_predicted.png'))
        logging.info("Saved actual vs predicted prices plot")
    except Exception as e:
        logging.error(f"Error visualizing predictions: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for dynamic pricing strategy.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load and preprocess data
        df = load_data(args.data_path)
        df = preprocess_data(df)

        if args.mode == 'analyze':
            # Perform EDA
            perform_eda(df, args.output_dir)
            print("🌟 Analysis Results:")
            print(f"- Average Ride Cost: ${df['Historical_Cost_of_Ride'].mean():.2f} "
                  f"(std: ${df['Historical_Cost_of_Ride'].std():.2f})")
            print(f"✅ High Demand Insights: Urban areas show {df[df['Location_Category'] == 'Urban']['Demand_Supply_Ratio'].mean():.2f} ratio")

        elif args.mode == 'predict':
            # Train model and predict price
            model, _, _ = train_model(df)
            predicted_price = predict_price(model, args.riders, args.drivers, args.vehicle_type, args.duration)
            print(f"📈 Predicted Price: ${predicted_price:.2f}")

        elif args.mode == 'visualize':
            # Perform EDA and visualize predictions
            perform_eda(df, args.output_dir)
            model, X_test, y_test = train_model(df)
            visualize_predictions(model, X_test, y_test, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic Pricing Strategy: Optimize revenue with Python")
    parser.add_argument('--mode', choices=['analyze', 'predict', 'visualize'], default='analyze',
                        help="Mode: analyze, predict, or visualize")
    parser.add_argument('--data_path', default='dynamic_pricing.csv', help="Path to the dataset")
    parser.add_argument('--riders', type=int, default=50, help="Number of riders for prediction")
    parser.add_argument('--drivers', type=int, default=25, help="Number of drivers for prediction")
    parser.add_argument('--vehicle_type', default='Economy', help="Vehicle type (Premium or Economy)")
    parser.add_argument('--duration', type=int, default=30, help="Expected ride duration in minutes")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
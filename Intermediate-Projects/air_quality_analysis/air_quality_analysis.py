import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pio.templates.default = "plotly_white"

def load_data(data_path):
    """
    Load the air quality dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset with datetime conversion.
    """
    try:
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['date', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def calculate_aqi(df):
    """
    Calculate AQI and categorize air quality for each row.
    
    Args:
        df (pd.DataFrame): Input dataset with pollutant columns.
    
    Returns:
        pd.DataFrame: Dataset with added 'AQI' and 'AQI Category' columns.
    """
    try:
        # Define AQI breakpoints (generic ranges for demonstration)
        aqi_breakpoints = [
            (0, 12.0, 50), (12.1, 35.4, 100), (35.5, 55.4, 150),
            (55.5, 150.4, 200), (150.5, 250.4, 300), (250.5, 350.4, 400),
            (350.5, 500.4, 500)
        ]
        aqi_categories = [
            (0, 50, 'Good'), (51, 100, 'Moderate'), (101, 150, 'Unhealthy for Sensitive Groups'),
            (151, 200, 'Unhealthy'), (201, 300, 'Very Unhealthy'), (301, 500, 'Hazardous')
        ]

        def calculate_aqi_value(pollutant_name, concentration):
            for low, high, aqi in aqi_breakpoints:
                if low <= concentration <= high:
                    return aqi
            return 500  # Default to max if out of range

        def calculate_overall_aqi(row):
            pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
            aqi_values = [calculate_aqi_value(pollutant, row[pollutant]) for pollutant in pollutants]
            return max(aqi_values)

        def categorize_aqi(aqi_value):
            for low, high, category in aqi_categories:
                if low <= aqi_value <= high:
                    return category
            return 'Hazardous'

        df['AQI'] = df.apply(calculate_overall_aqi, axis=1)
        df['AQI Category'] = df['AQI'].apply(categorize_aqi)
        logging.info("AQI calculation and categorization completed")
        return df
    except Exception as e:
        logging.error(f"Error calculating AQI: {str(e)}")
        raise

def visualize_data(df, output_dir):
    """
    Generate interactive visualizations for air quality analysis.
    
    Args:
        df (pd.DataFrame): Dataset with AQI and pollutant data.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

        # Time series of pollutants
        fig = go.Figure()
        for pollutant in pollutants:
            fig.add_trace(go.Scatter(x=df['date'], y=df[pollutant], mode='lines', name=pollutant))
        fig.update_layout(title='Time Series Analysis of Air Pollutants in Delhi',
                          xaxis_title='Date', yaxis_title='Concentration (µg/m³)')
        fig.write(os.path.join(output_dir, 'pollutant_time_series.png'))
        logging.info("Saved pollutant time series plot")

        # AQI over time
        fig = px.bar(df, x='date', y='AQI', title='AQI of Delhi in January')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='AQI')
        fig.write(os.path.join(output_dir, 'aqi_over_time.png'))
        logging.info("Saved AQI over time plot")

        # AQI category distribution
        fig = px.histogram(df, x='date', color='AQI Category', title='AQI Category Distribution Over Time')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Count')
        fig.write(os.path.join(output_dir, 'aqi_category_distribution.png'))
        logging.info("Saved AQI category distribution plot")

        # Pollutant concentrations (donut plot)
        total_concentrations = df[pollutants].sum()
        concentration_data = pd.DataFrame({"Pollutant": pollutants, "Concentration": total_concentrations})
        fig = px.pie(concentration_data, names='Pollutant', values='Concentration',
                     title='Pollutant Concentrations in Delhi', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(legend_title='Pollutant')
        fig.write(os.path.join(output_dir, 'pollutant_concentrations.png'))
        logging.info("Saved pollutant concentrations plot")

        # Correlation between pollutants
        correlation_matrix = df[pollutants].corr()
        fig = px.imshow(correlation_matrix, x=pollutants, y=pollutants, title='Correlation Between Pollutants')
        fig.write(os.path.join(output_dir, 'pollutant_correlation.png'))
        logging.info("Saved pollutant correlation plot")

        # Hourly AQI trends
        df['Hour'] = df['date'].dt.hour
        hourly_avg_aqi = df.groupby('Hour')['AQI'].mean().reset_index()
        fig = px.line(hourly_avg_aqi, x='Hour', y='AQI', title='Hourly Average AQI Trends in Delhi (Jan 2023)')
        fig.update_xaxes(title='Hour of the Day')
        fig.update_yaxes(title='Average AQI')
        fig.write(os.path.join(output_dir, 'hourly_aqi_trends.png'))
        logging.info("Saved hourly AQI trends plot")

        # Weekly AQI trends
        df['Day_of_Week'] = df['date'].dt.day_name()
        average_aqi_by_day = df.groupby('Day_of_Week')['AQI'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig = px.bar(average_aqi_by_day, x=average_aqi_by_day.index, y='AQI',
                     title='Average AQI by Day of the Week')
        fig.update_xaxes(title='Day of the Week')
        fig.update_yaxes(title='Average AQI')
        fig.write(os.path.join(output_dir, 'weekly_aqi_trends.png'))
        logging.info("Saved weekly AQI trends plot")

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for air quality analysis.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        # Calculate AQI and categories
        df = calculate_aqi(df)

        if args.mode == 'analyze':
            # Print summary metrics
            print("🌟 Analysis Results:")
            print(f"- Average AQI: {df['AQI'].mean():.1f} ({df['AQI Category'].mode()[0]})")
            print(f"- Dominant AQI Category: {df['AQI Category'].value_counts().idxmax()} "
                  f"({df['AQI Category'].value_counts().max()/len(df)*100:.1f}%)")
            corr_matrix = df[['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']].corr()
            max_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().idxmax()
            print(f"✅ Max Correlation: {max_corr[0]} and {max_corr[1]} ({corr_matrix.loc[max_corr].values[0]:.2f})")

        if args.mode == 'visualize':
            # Generate visualizations
            visualize_data(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Air Quality Index Analysis: Decode environmental health with Python")
    parser.add_argument('--mode', choices=['analyze', 'visualize'], default='analyze', help="Mode: analyze or visualize")
    parser.add_argument('--data_path', default='delhiaqi.csv', help="Path to the dataset")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
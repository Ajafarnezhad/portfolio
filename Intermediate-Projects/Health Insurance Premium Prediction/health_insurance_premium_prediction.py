import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.io as pio
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pio.templates.default = "plotly_white"

def load_data(data_path):
    """
    Load the health insurance dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset with processed features.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        
        # Encode categorical variables
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
        df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})
        
        if df.isnull().sum().any():
            logging.warning("Dataset contains null values; filling with mean for numeric columns")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def perform_eda(df, output_dir):
    """
    Perform exploratory data analysis and save visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Premium distribution
        fig = px.histogram(df, x='charges', nbins=30, title='Distribution of Insurance Premiums')
        fig.update_xaxes(title_text='Premium Amount ($)')
        fig.update_yaxes(title_text='Frequency')
        fig.write(os.path.join(output_dir, 'premium_distribution.png'))
        logging.info("Saved premium distribution plot")

        # Premiums vs. age, colored by smoker status
        fig = px.scatter(df, x='age', y='charges', color='smoker', 
                         title='Premiums vs. Age (Colored by Smoker Status)',
                         color_continuous_scale=['blue', 'red'])
        fig.update_xaxes(title_text='Age')
        fig.update_yaxes(title_text='Premium Amount ($)')
        fig.write(os.path.join(output_dir, 'premium_vs_age.png'))
        logging.info("Saved premiums vs. age plot")

        # Correlation heatmap
        correlation = df.corr()
        fig = px.imshow(correlation, text_auto=True, title='Correlation Heatmap of Features',
                        color_continuous_scale='Viridis')
        fig.write(os.path.join(output_dir, 'correlation_heatmap.png'))
        logging.info("Saved correlation heatmap plot")

        return correlation
    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def train_model(df):
    """
    Train a Random Forest Regressor for premium prediction.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        RandomForestRegressor: Trained model.
    """
    try:
        features = ['age', 'sex', 'bmi', 'smoker']
        X = df[features]
        y = df['charges'].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest Regressor trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def predict_premium(model):
    """
    Predict insurance premium based on user input.
    
    Args:
        model: Trained RandomForestRegressor model.
    
    Returns:
        float: Predicted premium amount.
    """
    try:
        print("Enter the following details for premium prediction:")
        features = [
            ("Age", int),
            ("Sex (0 for male, 1 for female)", int),
            ("BMI", float),
            ("Smoker (0 for no, 1 for yes)", int)
        ]
        input_data = []
        for name, dtype in features:
            value = input(f"{name}: ")
            input_data.append(dtype(value))
        
        input_data = np.array([input_data])
        predicted_premium = model.predict(input_data)[0]
        logging.info(f"Predicted premium: ${predicted_premium:.2f}")
        return predicted_premium
    except Exception as e:
        logging.error(f"Error predicting premium: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for health insurance premium prediction.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        if args.mode == 'analyze':
            # Perform EDA
            correlation = perform_eda(df, args.output_dir)
            print("🌟 Analysis Results:")
            print(f"- Average Premium: ${df['charges'].mean():.2f} (std: ${df['charges'].std():.2f})")
            print(f"✅ Key Insight: Smoking status has a strong correlation ({correlation['charges']['smoker']:.2f}) with premium amounts")

        elif args.mode == 'predict':
            # Train model and predict
            model = train_model(df)
            predicted_premium = predict_premium(model)
            print(f"📈 Predicted Premium: ${predicted_premium:.2f}")

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Health Insurance Premium Prediction: Forecast costs with Python")
    parser.add_argument('--mode', choices=['analyze', 'predict', 'visualize'], default='analyze',
                        help="Mode: analyze, predict, or visualize")
    parser.add_argument('--data_path', default='Health_insurance.csv', help="Path to the dataset")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
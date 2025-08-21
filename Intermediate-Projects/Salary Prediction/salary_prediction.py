import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    Load the salary dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['YearsExperience', 'Salary']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        if df.isnull().sum().any():
            logging.warning("Dataset contains null values; consider preprocessing")
            df = df.dropna()
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

        # Scatter plot with trendline
        fig = px.scatter(df, x='Salary', y='YearsExperience', size='YearsExperience',
                         trendline='ols', title='Salary vs. Years of Experience')
        fig.update_xaxes(title_text='Salary ($)')
        fig.update_yaxes(title_text='Years of Experience')
        fig.write(os.path.join(output_dir, 'salary_vs_experience.png'))
        logging.info("Saved salary vs. experience plot")

        # Calculate correlation
        correlation = df['YearsExperience'].corr(df['Salary'])
        return correlation
    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def train_model(df):
    """
    Train a Linear Regression model to predict salaries.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        LinearRegression: Trained model.
    """
    try:
        X = np.array(df[['YearsExperience']])
        y = np.array(df[['Salary']]).ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Linear Regression model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def predict_salary(model, experience):
    """
    Predict salary based on years of experience.
    
    Args:
        model: Trained LinearRegression model.
        experience (float): Years of experience.
    
    Returns:
        float: Predicted salary.
    """
    try:
        if experience < 0:
            raise ValueError("Years of experience cannot be negative")
        features = np.array([[experience]])
        predicted_salary = model.predict(features)[0]
        logging.info(f"Predicted salary for {experience} years of experience: ${predicted_salary:.2f}")
        return predicted_salary
    except Exception as e:
        logging.error(f"Error predicting salary: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for salary prediction.
    
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
            print(f"- Average Salary: ${df['Salary'].mean():.2f} (std: ${df['Salary'].std():.2f})")
            print(f"✅ Key Insight: Strong linear correlation ({correlation:.2f}) between years of experience and salary")

        elif args.mode == 'predict':
            # Train model and predict
            model = train_model(df)
            predicted_salary = predict_salary(model, args.experience)
            print(f"📈 Predicted Salary for {args.experience} years of experience: ${predicted_salary:.2f}")

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Salary Prediction: Forecast earnings with Python")
    parser.add_argument('--mode', choices=['analyze', 'predict', 'visualize'], default='analyze',
                        help="Mode: analyze, predict, or visualize")
    parser.add_argument('--data_path', default='Salary_Data.csv', help="Path to the dataset")
    parser.add_argument('--experience', type=float, default=2, help="Years of experience for prediction")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
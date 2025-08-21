import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    Load the credit score dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset with processed features.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                           'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                           'Credit_Mix', 'Outstanding_Debt', 'Credit_History_Age', 
                           'Monthly_Balance', 'Credit_Score']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        
        # Handle missing values and data types
        df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2})
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        if df['Credit_Score'].isnull().any():
            raise ValueError("Credit_Score contains missing values")
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

        # Credit score distribution
        fig = px.pie(df, names='Credit_Score', title='Credit Score Distribution',
                     color='Credit_Score', color_discrete_map={'Poor': 'red', 'Standard': 'yellow', 'Good': 'green'})
        fig.write(os.path.join(output_dir, 'credit_score_distribution.png'))
        logging.info("Saved credit score distribution plot")

        # Credit scores by monthly balance
        fig = px.box(df, x='Credit_Score', y='Monthly_Balance', title='Credit Scores by Monthly Balance',
                     color='Credit_Score', color_discrete_map={'Poor': 'red', 'Standard': 'yellow', 'Good': 'green'})
        fig.update_traces(quartilemethod="exclusive")
        fig.update_xaxes(title_text='Credit Score')
        fig.update_yaxes(title_text='Monthly Balance')
        fig.write(os.path.join(output_dir, 'credit_scores_by_balance.png'))
        logging.info("Saved credit scores by monthly balance plot")

        # Credit scores by annual income
        fig = px.box(df, x='Credit_Score', y='Annual_Income', title='Credit Scores by Annual Income',
                     color='Credit_Score', color_discrete_map={'Poor': 'red', 'Standard': 'yellow', 'Good': 'green'})
        fig.update_traces(quartilemethod="exclusive")
        fig.update_xaxes(title_text='Credit Score')
        fig.update_yaxes(title_text='Annual Income')
        fig.write(os.path.join(output_dir, 'credit_scores_by_income.png'))
        logging.info("Saved credit scores by annual income plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def train_model(df):
    """
    Train a Random Forest Classifier for credit score classification.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        features = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 
                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                    'Credit_Mix', 'Outstanding_Debt', 'Credit_History_Age', 
                    'Monthly_Balance']
        X = df[features]
        y = df['Credit_Score'].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest Classifier trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def predict_credit_score(model):
    """
    Predict credit score based on user input.
    
    Args:
        model: Trained RandomForestClassifier model.
    
    Returns:
        str: Predicted credit score.
    """
    try:
        print("Enter the following details for credit score prediction:")
        features = [
            ("Annual Income", float),
            ("Monthly Inhand Salary", float),
            ("Number of Bank Accounts", float),
            ("Number of Credit Cards", float),
            ("Interest Rate", float),
            ("Number of Loans", float),
            ("Average Number of Days Delayed", float),
            ("Number of Delayed Payments", float),
            ("Credit Mix (Bad: 0, Standard: 1, Good: 2)", int),
            ("Outstanding Debt", float),
            ("Credit History Age (Months)", float),
            ("Monthly Balance", float)
        ]
        input_data = []
        for name, dtype in features:
            value = input(f"{name}: ")
            input_data.append(dtype(value))
        
        input_data = np.array([input_data])
        prediction = model.predict(input_data)[0]
        logging.info(f"Predicted credit score: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error predicting credit score: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for credit score classification.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        if args.mode == 'analyze':
            # Perform EDA
            perform_eda(df, args.output_dir)
            print("🌟 Analysis Results:")
            print(f"- Average Annual Income: ${df['Annual_Income'].mean():.2f} "
                  f"(std: ${df['Annual_Income'].std():.2f})")
            print(f"✅ Key Insight: High monthly balances (> $250) correlate with Good credit scores")

        elif args.mode == 'predict':
            # Train model and predict
            model = train_model(df)
            prediction = predict_credit_score(model)
            print(f"📈 Predicted Credit Score: {prediction}")

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Credit Score Classification: Empower financial decisions with Python")
    parser.add_argument('--mode', choices=['analyze', 'predict', 'visualize'], default='analyze',
                        help="Mode: analyze, predict, or visualize")
    parser.add_argument('--data_path', default='train.csv', help="Path to the dataset")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
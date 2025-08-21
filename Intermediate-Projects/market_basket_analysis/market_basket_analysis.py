import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pio.templates.default = "plotly_white"

def load_data(data_path):
    """
    Load the market basket dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} transactions)")
        required_columns = ['BillNo', 'Itemname', 'Quantity', 'Price', 'CustomerID']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        if df.isnull().sum().any():
            logging.warning("Dataset contains null values; consider preprocessing")
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

        # Item distribution
        fig = px.histogram(df, x='Itemname', title='Item Distribution')
        fig.update_xaxes(title='Item Name')
        fig.update_yaxes(title='Count')
        fig.write(os.path.join(output_dir, 'item_distribution.png'))
        logging.info("Saved item distribution plot")

        # Top 10 most popular items
        item_popularity = df.groupby('Itemname')['Quantity'].sum().sort_values(ascending=False)
        top_n = 10
        fig = go.Figure()
        fig.add_trace(go.Bar(x=item_popularity.index[:top_n], y=item_popularity.values[:top_n],
                            text=item_popularity.values[:top_n], textposition='auto',
                            marker=dict(color='skyblue')))
        fig.update_layout(title=f'Top {top_n} Most Popular Items',
                          xaxis_title='Item Name', yaxis_title='Total Quantity Sold')
        fig.write(os.path.join(output_dir, 'top_items.png'))
        logging.info("Saved top items plot")

        # Customer behavior
        customer_behavior = df.groupby('CustomerID').agg({'Quantity': 'mean', 'Price': 'sum'}).reset_index()
        table_data = pd.DataFrame({
            'CustomerID': customer_behavior['CustomerID'],
            'Average Quantity': customer_behavior['Quantity'],
            'Total Spending': customer_behavior['Price']
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=customer_behavior['Quantity'], y=customer_behavior['Price'],
                                 mode='markers', text=customer_behavior['CustomerID'],
                                 marker=dict(size=10, color='coral')))
        fig.add_trace(go.Table(
            header=dict(values=['CustomerID', 'Average Quantity', 'Total Spending']),
            cells=dict(values=[table_data['CustomerID'], table_data['Average Quantity'], table_data['Total Spending']])
        ))
        fig.update_layout(title='Customer Behavior',
                          xaxis_title='Average Quantity', yaxis_title='Total Spending')
        fig.write(os.path.join(output_dir, 'customer_behavior.png'))
        logging.info("Saved customer behavior plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def generate_association_rules(df, min_support=0.01, min_confidence=0.1, min_lift=0.5):
    """
    Generate association rules using the Apriori algorithm.
    
    Args:
        df (pd.DataFrame): Input dataset with 'BillNo' and 'Itemname'.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
        min_lift (float): Minimum lift threshold.
    
    Returns:
        pd.DataFrame: Association rules with support, confidence, and lift.
    """
    try:
        # Group items by BillNo
        basket = df.groupby('BillNo')['Itemname'].apply(list).reset_index()
        basket_encoded = basket['Itemname'].str.join('|').str.get_dummies('|')

        # Apply Apriori algorithm
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            raise ValueError("No frequent itemsets found; try lowering min_support")

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
        rules = rules[rules['confidence'] >= min_confidence]
        logging.info(f"Generated {len(rules)} association rules")
        return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    except Exception as e:
        logging.error(f"Error generating association rules: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for market basket analysis.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        if args.mode == 'analyze':
            # Perform EDA
            perform_eda(df, args.output_dir)

            # Generate association rules
            rules = generate_association_rules(df, args.min_support, args.min_confidence, args.min_lift)
            print("🌟 Association Rules:")
            print(rules.head(10).to_string(index=False))

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Market Basket Analysis: Uncover shopping patterns with Python")
    parser.add_argument('--mode', choices=['analyze', 'visualize'], default='analyze', help="Mode: analyze or visualize")
    parser.add_argument('--data_path', default='market_basket_dataset.csv', help="Path to the dataset")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    parser.add_argument('--min_support', type=float, default=0.01, help="Minimum support for Apriori")
    parser.add_argument('--min_confidence', type=float, default=0.1, help="Minimum confidence for rules")
    parser.add_argument('--min_lift', type=float, default=0.5, help="Minimum lift for rules")
    args = parser.parse_args()

    main(args)

# water_potability.py
# Integrated script for water potability analysis and prediction
# Author: Ajafarnezhad (aiamirjd@gmail.com)
# Last Updated: August 15, 2025

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.classification import setup, compare_models, create_model, predict_model, save_model
from imblearn.over_sampling import SMOTE
import streamlit as st
import os
from datetime import datetime

# Configuration
DATA_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "water_potability_model.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, "processed_water_data.csv")
EDA_HTML_PATH = os.path.join(OUTPUT_DIR, "eda_plots.html")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Fetch and preprocess the water potability dataset."""
    try:
        df = pd.read_csv(DATA_URL)
        # Handle missing values with median imputation
        for column in ['ph', 'Sulfate', 'Trihalomethanes']:
            df[column].fillna(df[column].median(), inplace=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis and save interactive visualizations."""
    # Initialize subplot grid for feature distributions
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=df.columns[:-1],
        specs=[[{"type": "histogram"}]*3]*3
    )
    
    features = df.columns[:-1]
    for i, feature in enumerate(features):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature, marker_color='#1f77b4'),
            row=row, col=col
        )
    
    fig.update_layout(
        title_text="Distribution of Water Quality Features",
        height=900, width=1200, showlegend=False
    )
    fig.write_html(EDA_HTML_PATH)
    
    # Correlation heatmap
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Water Quality Features"
    )
    fig_corr.write_html(os.path.join(OUTPUT_DIR, "correlation_plot.html"))
    
    # Potability distribution
    potability_counts = df['Potability'].value_counts()
    fig_potability = px.pie(
        values=potability_counts.values,
        names=['Non-Potable (0)', 'Potable (1)'],
        title="Potability Distribution",
        color_discrete_sequence=['#ff7f0e', '#2ca02c']
    )
    fig_potability.write_html(os.path.join(OUTPUT_DIR, "potability_distribution.html"))

def train_model(df):
    """Train and evaluate a machine learning model using PyCaret."""
    # Apply SMOTE to handle class imbalance
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    smote = SMOTE(random_state=786)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)
    
    # Setup PyCaret
    clf = setup(
        data=df_balanced,
        target='Potability',
        session_id=786,
        silent=True,
        verbose=False
    )
    
    # Compare models and select the best
    best_model = compare_models(sort='F1', n_select=1)
    predictions = predict_model(best_model, data=df)
    
    # Save model
    save_model(best_model, MODEL_PATH)
    
    # Feature importance plot
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    fig_imp = px.bar(
        feature_imp, x='Importance', y='Feature',
        title="Feature Importance in Water Potability Prediction",
        color='Importance', color_continuous_scale='Viridis'
    )
    fig_imp.write_html(os.path.join(OUTPUT_DIR, "feature_importance.html"))
    
    return best_model, predictions

def streamlit_app():
    """Run the Streamlit app for real-time predictions."""
    st.set_page_config(page_title="Water Potability Prediction", layout="wide")
    st.title("💧 Water Potability Prediction Dashboard")
    st.markdown("""
    This app predicts whether a water sample is safe for drinking based on its chemical and physical properties.
    Enter the parameters below or upload a CSV file for batch predictions.
    """)
    
    # Load data and model
    df = load_data()
    if df is None:
        return
    
    try:
        model = load_model(MODEL_PATH)
    except:
        st.warning("Model not found. Training a new model...")
        perform_eda(df)
        model, _ = train_model(df)
    
    # Sidebar for user input
    st.sidebar.header("Input Water Quality Parameters")
    inputs = {}
    feature_ranges = {
        'ph': (0, 14, 7.0),
        'Hardness': (0, 500, 200.0),
        'Solids': (0, 50000, 10000.0),
        'Chloramines': (0, 15, 7.0),
        'Sulfate': (0, 1000, 300.0),
        'Conductivity': (0, 1000, 400.0),
        'Organic_carbon': (0, 50, 10.0),
        'Trihalomethanes': (0, 200, 80.0),
        'Turbidity': (0, 10, 4.0)
    }
    
    for feature, (min_val, max_val, default) in feature_ranges.items():
        inputs[feature] = st.sidebar.slider(
            f"{feature}", float(min_val), float(max_val), float(default)
        )
    
    # Single prediction
    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([inputs])
        prediction = predict_model(model, data=input_df)
        label = prediction['Label'][0]
        score = prediction['Score'][0]
        result = "Potable (Safe)" if label == 1 else "Non-Potable (Unsafe)"
        st.subheader("Prediction Result")
        st.write(f"**Potability**: {result} (Confidence: {score:.2%})")
    
    # Batch prediction via file upload
    st.sidebar.header("Batch Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        for column in ['ph', 'Sulfate', 'Trihalomethanes']:
            if column in batch_df.columns:
                batch_df[column].fillna(batch_df[column].median(), inplace=True)
        predictions = predict_model(model, data=batch_df)
        st.subheader("Batch Prediction Results")
        st.dataframe(predictions[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                 'Conductivity', 'Organic_carbon', 'Trihalomethanes',
                                 'Turbidity', 'Label']].rename(columns={'Label': 'Potability'}))
        st.download_button(
            label="Download Predictions",
            data=predictions.to_csv(index=False),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Display EDA visualizations
    st.header("Exploratory Data Analysis")
    st.markdown("View interactive visualizations of the dataset:")
    for plot in ['eda_plots.html', 'correlation_plot.html', 'potability_distribution.html', 'feature_importance.html']:
        plot_path = os.path.join(OUTPUT_DIR, plot)
        if os.path.exists(plot_path):
            with open(plot_path, 'r') as f:
                st.components.v1.html(f.read(), height=600)

def main():
    """Main function to execute the pipeline."""
    df = load_data()
    if df is not None:
        perform_eda(df)
        model, predictions = train_model(df)
        print("Model training completed. Predictions saved in outputs/predictions.csv")
        predictions.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    
    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    main()

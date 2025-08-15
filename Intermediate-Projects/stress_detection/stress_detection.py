
# stress_detection.py
# Integrated script for stress detection from social media posts
# Author: Ajafarnezhad (aiamirjd@gmail.com)
# Last Updated: August 15, 2025

import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import plotly.express as px
from wordcloud import WordCloud
from pycaret.classification import setup, compare_models, create_model, predict_model, save_model, load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Configuration
DATA_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "stress_detection_model.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, "processed_stress_data.csv")
WORDCLOUD_PATH = os.path.join(OUTPUT_DIR, "wordcloud.png")
EDA_HTML_PATH = os.path.join(OUTPUT_DIR, "eda_plots.html")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text cleaning function
stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def clean_text(text):
    """Clean text by removing URLs, punctuation, numbers, and stopwords, and applying stemming."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

def load_data():
    """Fetch and preprocess the stress dataset."""
    try:
        df = pd.read_csv(DATA_URL)
        df['text'] = df['text'].apply(clean_text)
        df['label'] = df['label'].map({0: "No Stress", 1: "Stress"})
        df = df[['text', 'label']]
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis and save visualizations."""
    # Word cloud
    text = " ".join(df['text'])
    wordcloud = WordCloud(
        stopwords=set(stopwords.words('english')),
        background_color="white",
        width=800, height=400
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(WORDCLOUD_PATH, bbox_inches='tight')
    plt.close()

    # Label distribution
    label_counts = df['label'].value_counts()
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Stress vs. No Stress Distribution",
        color_discrete_sequence=['#ff7f0e', '#2ca02c']
    )
    fig.write_html(EDA_HTML_PATH)

def train_model(df):
    """Train and evaluate a machine learning model using PyCaret."""
    # Vectorize text
    cv = CountVectorizer()
    X = cv.fit_transform(df['text'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    # Convert to DataFrame for PyCaret
    train_df = pd.DataFrame(X_train.toarray(), columns=cv.get_feature_names_out())
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test.toarray(), columns=cv.get_feature_names_out())
    test_df['label'] = y_test.values
    
    # Setup PyCaret
    clf = setup(
        data=train_df,
        target='label',
        session_id=786,
        silent=True,
        verbose=False,
        text_features=cv.get_feature_names_out().tolist()
    )
    
    # Compare models and select the best
    best_model = compare_models(sort='F1', n_select=1)
    predictions = predict_model(best_model, data=test_df)
    
    # Save model and vectorizer
    save_model(best_model, MODEL_PATH)
    with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), 'wb') as f:
        import pickle
        pickle.dump(cv, f)
    
    return best_model, predictions, cv

def streamlit_app():
    """Run the Streamlit app for real-time stress detection."""
    st.set_page_config(page_title="Stress Detection Dashboard", layout="wide")
    st.title("🧠 Stress Detection from Social Media Posts")
    st.markdown("""
    This app detects stress in social media posts using machine learning.
    Enter a text or upload a CSV file with a 'text' column for batch predictions.
    """)
    
    # Load data and model
    df = load_data()
    if df is None:
        return
    
    try:
        model = load_model(MODEL_PATH)
        with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), 'rb') as f:
            import pickle
            cv = pickle.load(f)
    except:
        st.warning("Model or vectorizer not found. Training a new model...")
        perform_eda(df)
        model, _, cv = train_model(df)
    
    # Sidebar for user input
    st.sidebar.header("Input Text for Stress Detection")
    user_text = st.sidebar.text_area("Enter a social media post:", height=150)
    
    # Single prediction
    if st.sidebar.button("Predict"):
        if user_text:
            cleaned_text = clean_text(user_text)
            text_vector = cv.transform([cleaned_text]).toarray()
            text_df = pd.DataFrame(text_vector, columns=cv.get_feature_names_out())
            prediction = predict_model(model, data=text_df)
            label = prediction['Label'][0]
            score = prediction['Score'][0]
            st.subheader("Prediction Result")
            st.write(f"**Stress Level**: {label} (Confidence: {score:.2%})")
        else:
            st.error("Please enter a text to predict.")
    
    # Batch prediction
    st.sidebar.header("Batch Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'text' column", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if 'text' not in batch_df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            batch_df['text'] = batch_df['text'].apply(clean_text)
            text_vectors = cv.transform(batch_df['text']).toarray()
            batch_df_vectors = pd.DataFrame(text_vectors, columns=cv.get_feature_names_out())
            predictions = predict_model(model, data=batch_df_vectors)
            batch_df['Prediction'] = predictions['Label']
            batch_df['Confidence'] = predictions['Score']
            st.subheader("Batch Prediction Results")
            st.dataframe(batch_df[['text', 'Prediction', 'Confidence']])
            st.download_button(
                label="Download Predictions",
                data=batch_df.to_csv(index=False),
                file_name=f"stress_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Display EDA visualizations
    st.header("Exploratory Data Analysis")
    st.markdown("View insights from the dataset:")
    if os.path.exists(WORDCLOUD_PATH):
        st.image(WORDCLOUD_PATH, caption="Word Cloud of Social Media Posts")
    if os.path.exists(EDA_HTML_PATH):
        with open(EDA_HTML_PATH, 'r') as f:
            st.components.v1.html(f.read(), height=500)

def main():
    """Main function to execute the pipeline."""
    df = load_data()
    if df is not None:
        perform_eda(df)
        model, predictions, cv = train_model(df)
        print("Model training completed. Predictions saved in outputs/stress_predictions.csv")
        predictions.to_csv(os.path.join(OUTPUT_DIR, "stress_predictions.csv"), index=False)
    
    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    main()

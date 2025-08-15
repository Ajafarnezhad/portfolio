
# spam_detection.py
# Integrated script for spam detection in messages using machine learning
# Author: Ajafarnezhad (aiamirjd@gmail.com)
# Last Updated: August 15, 2025

import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Configuration
DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "spam_detection_model.pkl")
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, "vectorizer.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, "processed_spam_data.csv")
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
    """Fetch and preprocess the spam dataset."""
    try:
        df = pd.read_csv(DATA_URL, encoding='latin-1')
        df = df[['v1', 'v2']].rename(columns={'v1': 'class', 'v2': 'message'})
        df['class'] = df['class'].map({'ham': 'No Spam', 'spam': 'Spam'})
        df['message'] = df['message'].apply(clean_text)
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis and save visualizations."""
    # Label distribution
    label_counts = df['class'].value_counts()
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Spam vs. No Spam Distribution",
        color_discrete_sequence=['#ff7f0e', '#2ca02c']
    )
    fig.write_html(EDA_HTML_PATH)

    # Word cloud for spam messages
    spam_text = " ".join(df[df['class'] == 'Spam']['message'])
    wordcloud = WordCloud(
        stopwords=set(stopwords.words('english')),
        background_color="white",
        width=800, height=400
    ).generate(spam_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(WORDCLOUD_PATH, bbox_inches='tight')
    plt.close()

def train_model(df):
    """Train and evaluate a machine learning model using PyCaret."""
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['message'])
    y = df['class']

    # Convert to DataFrame for PyCaret
    X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    X_df['class'] = y.values

    # Setup PyCaret
    clf = setup(
        data=X_df,
        target='class',
        session_id=786,
        silent=True,
        verbose=False
    )

    # Compare models and select the best
    best_model = compare_models(sort='F1', n_select=1)
    predictions = predict_model(best_model, data=X_df)

    # Save model and vectorizer
    save_model(best_model, MODEL_PATH)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    return best_model, predictions, vectorizer

def streamlit_app():
    """Run the Streamlit app for real-time spam detection."""
    st.set_page_config(page_title="Spam Detection Dashboard", layout="wide")
    st.title("📧 Spam Detection Dashboard")
    st.markdown("""
    This app detects spam in messages or emails using machine learning.
    Enter a message or upload a CSV file with a 'message' column for batch predictions.
    """)

    # Load data and model
    df = load_data()
    if df is None:
        return

    try:
        model = load_model(MODEL_PATH)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
    except:
        st.warning("Model or vectorizer not found. Training a new model...")
        perform_eda(df)
        model, _, vectorizer = train_model(df)

    # Sidebar for user input
    st.sidebar.header("Input Message for Spam Detection")
    user_message = st.sidebar.text_area("Enter a message:", height=150)

    # Single prediction
    if st.sidebar.button("Predict"):
        if user_message:
            cleaned_message = clean_text(user_message)
            message_vector = vectorizer.transform([cleaned_message]).toarray()
            message_df = pd.DataFrame(message_vector, columns=vectorizer.get_feature_names_out())
            prediction = predict_model(model, data=message_df)
            label = prediction['Label'][0]
            score = prediction['Score'][0]
            st.subheader("Prediction Result")
            st.write(f"**Status**: {label} (Confidence: {score:.2%})")
        else:
            st.error("Please enter a message to predict.")

    # Batch prediction
    st.sidebar.header("Batch Prediction")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'message' column", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if 'message' not in batch_df.columns:
            st.error("CSV must contain a 'message' column.")
        else:
            batch_df['message'] = batch_df['message'].apply(clean_text)
            message_vectors = vectorizer.transform(batch_df['message']).toarray()
            batch_df_vectors = pd.DataFrame(message_vectors, columns=vectorizer.get_feature_names_out())
            predictions = predict_model(model, data=batch_df_vectors)
            batch_df['Prediction'] = predictions['Label']
            batch_df['Confidence'] = predictions['Score']
            st.subheader("Batch Prediction Results")
            st.dataframe(batch_df[['message', 'Prediction', 'Confidence']])
            st.download_button(
                label="Download Predictions",
                data=batch_df.to_csv(index=False),
                file_name=f"spam_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Display EDA visualizations
    st.header("Exploratory Data Analysis")
    st.markdown("View insights from the dataset:")
    if os.path.exists(WORDCLOUD_PATH):
        st.image(WORDCLOUD_PATH, caption="Word Cloud of Spam Messages")
    if os.path.exists(EDA_HTML_PATH):
        with open(EDA_HTML_PATH, 'r') as f:
            st.components.v1.html(f.read(), height=500)

def main():
    """Main function to execute the pipeline."""
    df = load_data()
    if df is not None:
        perform_eda(df)
        model, predictions, vectorizer = train_model(df)
        print("Spam detection model training completed. Predictions saved in outputs/spam_predictions.csv")
        predictions.to_csv(os.path.join(OUTPUT_DIR, "spam_predictions.csv"), index=False)

    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    main()

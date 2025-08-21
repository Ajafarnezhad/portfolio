
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle

# Function to load and preprocess dataset
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the fake news dataset.
    Args:
        file_path (str): Path to the CSV dataset.
    Returns:
        tuple: DataFrame, feature column (titles), and target column (labels).
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty or invalid.")

    # Check for required columns
    if 'title' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'title' and 'label' columns.")

    # Handle missing values
    df['title'] = df['title'].fillna('')
    df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})  # Convert labels to binary

    if df['label'].isnull().any():
        raise ValueError("Invalid or missing labels in dataset.")

    return df, df['title'], df['label']

# Function to vectorize text data
def vectorize_text(X_train, X_test, max_features=5000):
    """
    Convert text data to TF-IDF features.
    Args:
        X_train (Series): Training text data.
        X_test (Series): Testing text data.
        max_features (int): Maximum number of features for TF-IDF.
    Returns:
        tuple: Vectorized training and testing data, and the vectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Function to train the model
def train_model(X_train_tfidf, y_train):
    """
    Train a Multinomial Naive Bayes model.
    Args:
        X_train_tfidf (sparse matrix): Vectorized training data.
        y_train (Series): Training labels.
    Returns:
        MultinomialNB: Trained model.
    """
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the model and print performance metrics.
    Args:
        model: Trained model.
        X_test_tfidf (sparse matrix): Vectorized test data.
        y_test (Series): Test labels.
    Returns:
        float: Accuracy score.
    """
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
    return accuracy

# Function to predict on new data
def predict_news(model, vectorizer, title):
    """
    Predict whether a news title is fake or real.
    Args:
        model: Trained model.
        vectorizer: Fitted TF-IDF vectorizer.
        title (str): News title to predict.
    Returns:
        str: Prediction ('REAL' or 'FAKE').
    """
    title_tfidf = vectorizer.transform([title])
    prediction = model.predict(title_tfidf)[0]
    return 'REAL' if prediction == 1 else 'FAKE'

# Main function to run the fake news detection pipeline
def fake_news_detection(file_path, output_dir='output'):
    """
    Run the full fake news detection pipeline.
    Args:
        file_path (str): Path to the dataset.
        output_dir (str): Directory to save model and vectorizer.
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and preprocess data
    df, X, y = load_and_preprocess_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)

    # Train model
    model = train_model(X_train_tfidf, y_train)

    # Evaluate model
    evaluate_model(model, X_test_tfidf, y_test)

    # Save model and vectorizer
    with open(os.path.join(output_dir, 'fake_news_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Example predictions
    real_news = "Scientists discover new species in Pacific Ocean"
    fake_news = "Aliens invade New York City with laser beams"
    print(f"\nPrediction for real news: '{real_news}' -> {predict_news(model, vectorizer, real_news)}")
    print(f"Prediction for fake news: '{fake_news}' -> {predict_news(model, vectorizer, fake_news)}")

# Example usage
# fake_news_detection('fake_news_data.csv', 'output')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse
import logging
import os
import sys
from typing import Tuple
import kaggle
import nltk
import re
import string

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PlayStoreSentimentAnalyzer:
    """A class to manage sentiment analysis of Google Play Store reviews."""
    
    def __init__(self, data_path: str = None):
        """Initialize the analyzer, downloading dataset if not provided."""
        self.data_path = data_path
        self.data = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = MultinomialNB()
        self.analyzer = SentimentIntensityAnalyzer()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.load_data()

    def load_data(self) -> None:
        """Download or load the Google Play Store reviews dataset."""
        try:
            if self.data_path:
                self.data = pd.read_csv(self.data_path)
                logger.info(f"🌟 Loaded dataset from {self.data_path} ({len(self.data)} records)")
            else:
                # Download dataset from Kaggle
                kaggle.api.dataset_download_files(
                    'lava18/google-play-store-apps',
                    path='data/',
                    unzip=True
                )
                self.data = pd.read_csv('data/googleplaystore_user_reviews.csv')
                logger.info(f"🌟 Downloaded Google Play Store reviews dataset ({len(self.data)} records)")
            
            # Validate dataset
            required_columns = ['App', 'Translated_Review']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            
            # Drop rows with missing reviews
            self.data = self.data.dropna(subset=['Translated_Review'])
            logger.info(f"✅ Cleaned dataset, retained {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocess text: remove URLs, mentions, hashtags, punctuation, and apply tokenization/stemming."""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            # Stemming
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def preprocess_data(self) -> None:
        """Preprocess reviews and assign initial sentiment labels using VADER."""
        try:
            self.data['Processed_Review'] = self.data['Translated_Review'].apply(self.preprocess_text)
            self.data['VADER_Sentiment'] = self.data['Translated_Review'].apply(
                lambda x: 'positive' if self.analyzer.polarity_scores(x)['compound'] >= 0.05 else
                          'negative' if self.analyzer.polarity_scores(x)['compound'] <= -0.05 else 'neutral'
            )
            logger.info("✅ Preprocessed reviews and assigned VADER sentiment labels")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train_model(self, test_size: float = 0.2) -> None:
        """Train a Naive Bayes model on preprocessed reviews."""
        try:
            X = self.vectorizer.fit_transform(self.data['Processed_Review'])
            y = self.data['VADER_Sentiment']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            self.model.fit(self.X_train, self.y_train)
            logger.info("🧠 Trained Naive Bayes model")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self) -> dict:
        """Evaluate the Naive Bayes model on test data."""
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            logger.info(f"📈 Model Accuracy: {accuracy:.2%}")
            logger.info(f"✅ Classification Report:\n{classification_report(self.y_test, y_pred)}")
            return {'accuracy': accuracy, 'report': report}
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def predict_sentiment(self, review: str) -> Tuple[str, float]:
        """Predict sentiment for a single review."""
        try:
            processed_review = self.preprocess_text(review)
            X = self.vectorizer.transform([processed_review])
            sentiment = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0][np.argmax(self.model.classes_ == sentiment)]
            logger.info(f"📈 Predicted sentiment for review: {sentiment} (Probability: {prob:.2%})")
            return sentiment, prob
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def analyze_data(self) -> None:
        """Perform exploratory data analysis."""
        try:
            sentiment_counts = self.data['VADER_Sentiment'].value_counts()
            logger.info(f"🔍 Sentiment Distribution:\n{sentiment_counts}")
            logger.info(f"✅ Dataset Summary:\n{self.data.describe()}")
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise

    def visualize_data(self, output_dir: str = './plots') -> None:
        """Generate visualizations: sentiment distribution and word cloud."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Sentiment Distribution
            sentiment_counts = self.data['VADER_Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig = px.bar(
                sentiment_counts,
                x='Sentiment',
                y='Count',
                title="Sentiment Distribution of Google Play Store Reviews",
                color='Sentiment',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'blue'}
            )
            fig.update_layout(width=800, height=600)
            sentiment_path = os.path.join(output_dir, 'sentiment_distribution.html')
            fig.write_html(sentiment_path)
            logger.info(f"📊 Saved sentiment distribution to {sentiment_path}")

            # Word Cloud
            all_text = " ".join(self.data['Processed_Review'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            wordcloud_path = os.path.join(output_dir, 'wordcloud_reviews.png')
            plt.savefig(wordcloud_path, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 Saved word cloud to {wordcloud_path}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to handle CLI arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Google Play Store Sentiment Analysis System")
    parser.add_argument('--mode', choices=['analyze', 'train', 'predict', 'visualize'], default='analyze',
                        help="Mode of operation: analyze, train, predict, or visualize")
    parser.add_argument('--data_path', default=None,
                        help="Path to the dataset (optional; downloads Kaggle dataset if not provided)")
    parser.add_argument('--review', type=str, help="Review text for prediction (required for predict mode)")
    parser.add_argument('--output_dir', default='./plots',
                        help="Directory to save visualizations")
    
    args = parser.parse_args()

    try:
        analyzer = PlayStoreSentimentAnalyzer(args.data_path)
        analyzer.preprocess_data()

        if args.mode == 'analyze':
            analyzer.analyze_data()
        elif args.mode == 'train':
            analyzer.train_model()
            analyzer.evaluate_model()
        elif args.mode == 'predict':
            if not args.review:
                logger.error("Please provide --review for prediction")
                sys.exit(1)
            analyzer.train_model()
            sentiment, prob = analyzer.predict_sentiment(args.review)
            print(f"Prediction: {sentiment} (Probability: {prob:.2%})")
        elif args.mode == 'visualize':
            analyzer.train_model()
            analyzer.visualize_data(args.output_dir)
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
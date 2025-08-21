import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import argparse
import os
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    """
    Load the app reviews dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} reviews)")
        if 'Review' not in df.columns or 'Rating' not in df.columns:
            raise ValueError("Dataset must contain 'Review' and 'Rating' columns")
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

        # Rating distribution
        plt.figure(figsize=(9, 5))
        sns.countplot(data=df, x='Rating')
        plt.title('Distribution of Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
        plt.close()
        logging.info("Saved rating distribution plot")

        # Review length distribution
        df['Review Length'] = df['Review'].apply(len)
        plt.figure(figsize=(9, 6))
        sns.histplot(df['Review Length'], bins=50, kde=True)
        plt.title('Distribution of Review Lengths')
        plt.xlabel('Length of Review')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'review_length_distribution.png'))
        plt.close()
        logging.info("Saved review length distribution plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def label_sentiments(df):
    """
    Label reviews with sentiments using TextBlob.
    
    Args:
        df (pd.DataFrame): Input dataset with 'Review' column.
    
    Returns:
        pd.DataFrame: Dataset with added 'Sentiment' column.
    """
    try:
        def textblob_sentiment_analysis(review):
            sentiment = TextBlob(str(review)).sentiment
            if sentiment.polarity > 0.1:
                return 'Positive'
            elif sentiment.polarity < -0.1:
                return 'Negative'
            else:
                return 'Neutral'

        df['Sentiment'] = df['Review'].apply(textblob_sentiment_analysis)
        logging.info("Sentiment labeling completed")
        return df
    except Exception as e:
        logging.error(f"Error in sentiment labeling: {str(e)}")
        raise

def analyze_sentiments(df, output_dir):
    """
    Analyze sentiment distribution and relationships, and generate word clouds.
    
    Args:
        df (pd.DataFrame): Dataset with 'Sentiment' column.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Sentiment distribution
        sentiment_distribution = df['Sentiment'].value_counts()
        plt.figure(figsize=(9, 5))
        sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
        plt.title('Distribution of Sentiments')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
        plt.close()
        logging.info(f"Sentiment Distribution: {sentiment_distribution.to_dict()}")

        # Sentiment vs Rating
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='Rating', hue='Sentiment')
        plt.title('Sentiment Distribution Across Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.savefig(os.path.join(output_dir, 'sentiment_vs_rating.png'))
        plt.close()
        logging.info("Saved sentiment vs rating plot")

        # Word clouds for each sentiment
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            text = ' '.join(review for review in df[df['Sentiment'] == sentiment]['Review'])
            if text.strip():  # Ensure text is not empty
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Word Cloud for {sentiment} Reviews')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'wordcloud_{sentiment.lower()}.png'))
                plt.close()
                logging.info(f"Saved word cloud for {sentiment} reviews")
            else:
                logging.warning(f"No text available for {sentiment} word cloud")

    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for EDA or sentiment analysis.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        if args.mode == 'explore':
            # Perform exploratory data analysis
            perform_eda(df, args.output_dir)
        elif args.mode == 'analyze':
            # Perform sentiment analysis
            df = label_sentiments(df)
            analyze_sentiments(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="App Reviews Sentiment Analysis: Decode user feedback with NLP")
    parser.add_argument('--mode', choices=['explore', 'analyze'], default='explore', help="Mode: explore or analyze")
    parser.add_argument('--data_path', default='linkedin_reviews.csv', help="Path to the dataset")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
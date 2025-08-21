import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    Load the news dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['ID', 'News Category', 'Title', 'Summary']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        if df['Title'].isnull().any():
            logging.warning("Dataset contains null values in 'Title'; filling with empty strings")
            df['Title'] = df['Title'].fillna('')
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

        # News category distribution
        categories = df['News Category'].value_counts()
        fig = px.bar(x=categories.index, y=categories.values, 
                     labels={'x': 'News Category', 'y': 'Number of Articles'},
                     title='Distribution of News Categories')
        fig.update_xaxes(title_text='News Category')
        fig.update_yaxes(title_text='Number of Articles')
        fig.write(os.path.join(output_dir, 'news_categories.png'))
        logging.info("Saved news categories plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def build_recommendation_model(df):
    """
    Build a content-based recommendation model using TF-IDF and cosine similarity.
    
    Args:
        df (pd.DataFrame): Input dataset with 'Title' column.
    
    Returns:
        tuple: TF-IDF matrix, cosine similarity matrix, and indices mapping.
    """
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Title'])
        similarity = cosine_similarity(tfidf_matrix)
        indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
        logging.info("Recommendation model built successfully")
        return tfidf_matrix, similarity, indices
    except Exception as e:
        logging.error(f"Error building recommendation model: {str(e)}")
        raise

def recommend_news(news_title, df, similarity, indices, top_n=10):
    """
    Recommend news articles based on cosine similarity of titles.
    
    Args:
        news_title (str): News article title to base recommendations on.
        df (pd.DataFrame): Dataset with news information.
        similarity (array): Cosine similarity matrix.
        indices (pd.Series): Mapping of titles to indices.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.Series: Recommended article titles.
    """
    try:
        if news_title not in indices:
            raise ValueError(f"News title '{news_title}' not found in dataset")
        
        index = indices[news_title]
        similarity_scores = list(enumerate(similarity[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[:top_n]
        news_indices = [i[0] for i in similarity_scores]
        recommendations = df['Title'].iloc[news_indices]
        logging.info(f"Generated {len(recommendations)} recommendations for '{news_title}'")
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for news recommendation system.
    
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
            top_category = df['News Category'].value_counts().idxmax()
            print(f"- Top Category: {top_category} ({df['News Category'].value_counts().max()/len(df)*100:.1f}% of articles)")
            print(f"✅ Key Insight: {top_category} category dominates readership")

        elif args.mode == 'recommend':
            # Build model and recommend news
            _, similarity, indices = build_recommendation_model(df)
            recommendations = recommend_news(args.news_title, df, similarity, indices, args.top_n)
            print(f"📈 Recommendations for '{args.news_title}':")
            for i, title in enumerate(recommendations, 1):
                print(f"{i}. {title}")

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="News Recommendation System: Personalize news with Python")
    parser.add_argument('--mode', choices=['analyze', 'recommend', 'visualize'], default='analyze',
                        help="Mode: analyze, recommend, or visualize")
    parser.add_argument('--data_path', default='News.csv', help="Path to the dataset")
    parser.add_argument('--news_title', default='Walmart Slashes Prices on Last-Generation iPads',
                        help="News article title for recommendations")
    parser.add_argument('--top_n', type=int, default=10, help="Number of recommendations to return")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
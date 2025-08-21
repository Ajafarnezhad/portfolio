import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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
    Load the books dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset with processed ratings.
    """
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded dataset from {data_path} ({len(df)} books)")
        required_columns = ['bookID', 'title', 'authors', 'average_rating']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        
        # Convert average_rating to numeric
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
        if df['average_rating'].isnull().any():
            logging.warning("Some ratings could not be converted to numeric; filling with mean")
            df['average_rating'].fillna(df['average_rating'].mean(), inplace=True)
        
        # Create book_content column
        df['book_content'] = df['title'] + ' ' + df['authors']
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
        fig = px.histogram(df, x='average_rating', nbins=30, title='Distribution of Average Ratings')
        fig.update_xaxes(title_text='Average Rating')
        fig.update_yaxes(title_text='Frequency')
        fig.write(os.path.join(output_dir, 'rating_distribution.png'))
        logging.info("Saved rating distribution plot")

        # Top 10 authors by book count
        top_authors = df['authors'].value_counts().head(10)
        fig = px.bar(x=top_authors.values, y=top_authors.index, orientation='h',
                     labels={'x': 'Number of Books', 'y': 'Author'},
                     title='Number of Books per Author')
        fig.write(os.path.join(output_dir, 'top_authors.png'))
        logging.info("Saved top authors plot")

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def build_recommendation_model(df):
    """
    Build a content-based recommendation model using TF-IDF and cosine similarity.
    
    Args:
        df (pd.DataFrame): Input dataset with 'book_content' column.
    
    Returns:
        tuple: TF-IDF matrix and cosine similarity matrix.
    """
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['book_content'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        logging.info("Recommendation model built successfully")
        return tfidf_matrix, cosine_sim
    except Exception as e:
        logging.error(f"Error building recommendation model: {str(e)}")
        raise

def recommend_books(book_title, df, cosine_sim, top_n=10):
    """
    Recommend books based on cosine similarity.
    
    Args:
        book_title (str): Title of the book to base recommendations on.
        df (pd.DataFrame): Dataset with book information.
        cosine_sim (array): Cosine similarity matrix.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.Series: Titles of recommended books.
    """
    try:
        if book_title not in df['title'].values:
            raise ValueError(f"Book title '{book_title}' not found in dataset")
        
        idx = df[df['title'] == book_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude the input book
        book_indices = [i[0] for i in sim_scores]
        recommendations = df['title'].iloc[book_indices]
        logging.info(f"Generated {len(recommendations)} recommendations for '{book_title}'")
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for book recommendation system.
    
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
            print(f"- Average Rating: {df['average_rating'].mean():.2f} (std: {df['average_rating'].std():.2f})")
            top_author = df['authors'].value_counts().idxmax()
            print(f"✅ Top Author: {top_author} ({df['authors'].value_counts().max()} books)")

        elif args.mode == 'recommend':
            # Build recommendation model
            _, cosine_sim = build_recommendation_model(df)
            recommendations = recommend_books(args.book_title, df, cosine_sim, args.top_n)
            print(f"📚 Recommendations for '{args.book_title}':")
            for i, title in enumerate(recommendations, 1):
                print(f"{i}. {title}")

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Book Recommendation System: Personalize reading with Python")
    parser.add_argument('--mode', choices=['analyze', 'recommend', 'visualize'], default='analyze', 
                        help="Mode: analyze, recommend, or visualize")
    parser.add_argument('--data_path', default='books_data.csv', help="Path to the dataset")
    parser.add_argument('--book_title', default='Dubliners: Text  Criticism  and Notes', 
                        help="Book title for recommendations")
    parser.add_argument('--top_n', type=int, default=10, help="Number of recommendations to return")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
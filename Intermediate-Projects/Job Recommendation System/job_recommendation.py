import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import re
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(nltk.corpus.stopwords.words('english'))

def load_data(data_path):
    """
    Load the job postings dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path)
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['Job Salary', 'Job Experience Required', 'Key Skills', 
                          'Role Category', 'Functional Area', 'Industry', 'Job Title']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        if df.isnull().sum().any():
            logging.warning("Dataset contains null values; consider preprocessing")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def generate_wordcloud(text, title, output_path):
    """
    Generate and save a word cloud for the given text.
    
    Args:
        text (str): Text to generate word cloud from.
        title (str): Title for the word cloud.
        output_path (str): Path to save the word cloud image.
    """
    try:
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=20, pad=20)
        plt.axis("off")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved word cloud: {output_path}")
    except Exception as e:
        logging.error(f"Error generating word cloud: {str(e)}")
        raise

def perform_eda(df, output_dir):
    """
    Perform exploratory data analysis and save word cloud visualizations.
    
    Args:
        df (pd.DataFrame): Input dataset.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Word cloud for Key Skills
        skills_text = " ".join(df['Key Skills'])
        generate_wordcloud(skills_text, "Key Skills Word Cloud", 
                         os.path.join(output_dir, 'skills_wordcloud.png'))

        # Word cloud for Functional Area
        functional_text = " ".join(df['Functional Area'])
        generate_wordcloud(functional_text, "Functional Area Word Cloud", 
                         os.path.join(output_dir, 'functional_area_wordcloud.png'))

        # Word cloud for Job Title
        job_title_text = " ".join(df['Job Title'])
        generate_wordcloud(job_title_text, "Job Title Word Cloud", 
                         os.path.join(output_dir, 'job_title_wordcloud.png'))

    except Exception as e:
        logging.error(f"Error in EDA: {str(e)}")
        raise

def build_recommendation_model(df):
    """
    Build a content-based recommendation model using TF-IDF and cosine similarity.
    
    Args:
        df (pd.DataFrame): Input dataset with 'Key Skills' column.
    
    Returns:
        tuple: TF-IDF matrix, cosine similarity matrix, and indices mapping.
    """
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Key Skills'])
        similarity = cosine_similarity(tfidf_matrix)
        indices = pd.Series(df.index, index=df['Job Title']).drop_duplicates()
        logging.info("Recommendation model built successfully")
        return tfidf_matrix, similarity, indices
    except Exception as e:
        logging.error(f"Error building recommendation model: {str(e)}")
        raise

def recommend_jobs(job_title, df, similarity, indices, top_n=5):
    """
    Recommend jobs based on cosine similarity of skills.
    
    Args:
        job_title (str): Job title to base recommendations on.
        df (pd.DataFrame): Dataset with job information.
        similarity (array): Cosine similarity matrix.
        indices (pd.Series): Mapping of job titles to indices.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: Recommended jobs with title, experience, and skills.
    """
    try:
        if job_title not in indices:
            raise ValueError(f"Job title '{job_title}' not found in dataset")
        
        index = indices[job_title]
        similarity_scores = list(enumerate(similarity[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[:top_n]
        job_indices = [i[0] for i in similarity_scores]
        recommendations = df[['Job Title', 'Job Experience Required', 'Key Skills']].iloc[job_indices]
        logging.info(f"Generated {len(recommendations)} recommendations for '{job_title}'")
        return recommendations
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for job recommendation system.
    
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
            top_skill = df['Key Skills'].str.split('|').explode().mode()[0]
            print(f"- Most Common Skill: {top_skill} (mentioned in {df['Key Skills'].str.contains(top_skill, case=False).sum()/len(df)*100:.1f}% of jobs)")
            top_industry = df['Industry'].value_counts().idxmax()
            print(f"✅ Key Insight: {top_industry} dominates with {df['Industry'].value_counts().max()/len(df)*100:.1f}% of postings")

        elif args.mode == 'recommend':
            # Build model and recommend jobs
            _, similarity, indices = build_recommendation_model(df)
            recommendations = recommend_jobs(args.job_title, df, similarity, indices, args.top_n)
            print(f"📈 Recommendations for '{args.job_title}':")
            print(recommendations.to_string(index=False))

        elif args.mode == 'visualize':
            # Generate visualizations only
            perform_eda(df, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Job Recommendation System: Match talent with opportunity using Python")
    parser.add_argument('--mode', choices=['analyze', 'recommend', 'visualize'], default='analyze',
                        help="Mode: analyze, recommend, or visualize")
    parser.add_argument('--data_path', default='jobs.csv', help="Path to the dataset")
    parser.add_argument('--job_title', default='Software Developer', help="Job title for recommendations")
    parser.add_argument('--top_n', type=int, default=5, help="Number of recommendations to return")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
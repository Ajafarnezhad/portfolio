import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import argparse
import logging
import os
from typing import List, Tuple
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ArticleRecommender:
    """A class to manage article recommendation system using content-based filtering."""
    
    def __init__(self, data_path: str):
        """Initialize the recommender with dataset path."""
        self.data_path = data_path
        self.data = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.load_data()

    def load_data(self) -> None:
        """Load and validate the dataset."""
        try:
            self.data = pd.read_csv(self.data_path, encoding='latin1')
            required_columns = ['Title', 'Article']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            logger.info(f"🌟 Loaded article dataset ({len(self.data)} records)")
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        """Preprocess article text and compute TF-IDF and similarity matrices."""
        try:
            articles = self.data['Article'].tolist()
            self.tfidf_matrix = self.tfidf.fit_transform(articles)
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
            logger.info("✅ Computed TF-IDF and cosine similarity matrices")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def recommend_articles(self, index: int, top_n: int = 4) -> str:
        """Generate recommendations for a given article index."""
        try:
            if self.similarity_matrix is None:
                raise ValueError("Similarity matrix not computed. Run preprocess_data first.")
            sim_scores = self.similarity_matrix[index]
            top_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
            recommendations = self.data['Title'].iloc[top_indices].tolist()
            return ", ".join(recommendations)
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    def generate_all_recommendations(self) -> pd.DataFrame:
        """Generate recommendations for all articles."""
        try:
            self.data['Recommended Articles'] = [
                self.recommend_articles(i) for i in range(len(self.data))
            ]
            logger.info("📈 Generated recommendations for all articles")
            return self.data
        except Exception as e:
            logger.error(f"Error generating all recommendations: {str(e)}")
            raise

    def analyze_data(self) -> None:
        """Perform exploratory data analysis."""
        try:
            avg_similarity = np.mean(self.similarity_matrix[np.triu_indices(len(self.data), k=1)])
            logger.info(f"🔍 Average cosine similarity: {avg_similarity:.2f} (std: {np.std(self.similarity_matrix):.2f})")
            if 'Category' in self.data.columns:
                category_counts = self.data['Category'].value_counts()
                logger.info(f"✅ Category distribution:\n{category_counts}")
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise

    def visualize_data(self, output_dir: str = './plots') -> None:
        """Generate visualizations: similarity heatmap and word cloud."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Similarity Heatmap
            fig = px.imshow(
                self.similarity_matrix,
                labels=dict(x="Article Index", y="Article Index", color="Cosine Similarity"),
                title="Article Similarity Heatmap"
            )
            fig.write_layout(width=800, height=800)
            fig.write_xaxes(title="Article Index")
            fig.write_yaxes(title="Article Index")
            fig.write_coloraxes(showscale=True)
            fig.write()
            heatmap_path = os.path.join(output_dir, 'similarity_heatmap.html')
            fig.write_html(heatmap_path)
            logger.info(f"📊 Saved similarity heatmap to {heatmap_path}")

            # Word Cloud
            all_text = " ".join(self.data['Article'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            wordcloud_path = os.path.join(output_dir, 'wordcloud_articles.png')
            plt.savefig(wordcloud_path, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 Saved word cloud to {wordcloud_path}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to handle CLI arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Article Recommendation System")
    parser.add_argument('--mode', choices=['analyze', 'recommend', 'visualize'], default='analyze',
                        help="Mode of operation: analyze, recommend, or visualize")
    parser.add_argument('--data_path', default='articles.csv',
                        help="Path to the article dataset")
    parser.add_argument('--output_dir', default='./plots',
                        help="Directory to save visualizations")
    
    args = parser.parse_args()

    try:
        recommender = ArticleRecommender(args.data_path)
        recommender.preprocess_data()

        if args.mode == 'analyze':
            recommender.analyze_data()
        elif args.mode == 'recommend':
            results = recommender.generate_all_recommendations()
            print(results[['Title', 'Recommended Articles']].head())
        elif args.mode == 'visualize':
            recommender.visualize_data(args.output_dir)
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
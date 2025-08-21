import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import argparse
import logging
import string

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_data(data_path):
    """
    Load the articles dataset from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(data_path, encoding='latin1')
        logging.info(f"Loaded dataset from {data_path} ({len(df)} records)")
        required_columns = ['Article', 'Title']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain {required_columns}")
        if df['Article'].isnull().any():
            logging.warning("Dataset contains null values in 'Article'; filling with empty strings")
            df['Article'] = df['Article'].fillna('')
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing punctuation, stopwords, and lemmatizing.
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Preprocessed text.
    """
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize
        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(word) for word in tokens]
        # Join tokens
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        return ''

def perform_topic_modelling(df, n_topics=5):
    """
    Perform topic modelling using Latent Dirichlet Allocation (LDA).
    
    Args:
        df (pd.DataFrame): Input dataset with 'Article' column.
        n_topics (int): Number of topics to identify.
    
    Returns:
        pd.DataFrame: Dataset with topic labels.
        LatentDirichletAllocation: Trained LDA model.
        TfidfVectorizer: Fitted TF-IDF vectorizer.
    """
    try:
        # Preprocess text
        df['Article'] = df['Article'].apply(preprocess_text)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(df['Article'])
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
        topic_modelling = lda.transform(X)
        topic_labels = np.argmax(topic_modelling, axis=1)
        df['topic_labels'] = topic_labels
        
        logging.info(f"Assigned {n_topics} topics to {len(df)} articles")
        return df, lda, vectorizer
    except Exception as e:
        logging.error(f"Error in topic modelling: {str(e)}")
        raise

def visualize_topics(df, lda, vectorizer, output_dir, n_topics=5):
    """
    Visualize topic distribution and word clouds for each topic.
    
    Args:
        df (pd.DataFrame): Dataset with topic labels.
        lda (LatentDirichletAllocation): Trained LDA model.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        output_dir (str): Directory to save plots.
        n_topics (int): Number of topics.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Topic distribution
        topic_counts = df['topic_labels'].value_counts().sort_index()
        fig = px.pie(names=[f'Topic {i}' for i in topic_counts.index], 
                     values=topic_counts.values,
                     title='Topic Distribution Across Articles',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.write(os.path.join(output_dir, 'topic_distribution.png'))
        logging.info("Saved topic distribution plot")

        # Word clouds for each topic
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx in range(n_topics):
            top_words_idx = lda.components_[topic_idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            wordcloud_text = ' '.join(top_words)
            wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color='white', width=800, height=400).generate(wordcloud_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Top Words in Topic {topic_idx}', fontsize=16)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'topic_wordcloud_{topic_idx}.png'), bbox_inches='tight')
            plt.close()
            logging.info(f"Saved word cloud for Topic {topic_idx}")

    except Exception as e:
        logging.error(f"Error visualizing topics: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for topic modelling.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load dataset
        df = load_data(args.data_path)

        if args.mode == 'analyze':
            # Perform topic modelling
            df, lda, vectorizer = perform_topic_modelling(df, args.n_topics)
            print("🌟 Topic Modelling Results:")
            print(df[['Title', 'topic_labels']].head().to_string(index=False))
            print(f"🔍 Topics Assigned: {args.n_topics} unique topics identified")
            topic_counts = df['topic_labels'].value_counts()
            dominant_topic = topic_counts.idxmax()
            print(f"✅ Key Insight: Topic {dominant_topic} dominates with {topic_counts[dominant_topic]/len(df)*100:.1f}% of articles")

        elif args.mode == 'visualize':
            # Perform topic modelling and visualize
            df, lda, vectorizer = perform_topic_modelling(df, args.n_topics)
            visualize_topics(df, lda, vectorizer, args.output_dir, args.n_topics)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Topic Modelling: Uncover hidden themes with Python")
    parser.add_argument('--mode', choices=['analyze', 'visualize'], default='analyze',
                        help="Mode: analyze or visualize")
    parser.add_argument('--data_path', default='articles.csv', help="Path to the dataset")
    parser.add_argument('--n_topics', type=int, default=5, help="Number of topics for LDA")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
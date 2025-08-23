import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import logging
import argparse
import os
from typing import Tuple
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndianFoodAnalyzer:
    """Class for EDA and classification of Indian food dataset."""
    
    def __init__(self, data_path: str, output_dir: str = './plots'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> None:
        """Load and preprocess the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            # Handle missing values
            self.df = self.df.replace('-1', np.nan)
            self.df['flavor_profile'] = self.df['flavor_profile'].fillna('unknown')
            self.df['state'] = self.df['state'].fillna('Unknown')
            self.df['region'] = self.df['region'].fillna('Unknown')
            logging.info(f"Loaded Indian Food dataset ({len(self.df)} records)")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def perform_eda(self) -> None:
        """Perform exploratory data analysis."""
        try:
            if self.df is None:
                self.load_data()
            
            # Log distributions
            logging.info("Course Distribution:\n" + str(self.df['course'].value_counts()))
            logging.info("Flavor Profile Summary:\n" + str(self.df['flavor_profile'].value_counts()))
            logging.info("Diet Distribution:\n" + str(self.df['diet'].value_counts()))
            logging.info("State Distribution:\n" + str(self.df['state'].value_counts()))
            logging.info("Region Distribution:\n" + str(self.df['region'].value_counts()))

            # Save static plots (optional, for local use)
            for column in ['course', 'flavor_profile', 'diet', 'state', 'region']:
                fig = px.histogram(self.df, y=column, title=f'{column.capitalize()} Distribution')
                fig.write_html(f"{self.output_dir}/{column}_distribution.html")
            
        except Exception as e:
            logging.error(f"Error in EDA: {str(e)}")
            raise

    def train_classifier(self) -> None:
        """Train a Random Forest classifier to predict flavor profiles."""
        try:
            if self.df is None:
                self.load_data()
            
            # Prepare features
            tfidf = TfidfVectorizer(max_features=100)
            X_text = tfidf.fit_transform(self.df['ingredients'].fillna('')).toarray()
            X_diet = pd.get_dummies(self.df['diet'], prefix='diet').values
            X_state = pd.get_dummies(self.df['state'], prefix='state').values
            X = np.hstack([X_text, X_diet, X_state])
            y = self.df['flavor_profile']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred)
            logging.info(f"Random Forest Classifier Accuracy: {clf.score(X_test, y_test)*100:.2f}%")
            logging.info(f"Classification Report:\n{report}")
            
        except Exception as e:
            logging.error(f"Error in training classifier: {str(e)}")
            raise

    def generate_report(self) -> None:
        """Generate an HTML report with Recharts visualizations."""
        try:
            if self.df is None:
                self.load_data()
            
            # Generate JSON data for HTML report
            course_counts = self.df['course'].value_counts().to_dict()
            flavor_counts = self.df['flavor_profile'].value_counts().to_dict()
            diet_counts = self.df['diet'].value_counts().to_dict()
            state_counts = self.df['state'].value_counts().to_dict()
            region_counts = self.df['region'].value_counts().to_dict()
            
            data = {
                'course': [{'name': k, 'value': v} for k, v in course_counts.items()],
                'flavor': [{'name': k, 'value': v} for k, v in flavor_counts.items()],
                'diet': [{'name': k, 'value': v} for k, v in diet_counts.items()],
                'state': [{'name': k, 'value': v} for k, v in state_counts.items()],
                'region': [{'name': k, 'value': v} for k, v in region_counts.items()]
            }
            
            with open(f"{self.output_dir}/data.json", 'w') as f:
                import json
                json.dump(data, f)
            
            logging.info(f"Data JSON saved to {self.output_dir}/data.json")
            
        except Exception as e:
            logging.error(f"Error generating report data: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Indian Cuisine Insights")
    parser.add_argument('--mode', choices=['analyze', 'train', 'visualize'], default='analyze')
    parser.add_argument('--data_path', default='food_menu_data.csv')
    parser.add_argument('--output_dir', default='./plots')
    args = parser.parse_args()

    analyzer = IndianFoodAnalyzer(args.data_path, args.output_dir)
    
    if args.mode == 'analyze':
        analyzer.perform_eda()
    elif args.mode == 'train':
        analyzer.train_classifier()
    elif args.mode == 'visualize':
        analyzer.generate_report()

if __name__ == "__main__":
    main()
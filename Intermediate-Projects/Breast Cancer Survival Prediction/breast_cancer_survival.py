import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import argparse
import logging
import os
import sys
from typing import Tuple
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('survival_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BreastCancerSurvivalPredictor:
    """A class to manage breast cancer survival prediction using machine learning."""
    
    def __init__(self, data_path: str = None):
        """Initialize the predictor, downloading dataset if not provided."""
        self.data_path = data_path
        self.data = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.load_data()

    def load_data(self) -> None:
        """Download or load the Haberman's Survival Dataset."""
        try:
            if self.data_path:
                self.data = pd.read_csv(self.data_path)
                logger.info(f"🌟 Loaded dataset from {self.data_path} ({len(self.data)} records)")
            else:
                # Download Haberman's Survival Dataset from UCI
                haberman = fetch_ucirepo(id=43)
                self.data = pd.DataFrame(
                    data=haberman.data.features,
                    columns=['age', 'operation_year', 'positive_axillary_nodes']
                )
                self.data['survival_status'] = haberman.data.targets['survival_status']
                logger.info(f"🌟 Downloaded Haberman's Survival Dataset ({len(self.data)} records)")
            
            # Validate dataset
            required_columns = ['age', 'operation_year', 'positive_axillary_nodes', 'survival_status']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            
            # Map survival_status (1: survived >=5 years, 2: died <5 years) to binary (1: survived, 0: died)
            self.data['survival_status'] = self.data['survival_status'].map({1: 1, 2: 0})
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def preprocess_data(self, test_size: float = 0.2) -> None:
        """Preprocess data and split into train/test sets."""
        try:
            X = self.data[['age', 'operation_year', 'positive_axillary_nodes']]
            y = self.data['survival_status']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("✅ Split data into training and test sets")
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train_model(self) -> None:
        """Train the Random Forest model."""
        try:
            self.model.fit(self.X_train, self.y_train)
            logger.info("🧠 Trained Random Forest model")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self) -> dict:
        """Evaluate the model on test data."""
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

    def predict_survival(self, age: int, operation_year: int, nodes: int) -> Tuple[float, str]:
        """Predict survival probability for a single patient."""
        try:
            input_data = np.array([[age, operation_year, nodes]])
            prob = self.model.predict_proba(input_data)[0][1]  # Probability of survival (class 1)
            prediction = "Survived ≥5 years" if prob >= 0.5 else "Died <5 years"
            logger.info(f"📈 Predicted survival for input (age={age}, year={operation_year}, nodes={nodes}): {prediction} (Probability: {prob:.2%})")
            return prob, prediction
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def analyze_data(self) -> None:
        """Perform exploratory data analysis."""
        try:
            logger.info(f"🔍 Dataset Summary:\n{self.data.describe()}")
            survival_rate = self.data['survival_status'].mean()
            logger.info(f"✅ Survival Rate (>=5 years): {survival_rate:.2%}")
            correlation = self.data.corr()
            logger.info(f"✅ Correlation Matrix:\n{correlation}")
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise

    def visualize_data(self, output_dir: str = './plots') -> None:
        """Generate visualizations: survival distribution and feature importance."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Survival Distribution
            fig = px.histogram(
                self.data, x='age', color='survival_status',
                title="Age Distribution by Survival Status",
                labels={'survival_status': 'Survival Status', 'age': 'Age'},
                category_orders={'survival_status': [1, 0]},
                color_discrete_map={1: 'green', 0: 'red'}
            )
            fig.update_layout(bargap=0.1, width=800, height=600)
            survival_path = os.path.join(output_dir, 'survival_distribution.html')
            fig.write_html(survival_path)
            logger.info(f"📊 Saved survival distribution to {survival_path}")

            # Feature Importance
            importance = self.model.feature_importances_
            feature_names = ['Age', 'Operation Year', 'Positive Axillary Nodes']
            fig = go.Figure(data=[
                go.Bar(x=feature_names, y=importance, marker_color='skyblue')
            ])
            fig.update_layout(
                title="Feature Importance in Survival Prediction",
                xaxis_title="Features",
                yaxis_title="Importance",
                width=800, height=600
            )
            importance_path = os.path.join(output_dir, 'feature_importance.html')
            fig.write_html(importance_path)
            logger.info(f"📊 Saved feature importance plot to {importance_path}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def main():
    """Main function to handle CLI arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Breast Cancer Survival Prediction System")
    parser.add_argument('--mode', choices=['analyze', 'train', 'predict', 'visualize'], default='analyze',
                        help="Mode of operation: analyze, train, predict, or visualize")
    parser.add_argument('--data_path', default=None,
                        help="Path to the dataset (optional; downloads Haberman's dataset if not provided)")
    parser.add_argument('--output_dir', default='./plots',
                        help="Directory to save visualizations")
    parser.add_argument('--age', type=int, help="Patient age for prediction (required for predict mode)")
    parser.add_argument('--operation_year', type=int, help="Year of operation for prediction (required for predict mode)")
    parser.add_argument('--nodes', type=int, help="Number of positive axillary nodes for prediction (required for predict mode)")
    
    args = parser.parse_args()

    try:
        predictor = BreastCancerSurvivalPredictor(args.data_path)
        predictor.preprocess_data()

        if args.mode == 'analyze':
            predictor.analyze_data()
        elif args.mode == 'train':
            predictor.train_model()
            predictor.evaluate_model()
        elif args.mode == 'predict':
            if not all([args.age, args.operation_year, args.nodes]):
                logger.error("Please provide --age, --operation_year, and --nodes for prediction")
                sys.exit(1)
            predictor.train_model()
            prob, prediction = predictor.predict_survival(args.age, args.operation_year, args.nodes)
            print(f"Prediction: {prediction} (Survival Probability: {prob:.2%})")
        elif args.mode == 'visualize':
            predictor.train_model()
            predictor.visualize_data(args.output_dir)
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
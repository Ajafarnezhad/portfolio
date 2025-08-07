import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from joblib import dump, load
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FruitClassifier:
    def __init__(self, data_path: str = "fruit.txt", model_path: str = "models/fruit_classifier_model.joblib"):
        """Initialize the FruitClassifier with data and model paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.label_map = {1: "Apple", 2: "Mandarin", 3: "Orange", 4: "Lemon"}

    def load_data(self) -> None:
        """Load and preprocess the fruit dataset."""
        try:
            self.data = pd.read_table(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Handle missing values
            if self.data.isnull().sum().any():
                self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                logger.info("Handled missing values by filling with mean.")
            
            # Select features and target
            self.X = self.data[['mass', 'width', 'height', 'color_score']]
            self.y = self.data['fruit_label']
            
            # Validate data
            if not self.X.select_dtypes(include=np.number).columns.equals(self.X.columns):
                raise ValueError("Non-numeric data found in features.")
            if self.y.nunique() != len(self.label_map):
                raise ValueError(f"Unexpected number of classes: {self.y.nunique()}")
            
            logger.info("Data preprocessed successfully.")
        except FileNotFoundError:
            logger.error(f"Dataset file '{self.data_path}' not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)

    def explore_data(self, output_dir: str = "plots/") -> None:
        """Explore the dataset with visualizations."""
        if self.data is None:
            self.load_data()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Pairplot
        sns.pairplot(self.data, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])
        plt.suptitle("Pairplot of Features by Fruit Type")
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()
        logger.info("Saved pairplot.")
        
        # Feature distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, feature in enumerate(['mass', 'width', 'height', 'color_score']):
            sns.histplot(data=self.data, x=feature, hue='fruit_name', ax=axes[idx//2, idx%2], kde=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
        plt.close()
        logger.info("Saved feature distribution plots.")

    def train_model(self, test_size: float = 0.2, cv_folds: int = 5) -> None:
        """Train the RandomForestClassifier with hyperparameter tuning."""
        if self.X is None or self.y is None:
            self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Pipeline with scaling and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='accuracy')
        logger.info(f"Cross-validation accuracy scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        self._evaluate(y_test, y_pred)
        
        # Save model
        dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_trained_model(self) -> None:
        """Load a pre-trained model from file."""
        try:
            self.model = load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file '{self.model_path}' not found. Train the model first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Predict fruit labels for new input data."""
        if self.model is None:
            self.load_trained_model()
        
        try:
            required_cols = ['mass', 'width', 'height', 'color_score']
            if not all(col in input_data.columns for col in required_cols):
                raise ValueError(f"Input data must contain columns: {required_cols}")
            
            predictions = self.model.predict(input_data[required_cols])
            predicted_labels = [self.label_map.get(pred, "Unknown") for pred in predictions]
            logger.info(f"Predictions: {predicted_labels}")
            return predicted_labels
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            sys.exit(1)

    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Evaluate the model with metrics and visualizations."""
        acc = accuracy_score(y_true, y_pred)
        logger.info(f"Test Accuracy: {acc:.2f}")
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=self.label_map.values())}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_map.values(), yticklabels=self.label_map.values())
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("plots/confusion_matrix.png")
        plt.close()
        logger.info("Saved confusion matrix plot.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fruit Classifier using Random Forest on fruit dataset.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explore"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference, 'explore' for data visualization.")
    parser.add_argument("--data_path", type=str, default="fruit.txt", help="Path to the dataset file.")
    parser.add_argument("--model_path", type=str, default="models/fruit_classifier_model.joblib", help="Path to save/load the model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for train-test split.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--input_data", type=str, help="Path to CSV for prediction (required in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    classifier = FruitClassifier(data_path=args.data_path, model_path=args.model_path)
    
    if args.mode == "explore":
        classifier.explore_data()
    elif args.mode == "train":
        classifier.train_model(test_size=args.test_size, cv_folds=args.cv_folds)
    elif args.mode == "predict":
        if not args.input_data:
            logger.error("Input data file required for predict mode.")
            sys.exit(1)
        input_df = pd.read_csv(args.input_data)
        predictions = classifier.predict(input_df)
        print(f"Predictions: {predictions}")
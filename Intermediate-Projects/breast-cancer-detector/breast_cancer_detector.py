import argparse
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from joblib import dump, load
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BreastCancerDetector:
    def __init__(self, data_path: str = "cancer.csv", model_path: str = "models/breast_cancer_model.joblib"):
        """Initialize the BreastCancerDetector with data and model paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None

    def load_data(self) -> None:
        """Load and preprocess the breast cancer dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Drop unnecessary columns
            self.data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True, errors='ignore')
            
            # Convert diagnosis to binary (1: Malignant, 0: Benign)
            self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})
            
            # Handle missing values if any
            if self.data.isnull().sum().any():
                self.data.fillna(self.data.median(), inplace=True)
                logger.info("Handled missing values by filling with median.")
            
            self.y = self.data['diagnosis']
            self.X = self.data.drop('diagnosis', axis=1)
            
            # Feature scaling
            scaler = StandardScaler()
            self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
            logger.info("Data preprocessed and scaled.")
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
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.data.corr(), cmap='coolwarm', annot=False)
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        logger.info("Saved correlation heatmap.")
        
        # Diagnosis distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='diagnosis', data=self.data)
        plt.title("Diagnosis Distribution (0: Benign, 1: Malignant)")
        plt.savefig(os.path.join(output_dir, "diagnosis_distribution.png"))
        plt.close()
        logger.info("Saved diagnosis distribution plot.")

    def train_model(self, test_size: float = 0.2, cv_folds: int = 5) -> None:
        """Train the logistic regression model with hyperparameter tuning and cross-validation."""
        if self.X is None or self.y is None:
            self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Pipeline with PCA and Logistic Regression
        pipeline = Pipeline([
            ('pca', PCA(n_components=0.95)),  # Retain 95% variance
            ('classifier', LogisticRegression(max_iter=10000))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring='f1')
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
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
        """Predict diagnosis for new input data."""
        if self.model is None:
            self.load_trained_model()
        
        if self.X is None:
            self.load_data()  # To ensure scaler consistency, but since we use pipeline, it's handled
        
        predictions = self.model.predict(input_data)
        logger.info(f"Predictions: {predictions}")
        return predictions

    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Evaluate the model with multiple metrics."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        logger.info(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("plots/confusion_matrix.png")
        plt.close()
        logger.info("Saved confusion matrix plot.")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("plots/roc_curve.png")
        plt.close()
        logger.info("Saved ROC curve plot.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Breast Cancer Detector using Logistic Regression.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explore"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference, 'explore' for data visualization.")
    parser.add_argument("--data_path", type=str, default="cancer.csv", help="Path to the dataset CSV.")
    parser.add_argument("--model_path", type=str, default="models/breast_cancer_model.joblib", help="Path to save/load the model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for train-test split.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--input_data", type=str, help="Path to CSV for prediction (required in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    detector = BreastCancerDetector(data_path=args.data_path, model_path=args.model_path)
    
    if args.mode == "explore":
        detector.explore_data()
    elif args.mode == "train":
        detector.train_model(test_size=args.test_size, cv_folds=args.cv_folds)
    elif args.mode == "predict":
        if not args.input_data:
            logger.error("Input data CSV required for predict mode.")
            sys.exit(1)
        input_df = pd.read_csv(args.input_data)
        predictions = detector.predict(input_df)
        print(f"Predictions: {predictions}")
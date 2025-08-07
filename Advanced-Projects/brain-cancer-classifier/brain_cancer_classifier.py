import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import dump, load
from typing import Optional, Tuple, List
import shap
import umap
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BrainCancerClassifier:
    def __init__(self, data_path: str = "data.csv", model_path: str = "models/brain_cancer_model.joblib", results_excel_path: str = "results/brain_cancer_results.xlsx"):
        """Initialize the BrainCancerClassifier with data, model, and results paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.results_excel_path = results_excel_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.categorical_cols = ['gender', 'TMZ', 'Radiology', 'DxWHO2007', 'IDH1_molecular', 'H3.3K27M', 
                              'H3.3G34R', 'BRAFV600E', 'IntegratedDxStep1', 'EGFR', 'IDH1_2', 'IDH1_tarkibi', 
                              'IDH2', 'H3F3A', 'MGMT', 'V600E', 'EGFR_A', '@1p19q', 'PTEN', 'Integrateddxstep2', 'CD44', 'MGMT_new']
        self.numerical_cols = ['Age', 'timefordeath', 'time_Recurrence', 'PCV_new', 'Event_reccurrence', 'Event_death']

    def load_data(self) -> None:
        """Load and preprocess the brain cancer dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Drop unnecessary columns
            self.data.drop(['Patient'], axis=1, inplace=True, errors='ignore')
            
            # Define target (Event_death: 1 for death, 0 for survival)
            self.y = self.data['Event_death']
            self.X = self.data.drop('Event_death', axis=1)
            
            # Handle missing values
            imputer_cat = SimpleImputer(strategy='most_frequent')
            imputer_num = SimpleImputer(strategy='median')
            self.X[self.categorical_cols] = pd.DataFrame(imputer_cat.fit_transform(self.X[self.categorical_cols]), columns=self.categorical_cols)
            self.X[self.numerical_cols] = pd.DataFrame(imputer_num.fit_transform(self.X[self.numerical_cols]), columns=self.numerical_cols)
            logger.info("Handled missing values.")
            
            # Encode categorical variables
            label_encoders = {}
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                label_encoders[col] = le
            
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
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.X.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        logger.info("Saved correlation heatmap.")
        
        # Event_death distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Event_death', data=self.data)
        plt.title("Event Death Distribution (0: Survived, 1: Died)")
        plt.savefig(os.path.join(output_dir, "event_death_distribution.png"))
        plt.close()
        logger.info("Saved event death distribution plot.")
        
        # UMAP for dimensionality reduction visualization
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(self.X[self.numerical_cols])
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=self.data['Event_death'], palette='deep')
        plt.title("UMAP Projection of Numerical Features")
        plt.savefig(os.path.join(output_dir, "umap_projection.png"))
        plt.close()
        logger.info("Saved UMAP projection plot.")

    def train_model(self, test_size: float = 0.2, cv_folds: int = 5) -> None:
        """Train the GradientBoostingClassifier with hyperparameter tuning and feature selection."""
        if self.X is None or self.y is None:
            self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Pipeline with scaling, feature selection, and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'feature_selection__k': [5, 10, 15],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv_folds, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        self._evaluate(y_test, y_pred, X_test)
        
        # Save model and results
        dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        self._save_results(y_test, y_pred)

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
        """Predict survival outcome for new input data."""
        if self.model is None:
            self.load_trained_model()
        
        try:
            # Ensure input has required columns
            required_cols = self.categorical_cols + self.numerical_cols
            if not all(col in input_data.columns for col in required_cols):
                raise ValueError(f"Input data must contain columns: {required_cols}")
            
            # Preprocess input (impute and encode as in training)
            input_data_copy = input_data.copy()
            input_data_copy[self.categorical_cols] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(input_data_copy[self.categorical_cols]), columns=self.categorical_cols)
            input_data_copy[self.numerical_cols] = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(input_data_copy[self.numerical_cols]), columns=self.numerical_cols)
            
            for col in self.categorical_cols:
                le = LabelEncoder()
                input_data_copy[col] = le.fit_transform(input_data_copy[col])
            
            scaler = StandardScaler()
            input_data_copy[self.numerical_cols] = scaler.fit_transform(input_data_copy[self.numerical_cols])
            
            predictions = self.model.predict(input_data_copy)
            logger.info(f"Predictions: {predictions}")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            sys.exit(1)

    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray, X_test: pd.DataFrame) -> None:
        """Evaluate the model with multiple metrics and visualizations."""
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
        fpr, tpr, _ = roc_curve(y_true, self.model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("plots/roc_curve.png")
        plt.close()
        logger.info("Saved ROC curve plot.")
        
        # SHAP feature importance
        explainer = shap.TreeExplainer(self.model.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.title("SHAP Feature Importance")
        plt.savefig("plots/shap_feature_importance.png")
        plt.close()
        logger.info("Saved SHAP feature importance plot.")

    def _save_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        """Save evaluation metrics and predictions to Excel."""
        os.makedirs(os.path.dirname(self.results_excel_path), exist_ok=True)
        results = pd.DataFrame({
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1_Score': f1_score(y_true, y_pred)
        })
        results.to_excel(self.results_excel_path, index=False)
        logger.info(f"Results saved to {self.results_excel_path}")
        
        # Copy original dataset for reference
        original_data_filename = os.path.basename(self.data_path)
        destination_path = os.path.join(os.path.dirname(self.results_excel_path), f"Original_{original_data_filename}")
        shutil.copyfile(self.data_path, destination_path)
        logger.info(f"Copied original dataset to {destination_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Brain Cancer Outcome Classifier using Gradient Boosting.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explore"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference, 'explore' for data visualization.")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to the dataset CSV.")
    parser.add_argument("--model_path", type=str, default="models/brain_cancer_model.joblib", help="Path to save/load the model.")
    parser.add_argument("--results_excel_path", type=str, default="results/brain_cancer_results.xlsx", help="Path to save results Excel file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for train-test split.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--input_data", type=str, help="Path to CSV for prediction (required in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    classifier = BrainCancerClassifier(data_path=args.data_path, model_path=args.model_path, results_excel_path=args.results_excel_path)
    
    if args.mode == "explore":
        classifier.explore_data()
    elif args.mode == "train":
        classifier.train_model(test_size=args.test_size, cv_folds=args.cv_folds)
    elif args.mode == "predict":
        if not args.input_data:
            logger.error("Input data CSV required for predict mode.")
            sys.exit(1)
        input_df = pd.read_csv(args.input_data)
        predictions = classifier.predict(input_df)
        print(f"Predictions (1: Death, 0: Survival): {predictions}")
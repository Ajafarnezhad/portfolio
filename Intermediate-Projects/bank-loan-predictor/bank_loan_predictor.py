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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BankLoanPredictor:
    def __init__(self, data_path: str = "loan.csv", model_path: str = "models/bank_loan_model.joblib"):
        """Initialize the BankLoanPredictor with data and model paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        self.numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    def load_data(self) -> None:
        """Load and preprocess the bank loan dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Drop unnecessary columns
            self.data.drop(['Loan_ID'], axis=1, inplace=True, errors='ignore')
            
            # Convert Loan_Status to binary (1: Y, 0: N)
            self.data['Loan_Status'] = self.data['Loan_Status'].map({'Y': 1, 'N': 0})
            
            # Handle missing values
            imputer_cat = SimpleImputer(strategy='most_frequent')
            imputer_num = SimpleImputer(strategy='median')
            self.data[self.categorical_cols] = pd.DataFrame(imputer_cat.fit_transform(self.data[self.categorical_cols]), columns=self.categorical_cols)
            self.data[self.numerical_cols] = pd.DataFrame(imputer_num.fit_transform(self.data[self.numerical_cols]), columns=self.numerical_cols)
            logger.info("Handled missing values.")
            
            # Encode categorical variables
            label_encoders = {}
            for col in self.categorical_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                label_encoders[col] = le
            
            self.y = self.data['Loan_Status']
            self.X = self.data.drop('Loan_Status', axis=1)
            
            # Feature scaling for numerical columns
            scaler = StandardScaler()
            self.X[self.numerical_cols] = scaler.fit_transform(self.X[self.numerical_cols])
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
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        logger.info("Saved correlation heatmap.")
        
        # Loan Status distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Loan_Status', data=self.data)
        plt.title("Loan Status Distribution (0: No, 1: Yes)")
        plt.savefig(os.path.join(output_dir, "loan_status_distribution.png"))
        plt.close()
        logger.info("Saved loan status distribution plot.")
        
        # Boxplots for numerical features by Loan_Status
        for col in self.numerical_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Loan_Status', y=col, data=self.data)
            plt.title(f"{col} by Loan Status")
            plt.savefig(os.path.join(output_dir, f"{col}_by_loan_status.png"))
            plt.close()
        logger.info("Saved boxplots for numerical features.")

    def train_model(self, test_size: float = 0.2, cv_folds: int = 5) -> None:
        """Train the RandomForestClassifier with hyperparameter tuning and cross-validation."""
        if self.X is None or self.y is None:
            self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
        # Pipeline with imputation (though handled) and classifier
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
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
        """Predict loan approval for new input data."""
        if self.model is None:
            self.load_trained_model()
        
        try:
            # Ensure input has required columns
            required_cols = self.categorical_cols + self.numerical_cols
            if not all(col in input_data.columns for col in required_cols):
                raise ValueError(f"Input data must contain columns: {required_cols}")
            
            # Encode categoricals (assuming same encoding as training)
            for col in self.categorical_cols:
                input_data[col] = LabelEncoder().fit_transform(input_data[col])  # Note: In production, use fitted encoders
            
            # Scale numerical
            scaler = StandardScaler()
            input_data[self.numerical_cols] = scaler.fit_transform(input_data[self.numerical_cols])  # Note: Use training scaler in production
            
            predictions = self.model.predict(input_data)
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
        
        # Feature Importance (for RandomForest)
        importances = self.model.named_steps['classifier'].feature_importances_
        features = self.X.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title("Feature Importances")
        plt.savefig("plots/feature_importances.png")
        plt.close()
        logger.info("Saved feature importances plot.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bank Loan Approval Predictor using Random Forest.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explore"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference, 'explore' for data visualization.")
    parser.add_argument("--data_path", type=str, default="loan.csv", help="Path to the dataset CSV.")
    parser.add_argument("--model_path", type=str, default="models/bank_loan_model.joblib", help="Path to save/load the model.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for train-test split.")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--input_data", type=str, help="Path to CSV for prediction (required in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predictor = BankLoanPredictor(data_path=args.data_path, model_path=args.model_path)
    
    if args.mode == "explore":
        predictor.explore_data()
    elif args.mode == "train":
        predictor.train_model(test_size=args.test_size, cv_folds=args.cv_folds)
    elif args.mode == "predict":
        if not args.input_data:
            logger.error("Input data CSV required for predict mode.")
            sys.exit(1)
        input_df = pd.read_csv(args.input_data)
        predictions = predictor.predict(input_df)
        print(f"Predictions (1: Approved, 0: Rejected): {predictions}")
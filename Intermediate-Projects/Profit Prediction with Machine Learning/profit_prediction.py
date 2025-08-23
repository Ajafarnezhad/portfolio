import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import argparse
import logging
import os
import sys
from typing import Tuple, Dict, Any, Optional
import kaggle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('profit_prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles dataset loading and validation."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Download or load the 50 Startups dataset."""
        try:
            if self.data_path:
                self.data = pd.read_csv(self.data_path)
                logger.info(f"🌟 Loaded dataset from {self.data_path} ({len(self.data)} records)")
            else:
                kaggle.api.dataset_download_files(
                    'ahsan81/startup-success-prediction',
                    path='data/',
                    unzip=True
                )
                self.data = pd.read_csv('data/50_Startups.csv')
                logger.info(f"🌟 Downloaded 50 Startups dataset ({len(self.data)} records)")
            
            # Validate dataset
            required_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            
            # Handle missing values
            initial_len = len(self.data)
            self.data = self.data.dropna()
            if len(self.data) < initial_len:
                logger.warning(f"Removed {initial_len - len(self.data)} rows with missing values")
            logger.info(f"✅ Cleaned dataset, retained {len(self.data)} records")
            return self.data
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

class DataPreprocessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def preprocess(self, data: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Preprocess data: encode categorical variables, scale features, and split."""
        try:
            # Feature engineering: Add interaction term
            data['R&D_Marketing_Interaction'] = data['R&D Spend'] * data['Marketing Spend']
            
            # One-hot encode 'State' column
            data = pd.get_dummies(data, columns=['State'], drop_first=True)
            
            # Define features and target
            X = data.drop('Profit', axis=1)
            y = data['Profit']
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            logger.info("✅ Preprocessed data with feature engineering and split into training and test sets")
            return X_train, X_test, y_train, y_test, self.feature_names
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def transform_input(self, input_data: pd.DataFrame) -> np.ndarray:
        """Transform input data for prediction."""
        try:
            input_data['R&D_Marketing_Interaction'] = input_data['R&D Spend'] * input_data['Marketing Spend']
            input_scaled = self.scaler.transform(input_data[self.feature_names])
            return input_scaled
        except Exception as e:
            logger.error(f"Error transforming input: {str(e)}")
            raise

class ProfitPredictor:
    """Manages profit prediction using a Random Forest Regressor."""
    
    def __init__(self, model: Any = None, preprocessor: DataPreprocessor = None):
        """Initialize with optional model and preprocessor for dependency injection."""
        self.model = model if model else RandomForestRegressor(random_state=42)
        self.preprocessor = preprocessor if preprocessor else DataPreprocessor()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model with hyperparameter tuning."""
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            logger.info(f"🧠 Trained Random Forest model with best params: {grid_search.best_params_}")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        try:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"📈 Model Performance: MSE = {mse:.2f}, R² = {r2:.2%}")
            return {'mse': mse, 'r2': r2}
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def predict_profit(self, rd_spend: float, administration: float, marketing_spend: float, state: str) -> float:
        """Predict profit for a single startup."""
        try:
            input_data = pd.DataFrame({
                'R&D Spend': [rd_spend],
                'Administration': [administration],
                'Marketing Spend': [marketing_spend]
            })
            
            # Add dummy variables for state
            state_columns = [col for col in self.feature_names if col.startswith('State_')]
            for col in state_columns:
                input_data[col] = 0
            if state in ['New York', 'Florida']:  # Assuming 'California' is the reference
                input_data[f'State_{state}'] = 1
            
            # Transform input
            input_scaled = self.preprocessor.transform_input(input_data)
            
            # Predict
            profit = self.model.predict(input_scaled)[0]
            logger.info(f"📈 Predicted profit for input (R&D={rd_spend}, Admin={administration}, Marketing={marketing_spend}, State={state}): ${profit:.2f}")
            return profit
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def analyze_data(self, data: pd.DataFrame) -> None:
        """Perform exploratory data analysis."""
        try:
            logger.info(f"🔍 Dataset Summary:\n{data.describe()}")
            correlation = data.corr()
            logger.info(f"✅ Correlation Matrix:\n{correlation}")
            logger.info(f"✅ Average Profit: ${data['Profit'].mean():.2f} (std: ${data['Profit'].std():.2f})")
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            raise

    def visualize_data(self, data: pd.DataFrame, output_dir: str = './plots') -> None:
        """Generate visualizations: correlation heatmap and feature importance."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Correlation Heatmap
            correlation = data.corr()
            fig = px.imshow(
                correlation,
                labels=dict(x="Features", y="Features", color="Correlation"),
                title="Correlation Heatmap of Startup Features",
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(width=800, height=800)
            heatmap_path = os.path.join(output_dir, 'correlation_heatmap.html')
            fig.write_html(heatmap_path)
            logger.info(f"📊 Saved correlation heatmap to {heatmap_path}")

            # Feature Importance
            importance = self.model.feature_importances_
            fig = go.Figure(data=[
                go.Bar(x=self.feature_names, y=importance, marker_color='skyblue')
            ])
            fig.update_layout(
                title="Feature Importance in Profit Prediction",
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
    parser = argparse.ArgumentParser(description="Startup Profit Prediction System")
    parser.add_argument('--mode', choices=['analyze', 'train', 'predict', 'visualize'], default='analyze',
                        help="Mode of operation: analyze, train, predict, or visualize")
    parser.add_argument('--data_path', default=None,
                        help="Path to the dataset (optional; downloads Kaggle dataset if not provided)")
    parser.add_argument('--rd_spend', type=float, help="R&D spend for prediction (required for predict mode)")
    parser.add_argument('--administration', type=float, help="Administration cost for prediction (required for predict mode)")
    parser.add_argument('--marketing_spend', type=float, help="Marketing spend for prediction (required for predict mode)")
    parser.add_argument('--state', type=str, choices=['New York', 'California', 'Florida'],
                        help="State for prediction (required for predict mode)")
    parser.add_argument('--output_dir', default='./plots',
                        help="Directory to save visualizations")
    
    args = parser.parse_args()

    try:
        # Initialize components
        loader = DataLoader(args.data_path)
        data = loader.load_data()
        preprocessor = DataPreprocessor()
        predictor = ProfitPredictor(preprocessor=preprocessor)
        
        # Preprocess data
        predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test, predictor.feature_names = preprocessor.preprocess(data)

        if args.mode == 'analyze':
            predictor.analyze_data(data)
        elif args.mode == 'train':
            predictor.train_model(predictor.X_train, predictor.y_train)
            predictor.evaluate_model(predictor.X_test, predictor.y_test)
        elif args.mode == 'predict':
            if not all([args.rd_spend, args.administration, args.marketing_spend, args.state]):
                logger.error("Please provide --rd_spend, --administration, --marketing_spend, and --state for prediction")
                sys.exit(1)
            predictor.train_model(predictor.X_train, predictor.y_train)
            profit = predictor.predict_profit(args.rd_spend, args.administration, args.marketing_spend, args.state)
            print(f"Predicted Profit: ${profit:.2f}")
        elif args.mode == 'visualize':
            predictor.train_model(predictor.X_train, predictor.y_train)
            predictor.visualize_data(data, args.output_dir)
    except Exception as e:
        logger.error(f"Program terminated due to error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import argparse
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from joblib import dump, load

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="CO2 Emissions Predictor: Train and predict vehicle CO2 emissions.")
    parser.add_argument("--data_path", type=str, default="co2.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--model_type", type=str, choices=["linear", "random_forest", "gradient_boosting"], default="linear", help="Model type to use.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--predict", type=float, nargs=3, metavar=("ENGINE", "CYLINDERS", "FUELCOMB"), help="Predict CO2 for given engine size, cylinders, and fuel consumption.")
    parser.add_argument("--model_path", type=str, default="model.joblib", help="Path to save/load the model.")
    parser.add_argument("--plot", action="store_true", help="Generate and show plots.")
    return parser.parse_args()

def load_and_explore_data(file_path, plot=False):
    """
    Load the dataset, perform EDA, and optionally plot visualizations.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")
        logger.info(f"Shape: {df.shape}")
        logger.info("\nFirst 5 rows:\n%s", df.head())
        logger.info("\nDescription:\n%s", df.describe())
        logger.info("\nMissing values:\n%s", df.isnull().sum())
        
        if plot:
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.show()
            
            # Pairplot
            sns.pairplot(df)
            plt.suptitle("Pairplot of Features")
            plt.show()
            
            # Distribution of target
            plt.figure(figsize=(10, 6))
            sns.histplot(df["out1"], kde=True)
            plt.title("Distribution of CO2 Emissions")
            plt.show()
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def prepare_data(df):
    """
    Prepare features and target, with feature engineering.
    """
    X = df.drop("out1", axis=1)
    y = df["out1"]
    
    # Feature engineering: Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled, y, scaler, poly

def get_model(model_type):
    """
    Get the specified model with hyperparameter tuning.
    """
    if model_type == "linear":
        model = LinearRegression()
        param_grid = {}
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10, 20]}
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    
    if param_grid:
        model = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
    
    return model

def train_and_evaluate_model(X, y, model_type, model_path):
    """
    Train the model, evaluate with cross-validation, and save it.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    logger.info(f"Cross-validation R2 scores: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Mean Squared Error: {mse:.2f}")
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"R-squared: {r2:.2f}")
    
    # Save model
    dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.title("Predicted vs Actual CO2 Emissions")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    
    return model

def make_prediction(model, scaler, poly, engine, cylinders, fuelcomb):
    """
    Make a prediction for given inputs.
    """
    input_data = np.array([[engine, cylinders, fuelcomb]])
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)
    prediction = model.predict(input_scaled)
    logger.info(f"Predicted CO2: {prediction[0]:.2f}")
    return prediction[0]

if __name__ == "__main__":
    args = parse_arguments()
    
    df = load_and_explore_data(args.data_path, plot=args.plot)
    X, y, scaler, poly = prepare_data(df)
    
    if args.train:
        model = train_and_evaluate_model(X, y, args.model_type, args.model_path)
    else:
        try:
            model = load(args.model_path)
            logger.info(f"Model loaded from {args.model_path}")
        except FileNotFoundError:
            logger.error("Model file not found. Please train the model first.")
            exit(1)
    
    if args.predict:
        engine, cylinders, fuelcomb = args.predict
        make_prediction(model, scaler, poly, engine, cylinders, fuelcomb)
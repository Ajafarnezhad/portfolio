import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from joblib import dump, load
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BitcoinPredictor:
    def __init__(self, data_path: str = "bitcoin.csv", model_path: str = "models/bitcoin_predictor_model.h5", scaler_path: str = "models/bitcoin_scaler.joblib"):
        """Initialize the BitcoinPredictor with data, model, and scaler paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.window_size = 60  # Default look-back window for LSTM

    def load_data(self) -> None:
        """Load and preprocess the Bitcoin dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Clean 'Vol.' column: remove 'K'/'M' and commas, convert to float
            def clean_vol(vol):
                vol = str(vol).replace(',', '')
                if 'K' in vol:
                    return float(vol.replace('K', '')) * 1000
                elif 'M' in vol:
                    return float(vol.replace('M', '')) * 1000000
                else:
                    return float(vol)
            
            self.data['Vol.'] = self.data['Vol.'].apply(clean_vol)
            
            # Convert Date to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data.set_index('Date', inplace=True)
            
            # Handle missing values
            if self.data.isnull().sum().any():
                self.data.fillna(method='ffill', inplace=True)
                logger.info("Handled missing values with forward fill.")
            
            # Feature engineering: Add moving averages, RSI, etc.
            self.data['MA_7'] = self.data['Close'].rolling(window=7).mean()
            self.data['MA_30'] = self.data['Close'].rolling(window=30).mean()
            self.data['RSI'] = self._compute_rsi(self.data['Close'])
            self.data.dropna(inplace=True)
            
            logger.info("Data preprocessed with feature engineering.")
        except FileNotFoundError:
            logger.error(f"Dataset file '{self.data_path}' not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)

    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def explore_data(self, output_dir: str = "plots/") -> None:
        """Explore the dataset with visualizations."""
        if self.data is None:
            self.load_data()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Price over time
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data['MA_7'], label='7-Day MA')
        plt.plot(self.data['MA_30'], label='30-Day MA')
        plt.title("Bitcoin Close Price and Moving Averages")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "price_trend.png"))
        plt.close()
        logger.info("Saved price trend plot.")
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        logger.info("Saved correlation heatmap.")
        
        # RSI plot
        plt.figure(figsize=(12, 4))
        plt.plot(self.data['RSI'], label='RSI')
        plt.axhline(70, color='r', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='g', linestyle='--', label='Oversold (30)')
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "rsi_plot.png"))
        plt.close()
        logger.info("Saved RSI plot.")

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)

    def train_model(self, test_size: float = 0.2, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.1) -> None:
        """Train the LSTM model for price prediction."""
        if self.data is None:
            self.load_data()
        
        # Use 'Close' as target, scale data
        prices = self.data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_prices)
        
        # Split data
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.window_size, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        logger.info("LSTM model built and compiled.")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        logger.info("Model training completed.")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_test_inv = self.scaler.inverse_transform(y_test)
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        logger.info(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label='Actual')
        plt.plot(y_pred_inv, label='Predicted')
        plt.title("Bitcoin Price Predictions")
        plt.legend()
        plt.savefig("plots/predictions.png")
        plt.close()
        logger.info("Saved predictions plot.")

    def load_trained_model(self) -> None:
        """Load a pre-trained model and scaler from files."""
        try:
            self.model = load_model(self.model_path)
            self.scaler = load(self.scaler_path)
            logger.info(f"Loaded model from {self.model_path} and scaler from {self.scaler_path}")
        except FileNotFoundError:
            logger.error(f"Model or scaler file not found. Train the model first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model/scaler: {e}")
            sys.exit(1)

    def predict(self, steps: int = 30) -> np.ndarray:
        """Predict future Bitcoin prices for given steps."""
        if self.model is None:
            self.load_trained_model()
        
        if self.data is None:
            self.load_data()
        
        # Use last window for prediction
        last_sequence = self.scaler.transform(self.data['Close'][-self.window_size:].values.reshape(-1, 1))
        predictions = []
        
        for _ in range(steps):
            input_seq = last_sequence.reshape(1, self.window_size, 1)
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0][0])
            last_sequence = np.append(last_sequence[1:], pred, axis=0)
        
        predictions_inv = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        logger.info(f"Predicted prices for next {steps} days: {predictions_inv.flatten()}")
        return predictions_inv.flatten()

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bitcoin Price Predictor using LSTM.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explore"], required=True,
                        help="Mode: 'train' to train model, 'predict' for future prices, 'explore' for data visualization.")
    parser.add_argument("--data_path", type=str, default="bitcoin.csv", help="Path to the Bitcoin CSV dataset.")
    parser.add_argument("--model_path", type=str, default="models/bitcoin_predictor_model.h5", help="Path to save/load the model.")
    parser.add_argument("--scaler_path", type=str, default="models/bitcoin_scaler.joblib", help="Path to save/load the scaler.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for train-test split.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--steps", type=int, default=30, help="Number of future steps to predict (in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    predictor = BitcoinPredictor(data_path=args.data_path, model_path=args.model_path, scaler_path=args.scaler_path)
    
    if args.mode == "explore":
        predictor.explore_data()
    elif args.mode == "train":
        predictor.train_model(test_size=args.test_size, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "predict":
        predictions = predictor.predict(steps=args.steps)
        print(f"Predicted Prices: {predictions}")
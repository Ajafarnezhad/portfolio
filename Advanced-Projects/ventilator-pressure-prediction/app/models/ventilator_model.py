import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the pre-trained LSTM model."""
    try:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise

def preprocess_data(file_path):
    """Preprocess CSV data for LSTM prediction."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in CSV")

        # Group by breath_id and ensure each breath has 80 time steps
        grouped = df.groupby('breath_id').apply(lambda x: x[required_columns].head(80)).reset_index()
        X = grouped[required_columns].values.reshape(-1, 80, len(required_columns))
        X = X.astype(np.float32)
        
        # Normalize features
        X[:, :, 0] = X[:, :, 0] / 50.0  # R
        X[:, :, 1] = X[:, :, 1] / 50.0  # C
        X[:, :, 2] = X[:, :, 2] / 3.0   # time_step
        X[:, :, 3] = X[:, :, 3] / 100.0 # u_in
        X[:, :, 4] = X[:, :, 4]         # u_out (binary)
        
        return X
    except Exception as e:
        logger.error("Error preprocessing data: %s", str(e))
        raise

def predict_pressure(file_path, model):
    """Predict ventilator pressure from CSV data."""
    try:
        X = preprocess_data(file_path)
        predictions = model.predict(X)
        predictions = predictions.flatten()
        logger.info("Predictions generated for %s", file_path)
        return predictions.tolist()
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
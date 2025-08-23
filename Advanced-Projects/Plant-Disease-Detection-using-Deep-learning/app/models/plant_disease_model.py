import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the pre-trained InceptionV3 model."""
    try:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
        model = load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise

def predict_image(img_path, model):
    """Predict plant disease from an image."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]
        
        class_labels = {0: "Healthy", 1: "Powdery", 2: "Rust"}
        result = class_labels.get(pred_class, "Unknown")
        
        logger.info("Prediction for %s: %s", img_path, result)
        return result
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
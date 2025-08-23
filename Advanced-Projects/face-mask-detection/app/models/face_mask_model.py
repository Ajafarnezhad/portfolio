import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the pre-trained CNN model."""
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

def predict_image(img_path, model):
    """Predict if an image shows a face with or without a mask."""
    try:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        prediction = "No Mask" if result[0][0] > 0.5 else "Mask"
        
        logger.info("Prediction for %s: %s", img_path, prediction)
        return prediction
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise
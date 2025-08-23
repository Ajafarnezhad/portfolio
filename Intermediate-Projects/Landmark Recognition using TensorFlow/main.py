import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image
from geopy.geocoders import Nominatim
import logging
import argparse
import json
import os
from typing import Tuple, Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LandmarkRecognizer:
    """Class for recognizing landmarks in images and retrieving geolocation data."""
    
    def __init__(self, model_url: str, labels_path: str, output_dir: str = './results'):
        self.model_url = model_url
        self.labels_path = labels_path
        self.output_dir = output_dir
        self.labels = None
        self.classifier = None
        self.geolocator = Nominatim(user_agent="LandmarkRecognizerApp")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_labels(self) -> None:
        """Load and process the labels CSV file."""
        try:
            df = pd.read_csv(self.labels_path)
            self.labels = dict(zip(df.id, df.name))
            logging.info(f"Loaded {len(self.labels)} labels from {self.labels_path}")
        except Exception as e:
            logging.error(f"Error loading labels: {str(e)}")
            raise

    def load_model(self) -> None:
        """Load the TensorFlow Hub model."""
        try:
            img_shape = (321, 321, 3)
            self.classifier = tf.keras.Sequential([
                hub.KerasLayer(self.model_url, input_shape=img_shape, output_key="predictions:logits")
            ])
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def process_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess the input image."""
        try:
            # Validate image format
            valid_extensions = {'.jpg', '.jpeg', '.png'}
            if not os.path.splitext(image_path)[1].lower() in valid_extensions:
                raise ValueError(f"Unsupported image format. Use {valid_extensions}")

            img = Image.open(image_path).convert('RGB')
            img = img.resize((321, 321))
            img_array = np.array(img) / 255.0
            img_array = img_array[np.newaxis]
            logging.info(f"Processed image: {image_path}")
            return img_array
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def predict_landmark(self, image_path: str) -> Tuple[str, float]:
        """Predict the landmark from the image."""
        try:
            if self.classifier is None:
                self.load_model()
            if self.labels is None:
                self.load_labels()

            img_array = self.process_image(image_path)
            result = self.classifier.predict(img_array, verbose=0)
            pred_id = np.argmax(result, axis=1)[0]
            confidence = float(np.max(tf.nn.softmax(result, axis=1)))
            landmark = self.labels.get(pred_id, "Unknown")
            logging.info(f"Predicted landmark: {landmark} (Confidence: {confidence:.2%})")
            return landmark, confidence
        except Exception as e:
            logging.error(f"Error predicting landmark: {str(e)}")
            raise

    def get_geolocation(self, landmark: str) -> Dict:
        """Retrieve geolocation data for the predicted landmark."""
        try:
            location = self.geolocator.geocode(landmark)
            if location:
                geodata = {
                    'address': location.address,
                    'latitude': location.latitude,
                    'longitude': location.longitude
                }
                logging.info(f"Geolocation for {landmark}: {geodata}")
            else:
                geodata = {'address': 'Not found', 'latitude': None, 'longitude': None}
                logging.warning(f"No geolocation found for {landmark}")
            return geodata
        except Exception as e:
            logging.error(f"Error retrieving geolocation for {landmark}: {str(e)}")
            return {'address': 'Error', 'latitude': None, 'longitude': None}

    def save_results(self, image_path: str, landmark: str, confidence: float, geodata: Dict) -> None:
        """Save prediction and geolocation results to a JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"result_{timestamp}.json")
            result = {
                'image': os.path.basename(image_path),
                'landmark': landmark,
                'confidence': confidence,
                'geolocation': geodata,
                'timestamp': timestamp
            }
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
            logging.info(f"Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

    def recognize(self, image_path: str) -> Dict:
        """Run the full recognition pipeline."""
        try:
            landmark, confidence = self.predict_landmark(image_path)
            geodata = self.get_geolocation(landmark)
            self.save_results(image_path, landmark, confidence, geodata)
            return {
                'landmark': landmark,
                'confidence': confidence,
                'geolocation': geodata
            }
        except Exception as e:
            logging.error(f"Recognition pipeline error: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Landmark Recognition")
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--model_url', default='https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1', help='TensorFlow Hub model URL')
    parser.add_argument('--labels', default='landmarks_classifier_asia_V1_label_map.csv', help='Path to labels CSV')
    parser.add_argument('--output_dir', default='./results', help='Directory for saving results')
    args = parser.parse_args()

    recognizer = LandmarkRecognizer(args.model_url, args.labels, args.output_dir)
    result = recognizer.recognize(args.image)
    
    print(f"Predicted Landmark: {result['landmark']} (Confidence: {result['confidence']:.2%})")
    print(f"Geolocation: {result['geolocation']['address']}")
    print(f"Latitude: {result['geolocation']['latitude']}, Longitude: {result['geolocation']['longitude']}")

if __name__ == "__main__":
    main()
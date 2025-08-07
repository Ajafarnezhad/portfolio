import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AgeDetector:
    def __init__(self, dataset_dir: str = "datasets/UTKFace/", model_path: str = "models/age_detector_model.h5"):
        """Initialize the AgeDetector with dataset directory and model path."""
        self.dataset_dir = dataset_dir
        self.model_path = model_path
        self.model = None
        self.image_size = (128, 128)
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the UTKFace dataset."""
        if not os.path.exists(self.dataset_dir):
            logger.error(f"Dataset directory '{self.dataset_dir}' not found. Download from https://drive.google.com/drive/folders/19zV45_NQzrBPLzFymXeNZiufxbeENGth and unzip.")
            sys.exit(1)
        
        image_paths = []
        age_labels = []
        images = os.listdir(self.dataset_dir)
        
        for filename in images:
            image_path = os.path.join(self.dataset_dir, filename)
            temp = filename.split('_')
            if len(temp) >= 2 and temp[0].isnumeric():
                age = int(temp[0])
                image_paths.append(image_path)
                age_labels.append(age)
        
        features = []
        for image_path in image_paths:
            img = load_img(image_path, color_mode="grayscale")
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            img = img_to_array(img) / 255.0
            features.append(img)
        
        features = np.array(features)
        age_labels = np.array(age_labels)
        
        logger.info(f"Loaded {len(features)} images from dataset.")
        return features, age_labels

    def build_model(self) -> Sequential:
        """Build and compile the CNN model for age detection."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='linear')  # Regression for age
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        logger.info("Model built and compiled.")
        return model

    def train_model(self, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> None:
        """Train the model on the dataset."""
        features, age_labels = self.load_dataset()
        X_train, X_val, y_train, y_val = train_test_split(features, age_labels, test_size=validation_split, random_state=42)
        
        self.model = self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        logger.info("Model training completed.")
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_trained_model(self) -> None:
        """Load a pre-trained model from file."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file '{self.model_path}' not found. Train the model first.")
            sys.exit(1)
        self.model = load_model(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

    def predict_age_from_image(self, image_path: str) -> float:
        """Predict age from a single image file."""
        if self.model is None:
            self.load_trained_model()
        
        img = load_img(image_path, color_mode="grayscale")
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        
        age = self.model.predict(img, verbose=0)[0][0]
        logger.info(f"Predicted age for {image_path}: {age:.2f}")
        return age

    def live_detection(self) -> None:
        """Run live age detection using webcam."""
        if self.model is None:
            self.load_trained_model()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Could not open webcam.")
            sys.exit(1)
        
        logger.info("Starting live age detection. Press 'Esc' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            faces = self.detector.detectMultiScale(frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, self.image_size) / 255.0
                face_input = np.expand_dims(face_resized, axis=(0, -1))
                
                age = self.model.predict(face_input, verbose=0)[0][0]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y+h), (x+w, y+h+50), (255, 0, 0), -1)
                cv2.putText(frame, f"Age: {age:.0f}", (x + 10, y + h + 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Live Age Detection', frame)
            
            if cv2.waitKey(1) == 27:  # Esc key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Live detection stopped.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Age Detector using CNN on UTKFace dataset.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "live"], required=True,
                        help="Mode: 'train' to train model, 'predict' for single image, 'live' for webcam.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/UTKFace/",
                        help="Path to UTKFace dataset directory.")
    parser.add_argument("--model_path", type=str, default="models/age_detector_model.h5",
                        help="Path to save/load the model.")
    parser.add_argument("--image_path", type=str, help="Path to image for prediction (required in predict mode).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    detector = AgeDetector(dataset_dir=args.dataset_dir, model_path=args.model_path)
    
    if args.mode == "train":
        detector.train_model(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "predict":
        if not args.image_path:
            logger.error("Image path required for predict mode.")
            sys.exit(1)
        age = detector.predict_age_from_image(args.image_path)
        print(f"Predicted Age: {age:.2f}")
    elif args.mode == "live":
        detector.live_detection()
import argparse
import logging
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FashionItemClassifier:
    def __init__(self, model_path: str = "models/fashion_classifier_model.h5"):
        """Initialize the FashionItemClassifier with model path."""
        self.model_path = model_path
        self.model = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.img_height, self.img_width = 28, 28  # Fashion MNIST image size

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess the Fashion MNIST dataset."""
        try:
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            
            # Normalize pixel values to [0, 1]
            train_images = train_images.astype('float32') / 255.0
            test_images = test_images.astype('float32') / 255.0
            
            # Reshape for CNN (add channel dimension)
            train_images = np.expand_dims(train_images, axis=-1)
            test_images = np.expand_dims(test_images, axis=-1)
            
            # Convert labels to categorical
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
            
            logger.info(f"Loaded Fashion MNIST dataset: Train {train_images.shape}, Test {test_images.shape}")
            return (train_images, train_labels), (test_images, test_labels)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            sys.exit(1)

    def build_model(self) -> Sequential:
        """Build the CNN model for fashion item classification."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model built and compiled.")
        return model

    def train_model(self, epochs: int = 20, batch_size: int = 64, validation_split: float = 0.2) -> None:
        """Train the model on the Fashion MNIST dataset."""
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            train_images, train_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Model training completed.")
        
        # Evaluate and visualize
        self._evaluate_and_plot(history.history, test_images, test_labels)

    def load_trained_model(self) -> None:
        """Load a pre-trained model from file."""
        try:
            self.model = load_model(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file '{self.model_path}' not found. Train the model first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def predict(self, image_path: str) -> Optional[str]:
        """Predict the fashion item from a single image."""
        if self.model is None:
            self.load_trained_model()
        
        try:
            # Load and preprocess the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            img = cv2.resize(img, (self.img_height, self.img_width))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            
            # Predict
            prediction = self.model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            result = f"{self.class_names[predicted_class]} (Confidence: {confidence:.2f})"
            logger.info(f"Predicted: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    def _evaluate_and_plot(self, history: dict, test_images: np.ndarray, test_labels: np.ndarray) -> None:
        """Evaluate the model and plot training history and confusion matrix."""
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
        
        # Predictions for confusion matrix
        y_pred = np.argmax(self.model.predict(test_images, verbose=0), axis=1)
        y_true = np.argmax(test_labels, axis=1)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("plots/confusion_matrix.png")
        plt.close()
        logger.info("Saved confusion matrix plot.")
        
        # Training History
        plt.figure(figsize=(12, 4))
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("plots/training_accuracy.png")
        plt.close()
        
        plt.figure(figsize=(12, 4))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("plots/training_loss.png")
        plt.close()
        logger.info("Saved training history plots.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fashion Item Classifier using CNN on Fashion MNIST.")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference on an image.")
    parser.add_argument("--model_path", type=str, default="models/fashion_classifier_model.h5", help="Path to save/load the model.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--image_path", type=str, help="Path to image for prediction (required in predict mode).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    classifier = FashionItemClassifier(model_path=args.model_path)
    
    if args.mode == "train":
        classifier.train_model(epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "predict":
        if not args.image_path:
            logger.error("Image path required for predict mode.")
            sys.exit(1)
        prediction = classifier.predict(args.image_path)
        if prediction:
            print(f"Prediction: {prediction}")
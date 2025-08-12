#!/usr/bin/env python3
"""
Kidney Stone Classifier - Advanced Deep Learning Solution

This script implements a state-of-the-art deep learning pipeline for classifying kidney stone images
using transfer learning with EfficientNet models, advanced augmentation, and interpretability tools.
Designed for high accuracy in medical imaging tasks with robust validation and visualization.

Author: Amirhossein jafarnezhad
Date: August 10, 2025
Version: 1.0.0
"""

import argparse
import logging
import sys
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7, EfficientNetV2L
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effnetv2_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from PIL import Image
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import mlflow
import mlflow.tensorflow
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KidneyStoneClassifier:
    def __init__(self, model_path: str = "models/kidney_stone_classifier_model.h5", 
                 dataset_dir: str = "kidney_stone_dataset", 
                 log_dir: str = "logs/", 
                 use_v2: bool = False):
        """Initialize the KidneyStoneClassifier with advanced configurations."""
        self.model_path = model_path
        self.dataset_dir = dataset_dir
        self.log_dir = log_dir
        self.use_v2 = use_v2
        self.img_height, self.img_width = 600, 600  # Optimized for EfficientNetB7/V2L
        self.batch_size = 16  # Reduced for GPU memory efficiency
        self.classes = ['Normal', 'Stone']
        self.model = None
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def download_dataset(self) -> None:
        """Download and extract the kidney stone dataset from Kaggle."""
        if not os.path.exists(self.dataset_dir):
            logger.info("Initiating dataset download from Kaggle...")
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("vivektalwar13071999/finalsplit", path=self.dataset_dir, unzip=True)
            logger.info("Dataset downloaded and extracted successfully.")
        else:
            logger.info("Dataset already exists, skipping download.")

    def load_data(self) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """Load and preprocess the kidney stone dataset with advanced augmentation."""
        train_datagen = ImageDataGenerator(
            preprocessing_function=effnetv2_preprocess if self.use_v2 else effnet_preprocess,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect',
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )

        test_datagen = ImageDataGenerator(preprocessing_function=effnetv2_preprocess if self.use_v2 else effnet_preprocess)
        test_generator = test_datagen.flow_from_directory(
            self.dataset_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        logger.info(f"Loaded data: Train {train_generator.samples}, Validation {validation_generator.samples}, Test {test_generator.samples}")
        return train_generator, validation_generator, test_generator

    def build_model(self) -> Sequential:
        """Build a state-of-the-art CNN model using EfficientNetB7 or EfficientNetV2L with fine-tuning."""
        if self.use_v2:
            base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        else:
            base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))

        # Unfreeze top layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.6),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        logger.info("Advanced CNN model built and compiled with fine-tuning.")
        return model

    def train_model(self, epochs: int = 100, validation_split: float = 0.2) -> None:
        """Train the model with MLflow tracking and advanced callbacks."""
        train_generator, validation_generator, test_generator = self.load_data()

        self.model = self.build_model()
        
        # Set up MLflow
        mlflow.set_experiment("Kidney_Stone_Classifier")
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": self.batch_size,
                "model_type": "EfficientNetV2L" if self.use_v2 else "EfficientNetB7",
                "img_size": (self.img_height, self.img_width)
            })

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                ModelCheckpoint(self.model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
                TensorBoard(log_dir=self.log_dir, histogram_freq=1, write_graph=True)
            ]
            
            # Train the model
            history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            logger.info("Model training completed with MLflow tracking.")
            
            # Log metrics
            val_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("val_accuracy", val_accuracy)
            logger.info(f"Logged validation accuracy: {val_accuracy:.4f}")
            
            # Evaluate and visualize
            self._evaluate_and_plot(history.history, test_generator)

    def load_trained_model(self) -> None:
        """Load a pre-trained model with error handling and validation."""
        try:
            self.model = load_model(self.model_path, compile=True)
            logger.info(f"Loaded model from {self.model_path} with compilation.")
            # Validate model architecture
            self.model.summary()
        except FileNotFoundError:
            logger.error(f"Model file '{self.model_path}' not found. Train the model first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def predict(self, image_path: str) -> Optional[Dict[str, float]]:
        """Predict kidney stone presence with confidence scores and visualization."""
        if self.model is None:
            self.load_trained_model()
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            img = cv2.resize(img, (self.img_height, self.img_width))
            img = img.astype('float32')
            if self.use_v2:
                img = effnetv2_preprocess(img)
            else:
                img = effnet_preprocess(img)
            img = np.expand_dims(img, axis=0)
            
            prediction = self.model.predict(img, verbose=0)[0][0]
            predicted_class = 'Stone' if prediction > 0.5 else 'Normal'
            confidence = prediction if predicted_class == 'Stone' else (1 - prediction)
            
            # Visualize prediction
            plt.figure(figsize=(6, 6))
            plt.imshow(Image.open(image_path))
            plt.title(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
            plt.axis('off')
            plt.savefig(os.path.join(self.plot_dir, f"prediction_{os.path.basename(image_path)}"))
            plt.close()
            
            result = {
                "class": predicted_class,
                "confidence": float(confidence),
                "probability_stone": float(prediction),
                "probability_normal": float(1 - prediction)
            }
            logger.info(f"Predicted: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None

    def _evaluate_and_plot(self, history: dict, test_generator: tf.keras.preprocessing.image.DirectoryIterator) -> None:
        """Evaluate the model with comprehensive metrics and advanced visualizations."""
        # Evaluate
        test_loss, test_accuracy, test_auc, test_precision, test_recall = self.model.evaluate(test_generator, verbose=0)
        logger.info(f"Test Results - Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        # Predictions for detailed metrics
        y_pred_prob = self.model.predict(test_generator, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = test_generator.classes
        
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        display(report_df.style.background_gradient(cmap='Blues'))
        logger.info("Displayed detailed classification report.")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(self.plot_dir, "confusion_matrix.png"))
        plt.close()
        logger.info("Saved confusion matrix plot.")
        
        # Training History
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        plt.savefig(os.path.join(self.plot_dir, "training_history.png"))
        plt.close()
        logger.info("Saved training history plot.")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.plot_dir, "roc_curve.png"))
        plt.close()
        logger.info("Saved ROC curve plot.")

    def explain_prediction(self, image_path: str, explainer_type: str = "shap") -> None:
        """Provide model interpretability using SHAP or LIME with high-resolution output."""
        if self.model is None:
            self.load_trained_model()
        
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_height, self.img_width))
        img = img.astype('float32')
        if self.use_v2:
            img = effnetv2_preprocess(img)
        else:
            img = effnet_preprocess(img)
        img = np.expand_dims(img, axis=0)
        
        if explainer_type == "shap":
            explainer = shap.DeepExplainer(self.model, img)
            shap_values = explainer.shap_values(img)
            shap.image_plot(shap_values, -img, show=False)
            plt.savefig(os.path.join(self.plot_dir, "shap_explanation.png"), dpi=300)
            plt.close()
            logger.info("Saved high-resolution SHAP explanation plot.")
        elif explainer_type == "lime":
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img[0], self.model.predict, top_labels=2, hide_color=0, num_samples=2000)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=True)
            plt.figure(figsize=(10, 10))
            plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            plt.title("LIME Explanation")
            plt.axis('off')
            plt.savefig(os.path.join(self.plot_dir, "lime_explanation.png"), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved high-resolution LIME explanation plot.")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with validation."""
    parser = argparse.ArgumentParser(description="Advanced Kidney Stone Classifier using Deep Learning.")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "explain"], required=True,
                        help="Mode: 'train' to train model, 'predict' for inference, 'explain' for interpretability.")
    parser.add_argument("--model_path", type=str, default="models/kidney_stone_classifier_model.h5",
                        help="Path to save/load the model.")
    parser.add_argument("--dataset_dir", type=str, default="kidney_stone_dataset",
                        help="Path to the dataset directory.")
    parser.add_argument("--use_v2", action="store_true",
                        help="Use EfficientNetV2L instead of EfficientNetB7.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--image_path", type=str,
                        help="Path to image for prediction/explanation (required in predict/explain mode).")
    parser.add_argument("--explainer_type", type=str, choices=["shap", "lime"], default="shap",
                        help="Explainer type for explain mode.")
    args = parser.parse_args()
    if args.mode in ["predict", "explain"] and not args.image_path:
        parser.error("argument --image_path is required for predict or explain modes")
    return args

if __name__ == "__main__":
    args = parse_arguments()
    classifier = KidneyStoneClassifier(model_path=args.model_path, dataset_dir=args.dataset_dir, use_v2=args.use_v2)
    
    try:
        if args.mode == "train":
            classifier.download_dataset()
            with mlflow.start_run():
                classifier.train_model(epochs=args.epochs, validation_split=0.2)
        elif args.mode == "predict":
            result = classifier.predict(args.image_path)
            if result:
                print(f"Prediction Result: {result}")
        elif args.mode == "explain":
            classifier.explain_prediction(args.image_path, args.explainer_type)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
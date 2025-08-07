\# CIFAR-10 Image Classifier



\## Overview

This intermediate Python project builds and trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes (e.g., airplane, bird, cat). It features data preprocessing, advanced model architecture with callbacks, comprehensive evaluation (accuracy, confusion matrix), training history visualization, and prediction on custom images. The project includes a CLI interface, model persistence, and robust error handling, making it an impressive portfolio piece for deep learning in computer vision.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Loads CIFAR-10, normalizes RGB images, and converts labels to categorical.

\- \*\*Model Training\*\*: Builds a deep CNN with multiple Conv2D, MaxPooling2D, Dense layers, and Dropout, trained with EarlyStopping and ModelCheckpoint.

\- \*\*Evaluation\*\*: Computes test accuracy, generates confusion matrix, and plots training/validation accuracy/loss.

\- \*\*Prediction\*\*: Classifies custom 32x32 color images.

\- \*\*CLI Interface\*\*: Supports modes for training and prediction with configurable epochs and batch size.

\- \*\*Model Persistence\*\*: Saves/loads model using HDF5.

\- \*\*Error Handling \& Logging\*\*: Comprehensive checks and detailed logs for debugging.



\## Requirements

\- Python 3.8+

\- Libraries: `tensorflow`, `matplotlib`, `seaborn`, `opencv-python`, `numpy`



Install dependencies:

```bash

pip install tensorflow matplotlib seaborn opencv-python numpy



Dataset



The CIFAR-10 dataset is automatically downloaded from TensorFlow:



50,000 training images, 10,000 test images.

10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.







How to Run



Train model:

bashpython cifar10\_classifier.py --mode train --epochs 50 --batch\_size 128



Predict on an image (prepare a 32x32 color image):

bashpython cifar10\_classifier.py --mode predict --image\_path path/to/image.jpg





Custom options:



--model\_path: Path to save/load model.

--epochs: Number of training epochs.

--batch\_size: Batch size for training.



Example Output



Training:

textINFO: Loaded CIFAR-10 dataset: Train (50000, 32, 32, 3), Test (10000, 32, 32, 3)

INFO: Model built and compiled.

INFO: Model training completed.

Test Accuracy: 0.8750, Test Loss: 0.4200



Prediction: Prediction: bird (Confidence: 0.89)



Plots saved in plots/ folder: confusion\_matrix.png, training\_accuracy.png, training\_loss.png.

Improvements and Future Work



Add data augmentation (e.g., rotation, zoom) with ImageDataGenerator.

Implement transfer learning (e.g., ResNet50) for higher accuracy.

Deploy as a web app with Flask/Streamlit for image uploads.

Add real-time classification from webcam.

Unit tests with pytest for model training and prediction.



License

MIT License


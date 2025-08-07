\# Handwritten Digit Recognizer



\## Overview

This intermediate Python project builds and trains a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset, containing 70,000 28x28 grayscale images of digits 0-9. It features data preprocessing with augmentation, an advanced CNN architecture with BatchNormalization, comprehensive evaluation (accuracy, confusion matrix), training history visualization, and prediction on custom images. The project includes a CLI interface, model persistence, and robust error handling, making it an outstanding portfolio piece for deep learning in computer vision.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Loads MNIST, normalizes images, and applies data augmentation (rotation, shift, zoom).

\- \*\*Model Training\*\*: Builds a deep CNN with Conv2D, MaxPooling2D, BatchNormalization, Dense layers, and Dropout, trained with EarlyStopping and ModelCheckpoint.

\- \*\*Evaluation\*\*: Computes test accuracy, generates confusion matrix, and plots training/validation accuracy/loss.

\- \*\*Prediction\*\*: Recognizes digits from custom 28x28 grayscale images.

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



The MNIST dataset is automatically downloaded from TensorFlow:



60,000 training images, 10,000 test images.

10 classes: digits 0 through 9.







How to Run



Train model:

bashpython handwritten\_digit\_recognizer.py --mode train --epochs 50 --batch\_size 128



Predict on an image (prepare a 28x28 grayscale image):

bashpython handwritten\_digit\_recognizer.py --mode predict --image\_path path/to/image.png





Custom options:



--model\_path: Path to save/load model.

--epochs: Number of training epochs.

--batch\_size: Batch size for training.



Example Output



Training:

textINFO: Loaded MNIST dataset: Train (60000, 28, 28, 1), Test (10000, 28, 28, 1)

INFO: Model built and compiled with BatchNormalization.

INFO: Model training completed.

Test Accuracy: 0.9850, Test Loss: 0.0450



Prediction: Prediction: 8 (Confidence: 0.97)



Plots saved in plots/ folder: confusion\_matrix.png, training\_accuracy.png, training\_loss.png.

Improvements and Future Work



Add transfer learning (e.g., pre-trained ResNet) for enhanced accuracy.

Implement real-time digit recognition from webcam.

Deploy as a web app with Flask/Streamlit for image uploads.

Add data augmentation with noise or elastic distortions.

Unit tests with pytest for model training and prediction.



License

MIT License


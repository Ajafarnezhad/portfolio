\# Age Detector



\## Overview

This intermediate Python project uses a Convolutional Neural Network (CNN) to detect age from facial images based on the UTKFace dataset. It supports training the model, predicting age from single images, and live detection via webcam using OpenCV. The project features a modular design, CLI interface, early stopping for training, and robust error handling, making it a strong portfolio piece for computer vision and deep learning.



\## Features

\- \*\*Dataset Processing\*\*: Load and preprocess UTKFace dataset for age regression.

\- \*\*Model Training\*\*: Build and train a CNN with convolutional layers, dropout, and callbacks for optimal performance.

\- \*\*Age Prediction\*\*: Predict age from a single image file.

\- \*\*Live Detection\*\*: Real-time age detection from webcam feed using Haar Cascade for face detection.

\- \*\*CLI Interface\*\*: Command-line arguments for mode selection, dataset/model paths, and training parameters.

\- \*\*Error Handling\*\*: Comprehensive checks for dataset existence, model loading, and hardware issues.

\- \*\*Logging\*\*: Detailed logs for debugging and monitoring.



\## Requirements

\- Python 3.8+

\- Libraries: `opencv-python`, `numpy`, `pandas`, `pillow`, `tensorflow`, `scikit-learn`



Install dependencies:

```bash

pip install opencv-python numpy pandas pillow tensorflow scikit-learn



Dataset



Download the UTKFace dataset from: Google Drive

Unzip and place in datasets/UTKFace/ (or specify custom path via CLI).



How to Run



Train the model:

bashpython age\_detector.py --mode train --epochs 50 --batch\_size 32



Predict age from an image:

bashpython age\_detector.py --mode predict --image\_path path/to/image.jpg



Live detection via webcam:

bashpython age\_detector.py --mode live



Press 'Esc' to quit.







Custom options:



--dataset\_dir: Path to dataset.

--model\_path: Path to save/load model.



Example Output



Training: Logs loss/MAE per epoch and saves best model.

Prediction: Predicted Age: 28.50

Live: Displays video feed with detected faces and estimated ages.



Improvements and Future Work



Add gender/ethnicity prediction (multi-output model).

Implement data augmentation with Keras ImageDataGenerator.

Deploy as a web app with Flask/Streamlit.

Use transfer learning (e.g., ResNet) for better accuracy.

Add unit tests with pytest for model evaluation.



License

MIT License


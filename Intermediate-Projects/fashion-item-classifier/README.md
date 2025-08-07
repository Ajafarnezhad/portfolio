\# Fashion Item Classifier



\## Overview

This intermediate Python project builds and trains a Convolutional Neural Network (CNN) to classify fashion items from the Fashion MNIST dataset, which includes 10 categories like T-shirt, Trouser, and Sneaker. It features data preprocessing, model training with callbacks, comprehensive evaluation (accuracy, confusion matrix), training history visualization, and image-based prediction. The project includes a CLI interface, model persistence, and robust error handling, making it a standout portfolio piece for deep learning in computer vision.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Loads Fashion MNIST, normalizes images, and reshapes for CNN.

\- \*\*Model Training\*\*: Builds a CNN with Conv2D, MaxPooling2D, Dense layers, and Dropout, trained with EarlyStopping and ModelCheckpoint.

\- \*\*Evaluation\*\*: Computes test accuracy, generates confusion matrix, and plots training/validation accuracy/loss.

\- \*\*Prediction\*\*: Classifies fashion items from custom grayscale images (28x28 pixels).

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



The Fashion MNIST dataset is automatically downloaded from TensorFlow:



60,000 training images, 10,000 test images.

10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.







How to Run



Train model:

bashpython fashion\_item\_classifier.py --mode train --epochs 30 --batch\_size 128



Predict on an image (prepare a 28x28 grayscale image):

bashpython fashion\_item\_classifier.py --mode predict --image\_path path/to/image.png





Custom options:



--model\_path: Path to save/load model.

--epochs: Number of training epochs.

--batch\_size: Batch size for training.



Example Output



Training:

textINFO: Loaded Fashion MNIST dataset: Train (60000, 28, 28, 1), Test (10000, 28, 28, 1)

INFO: Model built and compiled.

INFO: Model training completed.

Test Accuracy: 0.9050, Test Loss: 0.3250



Prediction: Prediction: Sneaker (Confidence: 0.92)



Plots saved in plots/ folder: confusion\_matrix.png, training\_accuracy.png, training\_loss.png.

Improvements and Future Work



Add data augmentation (e.g., rotation, flip) with ImageDataGenerator.

Implement transfer learning (e.g., pre-trained VGG16) for better accuracy.

Deploy as a web app with Flask/Streamlit for image uploads.

Add real-time webcam classification.

Unit tests with pytest for model training and prediction.



License

MIT License


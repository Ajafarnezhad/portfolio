# CIFAR-10 Image Classifier: Master Computer Vision with Deep Learning 🖼️✨

Welcome to the **CIFAR-10 Image Classifier**, an intermediate Python project that brings the power of Convolutional Neural Networks (CNNs) to classify 60,000 32x32 color images across 10 diverse classes (e.g., airplane, bird, cat) from the CIFAR-10 dataset. With a modular design, intuitive CLI interface, and robust error handling, this project is a standout addition to your portfolio, showcasing your skills in deep learning and computer vision.

---

## 🌟 Project Highlights
This project combines advanced CNN architecture, comprehensive data preprocessing, and insightful visualizations to deliver accurate image classification. Featuring model persistence and a user-friendly CLI, it’s perfect for demonstrating expertise in deep learning and image processing.

---

## 🚀 Features
- **Data Loading & Preprocessing**: Loads the CIFAR-10 dataset, normalizes RGB images, and converts labels to categorical format for efficient training.
- **Model Training**: Builds a deep CNN with Conv2D, MaxPooling2D, Dense layers, and Dropout, optimized with EarlyStopping and ModelCheckpoint.
- **Evaluation**: Computes test accuracy, generates a confusion matrix, and visualizes training/validation accuracy and loss curves.
- **Custom Predictions**: Classifies user-provided 32x32 color images with ease.
- **CLI Interface**: Seamlessly switch between training and prediction modes with configurable epochs and batch sizes.
- **Model Persistence**: Saves and loads models in HDF5 format for reusability.
- **Error Handling & Logging**: Robust checks and detailed logs ensure smooth operation and easy debugging.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `tensorflow`
  - `matplotlib`
  - `seaborn`
  - `opencv-python`
  - `numpy`

Install dependencies with:
```bash
pip install tensorflow matplotlib seaborn opencv-python numpy
```

---

## 📂 Dataset
- **CIFAR-10 Dataset**: Automatically loaded via `tensorflow.keras.datasets.cifar10`, containing 60,000 32x32 RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **No Manual Download Required**: The dataset is included with TensorFlow.

---

## 🎮 How to Run

### 1. Train the Model
Build and train a CNN with customizable parameters:
```bash
python cifar10_classifier.py --mode train --epochs 50 --batch_size 32
```

### 2. Predict on Custom Images
Classify a user-provided 32x32 color image:
```bash
python cifar10_classifier.py --mode predict --image_path path/to/image.jpg
```

### 3. Customize Your Workflow
- `--epochs`: Number of training epochs (e.g., `50`).
- `--batch_size`: Batch size for training (e.g., `32`).
- `--model_path`: Save/load the trained model (e.g., `models/cifar10_model.h5`).

---

## 📈 Example Output
- **Training**:
  ```
  Epoch 1/50: Loss: 1.85, Accuracy: 0.35
  ...
  INFO: Best model saved at models/cifar10_model.h5
  INFO: Test accuracy: 0.82
  ```
- **Prediction**:
  ```
  Predicted Class: Cat (Confidence: 0.92)
  ```
- **Visualizations**: Plots saved in `plots/` folder:
  - `accuracy_loss_curves.png`: Training and validation accuracy/loss over epochs.
  - `confusion_matrix.png`: Classification performance across all classes.

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Advanced Architectures**: Experiment with ResNet or EfficientNet for higher accuracy.
- **Data Augmentation**: Implement real-time data augmentation with `tensorflow.keras.preprocessing.image.ImageDataGenerator`.
- **Web App Deployment**: Build a Flask or Streamlit app for interactive image classification.
- **Transfer Learning**: Fine-tune pre-trained models like VGG16 for faster training and better performance.
- **Unit Testing**: Add `pytest` for robust validation of preprocessing and model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Unleash your computer vision skills with the **CIFAR-10 Image Classifier** and bring images to life! 🚀
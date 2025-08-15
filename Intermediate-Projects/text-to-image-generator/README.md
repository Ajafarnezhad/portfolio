# Handwritten Digit Recognizer: Unlock the Art of Digit Recognition 🖌️✨

Welcome to the **Handwritten Digit Recognizer**, an intermediate Python project that harnesses the power of Convolutional Neural Networks (CNNs) to identify handwritten digits (0-9) from the MNIST dataset, featuring 70,000 28x28 grayscale images. With advanced data augmentation, a robust CNN architecture, and a sleek CLI interface, this project is a stellar portfolio piece to showcase your expertise in deep learning and computer vision.

---

## 🌟 Project Highlights
This project combines cutting-edge CNN architecture, comprehensive data preprocessing, and insightful visualizations to achieve high-accuracy digit recognition. Featuring model persistence, error handling, and an intuitive CLI, it’s perfect for demonstrating skills in machine learning and image processing.

---

## 🚀 Features
- **Data Loading & Preprocessing**: Loads the MNIST dataset, normalizes images, and applies data augmentation (rotation, shift, zoom) for enhanced model robustness.
- **Model Training**: Builds a deep CNN with Conv2D, MaxPooling2D, BatchNormalization, Dense layers, and Dropout, optimized with EarlyStopping and ModelCheckpoint.
- **Evaluation**: Computes test accuracy, generates a confusion matrix, and visualizes training/validation accuracy and loss curves.
- **Custom Predictions**: Recognizes digits from user-provided 28x28 grayscale images.
- **CLI Interface**: Seamlessly switch between training and prediction modes with configurable epochs and batch sizes.
- **Model Persistence**: Saves and loads models in HDF5 format for easy reuse.
- **Error Handling & Logging**: Robust checks and detailed logs ensure smooth operation and efficient debugging.

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
- **MNIST Dataset**: Automatically loaded via `tensorflow.keras.datasets.mnist`, containing 70,000 28x28 grayscale images of handwritten digits (0-9).
- **No Manual Download Required**: The dataset is included with TensorFlow.

---

## 🎮 How to Run

### 1. Train the Model
Build and train a CNN with customizable parameters:
```bash
python handwritten_digit_recognizer.py --mode train --epochs 50 --batch_size 32
```

### 2. Predict on Custom Images
Recognize digits from a user-provided 28x28 grayscale image:
```bash
python handwritten_digit_recognizer.py --mode predict --image_path path/to/image.png
```

### 3. Customize Your Workflow
- `--epochs`: Number of training epochs (e.g., `50`).
- `--batch_size`: Batch size for training (e.g., `32`).
- `--model_path`: Save/load the trained model (e.g., `models/digit_recognizer_model.h5`).

---

## 📈 Example Output
- **Training**:
  ```
  Epoch 1/50: Loss: 0.25, Accuracy: 0.92
  ...
  INFO: Best model saved at models/digit_recognizer_model.h5
  INFO: Test accuracy: 0.99
  ```
- **Prediction**:
  ```
  Predicted Digit: 7 (Confidence: 0.98)
  ```
- **Visualizations**: Plots saved in `plots/` folder:
  - `accuracy_loss_curves.png`: Training and validation accuracy/loss over epochs.
  - `confusion_matrix.png`: Classification performance across all digits.

---

## 🔮 Future Enhancements
Elevate this project with these exciting ideas:
- **Advanced Architectures**: Experiment with LeNet, ResNet, or EfficientNet for improved performance.
- **Enhanced Data Augmentation**: Add advanced techniques like shearing or flipping with `tensorflow.keras.preprocessing.image.ImageDataGenerator`.
- **Web App Deployment**: Build a Flask or Streamlit app for interactive digit recognition.
- **Transfer Learning**: Fine-tune pre-trained models for faster training and better accuracy.
- **Unit Testing**: Implement `pytest` for robust validation of preprocessing and model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Unleash your computer vision skills with the **Handwritten Digit Recognizer** and bring handwritten digits to life! 🚀
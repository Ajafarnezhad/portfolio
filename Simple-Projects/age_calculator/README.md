# Age Detector: Unlock the Power of Facial Age Prediction 🧠✨

Welcome to the **Age Detector**, an exciting Python project that harnesses the power of Convolutional Neural Networks (CNNs) to predict age from facial images using the UTKFace dataset. Whether you're training a cutting-edge model, predicting ages from single images, or detecting ages in real-time via webcam, this project offers a robust and modular solution to showcase your computer vision and deep learning skills.

---

## 🌟 Project Highlights
This intermediate-level project is designed to impress, with a clean CLI interface, modular architecture, and advanced features like early stopping and real-time face detection. Perfect for your portfolio, it combines deep learning with practical computer vision applications.

---

## 🚀 Features
- **Dataset Processing**: Seamlessly load and preprocess the UTKFace dataset for accurate age regression.
- **Model Training**: Train a high-performance CNN with convolutional layers, dropout, and early stopping for optimal results.
- **Single Image Prediction**: Estimate age from any facial image with a single command.
- **Live Webcam Detection**: Real-time age detection using OpenCV and Haar Cascade for face detection.
- **CLI Interface**: Intuitive command-line controls for mode selection, dataset paths, and training parameters.
- **Robust Error Handling**: Comprehensive checks for dataset integrity, model loading, and hardware compatibility.
- **Detailed Logging**: Track progress and debug with clear, informative logs.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `pillow`
  - `tensorflow`
  - `scikit-learn`

Install dependencies with:
```bash
pip install opencv-python numpy pandas pillow tensorflow scikit-learn
```

---

## 📂 Dataset
- **UTKFace Dataset**: Download from [Google Drive](https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE).
- **Setup**: Unzip and place the dataset in `datasets/UTKFace/` or specify a custom path via CLI.

---

## 🎮 How to Run

### 1. Train the Model
Build and train your CNN model with customizable parameters:
```bash
python age_detector.py --mode train --epochs 50 --batch_size 32
```

### 2. Predict Age from an Image
Estimate age from a single image:
```bash
python age_detector.py --mode predict --image_path path/to/image.jpg
```

### 3. Live Webcam Detection
Detect ages in real-time using your webcam:
```bash
python age_detector.py --mode live
```
*Press `Esc` to exit.*

### 4. Customize Your Workflow
- `--dataset_dir`: Specify the path to the UTKFace dataset (e.g., `datasets/UTKFace/`).
- `--model_path`: Save or load your trained model (e.g., `models/age_model.h5`).

---

## 📈 Example Output
- **Training**:
  ```
  Epoch 1/50: Loss: 12.34, MAE: 8.76
  ...
  INFO: Best model saved at models/age_model.h5
  ```
- **Prediction**:
  ```
  Predicted Age: 28.50 years
  ```
- **Live Detection**: Real-time video feed with bounding boxes around faces and estimated ages displayed.

---

## 🔮 Future Enhancements
Take this project to the next level with these planned improvements:
- **Multi-Output Prediction**: Add gender and ethnicity prediction using a multi-output CNN.
- **Data Augmentation**: Boost performance with Keras `ImageDataGenerator` for image transformations.
- **Web App Deployment**: Create an interactive interface using Flask or Streamlit.
- **Transfer Learning**: Leverage pre-trained models like ResNet for enhanced accuracy.
- **Robust Testing**: Add unit tests with `pytest` for reliable model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Get started with the **Age Detector** and bring your computer vision projects to life! 🚀
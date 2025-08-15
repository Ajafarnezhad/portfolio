# Advanced Kidney Stone Classifier and Object Detection

## Overview
This cutting-edge deep learning solution provides a robust pipeline for both classifying kidney stone images and detecting kidney stones using transfer learning and object detection. It leverages EfficientNetB3, ResNet101, or DenseNet161 for classification and YOLOv8 for object detection, optimized for high accuracy in medical imaging. The project integrates advanced data augmentation, comprehensive evaluation metrics (accuracy, precision, recall, F1, AUROC), and visualization tools. With GPU-optimized performance on A100, it ensures scalability and efficiency, making it ideal for a world-class medical imaging portfolio.

## Features
- **Transfer Learning for Classification**: Utilizes EfficientNetB3, ResNet101, or DenseNet161 with fine-tuned layers for superior feature extraction in kidney stone classification.
- **Object Detection with YOLOv8**: Employs YOLOv8 for precise kidney stone localization in images, with bounding box visualization.
- **Advanced Augmentation**: Implements random flips, rotations, color jitter, and resized crops to enhance model robustness.
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1, AUROC, confusion matrix, and classification reports for classification, alongside bounding box visualizations for detection.
- **Visualization**: Generates high-resolution plots for training history, confusion matrix, and detected bounding boxes using Matplotlib and Seaborn.
- **GPU Optimization**: Optimized for A100 GPU with a minimum batch size of 50, using AMP (Automatic Mixed Precision) for faster training.
- **Model Saving**: Saves best-performing models with timestamps for both classification and detection pipelines.
- **Error Handling & Logging**: Robust exception management and detailed logging for operational transparency.
- **Scalability**: Supports large datasets via Kaggle integration and efficient DataLoader configurations.

## Requirements
- **Python**: 3.8+
- **Libraries**: `torch`, `torchvision`, `efficientnet-pytorch`, `torcheval`, `ultralytics`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `pandas`, `numpy`, `kaggle`
- **Kaggle API**: Configure with `kaggle.json` for dataset download (place in `~/.kaggle/` and run `chmod 600 ~/.kaggle/kaggle.json`).

Install dependencies:
```bash
pip install efficientnet-pytorch torch torchvision torcheval ultralytics scikit-learn matplotlib seaborn opencv-python pandas numpy kaggle
```

## Dataset
- **Source**: Kaggle dataset "Kidney Stone Classification and Object Detection" by imtkaggleteam.
- **Structure**: Contains 'Normal' and 'Stone' subdirectories for classification and an 'annotations' directory for YOLO-format object detection annotations.
- **Access**: Automatically downloaded via `kagglehub` during script execution.

## How to Run
1. **Setup Kaggle API**:
   - Place `kaggle.json` in `~/.kaggle/` and set permissions: `chmod 600 ~/.kaggle/kaggle.json`.

2. **Run the Pipeline**:
   ```bash
   python kidney_stone_classifier.py
   ```
   - Executes both classification and object detection pipelines.
   - Classification trains EfficientNetB3, ResNet101, and DenseNet161 models, evaluates them, and plots training history and confusion matrices.
   - Object detection trains a YOLOv8 model and visualizes bounding boxes for test images.

3. **Output**:
   - **Classification**:
     - Saved models: `saved_models/{model_name}_best_{timestamp}.pth`
     - Plots: Training history (loss, accuracy, AUROC) and confusion matrix for each model.
     - Metrics: Accuracy, precision, recall, F1, AUROC, and classification report.
   - **Object Detection**:
     - Saved model: `saved_models/yolov8_best_{timestamp}.pt`
     - Visualizations: Bounding box images saved in `output/` directory.

## Custom Options
- **Classification**:
  - Models: EfficientNetB3, ResNet101, DenseNet161 (configured in code).
  - `--epochs`: Number of training epochs (default: 15).
  - `--batch_size`: Batch size for training (default: 64, minimum: 50 for A100).
  - `--img_size`: Image resolution (default: 384x384).
- **Object Detection**:
  - Uses YOLOv8 with configurable `imgsz`, `batch`, and `epochs` in the script.
- **Output Directory**: Models and visualizations saved in `saved_models/` and `output/` respectively.

## Example Output
**Classification**:
```
2025-08-15 19:00:00 - INFO - Classification Dataset: Train=700, Val=150, Test=150
2025-08-15 19:10:00 - INFO - Training EFFICIENTNET model...
2025-08-15 19:20:00 - INFO - Saved best model to saved_models/efficientnet_best_20250815_192000.pth
Test Results - Accuracy: 0.9600, AUROC: 0.9750, Precision: 0.9500, Recall: 0.9650
```

**Object Detection**:
```
2025-08-15 19:30:00 - INFO - Starting Object Detection Pipeline...
2025-08-15 19:40:00 - INFO - Saved YOLOv8 model to saved_models/yolov8_best_20250815_194000.pt
```

## Artifacts
- **Plots**: Saved in `output/` (e.g., confusion_matrix.png, training_history.png) and displayed during execution.
- **Models**: Saved in `saved_models/` (e.g., `efficientnet_best_{timestamp}.pth`, `yolov8_best_{timestamp}.pt`).
- **Logs**: Printed to console with timestamps and metrics.

## Improvements and Future Work
- **Multi-Modal Integration**: Combine image data with clinical metadata for enhanced predictions.
- **Advanced Interpretability**: Add Grad-CAM for classification and improved bounding box visualization for detection.
- **Hyperparameter Tuning**: Implement Optuna for automated optimization of learning rates and model architectures.
- **Cloud Deployment**: Deploy on cloud platforms like AWS SageMaker for real-time diagnostics.
- **Regulatory Compliance**: Incorporate bias detection and fairness metrics for clinical validation.

## License
MIT License
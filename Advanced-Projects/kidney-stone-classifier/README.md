# Advanced Kidney Stone Classifier

## Overview
This pinnacle of deep learning innovation delivers a state-of-the-art solution for classifying kidney stone images using transfer learning with EfficientNetB7 or EfficientNetV2L. Engineered for unparalleled accuracy in medical imaging, it integrates advanced data augmentation, fine-tuned transfer learning, comprehensive evaluation metrics (accuracy, precision, recall, F1, AUC), and cutting-edge interpretability tools (SHAP and LIME). With MLflow for experiment tracking, TensorBoard for visualization, and GPU-optimized performance, this project represents the zenith of diagnostic AI in healthcare, ideal for a world-class portfolio.

## Features
- **Transfer Learning Excellence**: Leverages EfficientNetB7 or EfficientNetV2L with fine-tuning of top layers for superior feature extraction.
- **Advanced Augmentation**: Employs rotations, shifts, flips, brightness adjustments, and reflective filling to bolster model robustness.
- **Experiment Tracking**: Utilizes MLflow to log parameters, metrics, and artifacts, ensuring reproducibility and traceability.
- **Interpretability**: Offers high-resolution SHAP and LIME explanations, critical for medical trust and regulatory compliance.
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, F1, AUC, ROC curve, confusion matrix, and detailed classification reports.
- **Interactive Visualization**: Generates high-resolution, publication-quality plots for training history, confusion matrix, ROC curve, and prediction explanations.
- **CLI Interface**: Supports training, prediction, and explanation modes with configurable options and strict argument validation.
- **Error Handling & Logging**: Implements robust exception management and detailed logging for operational transparency.
- **Scalability & Performance**: Optimized for GPU acceleration, supports large datasets via Kaggle integration, and includes TensorBoard for performance monitoring.

## Requirements
- **Python**: 3.11+
- **Libraries**: `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `shap`, `lime`, `mlflow`, `kaggle`
- **Kaggle API**: Configure with `kaggle.json` for dataset download (place in `~/.kaggle/` and run `chmod 600 ~/.kaggle/kaggle.json`).

Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib seaborn opencv-python shap lime mlflow kaggle
Dataset

Source: Kaggle dataset "finalsplit" by vivektalwar13071999 (ID: 7794764).
Structure: Contains 'Normal' and 'Stone' image subdirectories with a balanced split.
Access: Automatically downloaded and extracted during training mode.

How to Run

Setup Kaggle API:

Place kaggle.json in ~/.kaggle/ and set permissions: chmod 600 ~/.kaggle/kaggle.json.


Train the Model:
bashpython kidney_stone_classifier.py --mode train --epochs 100 --use_v2

Monitors progress via TensorBoard: tensorboard --logdir logs/.


Predict on an Image:
bashpython kidney_stone_classifier.py --mode predict --image_path path/to/image.jpg

Explain Prediction:
bashpython kidney_stone_classifier.py --mode explain --image_path path/to/image.jpg --explainer_type lime


Custom Options

--model_path: Path to save/load the trained model (default: models/kidney_stone_classifier_model.h5).
--dataset_dir: Directory for dataset storage (default: kidney_stone_dataset).
--use_v2: Switch to EfficientNetV2L (default is EfficientNetB7).
--epochs: Number of training epochs (default: 100).
--batch_size: Batch size for training (default: 16).
--image_path: Path to input image for prediction or explanation.
--explainer_type: Choose "shap" or "lime" for interpretability (default: "shap").

Example Output

Training:
text2025-08-12 17:00:00 - INFO - Loaded data: Train 800, Validation 200, Test 1000
2025-08-12 17:05:00 - INFO - Advanced CNN model built and compiled with fine-tuning.
2025-08-12 17:15:00 - INFO - Model training completed with MLflow tracking.
Test Results - Accuracy: 0.9850, Loss: 0.0450, AUC: 0.9900, Precision: 0.9800, Recall: 0.9900

Prediction:
textPrediction Result: {'class': 'Stone', 'confidence': 0.95, 'probability_stone': 0.95, 'probability_normal': 0.05}

Explanation:
text2025-08-12 17:20:00 - INFO - Saved high-resolution LIME explanation plot.


Artifacts

Plots: Saved in plots/ directory (e.g., confusion_matrix.png, training_history.png).
Logs: Stored in logs/ directory for TensorBoard visualization.
Model: Saved as models/kidney_stone_classifier_model.h5.

Improvements and Future Work

Multi-Modal Learning: Integrate CT scan metadata with image data for multi-input models.
Cloud Deployment: Deploy on AWS SageMaker with REST API for real-time diagnostics.
Advanced Interpretability: Implement Grad-CAM and Integrated Gradients for localized insights.
Hyperparameter Optimization: Use Optuna for automated tuning of learning rates and layer sizes.
Regulatory Compliance: Add bias detection and fairness metrics for clinical validation.

License
MIT License

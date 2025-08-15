# Breast Cancer Detector: Empowering Healthcare with Machine Learning 🩺✨

Welcome to the **Breast Cancer Detector**, an intermediate Python project that leverages logistic regression to diagnose breast cancer using the Wisconsin Breast Cancer dataset. This powerful tool combines data exploration, model training, and prediction with a modular pipeline, intuitive CLI interface, and robust error handling. It’s a perfect portfolio piece to showcase your expertise in machine learning and healthcare analytics.

---

## 🌟 Project Highlights
This project delivers a comprehensive solution for breast cancer diagnosis, featuring advanced data preprocessing, insightful visualizations, and precise evaluation metrics. With model persistence and a user-friendly CLI, it’s ideal for demonstrating skills in medical data analysis and machine learning.

---

## 🚀 Features
- **Data Loading & Preprocessing**: Cleans data, handles missing values, encodes labels, and scales features using StandardScaler.
- **Data Exploration**: Visualizes correlations and diagnosis distributions with heatmaps and plots.
- **Model Training**: Builds a robust pipeline with PCA and LogisticRegression, optimized via GridSearchCV and cross-validation.
- **Evaluation Metrics**: Computes accuracy, precision, recall, F1 score, confusion matrix, and ROC curve with AUC.
- **Prediction**: Diagnoses new samples from CSV input (M: Malignant, B: Benign).
- **CLI Interface**: Easily switch between exploration, training, and prediction modes with customizable parameters.
- **Model Persistence**: Saves and loads models using joblib for seamless reuse.
- **Error Handling & Logging**: Robust checks and detailed logs ensure reliable operation and easy debugging.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `joblib`

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 📂 Dataset
- **Wisconsin Breast Cancer Dataset**: Available from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) or Kaggle.
- **Format**: CSV file (`cancer.csv`) with features like `radius_mean`, `texture_mean`, and target `diagnosis` (M: Malignant, B: Benign).
- **Setup**: Place the dataset in a folder (e.g., `data/cancer.csv`) or specify a custom path via CLI.

---

## 🎮 How to Run

### 1. Explore the Data
Uncover insights with visualizations:
```bash
python breast_cancer_detector.py --mode explore
```

### 2. Train the Model
Build and optimize a logistic regression model:
```bash
python breast_cancer_detector.py --mode train --test_size 0.15 --cv_folds 10
```

### 3. Predict on New Data
Diagnose new samples (prepare a CSV with features):
```bash
python breast_cancer_detector.py --mode predict --input_data new_data.csv
```

### 4. Customize Your Workflow
- `--data_path`: Path to the dataset (e.g., `data/cancer.csv`).
- `--model_path`: Save/load the trained model (e.g., `models/cancer_model.joblib`).

---

## 📈 Example Output
- **Exploration**:
  ```
  INFO: Generating visualizations...
  Plots saved: correlation_heatmap.png, diagnosis_distribution.png
  ```
- **Training**:
  ```
  INFO: Best parameters: {'classifier__C': 1, 'classifier__solver': 'liblinear'}
  INFO: Cross-validation F1 scores: 0.98 ± 0.01
  Accuracy: 0.99 | Precision: 0.98 | Recall: 0.99 | F1: 0.98
  ```
- **Prediction**:
  ```
  Predictions: [1 0 1 ...] (1: Malignant, 0: Benign)
  ```
- **Visualizations**: Plots saved in `plots/` folder:
  - `correlation_heatmap.png`: Feature relationships.
  - `confusion_matrix.png`: Model performance breakdown.
  - `roc_curve.png`: ROC curve with AUC score.

---

## 🔮 Future Enhancements
Elevate this project with these exciting ideas:
- **Ensemble Methods**: Integrate RandomForest or XGBoost for improved accuracy.
- **Feature Selection**: Use RFE or SHAP for interpretable feature importance.
- **Web App Deployment**: Build a Flask or Streamlit app for user-friendly predictions.
- **Multi-Dataset Support**: Extend compatibility to other medical datasets or multi-class tasks.
- **Unit Testing**: Add `pytest` for robust validation of data processing and model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Empower healthcare with the **Breast Cancer Detector** and make a difference with machine learning! 🚀
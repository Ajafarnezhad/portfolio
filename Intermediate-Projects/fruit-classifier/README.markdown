# Fruit Classifier

## Overview
This intermediate Python project classifies fruits (Apple, Mandarin, Orange, Lemon) based on physical features like mass, width, height, and color score using a Random Forest model. It includes data exploration with visualizations, model training with hyperparameter tuning and cross-validation, detailed evaluation metrics, and prediction on new data. The project features a robust CLI interface, model persistence, and comprehensive error handling, making it an excellent portfolio piece for machine learning applications in agriculture.

## Features
- **Data Loading & Preprocessing**: Handles missing values and scales features with StandardScaler.
- **Exploration**: Generates pairplots and feature distribution plots for data understanding.
- **Model Training**: Uses Pipeline with RandomForestClassifier, GridSearchCV for tuning, and cross-validation.
- **Evaluation**: Computes accuracy, classification report, and confusion matrix with visualizations.
- **Prediction**: Infers fruit types on new data from CSV.
- **CLI Interface**: Supports modes for training, prediction, and exploration with configurable parameters.
- **Model Persistence**: Saves/loads model using joblib.
- **Error Handling & Logging**: Robust checks and detailed logs for debugging.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Dataset
- The dataset (`fruit.txt`) contains features for fruits:
  - `fruit_label`: 1 (Apple), 2 (Mandarin), 3 (Orange), 4 (Lemon).
  - `fruit_name`, `fruit_subtype`: Descriptive names.
  - `mass`, `width`, `height`, `color_score`: Numerical features.
- Place `fruit.txt` in the project directory or specify via CLI.

## How to Run
1. **Explore data**:
   ```bash
   python fruit_classifier.py --mode explore
   ```
2. **Train model**:
   ```bash
   python fruit_classifier.py --mode train --test_size 0.25 --cv_folds 5
   ```
3. **Predict on new data** (prepare a CSV with `mass`, `width`, `height`, `color_score`):
   ```bash
   python fruit_classifier.py --mode predict --input_data new_data.csv
   ```

Custom options:
- `--data_path`: Path to dataset.
- `--model_path`: Path to save/load model.

## Example Output
- Training:
  ```
  INFO: Best parameters: {'classifier__max_depth': 10, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
  INFO: Cross-validation accuracy scores: 0.85 ± 0.03
  Test Accuracy: 0.87
  Classification Report:
                 precision    recall  f1-score   support
  Apple          0.85      0.90      0.87        10
  Mandarin       1.00      1.00      1.00         3
  Orange         0.80      0.75      0.77         8
  Lemon          0.90      0.88      0.89         9
  ```
- Prediction: `Predictions: ['Apple', 'Orange', 'Lemon']`

Plots saved in `plots/` folder: pairplot.png, feature_distributions.png, confusion_matrix.png.

## Improvements and Future Work
- Add image-based classification using CNNs for real-world fruit sorting.
- Implement feature engineering (e.g., volume from width/height).
- Deploy as a web app with Flask/Streamlit for user input.
- Support additional classifiers (e.g., XGBoost, SVM).
- Unit tests with pytest for data processing and model evaluation.

## License
MIT License
\# Breast Cancer Detector



\## Overview

This intermediate Python project uses logistic regression for breast cancer diagnosis based on the Wisconsin Breast Cancer dataset. It includes data exploration with visualizations, model training with hyperparameter tuning and cross-validation, advanced evaluation metrics (including ROC and confusion matrix), and prediction on new data. The project features a modular pipeline, CLI interface, and model persistence, making it a comprehensive portfolio piece for machine learning in healthcare.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Handles missing values, label encoding, and feature scaling with StandardScaler.

\- \*\*Exploration\*\*: Generates correlation heatmap and diagnosis distribution plots.

\- \*\*Model Training\*\*: Uses Pipeline with PCA and LogisticRegression, GridSearchCV for tuning, and cross-validation.

\- \*\*Evaluation\*\*: Computes accuracy, precision, recall, F1, confusion matrix, and ROC curve with AUC.

\- \*\*Prediction\*\*: Infer diagnosis on new data from CSV.

\- \*\*CLI Interface\*\*: Supports modes for training, prediction, and exploration with configurable parameters.

\- \*\*Model Persistence\*\*: Saves/loads model using joblib.

\- \*\*Error Handling \& Logging\*\*: Robust checks and detailed logs for debugging.



\## Requirements

\- Python 3.8+

\- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`



Install dependencies:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn joblib





Dataset



The dataset (cancer.csv) is the Wisconsin Breast Cancer dataset.

Columns: Features like radius\_mean, texture\_mean, etc., and target 'diagnosis' (M: Malignant, B: Benign).

Download if needed from UCI ML Repository or Kaggle.



How to Run



Explore data:

bashpython breast\_cancer\_detector.py --mode explore



Train model:

bashpython breast\_cancer\_detector.py --mode train --test\_size 0.15 --cv\_folds 10



Predict on new data (prepare a CSV with features):

bashpython breast\_cancer\_detector.py --mode predict --input\_data new\_data.csv





Custom options:



--data\_path: Path to dataset.

--model\_path: Path to save/load model.



Example Output



Training:

textINFO: Best parameters: {'classifier\_\_C': 1, 'classifier\_\_solver': 'liblinear'}

INFO: Cross-validation F1 scores: 0.98 ± 0.01

Accuracy: 0.99, Precision: 0.98, Recall: 0.99, F1: 0.98



Prediction: Predictions: \[1 0 1 ...]



Plots saved in plots/ folder: correlation\_heatmap.png, confusion\_matrix.png, roc\_curve.png.

Improvements and Future Work



Add ensemble methods (e.g., RandomForest, XGBoost) for better accuracy.

Implement feature selection with RFE or SHAP for interpretability.

Deploy as a web app with Flask/Streamlit for user input.

Add support for other datasets or multi-class classification.

Unit tests with pytest for data processing and model evaluation.



License

MIT License


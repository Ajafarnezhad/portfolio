\# Brain Cancer Classifier



\## Overview

This advanced intermediate Python project develops an automated machine learning pipeline to classify brain cancer patient outcomes (death or survival) using a Gradient Boosting Classifier. It supports data preprocessing (imputation, encoding, scaling), feature selection with SelectKBest, dimensionality reduction with UMAP, model training with hyperparameter tuning, and comprehensive evaluation (accuracy, precision, recall, F1, ROC, SHAP). The project includes a CLI interface, model persistence, and results export to Excel, making it a robust portfolio piece for biomedical machine learning.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Handles missing values with imputation, encodes categorical features, scales numerical ones.

\- \*\*Exploration\*\*: Generates correlation heatmap, event death distribution, and UMAP projection for visualization.

\- \*\*Model Training\*\*: Uses Pipeline with GradientBoostingClassifier, GridSearchCV for tuning, and cross-validation.

\- \*\*Feature Selection\*\*: Applies SelectKBest for top features.

\- \*\*Evaluation\*\*: Computes accuracy, precision, recall, F1, confusion matrix, ROC curve with AUC, and SHAP feature importance.

\- \*\*Prediction\*\*: Infers outcomes on new data from CSV.

\- \*\*CLI Interface\*\*: Supports modes for training, prediction, and exploration with configurable parameters.

\- \*\*Model Persistence\*\*: Saves/loads model using joblib.

\- \*\*Results Export\*\*: Saves metrics and predictions to Excel, copies original dataset for reference.

\- \*\*Error Handling \& Logging\*\*: Comprehensive checks and detailed logs for debugging.



\## Requirements

\- Python 3.8+

\- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `shap`, `umap-learn`



Install dependencies:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn joblib shap umap-learn

Dataset



The dataset (data.csv) contains brain cancer patient features:



Categorical: gender, TMZ, Radiology, DxWHO2007, etc.

Numerical: Age, timefordeath, time\_Recurrence, PCV\_new, etc.

Target: Event\_death (1: Death, 0: Survival).







How to Run



Explore data:

bashpython brain\_cancer\_classifier.py --mode explore



Train model:

bashpython brain\_cancer\_classifier.py --mode train --test\_size 0.25 --cv\_folds 10



Predict on new data (prepare a CSV with features):

bashpython brain\_cancer\_classifier.py --mode predict --input\_data new\_data.csv





Custom options:



--data\_path: Path to dataset.

--model\_path: Path to save/load model.

--results\_excel\_path: Path to save results.



Example Output



Training:

textINFO: Best parameters: {'classifier\_\_learning\_rate': 0.1, 'classifier\_\_max\_depth': 5, 'classifier\_\_n\_estimators': 200, 'feature\_selection\_\_k': 10}

INFO: Cross-validation F1 scores: 0.89 ± 0.03

Accuracy: 0.90, Precision: 0.91, Recall: 0.88, F1: 0.89



Prediction: Predictions (1: Death, 0: Survival): \[1 0 0 ...]



Plots and results saved in plots/ and results/ folders respectively.

Improvements and Future Work



Add support for multi-class classification (e.g., tumor stages).

Implement neural networks (e.g., CNN) for image-based features.

Deploy as a web app with Flask/Streamlit for clinical use.

Add cross-validation with stratified sampling.

Unit tests with pytest for preprocessing and model evaluation.



License

MIT License


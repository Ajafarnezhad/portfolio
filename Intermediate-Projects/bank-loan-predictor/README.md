\# Bank Loan Predictor



\## Overview

This intermediate Python project predicts bank loan approval based on the Loan Prediction dataset. It uses a Random Forest model with hyperparameter tuning, cross-validation, advanced evaluation metrics (including ROC, confusion matrix, feature importances), and data exploration visualizations. The project includes imputation for missing values, encoding, scaling, a CLI interface, model persistence, and prediction on new data, making it a comprehensive portfolio piece for machine learning in finance.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Handles missing values with imputation, encodes categorical features, scales numerical ones.

\- \*\*Exploration\*\*: Generates correlation heatmaps, loan status distributions, and boxplots for features by status.

\- \*\*Model Training\*\*: Pipeline with RandomForestClassifier, GridSearchCV for tuning, cross-validation.

\- \*\*Evaluation\*\*: Accuracy, precision, recall, F1, confusion matrix, ROC curve with AUC, feature importances plot.

\- \*\*Prediction\*\*: Infers loan approval on new data from CSV.

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



The dataset (loan.csv) contains features for loan applicants:



Categorical: Gender, Married, Dependents, Education, Self\_Employed, Property\_Area.

Numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan\_Amount\_Term, Credit\_History.

Target: Loan\_Status (Y/N).







How to Run



Explore data:

bashpython bank\_loan\_predictor.py --mode explore



Train model:

bashpython bank\_loan\_predictor.py --mode train --test\_size 0.25 --cv\_folds 10



Predict on new data (prepare a CSV with features):

bashpython bank\_loan\_predictor.py --mode predict --input\_data new\_data.csv





Custom options:



--data\_path: Path to dataset.

--model\_path: Path to save/load model.



Example Output



Training:

textINFO: Best parameters: {'classifier\_\_max\_depth': None, 'classifier\_\_min\_samples\_split': 2, 'classifier\_\_n\_estimators': 100}

INFO: Cross-validation F1 scores: 0.82 ± 0.02

Accuracy: 0.83, Precision: 0.85, Recall: 0.90, F1: 0.87



Prediction: Predictions (1: Approved, 0: Rejected): \[1 0 1 ...]



Plots saved in plots/ folder: correlation\_heatmap.png, loan\_status\_distribution.png, boxplots, confusion\_matrix.png, roc\_curve.png, feature\_importances.png.

Improvements and Future Work



Add advanced classifiers (e.g., XGBoost, LightGBM) or ensemble methods.

Implement SMOTE for handling class imbalance.

Deploy as a web app with Flask/Streamlit for user input.

Add SHAP for model interpretability.

Unit tests with pytest for preprocessing and evaluation.



License



MIT License


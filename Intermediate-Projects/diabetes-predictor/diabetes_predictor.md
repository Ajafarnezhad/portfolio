

# Diabetes Predictor

This is a Python script that uses machine learning to predict diabetes based on the Pima Indians Diabetes dataset. It employs logistic regression with data preprocessing, model evaluation, and interactive prediction capabilities.

## Features
- Preprocesses data by handling missing values and standardizing features.
- Trains a logistic regression model to predict diabetes.
- Evaluates model performance with accuracy, confusion matrix, and classification report.
- Saves the trained model and scaler for reuse.
- Allows interactive prediction for new patient data with probability estimates.
- Saves predictions to a CSV file for record-keeping.

## Installation
1. Ensure you have Python 3.x installed.
2. Install the required libraries:
   ```
   pip install pandas scikit-learn joblib numpy
   ```
3. Download the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place `diabetes.csv` in the project directory.
4. Clone the repository:
   ```
   git clone https://github.com/Ajafarnezhad/portfolio.git
   ```
5. Navigate to the project directory:
   ```
   cd portfolio/Intermediate-Projects/diabetes-predictor
   ```

## Usage
1. Ensure `diabetes.csv` is in the project directory.
2. Run the script:
   ```
   python diabetes_predictor.py
   ```
3. The script trains the model (if not already trained) and evaluates it.
4. Choose to predict diabetes for a new patient or exit.

Example interaction:
```
Welcome to Diabetes Predictor!
Model Evaluation:
Accuracy: 77.27%
Confusion Matrix:
[[85 14]
 [21 34]]
Classification Report:
              precision    recall  f1-score   support
           0       0.80      0.86      0.83       99
           1       0.71      0.62      0.66       55
    accuracy                           0.77       154
   macro avg       0.76      0.74      0.75       154
weighted avg       0.77      0.77      0.77       154

Options:
1. Predict diabetes for a new patient
2. Exit
Enter your choice (1-2): 1
Enter patient data for diabetes prediction:
Number of pregnancies: 2
Glucose level (mg/dL): 120
Blood pressure (mm Hg): 70
Skin thickness (mm): 20
Insulin level (mu U/ml): 80
BMI: 32
Diabetes pedigree function: 0.5
Age: 30

Prediction: Not Diabetic
Probability of being diabetic: 23.45%
Prediction saved to predictions.csv
```

## How It Works
- Loads the Pima Indians Diabetes dataset (`diabetes.csv`).
- Preprocesses data by replacing invalid zeros with medians and standardizing features.
- Trains a logistic regression model and evaluates it using accuracy, confusion matrix, and classification report.
- Saves the model and scaler using `joblib` for reuse.
- Allows users to input new patient data for diabetes prediction.
- Saves predictions with probabilities to a CSV file.

## Dataset
The project uses the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which includes features like:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0 = non-diabetic, 1 = diabetic)

## Improvements Ideas
- Add support for other models (e.g., Random Forest, SVM) and compare performance.
- Create a GUI using Tkinter or Streamlit for a user-friendly interface.
- Implement cross-validation for more robust model evaluation.

## Notes
- Ensure `diabetes.csv` is in the project directory before running.
- The model assumes valid numerical inputs for predictions.

This project is part of my portfolio. Check out my other projects on GitHub: [Ajafarnezhad](https://github.com/Ajafarnezhad)

License: MIT (Free to use and modify)


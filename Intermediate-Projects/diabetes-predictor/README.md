# Diabetes Predictor: Empowering Health with Machine Learning 🩺✨

Welcome to the **Diabetes Predictor**, a Python project that leverages machine learning to predict diabetes using the Pima Indians Diabetes dataset. This project combines logistic regression with robust data preprocessing, comprehensive model evaluation, and interactive prediction capabilities, making it an excellent portfolio piece for showcasing your skills in healthcare analytics and machine learning.

---

## 🌟 Project Highlights
This project offers a streamlined pipeline for diabetes prediction, featuring data preprocessing, model training, and user-friendly interactive predictions. With model persistence and detailed evaluation metrics, it’s perfect for demonstrating expertise in machine learning and medical data analysis.

---

## 🚀 Features
- **Data Preprocessing**: Handles missing values and standardizes features for optimal model performance.
- **Model Training**: Trains a logistic regression model to predict diabetes with high accuracy.
- **Evaluation Metrics**: Provides accuracy, confusion matrix, and classification report for thorough model assessment.
- **Model Persistence**: Saves the trained model and scaler using `joblib` for seamless reuse.
- **Interactive Predictions**: Allows users to input new patient data and receive diabetes predictions with probability estimates.
- **Prediction Logging**: Saves predictions to a CSV file for record-keeping and analysis.
- **Error Handling & Logging**: Ensures robust operation with comprehensive checks and detailed logs.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `numpy`

Install dependencies with:
```bash
pip install pandas scikit-learn joblib numpy
```

---

## 📂 Dataset
- **Pima Indians Diabetes Dataset**: Available from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- **Format**: CSV file (`diabetes.csv`) with features like `Glucose`, `BloodPressure`, `BMI`, and target `Outcome` (0: Non-diabetic, 1: Diabetic).
- **Setup**: Place `diabetes.csv` in the project directory or specify a custom path.

---

## 🎮 How to Install
1. Ensure Python 3.8+ is installed.
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn joblib numpy
   ```
3. Download the Pima Indians Diabetes dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place `diabetes.csv` in the project directory.
4. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   ```
5. Navigate to the project directory:
   ```bash
   cd portfolio/Intermediate-Projects/diabetes-predictor
   ```

---

## 🎯 How to Run
1. Ensure `diabetes.csv` is in the project directory.
2. Run the script:
   ```bash
   python diabetes_predictor.py
   ```
3. The script will:
   - Train the model (if not already trained) and evaluate its performance.
   - Prompt for interactive prediction or exit.

---

## 📈 Example Output
```
Welcome to Diabetes Predictor!
Model Evaluation:
Accuracy: 77.27%
Confusion Matrix:
[[85 14]
 [21 34]]
Classification Report:
              precision    recall  f1-score   support
           0       0.80      0.86      0.83        99
           1       0.71      0.62      0.66        55
    accuracy                           0.77       154
   macro avg       0.76      0.74      0.75       154
weighted avg       0.77      0.77      0.77       154
```

- **Interactive Prediction**:
  ```
  Enter patient data:
  Glucose: 120
  BloodPressure: 70
  ...
  Predicted Outcome: Non-diabetic (Probability: 0.82)
  Prediction saved to predictions.csv
  ```

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Advanced Models**: Incorporate ensemble methods like Random Forest or XGBoost for improved accuracy.
- **Feature Engineering**: Add new features or use techniques like PCA for dimensionality reduction.
- **Web App Deployment**: Build a Flask or Streamlit app for user-friendly predictions.
- **Visualization**: Add plots for feature distributions, ROC curves, or confusion matrices.
- **Unit Testing**: Implement `pytest` for robust validation of data processing and model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Transform healthcare with the **Diabetes Predictor** and make a difference with machine learning! 🚀
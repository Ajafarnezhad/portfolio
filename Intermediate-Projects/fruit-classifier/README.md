# 🌟 Fruit Classifier: Revolutionizing Fruit Identification with Machine Learning

## 🌱 Project Vision
Welcome to **Fruit Classifier**, an elegant and powerful Python-based machine learning project crafted to identify fruits—Apple, Mandarin, Orange, and Lemon—with precision and ease. By harnessing physical attributes like mass, width, height, and color score, this project employs a Random Forest model to deliver accurate classifications. With its sleek command-line interface (CLI), stunning visualizations, and robust functionality, **Fruit Classifier** is a standout portfolio piece for showcasing expertise in machine learning and agricultural innovation.

## ✨ Key Features
- **Smart Data Preprocessing**: Seamlessly handles missing values and scales features with `StandardScaler` for optimal model performance.
- **Insightful Data Exploration**: Unveils hidden patterns through vibrant pairplots and feature distribution visualizations.
- **Advanced Model Training**: Leverages a `Pipeline` with `RandomForestClassifier`, fine-tuned via `GridSearchCV` and validated with cross-validation for top-tier accuracy.
- **Comprehensive Evaluation**: Delivers detailed metrics (accuracy, precision, recall, F1-score) alongside eye-catching confusion matrix visualizations.
- **Effortless Predictions**: Classifies new data from CSV files with a streamlined prediction pipeline.
- **Intuitive CLI Experience**: Offers a user-friendly interface with modes for exploration, training, and prediction, customizable to your needs.
- **Model Persistence**: Saves and loads models effortlessly using `joblib` for future use.
- **Robust Error Handling & Logging**: Ensures smooth operation with meticulous error checks and detailed logs for transparency.

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.8 or higher
- **Dependencies**: A curated set of libraries to power your project:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib

Install them with a single command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Dataset Overview
The `fruit.txt` dataset is your key to unlocking fruit classification:
- **fruit_label**: Numeric identifier (1: Apple, 2: Mandarin, 3: Orange, 4: Lemon).
- **fruit_name**, **fruit_subtype**: Descriptive names for fruit types and subtypes.
- **mass**, **width**, **height**, **color_score**: Numerical features capturing physical characteristics.

Place `fruit.txt` in the project directory or specify its path via the CLI.

## 🎉 How to Use

### 1. Explore the Data
Dive into the dataset with stunning visualizations:
```bash
python fruit_classifier.py --mode explore
```

### 2. Train the Model
Build a high-performance model with customizable options:
```bash
python fruit_classifier.py --mode train --test_size 0.25 --cv_folds 5
```

### 3. Predict with Ease
Classify new fruits from a CSV file (with `mass`, `width`, `height`, `color_score`):
```bash
python fruit_classifier.py --mode predict --input_data new_data.csv
```

### Customizable CLI Options
- `--data_path`: Path to your dataset (default: `fruit.txt`).
- `--model_path`: Where to save/load your model (default: `fruit_classifier_model.joblib`).
- `--test_size`: Test data proportion (default: 0.25).
- `--cv_folds`: Number of cross-validation folds (default: 5).

## 📊 Sample Output

### Training Output
```
🌟 Loading dataset from fruit.txt
🔍 Preprocessing complete!
⚙️ Best parameters: {'classifier__max_depth': 10, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
📈 Cross-validation accuracy: 0.85 ± 0.03
✅ Test set accuracy: 0.87
📋 Classification Report:
               precision    recall  f1-score   support
Apple          0.85      0.90      0.87        10
Mandarin       1.00      1.00      1.00         3
Orange         0.80      0.75      0.77         8
Lemon          0.90      0.88      0.89         9
```

### Prediction Output
```
🎉 Predictions: ['Apple', 'Orange', 'Lemon']
```

### Visualizations
Find stunning plots (pairplot.png, feature_distributions.png, confusion_matrix.png) in the `plots/` folder, ready to impress!

## 🌈 Future Enhancements
- **Image-Based Classification**: Integrate CNNs for cutting-edge fruit sorting with images.
- **Feature Engineering**: Add derived features like fruit volume for enhanced accuracy.
- **Web App Deployment**: Transform into an interactive app with Flask or Streamlit.
- **Expanded Classifiers**: Experiment with XGBoost, SVM, and more for diverse modeling options.
- **Unit Testing**: Implement `pytest` for robust data and model validation.

## 📜 License
This project is proudly licensed under the **MIT License**, fostering open collaboration and innovation.

---

🌟 **Fruit Classifier**: Where machine learning meets the vibrant world of fruits! 🌟
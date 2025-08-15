# Heart Disease Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Heart+Disease+Prediction)  
*Advancing early diagnosis with machine learning and interpretable visualizations*

## 📖 Project Overview

This project develops a state-of-the-art machine learning pipeline to predict heart disease using the UCI Heart Disease dataset. By integrating Random Forest, XGBoost, and Logistic Regression models with SHAP explanations, it achieves high accuracy and interpretability. Interactive Plotly visualizations highlight key predictors and model performance, making it ideal for clinical stakeholders and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Predict Heart Disease**: Build accurate models for early diagnosis.
- **Identify Key Biomarkers**: Highlight critical features (e.g., chest pain, cholesterol) driving predictions.
- **Provide Interpretable Visuals**: Create interactive visualizations for clinical and academic audiences.
- **Support Clinical Decision-Making**: Offer insights for preventive interventions.

## 📊 Dataset Description

The dataset is sourced from the UCI Machine Learning Repository (`processed.cleveland.data`) and contains 303 records with 13 features:

- **Key Features**:
  - `age`: Age of the patient.
  - `sex`: Gender (1=male, 0=female).
  - `cp`: Chest pain type (1-4).
  - `trestbps`: Resting blood pressure.
  - `chol`: Serum cholesterol.
  - `fbs`: Fasting blood sugar (>120 mg/dl, 1=true, 0=false).
  - `restecg`: Resting ECG results (0-2).
  - `thalach`: Maximum heart rate achieved.
  - `exang`: Exercise-induced angina (1=yes, 0=no).
  - `oldpeak`: ST depression induced by exercise.
  - `slope`: Slope of peak exercise ST segment (1-3).
  - `ca`: Number of major vessels (0-3).
  - `thal`: Thalassemia (3=normal, 6=fixed defect, 7=reversible defect).
  - `target`: Binary (1=heart disease, 0=no disease).
- **Insights**:
  - Size: 303 rows, 14 columns.
  - Missing Data: Handled by imputing median for `ca` and `thal`.
  - Context: Cleveland dataset, widely used for heart disease research.

## 🛠 Methodology

The analysis is implemented in `Heart_Disease_Prediction.ipynb` with the following pipeline:

1. **Data Acquisition**:
   - Downloaded dataset from UCI repository.
   - Handled missing values and converted target to binary (0=no disease, 1=disease).

2. **Model Optimization**:
   - Trained Random Forest, XGBoost, and Logistic Regression with GridSearchCV for hyperparameter tuning.
   - Applied SMOTE to address class imbalance and scaled features with `StandardScaler`.

3. **Clinical Visualizations**:
   - Visualized feature importance to identify key biomarkers.
   - Used SHAP explanations to interpret feature impact.
   - Plotted confusion matrix and ROC curves for performance assessment.

4. **Outputs**:
   - Saved processed dataset as `processed_heart_disease_data.csv`.
   - Exported interactive visualizations as HTML (e.g., `feature_importance.html`) and SHAP plot as PNG.

## 📈 Key Results

- **Model Performance**:
  - XGBoost outperforms Random Forest and Logistic Regression, balancing sensitivity and specificity.
  - High accuracy supports reliable heart disease prediction.
- **Key Biomarkers**:
  - Chest pain type (`cp`), thalassemia (`thal`), and number of vessels (`ca`) are top predictors, based on feature importance and SHAP analysis.
- **Visualizations**:
  - Interactive bar chart of feature importance (XGBoost).
  - SHAP summary plot showing feature impact.
  - Heatmap of confusion matrix and combined ROC curves for all models.
- **Clinical Insights**:
  - Accurate prediction enables early heart disease detection, supporting preventive measures.
  - Interpretable models and visualizations aid clinical adoption.
  - Biomarkers like chest pain type offer non-invasive diagnostic potential.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imblearn`, `shap`, `plotly`
- Internet access to download the UCI dataset

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Intermediate-Projects
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   pandas==1.5.3
   numpy==1.23.5
   matplotlib==3.7.1
   seaborn==0.12.2
   scikit-learn==1.2.2
   xgboost==1.7.5
   imblearn==0.10.1
   shap==0.41.0
   plotly==5.15.0
   ```

3. The dataset is automatically downloaded from the UCI repository during notebook execution.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Heart_Disease_Prediction.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `feature_importance.html`) or view `shap_summary.png` for visualizations.

## 📋 Usage

- **For Medical Stakeholders**: Present the notebook’s visualizations and “Clinical Insights” section to highlight diagnostic accuracy and biomarker insights.
- **For Data Scientists**: Extend the analysis with additional models (e.g., SVM) or feature engineering (e.g., interaction terms).
- **For Developers**: Integrate the models into a clinical decision support system using Flask or Streamlit.

## 🔮 Future Improvements

- **Advanced Models**: Explore neural networks or gradient boosting variants for enhanced accuracy.
- **Feature Engineering**: Derive new features (e.g., cholesterol-to-age ratio).
- **Clinical Validation**: Test models on larger datasets or real-world clinical data.
- **Real-Time Diagnosis**: Develop a Streamlit app for interactive biomarker analysis.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: UCI Heart Disease dataset (Cleveland) from `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`.
- **Tools**: Built with `pandas`, `scikit-learn`, `xgboost`, `shap`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to the medical data science community for advancing heart disease research.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in medical data science and machine learning. Last updated: August 15, 2025.*
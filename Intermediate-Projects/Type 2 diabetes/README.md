# Type 2 Diabetes Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Type+2+Diabetes+Prediction)  
*Advancing early diagnosis with machine learning and interpretable visualizations*

## 📖 Project Overview

This project develops a cutting-edge machine learning pipeline to predict Type 2 diabetes using the UCI Pima Indians Diabetes dataset. By integrating Random Forest, XGBoost, and LightGBM models with SHAP explanations, it achieves high accuracy and interpretability. Interactive Plotly visualizations highlight key predictors and model performance, making it ideal for clinical stakeholders and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Predict Type 2 Diabetes**: Build accurate models for early diagnosis.
- **Identify Key Biomarkers**: Highlight critical features (e.g., Glucose, BMI) driving predictions.
- **Provide Interpretable Visuals**: Create interactive visualizations for clinical and academic audiences.
- **Support Clinical Decision-Making**: Offer insights for preventive care and interventions.

## 📊 Dataset Description

The dataset is sourced from the UCI Machine Learning Repository (`pima-indians-diabetes.data.csv`) and contains 768 records with 8 features:

- **Key Features**:
  - `Pregnancies`: Number of pregnancies.
  - `Glucose`: Plasma glucose concentration.
  - `BloodPressure`: Diastolic blood pressure.
  - `SkinThickness`: Triceps skin fold thickness.
  - `Insulin`: 2-hour serum insulin.
  - `BMI`: Body mass index.
  - `DiabetesPedigreeFunction`: Diabetes pedigree function.
  - `Age`: Age of the patient.
  - `Outcome`: Binary (1=Diabetes, 0=No Diabetes).
- **Insights**:
  - Size: 768 rows, 9 columns.
  - Missing Data: Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` treated as missing and imputed with median.
  - Context: Pima Indian population, high-risk for Type 2 diabetes.

## 🛠 Methodology

The analysis is implemented in `Type2_Diabetes_Prediction.ipynb` with the following pipeline:

1. **Data Acquisition and Preparation**:
   - Downloaded dataset from UCI repository.
   - Handled missing values (zeros) with median imputation.
   - Applied SMOTE for class imbalance and scaled features with `StandardScaler`.

2. **Model Optimization and Training**:
   - Trained Random Forest, XGBoost, and LightGBM with GridSearchCV for hyperparameter tuning.
   - Evaluated models using classification reports and confusion matrices.

3. **Diagnostic Visualizations**:
   - Visualized feature importance to identify key biomarkers.
   - Used SHAP explanations to interpret feature impact.
   - Plotted confusion matrix and ROC curves for performance assessment.

4. **Outputs**:
   - Saved processed dataset as `processed_diabetes_data.csv`.
   - Exported interactive visualizations as HTML (e.g., `feature_importance.html`) and SHAP plot as PNG.

## 📈 Key Results

- **Model Performance**:
  - LightGBM and XGBoost outperform Random Forest, with high sensitivity for diabetes detection.
  - Balanced accuracy supports reliable prediction.
- **Key Biomarkers**:
  - Glucose, BMI, and Diabetes Pedigree Function are top predictors, based on feature importance and SHAP analysis.
- **Visualizations**:
  - Interactive bar chart of feature importance (LightGBM).
  - SHAP summary plot showing feature impact.
  - Heatmap of confusion matrix and combined ROC curves for all models.
- **Clinical Insights**:
  - Accurate prediction enables early Type 2 diabetes diagnosis, supporting preventive care.
  - Interpretable models and visualizations aid clinical adoption.
  - Biomarkers like Glucose and BMI offer actionable screening targets.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`, `shap`, `plotly`
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
   lightgbm==3.3.5
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
2. Run `Type2_Diabetes_Prediction.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `feature_importance.html`) or view `shap_summary.png` for visualizations.

## 📋 Usage

- **For Medical Stakeholders**: Present the notebook’s visualizations and “Clinical Implications” section to highlight diagnostic accuracy and biomarker insights.
- **For Data Scientists**: Extend the analysis with neural networks or feature engineering (e.g., biomarker ratios).
- **For Developers**: Integrate the models into a clinical screening tool using Flask or Streamlit.

## 🔮 Future Improvements

- **Advanced Models**: Explore deep learning (e.g., neural networks) for enhanced accuracy.
- **Feature Engineering**: Derive new features (e.g., Glucose-to-BMI ratio).
- **Clinical Validation**: Test models on diverse populations or larger datasets.
- **Real-Time Screening**: Develop a Streamlit app for interactive biomarker analysis.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: UCI Pima Indians Diabetes dataset from `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`.
- **Tools**: Built with `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to the medical data science community for advancing diabetes research.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in medical data science and machine learning. Last updated: August 15, 2025.*
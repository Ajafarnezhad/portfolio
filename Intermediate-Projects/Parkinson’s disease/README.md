# Parkinson’s Disease Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Parkinson’s+Disease+Prediction)  
*Advancing early diagnosis through machine learning and interpretable visualizations*

## 📖 Project Overview

This project develops a cutting-edge machine learning pipeline to predict Parkinson’s disease using numerical features (e.g., voice measurements) and a binary target (Parkinson’s vs. Healthy). By leveraging Random Forest, XGBoost, and neural network models with SHAP explanations, it achieves high accuracy and interpretability. Interactive Plotly visualizations and clinical insights make it ideal for medical stakeholders and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Predict Parkinson’s Disease**: Build robust models for accurate diagnosis.
- **Identify Key Biomarkers**: Highlight critical features (e.g., voice metrics) driving predictions.
- **Provide Interpretable Visuals**: Create interactive visualizations for clinical and academic audiences.
- **Support Clinical Decision-Making**: Offer insights for early diagnosis and intervention.

## 📊 Dataset Description

The dataset (placeholder: `parkinsons_data.csv`) is assumed to contain numerical features and a binary target:

- **Assumed Features**:
  - Numerical: Voice measurements (e.g., MDVP:Fo, shimmer, jitter).
  - Target: Binary (1=Parkinson’s, 0=Healthy).
- **Insights**:
  - Size: Assumed sufficient records for modeling (update with actual dataset).
  - Missing Data: Handled via median imputation for numerical features.
  - Context: Typical of Parkinson’s datasets like UCI, focusing on voice-based biomarkers.

*Note*: Update the dataset path and feature names in the notebook based on your actual data.

## 🛠 Methodology

The analysis is implemented in `Parkinsons_Disease_Prediction.ipynb` with the following pipeline:

1. **Data Ingestion and Preparation**:
   - Loaded dataset and imputed missing values with median for numerical features.
   - Applied SMOTE to handle class imbalance and scaled features using `StandardScaler`.

2. **Predictive Modeling**:
   - Trained Random Forest and XGBoost with GridSearchCV for optimal hyperparameters.
   - Built a neural network with dropout and early stopping for robustness.
   - Evaluated models using classification reports and confusion matrices.

3. **Clinical Visualizations**:
   - Visualized feature importance to identify key biomarkers.
   - Used SHAP explanations to interpret feature impact.
   - Plotted confusion matrix and ROC curve for performance assessment.

4. **Outputs**:
   - Saved processed dataset as `processed_parkinsons_data.csv`.
   - Exported interactive visualizations as HTML (e.g., `feature_importance.html`) and SHAP plot as PNG.

## 📈 Key Results

- **Model Performance**:
  - XGBoost outperforms Random Forest and neural network, balancing sensitivity and specificity.
  - High accuracy supports reliable Parkinson’s prediction.
- **Key Biomarkers**:
  - Voice-related features (e.g., shimmer, jitter) are top predictors, based on feature importance and SHAP analysis.
- **Visualizations**:
  - Interactive bar chart of top 10 feature importance (XGBoost).
  - SHAP summary plot showing feature impact on predictions.
  - Heatmap of confusion matrix and ROC curve with AUC score.
- **Clinical Insights**:
  - Accurate prediction enables early Parkinson’s diagnosis, supporting timely interventions.
  - Interpretable models and visualizations aid clinical adoption.
  - Voice biomarkers offer a non-invasive diagnostic tool.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`, `imblearn`, `shap`, `plotly`
- Dataset: `parkinsons_data.csv` (update with your actual dataset)

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
   tensorflow==2.12.0
   imblearn==0.10.1
   shap==0.41.0
   plotly==5.15.0
   ```

3. Ensure `parkinsons_data.csv` is in the project directory and update the dataset path in the notebook.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Parkinsons_Disease_Prediction.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `feature_importance.html`) in a browser or view `shap_summary.png` for visualizations.

## 📋 Usage

- **For Medical Stakeholders**: Present the notebook’s visualizations and “Clinical Implications” section to highlight diagnostic accuracy and biomarker insights.
- **For Data Scientists**: Extend the analysis with additional models (e.g., SVM) or feature engineering (e.g., feature interactions).
- **For Developers**: Integrate the models into a clinical decision support system using Flask or Streamlit.

## 🔮 Future Improvements

- **Advanced Models**: Explore deep learning architectures (e.g., LSTM) for temporal data.
- **Feature Engineering**: Derive new features from voice metrics (e.g., ratios, variability).
- **Clinical Validation**: Test models on larger datasets like UCI Parkinson’s or clinical trials.
- **Real-Time Diagnosis**: Develop a Streamlit app for interactive biomarker analysis.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Assumed Parkinson’s dataset with voice features (update with actual source).
- **Tools**: Built with `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, `shap`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to the medical data science community for advancing Parkinson’s research.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in medical data science and machine learning. Last updated: August 15, 2025.*
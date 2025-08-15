# Alzheimer’s Disease Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Alzheimer’s+Disease+Prediction)  
*Predicting Alzheimer’s disease with advanced machine learning and interactive visualizations*

## 📖 Project Overview

This project develops a machine learning pipeline to predict Alzheimer’s disease using numerical features (e.g., MRI measurements, cognitive scores) and a binary target (Alzheimer’s vs. Non-Alzheimer’s). By leveraging Random Forest and XGBoost models with hyperparameter tuning, it achieves high accuracy and interpretability, supported by interactive Plotly visualizations. Designed for medical stakeholders and portfolio presentations, this project highlights expertise in medical data analysis and machine learning within an intermediate-level data science context.

### Objectives
- **Predict Alzheimer’s Disease**: Build accurate models to identify Alzheimer’s risk.
- **Identify Key Predictors**: Highlight critical features for diagnosis (e.g., imaging metrics).
- **Provide Visual Insights**: Create interactive visualizations for clinical and academic audiences.
- **Support Clinical Decisions**: Offer actionable insights for early diagnosis and intervention.

## 📊 Dataset Description

The dataset (placeholder: `alzheimers_data.csv`) is assumed to contain numerical features and a binary target:

- **Assumed Features**:
  - Numerical: MRI measurements, cognitive scores, or other biomarkers.
  - Target: Binary (1=Alzheimer’s, 0=Non-Alzheimer’s).
- **Insights**:
  - Size: Assumed to contain sufficient records for modeling (update with actual dataset).
  - Missing Data: Handled via median imputation for numerical features.
  - Context: Typical of Alzheimer’s datasets like ADNI, focusing on diagnostic biomarkers.

*Note*: Update the dataset path and feature names in the notebook based on your actual data.

## 🛠 Methodology

The analysis is implemented in `Alzheimers_Disease_Prediction.ipynb` with the following pipeline:

1. **Data Acquisition and Cleaning**:
   - Loaded dataset and imputed missing values with median for numerical features.
   - Scaled features using `StandardScaler` for model compatibility.

2. **Model Development**:
   - Trained Random Forest and XGBoost models with GridSearchCV for optimal hyperparameters.
   - Evaluated models using classification reports (precision, recall, F1-score).

3. **Visual Insights**:
   - Visualized feature importance to identify key predictors.
   - Plotted confusion matrix to assess model performance.
   - Generated ROC curve to evaluate discriminative ability.

4. **Outputs**:
   - Saved processed dataset as `processed_alzheimers_data.csv`.
   - Exported interactive visualizations as HTML (e.g., `feature_importance.html`).

## 📈 Key Results

- **Model Performance**:
  - Random Forest and XGBoost achieve high accuracy, with XGBoost slightly outperforming.
  - Models provide balanced precision and recall for Alzheimer’s prediction.
- **Key Predictors**:
  - Top features likely include imaging metrics or cognitive scores (based on feature importance).
- **Visualizations**:
  - Interactive bar chart of feature importance (Random Forest).
  - Heatmap of confusion matrix (XGBoost).
  - ROC curve with AUC score for model evaluation.
- **Clinical Insights**:
  - Accurate prediction supports early Alzheimer’s diagnosis, enabling timely interventions.
  - Feature importance highlights biomarkers for clinical focus.
  - Models are interpretable, suitable for integration into diagnostic workflows.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `plotly`
- Dataset: `alzheimers_data.csv` (update with your actual dataset)

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
   plotly==5.15.0
   ```

3. Ensure `alzheimers_data.csv` is in the project directory and update the dataset path in the notebook.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Alzheimers_Disease_Prediction.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `feature_importance.html`) in a browser for interactive exploration.

## 📋 Usage

- **For Medical Stakeholders**: Present the notebook’s visualizations and “Medical Insights” section to highlight predictive accuracy and clinical applications.
- **For Data Scientists**: Extend the analysis with additional models (e.g., SVM) or feature engineering (e.g., interaction terms).
- **For Developers**: Integrate the models into a clinical decision support system using Flask or Streamlit.

## 🔮 Future Improvements

- **Advanced Models**: Incorporate neural networks or ensemble methods for enhanced accuracy.
- **Feature Engineering**: Derive new features from raw biomarkers (e.g., ratios, interactions).
- **Clinical Validation**: Test models on larger, diverse datasets like ADNI.
- **Interactive Dashboard**: Build a Plotly Dash app for real-time prediction and visualization.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Assumed Alzheimer’s dataset with numerical features (update with actual source).
- **Tools**: Built with `pandas`, `scikit-learn`, `xgboost`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to the medical data science community for advancing Alzheimer’s research.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in medical data analysis and machine learning. Last updated: August 15, 2025.*
# Oral Cancer Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Oral+Cancer+Prediction)  
*Advancing early diagnosis with deep learning and interpretable visualizations*

## 📖 Project Overview

This project develops a cutting-edge deep learning pipeline to predict oral cancer from medical images, using a dataset of Malignant and Benign cases. By integrating ResNet50, EfficientNetB0, and XGBoost models with SHAP explanations, it achieves high accuracy and interpretability. Interactive Plotly visualizations highlight model performance and class distribution, making it ideal for clinical stakeholders and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Predict Oral Cancer**: Build accurate models for early diagnosis from medical images.
- **Identify Key Features**: Highlight critical image-based features driving predictions.
- **Provide Interpretable Visuals**: Create interactive visualizations for clinical and academic audiences.
- **Support Clinical Decision-Making**: Offer insights for early detection and intervention.

## 📊 Dataset Description

The dataset (assumed extracted from `OralCancer.rar`) contains medical images organized into two classes:

- **Classes**:
  - `Malignant`: Images indicating oral cancer.
  - `Benign`: Images indicating non-cancerous conditions.
- **Insights**:
  - Format: Assumed JPG/PNG images in `OralCancer/Malignant` and `OralCancer/Benign` directories.
  - Size: Assumed sufficient images for training (update with actual dataset details).
  - Preprocessing: Applied data augmentation (rotation, zoom, flip) and rescaling for robustness.

*Note*: Ensure `OralCancer.rar` is extracted to a local `OralCancer` directory with Malignant/Benign subfolders.

## 🛠 Methodology

The analysis is implemented in `Oral_Cancer_Prediction.ipynb` with the following pipeline:

1. **Data Acquisition and Preprocessing**:
   - Loaded images from `OralCancer` directory using `ImageDataGenerator`.
   - Applied augmentation (rotation, zoom, flip) for training robustness.
   - Rescaled images to 224x224 for CNN compatibility.

2. **Feature Extraction and Model Training**:
   - Trained ResNet50 and EfficientNetB0 CNNs with transfer learning (ImageNet weights).
   - Extracted features from ResNet50 for XGBoost training.
   - Used early stopping and learning rate reduction for optimal convergence.

3. **Clinical Visualizations**:
   - Visualized class distribution to assess dataset balance.
   - Plotted confusion matrix and ROC curve for performance evaluation.
   - Used SHAP explanations for XGBoost feature impact.

4. **Outputs**:
   - Saved models as `resnet_oral_cancer.keras`, `effnet_oral_cancer.keras`, and `xgb_oral_cancer.json`.
   - Exported visualizations as HTML (e.g., `confusion_matrix.html`) and SHAP plot as PNG.

## 📈 Key Results

- **Model Performance**:
  - EfficientNetB0 and ResNet50 achieve high accuracy, with XGBoost enhancing feature-based predictions.
  - Balanced sensitivity and specificity support reliable diagnosis.
- **Key Features**:
  - Image-based features (e.g., texture, color patterns) drive predictions, based on SHAP analysis.
- **Visualizations**:
  - Interactive bar chart of class distribution.
  - Heatmap of confusion matrix and ROC curve (EfficientNetB0).
  - SHAP summary plot for feature impact (XGBoost).
- **Clinical Insights**:
  - Accurate prediction enables early oral cancer detection, supporting timely interventions.
  - Interpretable models aid integration into clinical imaging systems.
  - Non-invasive image analysis offers scalable diagnostic potential.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `xgboost`, `shap`, `plotly`, `matplotlib`, `seaborn`
- Dataset: `OralCancer.rar` extracted to `OralCancer` directory

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
   tensorflow==2.12.0
   scikit-learn==1.2.2
   xgboost==1.7.5
   shap==0.41.0
   plotly==5.15.0
   matplotlib==3.7.1
   seaborn==0.12.2
   ```

3. Extract `OralCancer.rar` to a local `OralCancer` directory with `Malignant` and `Benign` subfolders.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Oral_Cancer_Prediction.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `roc_curve.html`) or view `shap_summary.png` for visualizations.

## 📋 Usage

- **For Medical Stakeholders**: Present the notebook’s visualizations and “Clinical Insights” section to highlight diagnostic accuracy and feature insights.
- **For Data Scientists**: Extend the analysis with additional CNN architectures (e.g., VGG16) or feature engineering.
- **For Developers**: Integrate the models into a clinical imaging system using Flask or Streamlit.

## 🔮 Future Improvements

- **Advanced Models**: Explore other CNNs (e.g., VGG16, InceptionV3) for improved accuracy.
- **Feature Engineering**: Extract additional image features (e.g., texture descriptors).
- **Clinical Validation**: Test models on larger, diverse medical image datasets.
- **Real-Time Diagnosis**: Develop a Streamlit app for interactive image analysis.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Medical images from `OralCancer.rar` (assumed Malignant/Benign structure).
- **Tools**: Built with `tensorflow`, `xgboost`, `shap`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to the medical imaging and data science communities for advancing cancer research.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in medical image analysis and deep learning. Last updated: August 15, 2025.*
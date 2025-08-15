# Insurance Fraud Detection Analysis

![Project Banner](https://via.placeholder.com/1200x200.png?text=Insurance+Fraud+Detection)  
*Detecting fraudulent insurance claims with advanced machine learning techniques*

## 📖 Project Overview

This project presents a comprehensive analysis for detecting fraudulent insurance claims using a dataset of 1,000 insurance records. By leveraging data preprocessing, exploratory data analysis (EDA), and machine learning models, the project identifies patterns and predictors of fraud. The analysis employs both traditional (Random Forest) and advanced (Deep Learning) modeling approaches to achieve robust performance, making it a valuable tool for insurance companies aiming to mitigate financial losses due to fraudulent claims.

The project is designed to be professional and presentable, with clear visualizations and actionable insights, suitable for stakeholders in the insurance industry or data science portfolios.

### Key Objectives
- **Identify Fraud Patterns**: Uncover relationships between claim characteristics (e.g., incident severity, claim amount) and fraudulent behavior.
- **Build Predictive Models**: Develop and compare machine learning models to predict fraud with high accuracy.
- **Provide Actionable Insights**: Deliver clear visualizations and findings to support decision-making in fraud detection.

## 📊 Dataset Description

The dataset (`insurance_claims.csv`) contains 1,000 records with 39 features describing insurance claims, sourced from an insurance claims dataset. Key features include:

- **Demographic Information**: Age, gender, education level, occupation, and hobbies of the insured.
- **Policy Details**: Policy number, bind date, state, coverage limit, deductible, and annual premium.
- **Incident Details**: Incident date, type (e.g., single-vehicle collision, vehicle theft), severity, and authorities contacted.
- **Financial Metrics**: Total claim amount, injury claim, property claim, vehicle claim, capital gains, and losses.
- **Target Variable**: `fraud_reported` (Y for fraudulent, N for non-fraudulent).

### Dataset Insights
- **Size**: 1,000 rows, 39 columns.
- **Missing Values**: 9.1% missing in `authorities_contacted`, imputed with "Unknown".
- **Class Imbalance**: Approximately 25% of claims are fraudulent, requiring careful handling in modeling.

For a detailed description of each feature, refer to `datainfo.pdf` in the repository.

## 🛠 Methodology

The analysis follows a structured pipeline implemented in `Insurance Fraud Detection Analysis.ipynb`:

1. **Data Preprocessing**:
   - Removed irrelevant column `_c39`.
   - Imputed missing values in `authorities_contacted` with "Unknown".
   - Encoded categorical variables using `LabelEncoder`.
   - Scaled numerical features with `StandardScaler`.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized fraud distribution by incident type and severity using bar plots.
   - Generated a correlation heatmap to identify relationships between numerical features.
   - Highlighted key predictors like total claim amount and incident severity.

3. **Modeling**:
   - **Random Forest Classifier**: Trained with 100 estimators and a max depth of 10 to predict fraud, emphasizing interpretability.
   - **Deep Learning Model**: Built a neural network with three layers (128, 64, 32 neurons), ReLU activation, and dropout (0.3) to prevent overfitting.
   - Both models were evaluated using accuracy, classification reports, and confusion matrices.

4. **Evaluation**:
   - Assessed model performance on a 30% test set with stratified sampling.
   - Visualized results with confusion matrices and feature importance plots (Random Forest).
   - Plotted training and validation accuracy/loss for the deep learning model.

## 📈 Key Results

- **EDA Insights**:
  - Fraudulent claims are more prevalent in major damage incidents and single-vehicle collisions.
  - Total claim amount, incident severity, and vehicle claim are strong predictors of fraud.

- **Model Performance**:
  - **Random Forest**: Achieved ~80-85% accuracy, with stable performance and interpretable feature importance.
  - **Deep Learning**: Comparable accuracy (~80%) with potential for improvement through hyperparameter tuning.
  - Confusion matrices show balanced performance, though class imbalance slightly impacts precision for fraudulent cases.

- **Feature Importance** (Random Forest):
  - Top predictors: `total_claim_amount`, `incident_severity`, `vehicle_claim`, `injury_claim`.

- **Visualizations**:
  - Bar plots for fraud distribution by incident type and severity.
  - Heatmap for feature correlations.
  - Feature importance plot and confusion matrices for model evaluation.
  - Training history plots for deep learning model performance.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`
- Dataset: `insurance_claims.csv` (included in the repository)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Advanced-Projects
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Create a `requirements.txt` with:
   ```
   pandas==1.5.3
   numpy==1.23.5
   matplotlib==3.7.1
   seaborn==0.12.2
   scikit-learn==1.2.2
   tensorflow==2.12.0
   ```

3. Ensure `insurance_claims.csv` is in the project directory.

### Running the Analysis
1. Open `Insurance Fraud Detection Analysis.ipynb` in Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run all cells to execute the analysis and generate visualizations.
3. Review the output for EDA plots, model performance metrics, and key insights.

## 📋 Usage

- **For Stakeholders**: Use the notebook to present fraud detection insights, focusing on the "Key Insights" section and visualizations.
- **For Data Scientists**: Extend the analysis by experimenting with hyperparameter tuning, additional models (e.g., XGBoost), or addressing class imbalance with techniques like SMOTE.
- **For Developers**: Integrate the models into a production environment using the saved `RandomForestClassifier` or `Sequential` model objects.

## 🔮 Future Improvements

- **Address Class Imbalance**: Implement oversampling (e.g., SMOTE) or class-weight adjustments to improve fraud detection.
- **Model Enhancement**: Experiment with ensemble methods or advanced neural network architectures.
- **Feature Engineering**: Derive new features, such as time since policy bind or claim-to-premium ratio.
- **Deployment**: Develop a web-based dashboard for real-time fraud detection using Flask or Streamlit.

## 📧 Contact

For questions or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: [Your Email Address]

## 🙏 Acknowledgments

- Dataset inspired by real-world insurance claim scenarios.
- Built with open-source tools: Python, Scikit-learn, TensorFlow, and Seaborn.

---

*This project is part of the [Advanced Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Advanced-Projects) portfolio by Ajafarnezhad, showcasing expertise in data science and machine learning.*
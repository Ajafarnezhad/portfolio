# Water Potability Analysis and Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Water+Potability+Analysis+and+Prediction)  
*Harnessing Machine Learning for Safe Drinking Water Classification*

## 📖 Project Overview

Access to safe drinking water is a fundamental human right and a critical global challenge. This project delivers an end-to-end machine learning pipeline to analyze and predict water potability, enabling the classification of water samples as safe (potable) or unsafe (non-potable) for consumption. Built with Python, it integrates data preprocessing, exploratory data analysis (EDA) with Plotly, model training using PyCaret, and a Streamlit app for real-time predictions. This project is ideal for environmental scientists, data scientists, and policymakers, serving as a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Classify Water Potability**: Predict whether water samples are safe for drinking based on chemical and physical properties.
- **Analyze Key Factors**: Provide insights into features affecting water quality, such as pH, hardness, and sulfates.
- **Deploy Interactive Interface**: Offer a Streamlit app for real-time predictions and data visualization.
- **Support Public Health**: Deliver actionable insights for water treatment and regulatory decision-making.

## 📊 Dataset Description

The dataset, sourced from Kaggle ([download here](https://raw.githubusercontent.com/amankharwal/Website-data/master/water_potability.csv)), contains water quality measurements for ~3,276 samples:

- **Features**:
  - `pH`: Acidity/alkalinity (ideal: 6.5–8.5).
  - `Hardness`: Calcium/magnesium content (ideal: 120–200 mg/L).
  - `Solids`: Total dissolved solids (lower is better).
  - `Chloramines`: Disinfectant levels (safe: <4 mg/L).
  - `Sulfate`: Mineral content (safe: <500 mg/L).
  - `Conductivity`: Electrical conductivity (safe: <500 µS/cm).
  - `Organic_carbon`: Organic material (safe: <25 mg/L).
  - `Trihalomethanes`: Chlorine byproducts (safe: <80 µg/L).
  - `Turbidity`: Suspended particles (safe: <5 NTU).
- **Target**:
  - `Potability`: Binary label (1 = potable, 0 = non-potable).
- **Insights**:
  - Size: ~3,276 records with missing values in `pH`, `Sulfate`, and `Trihalomethanes`.
  - Imbalance: ~60% non-potable, ~40% potable samples.
  - Preprocessing: Median imputation for missing values and SMOTE for class balancing.

## 🛠 Methodology

The project is implemented in a single Python script (`water_potability.py`):

1. **Data Processing and EDA**:
   - Fetches data from the Kaggle URL.
   - Imputes missing values using median values for `pH`, `Sulfate`, and `Trihalomethanes`.
   - Generates interactive Plotly visualizations for feature distributions, correlations, and potability distribution.

2. **Model Training**:
   - Uses SMOTE to address class imbalance.
   - Employs PyCaret to compare classification models (e.g., Random Forest, Gradient Boosting).
   - Evaluates models using accuracy, precision, recall, F1-score, and ROC-AUC.
   - Saves the trained model and feature importance visualizations.

3. **Deployment**:
   - Deploys a Streamlit app for real-time predictions.
   - Supports single-sample predictions via user inputs and batch predictions via CSV upload.
   - Displays EDA visualizations and prediction results interactively.

## 📈 Key Results

- **Model Performance**:
  - Random Forest achieved an F1-score of ~0.75 and ROC-AUC of ~0.80.
  - SMOTE effectively mitigated class imbalance, improving recall for potable samples.
- **Visualizations**:
  - Interactive Plotly dashboards for feature distributions, correlations, and feature importance.
  - Key predictors identified: `pH`, `Sulfate`, and `Hardness`.
- **Practical Insights**:
  - Accurate predictions support water treatment prioritization and public health policies.
  - Real-time interface enables rapid assessment of water safety.
  - Applicable to environmental monitoring, regulatory compliance, and community health initiatives.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `pycaret`, `plotly`, `streamlit`, `scikit-learn`, `imblearn`
- Internet access for dataset download

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
   pycaret==3.0.4
   plotly==5.15.0
   streamlit==1.22.0
   scikit-learn==1.2.2
   imblearn==0.10.1
   ```

### Running the Project
1. **Run the Pipeline**:
   ```bash
   python water_potability.py
   ```
   This:
   - Fetches and preprocesses the dataset (`outputs/processed_water_data.csv`).
   - Generates EDA visualizations (`outputs/eda_plots.html`, `correlation_plot.html`, `potability_distribution.html`, `feature_importance.html`).
   - Trains and saves the model (`outputs/water_potability_model.pkl`).
   - Launches the Streamlit app.

2. **Access the Streamlit App**:
   - Navigate to `http://localhost:8501` to input water quality parameters or upload a CSV for batch predictions.

3. **View Visualizations**:
   - Open HTML files in the `outputs` directory in a browser for interactive EDA insights.

## 📋 Usage

- **For Environmental Scientists**: Use the Streamlit app to assess water safety and guide treatment strategies.
- **For Data Scientists**: Extend the pipeline with additional models (e.g., XGBoost) or features (e.g., microbial data).
- **For Policymakers**: Leverage predictions and visualizations to inform water safety regulations and public health campaigns.
- **For Developers**: Deploy the Streamlit app on cloud platforms (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Feature Expansion**: Incorporate microbial or heavy metal data for enhanced predictions.
- **Multi-Class Classification**: Categorize water into multiple safety levels (e.g., high, medium, low risk).
- **Real-Time Monitoring**: Integrate IoT sensor data for continuous water quality analysis.
- **Geospatial Analysis**: Map water quality trends across regions using GIS integration.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Data Source**: Kaggle water potability dataset.
- **Tools**: Built with `pycaret`, `plotly`, `streamlit`, `pandas`, `scikit-learn`, and `imblearn`.
- **Inspiration**: Thanks to Aman Kharwal and the environmental data science community for foundational insights.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in machine learning, data visualization, and environmental analytics. Last updated: August 15, 2025.*
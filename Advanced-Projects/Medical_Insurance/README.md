# Medical Insurance Dataset Analysis

## Project Overview
This project focuses on analyzing a medical insurance dataset to predict healthcare costs and understand the impact of various demographic and financial factors on insurance expenses and premiums. The dataset includes approximately 1,338 records with attributes such as age, gender, BMI, number of children, discount eligibility, region, expenses, and premiums. The analysis is performed using a Jupyter Notebook (`Medical_Insurance.ipynb`) and leverages advanced machine learning techniques to model healthcare costs.

The repository is hosted at: [github.com/Ajafarnezhad/portfolio/Advanced-Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Advanced-Projects).

## Dataset Description
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset) and is provided in CSV format (`medical_insurance.csv`). It contains the following columns:

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| `age`               | Age of the policyholder (integer)              |
| `gender`            | Gender (male/female)                          |
| `bmi`               | Body Mass Index (float)                       |
| `children`          | Number of children covered under the policy (integer) |
| `discount_eligibility` | Whether the person is eligible for a discount (yes/no) |
| `region`            | Geographic region (e.g., southeast, northwest) |
| `expenses`          | Actual medical expenses incurred (float)       |
| `premium`           | Insurance premium charged (float)              |

### Dataset Statistics
- **Rows**: 1,338
- **Columns**: 8
- **Memory Usage**: ~264.3 KB
- **Key Metrics**:
  - Average age: ~39.2 years
  - Average BMI: ~30.7
  - Average expenses: ~$13,270
  - Average premium: ~$262.87

## Project Structure
- **`medical_insurance.csv`**: The raw dataset used for analysis.
- **`Medical_Insurance.ipynb`**: A Jupyter Notebook containing the full analysis pipeline, including data loading, preprocessing, visualization, model training, and evaluation.
- **`insurance_analysis.log`**: Log file capturing the execution details of the analysis.
- **`analysis_report.json`**: A JSON report summarizing dataset statistics, model performance, and feature importance.
- **`insurance_expenses_model_*.pkl`**: Serialized machine learning model for predicting expenses.

## Analysis Pipeline
The Jupyter Notebook (`Medical_Insurance.ipynb`) implements the following steps:
1. **Data Loading and Validation**:
   - Loads the dataset with optimized data types (e.g., `int16` for age, `float32` for expenses).
   - Validates data integrity and generates a dataset summary.
2. **Exploratory Data Analysis (EDA)**:
   - Summarizes descriptive statistics (e.g., mean, std, min, max).
   - Visualizes relationships between features and expenses using interactive Plotly charts.
3. **Data Preprocessing**:
   - Handles categorical variables (`gender`, `discount_eligibility`, `region`) using one-hot encoding.
   - Scales numerical features (`age`, `bmi`, `children`) using `StandardScaler`.
   - Removes outliers to improve model robustness.
4. **Model Development**:
   - Uses a `Pipeline` with `GradientBoostingRegressor`, `XGBoost`, and `LightGBM` in a `VotingRegressor` ensemble.
   - Applies feature selection using `SelectFromModel` and recursive feature elimination (`RFECV`).
   - Performs cross-validation to evaluate model performance.
5. **Model Evaluation**:
   - Metrics: R² score and RMSE for both test and cross-validation predictions.
   - Visualizes actual vs. predicted expenses using scatter plots.
   - Analyzes feature importance to identify key predictors of expenses.
6. **Model Serialization**:
   - Saves the trained model as a `.pkl` file for reproducibility.
7. **Reporting**:
   - Generates a comprehensive JSON report (`analysis_report.json`) with dataset summary, model performance, and feature importance.

## Key Findings
- **Feature Importance**: Factors like `discount_eligibility`, `age`, and `bmi` are significant predictors of medical expenses.
- **Model Performance**: The ensemble model achieves high R² scores and low RMSE, indicating robust predictive capability.
- **Interesting Insight**: Policyholders with discount eligibility tend to have significantly higher expenses, possibly indicating higher-risk profiles.

## Requirements
To run the analysis, install the required Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost lightgbm tensorflow joblib
```
Note: The `kaleido` package is required for exporting Plotly images. Install it with:
```bash
pip install -U kaleido
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Advanced-Projects
   ```
2. Ensure the required dependencies are installed (see above).
3. Open and run the `Medical_Insurance.ipynb` notebook in Jupyter:
   ```bash
   jupyter notebook Medical_Insurance.ipynb
   ```
4. Review the generated outputs:
   - Check `insurance_analysis.log` for execution details.
   - View `analysis_report.json` for a summary of results.
   - Inspect the saved model (`insurance_expenses_model_*.pkl`) for deployment.

## Future Improvements
- Incorporate additional features (e.g., lifestyle factors) to enhance model accuracy.
- Experiment with neural network architectures for improved predictions.
- Add interactive dashboards using Dash or Streamlit for real-time exploration.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/health-insurance-dataset).
- Built with Python, Scikit-learn, Plotly, and other open-source libraries.
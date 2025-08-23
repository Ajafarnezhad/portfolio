# Heart Disease Prediction Project 🩺✨

Welcome to the **Heart Disease Prediction Project**, a sophisticated machine learning and deep learning endeavor designed to predict heart disease risk using clinical data. This project showcases my expertise in data science, bioinformatics, and AI, combining advanced modeling techniques with impactful visualizations to address a critical healthcare challenge. Achieving an impressive **83.61% accuracy** with a neural network model, this work demonstrates my ability to deliver data-driven solutions for real-world problems. 🚀🔬

This project is part of the **Intermediate Projects** section of my [GitHub portfolio](https://github.com/Ajafarnezhad/portfolio), highlighting my commitment to advancing healthcare through technology.

---

## 🌟 Project Highlights

- **Objective**: Accurately predict heart disease presence to support early diagnosis and preventive care. 🩺
- **Methodology**: Integrates traditional machine learning algorithms with a deep learning neural network for robust classification. 🧠
- **Key Result**: Neural network model achieves **83.61% accuracy**, outperforming Random Forest, SVM, and other models. 📈
- **Impact**: Offers actionable insights for clinical decision-making and medical research, aligning with my passion for bioinformatics and AI-driven healthcare solutions.

---

## 📊 Dataset Overview

The dataset, `heart.csv`, comprises **303 patient records** with **14 clinical features**, providing a robust foundation for predictive modeling. Below is a detailed overview of the features:

| Feature      | Description                                                                 | Type       |
|--------------|-----------------------------------------------------------------------------|------------|
| `age`        | Age of the patient (years)                                                  | Numerical  |
| `sex`        | Gender (1 = male, 0 = female)                                               | Categorical|
| `cp`         | Chest pain type (0–3)                                                       | Categorical|
| `trestbps`   | Resting blood pressure (mm Hg)                                              | Numerical  |
| `chol`       | Serum cholesterol (mg/dl)                                                   | Numerical  |
| `fbs`        | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)                       | Categorical|
| `restecg`    | Resting electrocardiographic results (0–2)                                  | Categorical|
| `thalach`    | Maximum heart rate achieved                                                 | Numerical  |
| `exang`      | Exercise-induced angina (1 = yes, 0 = no)                                   | Categorical|
| `oldpeak`    | ST depression induced by exercise relative to rest                          | Numerical  |
| `slope`      | Slope of the peak exercise ST segment (0–2)                                 | Categorical|
| `ca`         | Number of major vessels (0–3) colored by fluoroscopy                        | Categorical|
| `thal`       | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)           | Categorical|
| `target`     | Heart disease presence (1 = disease, 0 = no disease)                        | Categorical|

- **Dataset Size**: 303 entries, balanced with 165 positive (heart disease) and 138 negative cases.
- **Source**: Clinical records, optimized for binary classification tasks.

---

## 🛠️ Technical Approach

This project follows a rigorous, reproducible data science pipeline:

1. **Data Preprocessing** 🧹
   - Loaded dataset using **Pandas**, ensuring data integrity with no missing values.
   - Applied **StandardScaler** (Scikit-learn) to normalize numerical features for consistent model input.
   - Split data into 80% training and 20% testing sets for robust evaluation.

2. **Exploratory Data Analysis (EDA)** 📉
   - Conducted comprehensive analysis using **Seaborn** and **Matplotlib** to visualize feature distributions and correlations.
   - Identified key predictors (`thalach`, `oldpeak`, `cp`) through correlation heatmaps and pair plots.

3. **Model Development** 🤖
   - **Machine Learning Models**:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - Support Vector Machines (SVM)
   - **Neural Network** (TensorFlow/Keras):
     - **Architecture**: Input layer (13 features), multiple dense layers with ReLU activation and dropout, sigmoid output for binary classification.
     - **Training**: Optimized with binary cross-entropy loss and Adam optimizer.
   - Models trained on 80% of data, validated on 20% test set.

4. **Model Evaluation** 📏
   - Evaluated using accuracy, precision, recall, and F1-score.
   - Visualized performance with a professional bar plot comparing algorithm accuracies.

5. **Visualization** 🎨
   - Generated a sleek bar plot using **Seaborn** to highlight the neural network’s superior performance.

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Neural Network", "Random Forest", "SVM", "Decision Tree", "Logistic Regression"],
    "datasets": [{
      "label": "Accuracy (%)",
      "data": [83.61, 80.0, 78.5, 75.0, 74.0],
      "backgroundColor": ["#4CAF50", "#2196F3", "#FFC107", "#FF5722", "#9C27B0"],
      "borderColor": ["#388E3C", "#1976D2", "#FFA000", "#D81B60", "#7B1FA2"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Accuracy (%)",
          "font": { "size": 14 }
        },
        "max": 100
      },
      "x": {
        "title": {
          "display": true,
          "text": "Algorithms",
          "font": { "size": 14 }
        }
      }
    },
    "plugins": {
      "legend": {
        "display": true,
        "position": "top",
        "labels": { "font": { "size": 12 } }
      },
      "title": {
        "display": true,
        "text": "Model Performance Comparison",
        "font": { "size": 16 }
      }
    }
  }
}
```

---

## 📈 Results & Insights

- **Neural Network Performance**: Achieved **83.61% accuracy**, outperforming traditional models.
- **Model Comparison**:
  | Algorithm           | Accuracy (%) |
  |--------------------|--------------|
  | Neural Network     | 83.61        |
  | Random Forest      | ~80.0        |
  | SVM                | ~78.5        |
  | Decision Tree      | ~75.0        |
  | Logistic Regression| ~74.0        |
- **Key Insights**:
  - Features like `thalach` (maximum heart rate), `oldpeak` (ST depression), and `cp` (chest pain type) are critical predictors.
  - Higher `thalach` values correlate with lower heart disease risk, suggesting cardiovascular fitness as a protective factor. 🏃‍♂️
  - The neural network excels at capturing complex, non-linear patterns in clinical data.

---

## 📁 Repository Structure

```
portfolio/Intermediate-Projects/Heart-Disease-Prediction/
├── heart.csv                           # Dataset file
├── Heart_disease_prediction.ipynb      # Jupyter notebook with code and analysis
├── README.md                           # This README file
└── visualizations/                     # Folder for generated plots
    └── model_comparison.png            # Bar plot of algorithm accuracies
```

---

## 🚀 Getting Started

To explore or replicate this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ajafarnezhad/portfolio.git
   cd portfolio/Intermediate-Projects/Heart-Disease-Prediction
   ```

2. **Install Dependencies**:
   Install required Python libraries (Python 3.8+ recommended):
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```

3. **Run the Jupyter Notebook**:
   Launch the notebook to explore the code, analysis, and visualizations:
   ```bash
   jupyter notebook Heart_disease_prediction.ipynb
   ```

4. **View Visualizations**:
   Check the `visualizations/` folder for the model comparison plot.

---

## 🌍 Real-World Impact

This project reflects my dedication to leveraging AI for healthcare advancements. Potential applications include:
- **Clinical Decision Support**: Assisting physicians in identifying at-risk patients. 🩺
- **Preventive Healthcare**: Guiding lifestyle interventions based on risk predictions. 🥗
- **Medical Research**: Providing data-driven insights into cardiovascular risk factors. 🔬

---

## 🔮 Future Enhancements

- **Feature Engineering**: Incorporate interaction terms and domain-specific features to enhance model performance. 🛠️
- **Hyperparameter Tuning**: Optimize neural network architecture using grid search or Bayesian optimization. ⚙️
- **Model Deployment**: Develop a web-based application for real-time heart disease risk assessment. 🌐
- **Dataset Expansion**: Integrate larger, diverse datasets to improve model generalizability. 🌍

---

## 🙌 Acknowledgments

- **Dataset**: Clinical records enabling robust predictive modeling.
- **Tools**: Powered by **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **TensorFlow**, and **Scikit-learn**.
- **Inspiration**: Driven by a mission to advance healthcare through data science and bioinformatics. ❤️

---

## 📬 Connect with Me

I’m eager to collaborate, discuss data science, or explore opportunities in AI and bioinformatics:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **LinkedIn**: [amiraijd](https://linkedin.com/in/amiraijd)
- **Email**: [aiamirjd@gmail.com](mailto:aiamirjd@gmail.com)

Explore more of my work in my [GitHub portfolio](https://github.com/Ajafarnezhad/portfolio), where I showcase projects at the intersection of data science, bioinformatics, and AI.

---

*Crafted with precision and a passion for transforming healthcare through data-driven innovation.* 💻❤️
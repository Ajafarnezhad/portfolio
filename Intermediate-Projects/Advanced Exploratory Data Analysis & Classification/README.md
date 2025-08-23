# 🍛 Indian Cuisine Insights: Advanced Exploratory Data Analysis & Classification

## 🌟 Project Vision
Embark on a culinary journey through Indian cuisine with the **Indian Cuisine Insights** project, a sophisticated Python-based application that performs Exploratory Data Analysis (EDA) and machine learning classification on a dataset of Indian dishes. Leveraging pandas for data processing, scikit-learn for Random Forest classification, and Recharts for interactive visualizations, this project uncovers patterns in dish characteristics like flavor, diet, course, state, and region. With a modular design, robust error handling, and a standalone HTML report, this project is a premier showcase of data science and visualization skills, ideal for portfolios aiming for international standards.

## ✨ Core Features
- **Automated Data Processing** 📥: Loads and preprocesses the Indian Food dataset, handling missing values and inconsistencies.
- **Exploratory Data Analysis (EDA)** 🔍: Analyzes distributions of course, flavor, diet, state, and region with statistical summaries.
- **Machine Learning Classification** 🧠: Uses a Random Forest classifier to predict flavor profiles based on ingredients, diet, and state.
- **Interactive Visualizations** 📊: Generates a standalone HTML report with Recharts for dynamic, responsive charts.
- **Robust Error Handling & Logging** 🛡️: Ensures reliability with comprehensive checks and detailed logs.
- **Scalable CLI Interface** 🖥️: Offers intuitive commands for analysis, classification, and visualization.
- **Extensible Design** ⚙️: Supports additional datasets and advanced ML models for future enhancements.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Dependencies**: A curated suite of libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `plotly` (for local EDA, optional)
- **JavaScript Dependencies** (for HTML report):
  - React, Recharts, PapaParse (loaded via CDN in the HTML report)
- **Dataset**: The `food_menu_data.csv` file, included in the repository or downloadable from [Kaggle: Indian Food 101](https://www.kaggle.com/nehaprabhavalkar/indian-food-101).

Install Python dependencies with:
```bash
pip install pandas numpy scikit-learn plotly
```

### Dataset Spotlight
The **Indian Food 101 Dataset** powers this analysis:
- **Source**: Kaggle ([Indian Food 101](https://www.kaggle.com/nehaprabhavalkar/indian-food-101))
- **Content**: 255 Indian dishes with columns for `name`, `ingredients`, `diet`, `prep_time`, `cook_time`, `flavor_profile`, `course`, `state`, and `region`.
- **Size**: Compact yet rich, ideal for EDA and classification tasks.
- **Setup**: Place `food_menu_data.csv` in the project root or let the script download it via Kaggle API.

## 🎉 How to Use

### 1. Setup Kaggle API (Optional)
To download the dataset automatically:
1. Create a Kaggle account and download `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings).
2. Place `kaggle.json` in `~/.kaggle/` and run:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 2. Run Exploratory Data Analysis
Perform EDA to explore dish distributions and statistics:
```bash
python main.py --mode analyze
```

### 3. Train and Evaluate Classifier
Train a Random Forest model to predict flavor profiles:
```bash
python main.py --mode train
```

### 4. Generate Interactive Report
Create an HTML report with Recharts visualizations:
```bash
python main.py --mode visualize
```

### CLI Options
- `--mode`: Choose `analyze` (EDA), `train` (train and evaluate model), or `visualize` (generate HTML report) (default: `analyze`).
- `--data_path`: Path to `food_menu_data.csv` (default: `./food_menu_data.csv`).
- `--output_dir`: Directory for saving visualizations (default: `./plots`).

## 📊 Sample Output

### Analysis Output
```
🌟 Loaded Indian Food dataset (255 records)
🔍 Course Distribution:
dessert        85
main course    85
snack          74
starter        11
Name: course, dtype: int64
✅ Flavor Profile Summary:
sweet    88
spicy    79
-1       79
bitter    5
sour      4
Name: flavor_profile, dtype: int64
```

### Classification Output
```
📈 Random Forest Classifier Accuracy: 92.16%
✅ Classification Report:
              precision    recall  f1-score   support
bitter        1.00      1.00      1.00         1
sour          1.00      1.00      1.00         1
spicy         0.89      0.94      0.91        16
sweet         0.94      0.89      0.91        18
accuracy                          0.92        51
```

### Visualizations
Find the interactive report in the `plots/` folder:
- `indian_food_report.html`: Includes bar charts for course, flavor, diet, state, and region distributions, plus a pie chart for diet proportions.

## 🌈 Future Enhancements
- **Advanced ML Models** 🚀: Integrate deep learning (e.g., LSTM for ingredient text analysis) or clustering for dish grouping.
- **Real-Time Data Collection** ⚡: Scrape additional food data from culinary websites for broader analysis.
- **Web App Deployment** 🌐: Transform into a Streamlit dashboard for interactive exploration.
- **Topic Modeling** 📚: Apply LDA to extract ingredient-based themes.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of data pipelines and models.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in culinary data science.

---

🌟 **Indian Cuisine Insights**: Unlocking the flavors of India with data and AI! 🌟
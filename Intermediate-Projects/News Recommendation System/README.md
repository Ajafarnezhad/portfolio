# 📰 News Recommendation System: Personalizing News with Python

## 🌟 Project Vision
Step into the world of personalized content delivery with the **News Recommendation System**, a sophisticated Python-based application that recommends news articles tailored to user interests. By leveraging content-based filtering and cosine similarity, this project analyzes article titles to suggest relevant news, inspired by leading news platforms. With vibrant visualizations, a sleek command-line interface (CLI), and robust error handling, it’s a dazzling showcase of data science expertise, crafted to elevate your portfolio to global standards.

## ✨ Core Features
- **Seamless Data Integration** 📊: Loads and validates news article data with robust checks for integrity.
- **Exploratory Data Analysis (EDA)** 🔍: Visualizes news category distributions through interactive Plotly charts.
- **Content-Based Recommendation** 🧠: Uses TF-IDF vectorization and cosine similarity to recommend articles based on title similarity.
- **Personalized News Suggestions** 📰: Recommends top-N articles tailored to user-selected news titles.
- **Elegant CLI Interface** 🖥️: Offers intuitive commands for data exploration, recommendation generation, and visualization.
- **Robust Error Handling & Logging** 🛡️: Ensures reliability with meticulous checks and detailed logs for transparency.
- **Scalable Design** ⚙️: Supports extensible recommendation algorithms and large datasets for diverse news applications.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Dependencies**: A curated suite of libraries to power your analysis:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `plotly`

Install them with a single command:
```bash
pip install pandas numpy scikit-learn plotly
```

### Dataset Spotlight
The **News Dataset** is your key to personalized news recommendations:
- **Source**: Available at [Kaggle Microsoft News Dataset](https://www.kaggle.com/datasets/shivamb/microsoft-news-recommendation-dataset).
- **Content**: Contains news articles with columns for `ID`, `News Category`, `Title`, and `Summary`.
- **Size**: Thousands of articles, ideal for content-based recommendation systems.
- **Setup**: Download and place `News.csv` in the project directory or specify its path via the CLI.

## 🎉 How to Use

### 1. Analyze News Data
Perform EDA to explore news category distributions:
```bash
python news_recommendation.py --mode analyze --data_path News.csv
```

### 2. Recommend News Articles
Generate personalized article recommendations for a specific news title:
```bash
python news_recommendation.py --mode recommend --data_path News.csv --news_title "Walmart Slashes Prices on Last-Generation iPads"
```

### 3. Visualize Insights
Generate interactive visualizations for news categories:
```bash
python news_recommendation.py --mode visualize --data_path News.csv
```

### CLI Options
- `--mode`: Choose `analyze` (EDA), `recommend` (news recommendations), or `visualize` (plot generation) (default: `analyze`).
- `--data_path`: Path to the dataset (default: `News.csv`).
- `--news_title`: News article title for recommendations (default: `Walmart Slashes Prices on Last-Generation iPads`).
- `--top_n`: Number of recommendations to return (default: 10).
- `--output_dir`: Directory for saving visualizations (default: `./plots`).

## 📊 Sample Output

### Analysis Output
```
🌟 Loaded news dataset (100,000 records)
🔍 Top Category: News (45% of articles)
✅ Key Insight: Lifestyle and Health categories dominate weekend readership
```

### Recommendation Output
```
📈 Recommendations for 'Walmart Slashes Prices on Last-Generation iPads':
1. Walmart's Black Friday 2019 ad: the best deals...
2. Walmart Black Friday 2019 deals unveiled: Huge...
3. Consumer prices rise most in 7 months on higher...
...
```

### Visualizations
Find interactive plots in the `plots/` folder:
- `news_categories.png`: Bar chart of news category distributions.

## 🌈 Future Enhancements
- **Collaborative Filtering** 🚀: Integrate user reading behavior for hybrid recommendation systems.
- **Summary-Based Recommendations** 📚: Incorporate article summaries for more granular suggestions.
- **Web App Deployment** 🌐: Transform into an interactive dashboard with Streamlit for user-friendly exploration.
- **Real-Time Recommendations** ⚡: Enable live recommendations based on real-time news feeds.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of data processing and recommendation logic.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in content analytics.

---

🌟 **News Recommendation System**: Where data science delivers personalized news! 🌟
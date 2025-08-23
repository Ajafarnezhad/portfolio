# 📱 Google Play Store Sentiment Analysis: Decoding User Feedback with Machine Learning

## 🌟 Project Vision
Step into the world of Natural Language Processing (NLP) with the **Google Play Store Sentiment Analysis** project, a sophisticated Python-based application that classifies user reviews into positive, negative, or neutral sentiments. By leveraging VADER for initial sentiment scoring and a Naive Bayes classifier for enhanced accuracy, this system analyzes user feedback from the Google Play Store to uncover insights into app performance and user satisfaction. With interactive Plotly visualizations, a robust CLI interface, and automated dataset downloading, this project is a standout addition to any data science portfolio, showcasing expertise in NLP and sentiment analysis.

## ✨ Core Features
- **Automated Data Acquisition** 📥: Downloads the Google Play Store reviews dataset from Kaggle using the `kaggle` API.
- **Exploratory Data Analysis (EDA)** 🔍: Visualizes sentiment distributions and key terms using interactive Plotly charts and word clouds.
- **Sentiment Classification** 🧠: Combines VADER sentiment analysis with a Naive Bayes classifier for accurate sentiment prediction.
- **Interactive Predictions** 💬: Allows users to input review text for real-time sentiment classification.
- **Elegant CLI Interface** 🖥️: Offers intuitive commands for data analysis, model training, prediction, and visualization.
- **Robust Error Handling & Logging** 🛡️: Ensures reliability with comprehensive checks and detailed logs for transparency.
- **Scalable Design** ⚙️: Supports extensible NLP models and large-scale datasets for diverse sentiment analysis applications.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **Kaggle API**: Requires a Kaggle account and API token (`kaggle.json`) in `~/.kaggle/`.
- **Dependencies**: A curated suite of libraries to power your analysis:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `nltk`
  - `vaderSentiment`
  - `plotly`
  - `wordcloud`
  - `matplotlib`
  - `kaggle`

Install them with a single command:
```bash
pip install pandas numpy scikit-learn nltk vaderSentiment plotly wordcloud matplotlib kaggle
```

### Dataset Spotlight
The **Google Play Store Reviews Dataset** is the foundation of this sentiment analysis system:
- **Source**: Kaggle ([Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps)).
- **Content**: Contains columns for `App`, `Translated_Review`, `Sentiment`, `Sentiment_Polarity`, and `Sentiment_Subjectivity`.
- **Size**: ~40,000 reviews, ideal for NLP and sentiment analysis tasks.
- **Setup**: Automatically downloaded via the `kaggle` API, or provide a local CSV file with the same structure.

## 🎉 How to Use

### 1. Setup Kaggle API
1. Create a Kaggle account and download your API token (`kaggle.json`) from Kaggle’s user settings.
2. Place `kaggle.json` in `~/.kaggle/` and run:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 2. Analyze Review Data
Perform EDA to explore sentiment distributions and key terms:
```bash
python play_store_sentiment.py --mode analyze
```

### 3. Train and Evaluate Model
Train the Naive Bayes model and evaluate its performance:
```bash
python play_store_sentiment.py --mode train
```

### 4. Predict Sentiment
Classify the sentiment of a user-provided review:
```bash
python play_store_sentiment.py --mode predict --review "This app is amazing and easy to use!"
```

### 5. Visualize Insights
Generate interactive visualizations for sentiment trends:
```bash
python play_store_sentiment.py --mode visualize
```

### CLI Options
- `--mode`: Choose `analyze` (EDA), `train` (train and evaluate model), `predict` (sentiment prediction), or `visualize` (plot generation) (default: `analyze`).
- `--data_path`: Path to a local dataset (optional; downloads Kaggle dataset if not provided).
- `--review`: Review text for prediction (required for `predict` mode).
- `--output_dir`: Directory for saving visualizations (default: `./plots`).

## 📊 Sample Output

### Analysis Output
```
🌟 Downloaded Google Play Store reviews dataset (37,833 records)
🔍 Sentiment Distribution:
positive    23,917
neutral      8,542
negative     5,374
Name: VADER_Sentiment, dtype: int64
✅ Dataset Summary:
       Sentiment_Polarity  Sentiment_Subjectivity
count         37,833.00           37,833.00
mean              0.18                0.49
...
```

### Prediction Output
```
📈 Predicted sentiment for review "This app is amazing and easy to use!":
Prediction: positive (Probability: 92.15%)
```

### Model Evaluation Output
```
📈 Model Accuracy: 89.67%
✅ Classification Report:
              precision    recall  f1-score   support
negative       0.87      0.82     0.84      1075
neutral        0.88      0.90     0.89      1708
positive       0.91      0.92     0.91      4784
accuracy                           0.90      7567
```

### Visualizations
Find interactive plots in the `plots/` folder:
- `sentiment_distribution.html`: Bar chart of sentiment distribution across reviews.
- `wordcloud_reviews.png`: Word cloud of frequent terms in processed reviews.

## 🌈 Future Enhancements
- **Advanced Models** 🚀: Integrate deep learning models like LSTM or BERT for improved sentiment classification.
- **Real-Time Scraping** ⚡: Incorporate `google-play-scraper` for live review data collection.
- **Web App Deployment** 🌐: Transform into a Streamlit dashboard for interactive sentiment analysis.
- **Topic Modeling** 📚: Apply LDA to identify key themes in reviews.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of NLP pipelines.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in sentiment analysis.

---

🌟 **Google Play Store Sentiment Analysis**: Where NLP unlocks user insights! 🌟
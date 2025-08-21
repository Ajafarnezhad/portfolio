# Fake News Detection with Machine Learning

![Project Banner](https://via.placeholder.com/1200x200.png?text=Fake+News+Detection)  
*Combating Misinformation with a Machine Learning-Based Detection System*

## 📖 Project Overview

This project develops a machine learning-based system to detect fake news by analyzing news headlines. Leveraging a dataset of labeled news articles, the system uses TF-IDF vectorization and a Multinomial Naive Bayes classifier to distinguish between real and fake news. The pipeline includes data preprocessing, model training, evaluation, and prediction, with results suitable for stakeholders combating misinformation and for showcasing intermediate-level data science skills in a portfolio.

### Objectives
- **Detect Fake News**: Build a model to classify news headlines as real or fake.
- **Analyze Headline Patterns**: Identify linguistic features distinguishing fake news from real news.
- **Provide Actionable Insights**: Enable stakeholders to combat misinformation effectively.
- **Showcase Data Science Skills**: Demonstrate proficiency in text processing, machine learning, and model evaluation.

## 📊 Dataset Description

The dataset (`fake_news_data.csv`) contains news articles with the following key features:

- **Key Features**:
  - `title`: Headline of the news article.
  - `label`: Binary label indicating whether the news is real (`REAL`) or fake (`FAKE`).
  - Additional columns (e.g., content) may be present but are unused in this analysis.
- **Insights**:
  - Size: Thousands of news articles (varies by dataset).
  - Missing Data: Handled by imputing empty strings for missing titles and validating labels.
  - Notable Patterns: Fake news headlines often use sensational or exaggerated language.

## 🛠 Methodology

The analysis is implemented in `fake_news_detection.py` with the following pipeline:

1. **Data Preprocessing**:
   - Loaded dataset using `pandas` and validated required columns (`title`, `label`).
   - Handled missing values by imputing empty strings for titles and mapping labels to binary (1 for `REAL`, 0 for `FAKE`).
   - Selected `title` as the feature and `label` as the target.

2. **Feature Engineering**:
   - Applied TF-IDF vectorization to convert news titles into numerical features, using a maximum of 5,000 features and English stop words.

3. **Model Training**:
   - Split data into 80% training and 20% testing sets.
   - Trained a Multinomial Naive Bayes classifier, suitable for text classification tasks.

4. **Model Evaluation**:
   - Evaluated model performance using accuracy and a detailed classification report (precision, recall, F1-score).
   - Saved the trained model and vectorizer for future use.

5. **Outputs**:
   - Saved trained model and TF-IDF vectorizer as `fake_news_model.pkl` and `tfidf_vectorizer.pkl`.
   - Generated performance metrics and example predictions for real and fake news headlines.

## 📈 Key Results

- **Model Performance**:
  - Achieved an accuracy of approximately 80.74% on the test set (example dataset).
  - Classification report provides precision, recall, and F1-score for both `FAKE` and `REAL` classes.
- **Example Predictions**:
  - Real News: "Scientists discover new species in Pacific Ocean" → Predicted `REAL`.
  - Fake News: "Aliens invade New York City with laser beams" → Predicted `FAKE`.
- **Insights**:
  - The model effectively identifies sensational language common in fake news headlines.
  - The pipeline is scalable and can be extended with additional features or models for improved accuracy.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `pickle`
- Dataset: `fake_news_data.csv` (available from sources like Kaggle or included in the repository)

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
   scikit-learn==1.2.2
   ```

3. Ensure `fake_news_data.csv` is in the project directory.

### Running the Project
1. Run the script:
   ```bash
   python fake_news_detection.py
   ```
   Modify the script to call `fake_news_detection('fake_news_data.csv')` with your dataset path.
2. View outputs in the `output` directory, including saved model and vectorizer files.
3. Test custom predictions by calling `predict_news(model, vectorizer, "your_news_title")`.

## 📋 Usage

- **For Stakeholders**: Use the model to identify potentially fake news in real-time applications or media monitoring systems.
- **For Data Scientists**: Extend the pipeline with advanced NLP techniques (e.g., BERT) or additional features like news content or metadata.
- **For Developers**: Integrate the model into a web app using Flask or Streamlit for user-friendly fake news detection.

## 🔮 Future Improvements

- **Advanced NLP Models**: Incorporate deep learning models like BERT for improved feature extraction and accuracy.
- **Additional Features**: Include news content, publication source, or temporal metadata for richer analysis.
- **Real-Time Detection**: Develop a real-time API for processing live news feeds.
- **Visualization**: Add interactive visualizations for model performance and feature importance.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Sourced from publicly available datasets (e.g., Kaggle).
- **Tools**: Built with `pandas`, `scikit-learn`, and other open-source Python libraries.
- **Inspiration**: Thanks to the data science community and Aman Kharwal for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in text classification, machine learning, and model deployment. Last updated: August 21, 2025.*
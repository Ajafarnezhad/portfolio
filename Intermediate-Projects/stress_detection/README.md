# Stress Detection from Social Media Posts

![Project Banner](https://via.placeholder.com/1200x200.png?text=Stress+Detection+from+Social+Media+Posts)  
*Machine Learning for Identifying Psychological Stress in Online Content*

## 📖 Project Overview

Mental health challenges, including stress, anxiety, and depression, are increasingly shared on social media platforms like Reddit and Instagram. This project presents a robust machine learning pipeline to detect stress in social media posts, enabling organizations to identify and support users experiencing psychological distress. Built with Python, it integrates text preprocessing, exploratory data analysis (EDA) with Plotly and WordCloud, model training using PyCaret, and a Streamlit app for real-time stress detection. This project is ideal for data scientists, mental health advocates, and social media analysts, making it a compelling addition to an intermediate-level data science portfolio.

### Objectives
- **Detect Stress in Text**: Classify social media posts as "Stress" or "No Stress" based on their content.
- **Analyze Text Patterns**: Provide insights into common words and themes in stress-related posts.
- **Deploy Interactive Interface**: Offer a Streamlit app for real-time and batch stress predictions.
- **Support Mental Health Initiatives**: Enable organizations to identify and assist stressed individuals.

## 📊 Dataset Description

The dataset, sourced from Kaggle ([download here](https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv)), contains posts from mental health-related subreddits:

- **Features**:
  - `text`: Social media post content.
  - `label`: Binary label (1 = Stress, 0 = No Stress).
- **Insights**:
  - Size: ~2,800 records (exact size may vary).
  - No missing values in `text` or `label` columns.
  - Preprocessing: Text cleaned by removing URLs, punctuation, numbers, stopwords, and applying stemming.
  - Potential imbalance: Distribution of Stress vs. No Stress labels analyzed during EDA.

## 🛠 Methodology

The project is implemented in a single Python script (`stress_detection.py`):

1. **Data Processing and EDA**:
   - Fetches data from the Kaggle URL.
   - Cleans text by removing URLs, punctuation, numbers, and stopwords, and applying stemming.
   - Generates a word cloud and label distribution visualizations using Plotly and WordCloud.

2. **Model Training**:
   - Vectorizes text using `CountVectorizer`.
   - Uses PyCaret to compare classification models (e.g., Bernoulli Naive Bayes, Random Forest) and select the best based on F1-score.
   - Evaluates models with accuracy, precision, recall, F1-score, and ROC-AUC.
   - Saves the trained model and vectorizer.

3. **Deployment**:
   - Deploys a Streamlit app for real-time stress detection.
   - Supports single-post predictions via text input and batch predictions via CSV upload.
   - Displays EDA visualizations and prediction results interactively.

## 📈 Key Results

- **Model Performance**:
  - Best model (e.g., Bernoulli Naive Bayes or Random Forest) achieves an F1-score of ~0.70–0.80.
  - Accurate classification of stress-related posts, with confidence scores for predictions.
- **Visualizations**:
  - Word cloud highlighting frequent terms in stress-related posts (e.g., "feel", "help", "anxiety").
  - Pie chart showing the distribution of Stress vs. No Stress labels.
- **Practical Insights**:
  - Enables organizations to monitor social media for stress signals and offer timely support.
  - Real-time predictions facilitate rapid intervention for mental health crises.
  - Applicable to social media analytics, mental health research, and community outreach.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `nltk`, `pycaret`, `plotly`, `wordcloud`, `streamlit`, `scikit-learn`
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
   nltk==3.8.1
   pycaret==3.0.4
   plotly==5.15.0
   wordcloud==1.9.2
   streamlit==1.22.0
   scikit-learn==1.2.2
   ```

### Running the Project
1. **Run the Pipeline**:
   ```bash
   python stress_detection.py
   ```
   This:
   - Fetches and preprocesses the dataset (`outputs/processed_stress_data.csv`).
   - Generates EDA visualizations (`outputs/wordcloud.png`, `outputs/eda_plots.html`).
   - Trains and saves the model and vectorizer (`outputs/stress_detection_model.pkl`, `outputs/vectorizer.pkl`).
   - Launches the Streamlit app.

2. **Access the Streamlit App**:
   - Navigate to `http://localhost:8501` to input text or upload a CSV for stress detection.

3. **View Visualizations**:
   - Open `outputs/eda_plots.html` in a browser for the label distribution.
   - View `outputs/wordcloud.png` for the word cloud.

## 📋 Usage

- **For Mental Health Advocates**: Use the Streamlit app to identify stress in social media posts and prioritize support efforts.
- **For Data Scientists**: Extend the pipeline with advanced NLP models (e.g., BERT) or additional features (e.g., sentiment scores).
- **For Social Media Analysts**: Leverage predictions to monitor platform-wide mental health trends.
- **For Developers**: Deploy the Streamlit app on cloud platforms (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Advanced NLP**: Incorporate transformer-based models like BERT for improved text understanding.
- **Multi-Class Classification**: Detect specific emotions (e.g., anxiety, depression) beyond binary stress.
- **Real-Time Monitoring**: Integrate social media APIs for live post analysis.
- **Sentiment Integration**: Add sentiment analysis to enhance stress detection accuracy.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Data Source**: Kaggle stress detection dataset.
- **Tools**: Built with `pycaret`, `plotly`, `wordcloud`, `streamlit`, `nltk`, and `scikit-learn`.
- **Inspiration**: Thanks to Aman Kharwal and the mental health data science community for foundational insights.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in natural language processing, machine learning, and mental health analytics. Last updated: August 15, 2025.*
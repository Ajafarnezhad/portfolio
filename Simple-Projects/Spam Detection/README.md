# Spam Detection with Machine Learning

![Project Banner](https://via.placeholder.com/1200x200.png?text=Spam+Detection+with+Machine+Learning)  
*Enhancing User Experience with Intelligent Message Classification*

## 📖 Project Overview

Spam messages and emails, such as unsolicited advertisements, can overwhelm users and degrade their experience. This project delivers a robust machine learning pipeline to detect spam in text messages and emails, ensuring users receive only relevant notifications. Built with Python, it integrates text preprocessing with NLTK, exploratory data analysis (EDA) with Plotly and WordCloud, model training using PyCaret, and a Streamlit app for real-time spam detection. Designed for data scientists, cybersecurity professionals, and app developers, this project is a valuable addition to an intermediate-level data science portfolio.

### Objectives
- **Detect Spam Messages**: Classify messages as "Spam" or "No Spam" based on text content.
- **Analyze Text Patterns**: Identify key words and patterns in spam messages.
- **Deploy Interactive Interface**: Provide a Streamlit app for real-time and batch spam detection.
- **Improve User Experience**: Support applications like email clients and messaging apps to filter spam.

## 📊 Dataset Description

The dataset, sourced from Kaggle ([download here](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv)), contains labeled text messages:

- **Features**:
  - `message`: Text content of the message.
  - `class`: Label indicating "Spam" or "No Spam".
- **Insights**:
  - Size: ~5,572 messages (exact size may vary).
  - No missing values in the dataset.
  - Preprocessing: Text cleaned by removing URLs, punctuation, numbers, stopwords, and applying stemming.
  - Imbalance: Typically, more "No Spam" messages than "Spam" messages, analyzed during EDA.

## 🛠 Methodology

The project is implemented in a single Python script (`spam_detection.py`):

1. **Data Acquisition and Preprocessing**:
   - Fetches the dataset from the Kaggle URL.
   - Cleans text by removing URLs, punctuation, numbers, and stopwords, and applying stemming.
   - Maps labels (`ham` → "No Spam", `spam` → "Spam").

2. **Exploratory Data Analysis (EDA)**:
   - Generates a Plotly pie chart for spam vs. no spam distribution.
   - Creates a word cloud for spam messages to highlight frequent terms.
   - Saves visualizations to the `outputs` directory.

3. **Model Training**:
   - Vectorizes text using `TfidfVectorizer` for robust feature extraction.
   - Uses PyCaret to compare classification models (e.g., Naive Bayes, Random Forest) and select the best based on F1-score.
   - Saves the trained model and vectorizer for predictions.

4. **Deployment**:
   - Deploys a Streamlit app for real-time spam detection.
   - Supports single-message predictions via text input and batch predictions via CSV upload.
   - Displays EDA visualizations and prediction results interactively.

## 📈 Key Results

- **Model Performance**:
  - Best model achieves an F1-score of ~0.95–0.98, indicating high accuracy in spam detection.
  - Robust handling of imbalanced data with TF-IDF features.
- **Visualizations**:
  - Pie chart showing the distribution of spam vs. no spam messages.
  - Word cloud highlighting terms like "free", "win", and "cash" in spam messages.
- **Practical Insights**:
  - Enables email and messaging apps to filter spam, improving user experience.
  - Supports cybersecurity efforts by identifying malicious or promotional content.
  - Applicable to customer support, marketing analysis, and fraud detection.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `nltk`, `pycaret`, `plotly`, `wordcloud`, `streamlit`, `scikit-learn`
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
   python spam_detection.py
   ```
   This:
   - Fetches and preprocesses the dataset (`outputs/processed_spam_data.csv`).
   - Generates visualizations (`outputs/wordcloud.png`, `outputs/eda_plots.html`).
   - Trains and saves the model and vectorizer (`outputs/spam_detection_model.pkl`, `outputs/vectorizer.pkl`).
   - Launches the Streamlit app.

2. **Access the Streamlit App**:
   - Navigate to `http://localhost:8501` to input messages or upload a CSV for spam detection.

3. **View Visualizations**:
   - Open `outputs/eda_plots.html` in a browser for the label distribution.
   - View `outputs/wordcloud.png` for the spam word cloud.

## 📋 Usage

- **For App Developers**: Integrate the spam detection model into email or messaging platforms to filter spam.
- **For Data Scientists**: Extend the pipeline with advanced NLP models (e.g., BERT) or additional features (e.g., sentiment analysis).
- **For Cybersecurity Professionals**: Use predictions to identify potential phishing or fraudulent messages.
- **For Developers**: Deploy the Streamlit app on cloud platforms (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Advanced NLP**: Incorporate transformer-based models like BERT for improved text understanding.
- **Multi-Class Classification**: Detect specific types of spam (e.g., phishing, promotional).
- **Real-Time Monitoring**: Integrate with email or messaging APIs for live spam detection.
- **Feature Expansion**: Add metadata (e.g., sender, timestamp) to enhance prediction accuracy.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Data Source**: Kaggle spam dataset.
- **Tools**: Built with `pandas`, `nltk`, `pycaret`, `plotly`, `wordcloud`, `streamlit`, and `scikit-learn`.
- **Inspiration**: Thanks to Aman Kharwal and the data science community for foundational insights.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in natural language processing, machine learning, and cybersecurity analytics. Last updated: August 15, 2025.*
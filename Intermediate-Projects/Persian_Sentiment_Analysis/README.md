# Intermediate Data Science Projects

![Portfolio Banner](https://via.placeholder.com/1200x200.png?text=Intermediate+Data+Science+Portfolio)  
*Showcasing foundational skills in NLP and data visualization*

## 📖 Overview

Welcome to the **Intermediate Projects** directory of my data science portfolio. This repository contains projects that demonstrate core competencies in data analysis, natural language processing (NLP), and visualization, designed to solve practical problems with clear, actionable insights. The featured project, **Persian Sentiment Analysis**, focuses on analyzing Persian mentorship comments to uncover sentiment and key themes, highlighting skills in text processing and data visualization.

This directory is ideal for showcasing intermediate-level data science projects to potential employers, collaborators, or academic audiences, with a focus on clarity and practical applications.

## 🗣 Persian Sentiment Analysis

### Description
This project analyzes 51 Persian mentorship comments to evaluate participant satisfaction and identify recurring themes. Using NLP techniques tailored for Persian text, the project performs sentiment analysis, generates a word cloud, and visualizes key patterns, providing insights into mentor performance for program evaluation.

### Key Features
- **Dataset**: 51 Persian comments from `nazar.csv`, reflecting feedback on mentorship quality.
- **Preprocessing**: Normalizes and tokenizes text using `hazm`, removes custom and default stopwords from `stopwords.txt`.
- **Sentiment Analysis**: Applies rule-based sentiment classification (Positive/Neutral/Negative) using `TextBlob` as a placeholder.
- **Visualizations**:
  - Word cloud highlighting frequent terms (e.g., "عالی" [excellent], "پیگیر" [follow-up]).
  - Sentiment distribution bar plot.
  - Top word frequency plot for key themes.
- **Insights**: Identifies strong positive sentiment and mentor strengths like responsiveness and professionalism.
- **Output**: Saves processed comments with sentiments in `processed_nazar.csv`.

### Files
- **Notebook**: [Persian_Sentiment_Analysis.ipynb](Persian_Sentiment_Analysis.ipynb)
- **Dataset**: `nazar.csv`
- **Stopwords**: `stopwords.txt`
- **Output**: `processed_nazar.csv`
- **Visualization**: `wordcloud.png`

### Dataset Insights
- **Size**: 51 comments, 1 column (`nazaar`).
- **Content**: Feedback on mentors, with terms like "عالی" (excellent), "خوش برخورد" (friendly), and "صبور" (patient).
- **Preprocessing**: Removes stopwords (e.g., "با", "به", "و") and normalizes Persian text for consistency.

## 🛠 Methodology

The analysis is implemented in `Persian_Sentiment_Analysis.ipynb` with the following pipeline:

1. **Data Loading**:
   - Loads `nazar.csv` containing Persian comments.
   - Uses `stopwords.txt` for custom stopword filtering.

2. **Text Preprocessing**:
   - Normalizes text with `hazm.Normalizer` (e.g., unifies characters, removes diacritics).
   - Tokenizes comments using `hazm.WordTokenizer`.
   - Removes stopwords combining `hazm` defaults and custom list.

3. **Sentiment Analysis**:
   - Applies rule-based sentiment classification with `TextBlob` (Positive/Neutral/Negative).
   - Note: Suggests using ParsBERT for future Persian-specific sentiment analysis.

4. **Exploratory Data Analysis (EDA)**:
   - Visualizes sentiment distribution with a bar plot.
   - Generates a word cloud for frequent terms.
   - Plots top 10 word frequencies to highlight key themes.

5. **Results**:
   - Saves processed data with sentiment labels.
   - Produces professional visualizations for presentation.

## 📈 Key Results

- **Sentiment Distribution**: Majority of comments are positive, reflecting high mentor satisfaction.
- **Key Themes**: Frequent words include "عالی" (excellent), "پیگیر" (follow-up), and "خوش برخورد" (friendly), indicating strong mentor performance.
- **Visualizations**:
  - **Word Cloud**: Highlights dominant terms in a visually appealing format.
  - **Sentiment Plot**: Shows predominance of positive feedback.
  - **Word Frequency**: Identifies top terms driving mentor evaluations.
- **Insights**: Program organizers can leverage these findings to highlight mentor strengths and improve training.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `hazm`, `wordcloud-fa`, `textblob`
- Persian font for word cloud (e.g., B Nazanin, downloadable online)

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
   matplotlib==3.7.1
   seaborn==0.12.2
   hazm==0.7.0
   wordcloud-fa==0.1.10
   textblob==0.17.1
   ```

3. Ensure `nazar.csv` and `stopwords.txt` are in the project directory.
4. Update the `font_path` in `Persian_Sentiment_Analysis.ipynb` with a valid Persian font path (e.g., `fonts/BNazanin.ttf`).

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Persian_Sentiment_Analysis.ipynb` to execute the analysis and generate visualizations.
3. Review the word cloud, sentiment distribution, and word frequency plots.

## 📋 Usage

- **For Stakeholders**: Present the notebook’s visualizations and “Key Insights” section to highlight mentor performance and program strengths.
- **For Data Scientists**: Extend the analysis with advanced NLP models (e.g., ParsBERT) or topic modeling (e.g., LDA).
- **For Developers**: Integrate the sentiment analysis pipeline into a web application using Flask or Streamlit.

## 🔮 Future Improvements

- **Enhanced Sentiment Analysis**: Replace `TextBlob` with a Persian-specific model like ParsBERT for better accuracy.
- **Topic Modeling**: Apply LDA to identify specific mentorship themes.
- **Interactive Dashboard**: Develop a Streamlit app for real-time comment analysis.
- **Expanded Dataset**: Incorporate more comments to improve robustness.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: [Your Email Address]

## 🙏 Acknowledgments

- Dataset: Persian mentorship comments from `nazar.csv`.
- Tools: Built with `hazm`, `wordcloud-fa`, and other open-source Python libraries.
- Inspiration: Thanks to the NLP community for resources and best practices.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in NLP and data visualization. Last updated: August 15, 2025.*
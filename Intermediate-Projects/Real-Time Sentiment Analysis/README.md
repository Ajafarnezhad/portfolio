# Real-Time Sentiment Analysis

![Project Banner](https://via.placeholder.com/1200x200.png?text=Real-Time+Sentiment+Analysis)  
*Harnessing advanced NLP for real-time user feedback analysis*

## 📖 Project Overview

This project develops a state-of-the-art pipeline for real-time sentiment analysis of user feedback, using a pre-trained BERT model for accurate classification. Integrated with a Streamlit interface and interactive Plotly visualizations, it enables businesses to analyze user opinions instantly and track sentiment trends over time. Designed for stakeholders in customer experience and marketing, this project is a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Analyze User Sentiment**: Classify feedback as Positive or Negative in real-time.
- **Visualize Feedback Trends**: Provide interactive visualizations for sentiment distribution and trends.
- **Enable Business Insights**: Offer actionable recommendations based on user feedback.
- **Deploy User-Friendly Interface**: Create a Streamlit app for seamless interaction.

## 📊 Dataset Description

The dataset is generated in real-time via user inputs:

- **Data Source**: User-provided text feedback (e.g., service ratings).
- **Stored Data**: Saved as `user_feedback.csv` with columns:
  - `Timestamp`: Date and time of input.
  - `Input`: User-provided text.
  - `Sentiment`: Positive or Negative (from BERT model).
  - `Score`: Confidence score of the sentiment.
- **Insights**:
  - Dynamic dataset grows with each user input.
  - Stored data enables historical analysis and trend visualization.

## 🛠 Methodology

The analysis is implemented in `Real_Time_Sentiment_Analysis.ipynb` with the following pipeline:

1. **Sentiment Analysis Setup**:
   - Initialized a pre-trained BERT model (`distilbert-base-uncased-finetuned-sst-2-english`) for sentiment classification.
   - Set up a CSV file for storing user feedback.

2. **Real-Time Sentiment Analysis**:
   - Processed user input through the BERT model to classify sentiment and compute confidence scores.
   - Integrated a Streamlit interface for real-time input and result display.

3. **Feedback Visualization**:
   - Visualized sentiment distribution using a bar chart.
   - Plotted sentiment scores over time with a scatter plot.

4. **Outputs**:
   - Saved feedback to `user_feedback.csv`.
   - Exported visualizations as HTML (e.g., `sentiment_distribution.html`).

## 📈 Key Results

- **Sentiment Analysis**:
  - BERT model accurately classifies user feedback as Positive or Negative with confidence scores.
- **Visualizations**:
  - Interactive bar chart of sentiment distribution.
  - Scatter plot of sentiment scores over time, highlighting trends.
- **Business Insights**:
  - Tracks positive and negative feedback rates for service evaluation.
  - Enables businesses to highlight strengths and address weaknesses based on real-time data.
  - Streamlit interface ensures accessibility for non-technical users.
- **Applications**:
  - Real-time feedback analysis for customer experience teams.
  - Scalable for product reviews, social media monitoring, or market research.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `transformers`, `streamlit`, `plotly`, `matplotlib`, `seaborn`
- Internet access for downloading the BERT model

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
   transformers==4.30.2
   streamlit==1.22.0
   plotly==5.15.0
   matplotlib==3.7.1
   seaborn==0.12.2
   ```

### Running the Project
1. Run the Streamlit app:
   ```bash
   streamlit run Real_Time_Sentiment_Analysis.ipynb
   ```
2. Access the app in your browser (default: `http://localhost:8501`).
3. Enter feedback to analyze sentiment and view visualizations.
4. Open HTML files (e.g., `sentiment_distribution.html`) for standalone visualizations.

## 📋 Usage

- **For Business Stakeholders**: Use the Streamlit app to collect and analyze customer feedback in real-time, presenting visualizations and insights to improve services.
- **For Data Scientists**: Extend the pipeline with multilingual BERT models or topic modeling (e.g., LDA).
- **For Developers**: Deploy the Streamlit app on a cloud platform (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Multilingual Support**: Add BERT models for non-English languages (e.g., Persian).
- **Topic Modeling**: Integrate LDA to identify themes in feedback.
- **Real-Time Streaming**: Connect to social media APIs for live sentiment analysis.
- **Enhanced Visualizations**: Add word clouds or sentiment heatmaps for deeper insights.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Model**: Pre-trained BERT model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`).
- **Tools**: Built with `transformers`, `streamlit`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to Aman Kharwal and the NLP community for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in NLP and real-time data analysis. Last updated: August 15, 2025.*
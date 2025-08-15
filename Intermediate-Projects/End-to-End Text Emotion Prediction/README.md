# End-to-End Text Emotion Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Text+Emotion+Prediction)  
*End-to-end machine learning for real-time text emotion analysis*

## 📖 Project Overview

This project delivers a complete end-to-end machine learning pipeline for predicting text emotions (e.g., happy, sad, anger) using the `emotion` dataset from Hugging Face. It includes data acquisition, preprocessing, fine-tuning a BERT model, evaluation, and deployment via a Streamlit app for real-time predictions. Interactive Plotly visualizations provide insights into dataset and model performance, making it ideal for business stakeholders in customer experience and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Collect and Prepare Data**: Load and preprocess the `emotion` dataset for training.
- **Train and Evaluate Model**: Fine-tune a BERT model for multi-class emotion classification.
- **Visualize Insights**: Create interactive visualizations for dataset and model analysis.
- **Deploy Interactive Interface**: Build a Streamlit app for real-time emotion prediction.
- **Enable Business Applications**: Offer actionable insights from user feedback.

## 📊 Dataset Description

The dataset is sourced from Hugging Face’s `emotion` dataset, containing text samples with emotion labels:

- **Features**:
  - `text`: User-provided text input.
  - `label`: Emotion labels (happy, sad, anger, fear, love, surprise).
- **Stored Data**: User feedback saved as `emotion_feedback.csv` with columns:
  - `Timestamp`: Date and time of input.
  - `Input`: User-provided text.
  - `Emotion`: Predicted emotion.
  - `Confidence`: Confidence score of the prediction.
- **Insights**:
  - Training: ~16,000 samples; Test: ~2,000 samples.
  - Dynamic feedback dataset grows with user inputs via Streamlit.

## 🛠 Methodology

The pipeline is implemented across two scripts:

1. **train.py** (Training and Evaluation):
   - Loaded the `emotion` dataset and tokenized text using `distilbert-base-uncased`.
   - Fine-tuned a BERT model for multi-class emotion classification.
   - Evaluated model with classification reports and confusion matrices.
   - Visualized emotion distribution and model performance with Plotly.
   - Saved model, tokenizer, and visualizations.

2. **app.py** (Deployment):
   - Deployed the fine-tuned BERT model via a Streamlit app for real-time predictions.
   - Stored user feedback in `emotion_feedback.csv`.
   - Visualized feedback trends with Plotly.

## 📈 Key Results

- **Model Performance**:
  - Fine-tuned BERT model achieves high accuracy across emotions (happy, sad, anger, etc.).
  - Robust performance on multi-class classification.
- **Visualizations**:
  - Interactive bar chart of emotion distribution in the dataset.
  - Confusion matrix highlighting model performance.
  - Real-time feedback visualization in Streamlit.
- **Business Insights**:
  - Tracks dominant emotions and positive/negative rates for user feedback.
  - Enables businesses to enhance customer experience based on emotional trends.
  - Streamlit app ensures accessibility for non-technical users.
- **Applications**:
  - Real-time analysis for customer feedback, social media monitoring, or market research.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `transformers`, `datasets`, `torch`, `streamlit`, `plotly`, `matplotlib`, `seaborn`
- Internet access for downloading the `emotion` dataset and BERT model

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
   datasets==2.12.0
   torch==2.0.1
   streamlit==1.22.0
   plotly==5.15.0
   matplotlib==3.7.1
   seaborn==0.12.2
   ```

### Running the Project
1. **Train the Model**:
   ```bash
   python train.py
   ```
   This generates the model (`emotion_model`), visualizations (`emotion_distribution.html`, `confusion_matrix.html`), and label encoder (`label_encoder_classes.npy`).

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501` to enter text and view predictions/visualizations.

3. **View Visualizations**:
   - Open HTML files (e.g., `emotion_distribution.html`) in a browser for standalone visualizations.

## 📋 Usage

- **For Business Stakeholders**: Use the Streamlit app to collect and analyze text feedback in real-time, presenting visualizations to improve services.
- **For Data Scientists**: Extend the pipeline with multilingual BERT models or topic modeling (e.g., LDA).
- **For Developers**: Deploy the Streamlit app on a cloud platform (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Multilingual Support**: Add BERT models for non-English languages (e.g., Persian).
- **Topic Modeling**: Integrate LDA to identify themes in text feedback.
- **Real-Time Streaming**: Connect to social media APIs for live emotion analysis.
- **Enhanced Visualizations**: Add word clouds or sentiment heatmaps for deeper insights.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: `emotion` dataset from Hugging Face.
- **Model**: Pre-trained BERT model (`distilbert-base-uncased`) from Hugging Face.
- **Tools**: Built with `transformers`, `datasets`, `streamlit`, `plotly`, and other open-source libraries.
- **Inspiration**: Thanks to Aman Kharwal and the NLP community for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in NLP and end-to-end machine learning. Last updated: August 15, 2025.*
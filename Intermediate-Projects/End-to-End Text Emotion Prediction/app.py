# app.py: Real-Time Text Emotion Prediction with Streamlit
# This script deploys a fine-tuned BERT model for real-time emotion prediction using a Streamlit interface,
# with interactive Plotly visualizations for user feedback analysis.

# Importing core libraries for NLP, visualization, and deployment
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import os

# --- 1. Model and Data Setup ---
# Load fine-tuned BERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('emotion_model')
tokenizer = AutoTokenizer.from_pretrained('emotion_model')
label_encoder_classes = np.load('label_encoder_classes.npy', allow_pickle=True)

# Initialize feedback storage
if not os.path.exists('emotion_feedback.csv'):
    pd.DataFrame(columns=['Timestamp', 'Input', 'Emotion', 'Confidence']).to_csv('emotion_feedback.csv', index=False)

# --- 2. Sentiment Analysis Function ---
def predict_emotion(text):
    """Predict emotion of user input using BERT model."""
    if not text.strip():
        return 'Neutral', 0.0
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()
    predicted_label = label_encoder_classes[np.argmax(probs)]
    confidence = np.max(probs)
    return predicted_label, confidence

# --- 3. Streamlit Interface ---
st.title('Real-Time Text Emotion Prediction')
st.markdown('Enter text to predict its emotion in real-time. Results are visualized and stored for analysis.')

user_input = st.text_input('Enter Text:', '')
if user_input:
    emotion, confidence = predict_emotion(user_input)
    st.write(f'Predicted Emotion: **{emotion}** (Confidence: {confidence:.2%})')

    # Save feedback to CSV
    feedback_df = pd.DataFrame({
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Input': [user_input],
        'Emotion': [emotion],
        'Confidence': [confidence]
    })
    feedback_df.to_csv('emotion_feedback.csv', mode='a', header=False, index=False)

# --- 4. Feedback Visualization ---
feedback_data = pd.read_csv('emotion_feedback.csv')
if not feedback_data.empty:
    st.subheader('Feedback Analysis')
    emotion_counts = feedback_data['Emotion'].value_counts()
    fig = px.bar(x=emotion_counts.index, y=emotion_counts.values,
                 title='Emotion Distribution of User Feedback',
                 labels={'x': 'Emotion', 'y': 'Number of Inputs'},
                 color=emotion_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=500, title_x=0.5, showlegend=False)
    st.plotly_chart(fig)
    fig.write_html('feedback_emotion_distribution.html')

    # Business insights
    top_emotion = feedback_data['Emotion'].mode()[0]
    pos_rate = (feedback_data['Emotion'].isin(['happy', 'love'])).mean() * 100
    neg_rate = (feedback_data['Emotion'].isin(['sad', 'anger', 'fear'])).mean() * 100
    st.write(f"**Dominant Emotion**: {top_emotion}")
    st.write(f"**Positive Emotion Rate**: {pos_rate:.1f}%")
    st.write(f"**Negative Emotion Rate**: {neg_rate:.1f}%")
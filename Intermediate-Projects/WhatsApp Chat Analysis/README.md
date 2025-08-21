# WhatsApp Chat Analysis

![Project Banner](https://via.placeholder.com/1200x200.png?text=WhatsApp+Chat+Analysis)  
*Unveiling Insights from WhatsApp Conversations with Python and Interactive Visualizations*

## 📖 Project Overview

This project develops a comprehensive WhatsApp chat analysis pipeline to extract and visualize insights from exported chat data. By processing raw chat logs, the system performs text analysis, emoji extraction, link detection, and statistical summaries, enhanced with interactive visualizations using Plotly and word clouds. Designed for data scientists, analysts, and businesses, this project uncovers communication patterns, sentiment trends, and user behavior, making it an excellent addition to an intermediate-level data science portfolio.

### Objectives
- **Analyze Chat Data**: Extract meaningful statistics such as message counts, media shared, emojis, and links.
- **Visualize Communication Patterns**: Create interactive charts and word clouds to highlight word usage and emoji distribution.
- **Enable NLP Applications**: Provide a foundation for advanced tasks like sentiment analysis or named entity recognition.
- **Showcase Data Science Skills**: Demonstrate proficiency in text preprocessing, data wrangling, and visualization.

## 📊 Dataset Description

The dataset is an exported WhatsApp chat log (`.txt` format) containing conversations between two individuals or a group. Key features include:

- **Key Features**:
  - `Date`: Timestamp of each message.
  - `Author`: Sender of the message.
  - `Message`: Text content, including emojis, media placeholders, and URLs.
- **Insights**:
  - Size: Varies by chat (e.g., 1,288 messages in the example dataset).
  - Missing Data: Handled by imputing ‘Unknown’ for missing authors or empty messages.
  - Notable Patterns: Frequent use of emojis, media sharing, and informal language.

## 🛠 Methodology

The analysis is implemented in `whatsapp_chat_analysis.py` with the following pipeline:

1. **Data Preprocessing**:
   - Parsed raw `.txt` chat logs using regular expressions to extract date, author, and message.
   - Handled inconsistent formats and missing data.
   - Converted data into a structured `pandas` DataFrame with columns: `Date`, `Author`, `Message`.

2. **Exploratory Data Analysis (EDA)**:
   - Calculated total messages, media shared, emojis, and links.
   - Analyzed per-author statistics (e.g., message count, average words per message).
   - Extracted emojis and URLs using `emoji` and `re` libraries.

3. **Visualization**:
   - Created interactive Plotly bar charts for emoji distribution by author.
   - Generated word clouds to visualize frequently used words (overall and per author).
   - Saved visualizations as HTML and PNG files for interactive exploration.

4. **Outputs**:
   - Saved processed dataset as `processed_chat_data.csv`.
   - Generated visualization files (e.g., `emoji_distribution.html`, `overall_wordcloud.png`).
   - Produced a summary report with key statistics (e.g., total messages, media, emojis, links).

## 📈 Key Results

- **Chat Statistics**:
  - Total Messages: 1,288 (example dataset).
  - Media Shared: 11 messages with `<Media omitted>`.
  - Emojis Shared: 367 unique emojis.
  - Links Shared: 1 URL.
- **Per-Author Insights**:
  - Example (Aman): 687 messages, 6.17 avg. words/message, 228 emojis, 1 link.
  - Example (Sapna): 590 messages, 6.38 avg. words/message, 139 emojis, 0 links.
- **Visualizations**:
  - Interactive bar chart of emoji usage by author.
  - Word clouds highlighting frequent words for the entire chat and per author.
- **Insights**:
  - The chat reflects informal communication with significant emoji usage, indicating expressive interactions.
  - Word clouds reveal key topics and conversational styles unique to each participant.
  - The pipeline is scalable for group chats or customer support analysis.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `wordcloud`, `emoji`, `re`
- Dataset: Exported WhatsApp chat log (`.txt` format)

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
   plotly==5.15.0
   wordcloud==1.9.3
   emoji==2.12.1
   ```

3. Export your WhatsApp chat:
   - Open WhatsApp, select a chat, and choose “Export Chat” (without media).
   - Email the `.txt` file to yourself and place it in the project directory.

### Running the Project
1. Run the script:
   ```bash
   python whatsapp_chat_analysis.py
   ```
   Modify the script to call `analyze_whatsapp_chat('your_chat.txt')` with your file path.
2. View outputs in the `output` directory, including CSV, HTML, and PNG files.

## 📋 Usage

- **For Stakeholders**: Use the visualizations and summary statistics to understand communication patterns, customer interactions, or group dynamics.
- **For Data Scientists**: Extend the pipeline with NLP tasks like sentiment analysis or topic modeling using libraries like `transformers` or `spaCy`.
- **For Developers**: Integrate the analysis into a web app using Flask or Streamlit for real-time chat analytics.

## 🔮 Future Improvements

- **Sentiment Analysis**: Incorporate NLP models (e.g., BERT) to analyze the tone of conversations.
- **Temporal Analysis**: Visualize message frequency over time to identify peak activity periods.
- **Group Chat Support**: Extend parsing logic to handle multi-participant group chats.
- **Interactive Dashboard**: Build a Plotly Dash app for real-time chat exploration.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: User-exported WhatsApp chat logs.
- **Tools**: Built with `pandas`, `plotly`, `wordcloud`, `emoji`, and other open-source Python libraries.
- **Inspiration**: Thanks to the data science community and Aman Kharwal for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in text analysis, data wrangling, and interactive visualization. Last updated: August 21, 2025.*
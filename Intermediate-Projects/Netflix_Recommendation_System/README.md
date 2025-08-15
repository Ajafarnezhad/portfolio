# Netflix Recommendation System

![Project Banner](https://via.placeholder.com/1200x200.png?text=Netflix+Recommendation+System)  
*Building a content-based recommendation engine with interactive visualizations*

## 📖 Project Overview

This project develops a content-based recommendation system for Netflix movies and TV shows, leveraging a dataset of 15,480 titles. By combining features like genres, descriptions, cast, and directors, it recommends similar content using TF-IDF vectorization and cosine similarity. The project includes exploratory data analysis (EDA) with interactive Plotly visualizations, making it ideal for stakeholders in the entertainment industry and a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Develop a Recommendation System**: Suggest relevant movies and TV shows based on content features.
- **Analyze Content Trends**: Explore genre popularity, rating distribution, and content types.
- **Deliver Visual Insights**: Create interactive visualizations for engaging presentations.
- **Showcase Data Science Skills**: Demonstrate proficiency in text processing, recommendation algorithms, and visualization.

## 📊 Dataset Description

The dataset (`netflixData.csv`) contains 15,480 Netflix titles as of 2021, with 11 features:

- **Key Features**:
  - `Title`: Name of the movie or TV show.
  - `Genres`: Categories (e.g., International Movies, Dramas).
  - `Description`: Brief summary of the content.
  - `Cast`, `Director`: Actors and directors involved.
  - `Content Type`: Movie or TV Show.
  - `Imdb Score`, `Rating`, `Duration`: Additional metadata.
- **Insights**:
  - Size: 15,480 rows, 11 columns.
  - Missing Data: Handled by imputing ‘Unknown’ for `Cast` and `Director`, empty strings for `Description` and `Genres`.
  - Notable Genres: International Movies, Dramas, Comedies dominate.

## 🛠 Methodology

The analysis is implemented in `Netflix_Recommendation_System.ipynb` with the following pipeline:

1. **Data Preprocessing**:
   - Loaded dataset and handled missing values.
   - Cleaned text data by removing punctuation and converting to lowercase.
   - Combined `Genres`, `Description`, `Cast`, and `Director` into a single feature for recommendation.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized top 10 genres to identify popular categories.
   - Analyzed content rating distribution (e.g., TV-MA, TV-14).
   - Compared movies vs. TV shows using a pie chart.

3. **Recommendation System**:
   - Used TF-IDF vectorization to convert combined features into a numerical matrix.
   - Computed cosine similarity to find similar titles.
   - Built a recommendation function that returns top-N similar titles with details (genres, IMDb score, duration).

4. **Visualization**:
   - Created interactive Plotly charts for genre distribution, rating distribution, and content type.
   - Saved visualizations as HTML for presentation flexibility.

5. **Outputs**:
   - Saved processed dataset as `processed_netflix_data.csv`.
   - Generated HTML visualization files (e.g., `genre_distribution.html`).

## 📈 Key Results

- **Content Trends**:
  - International Movies, Dramas, and Comedies are the most common genres.
  - TV-MA and TV-14 ratings dominate, indicating a focus on mature audiences.
  - Movies (vs. TV shows) form the majority of the catalog.
- **Recommendation System**:
  - Accurately recommends titles based on genres, descriptions, cast, and directors.
  - Example: For “#Alive”, recommends similar thrillers and horror movies.
- **Visualizations**:
  - Interactive bar chart of top genres.
  - Histogram of content ratings.
  - Pie chart comparing movies and TV shows.
- **Insights**:
  - Netflix prioritizes diverse, international content for broad appeal.
  - The recommendation system effectively captures content similarity, suitable for enhancing user experience.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`
- Dataset: `netflixData.csv` (included in the repository)

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
   scikit-learn==1.2.2
   plotly==5.15.0
   ```

3. Ensure `netflixData.csv` is in the project directory.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `Netflix_Recommendation_System.ipynb` to execute the analysis and generate visualizations.
3. Open HTML files (e.g., `genre_distribution.html`) in a browser for interactive exploration.

## 📋 Usage

- **For Stakeholders**: Present the notebook’s visualizations and “Key Insights” section to highlight content trends and recommendation capabilities for streaming platforms.
- **For Data Scientists**: Extend the system with collaborative filtering or deep learning-based embeddings (e.g., BERT).
- **For Developers**: Integrate the recommendation system into a web app using Flask or Streamlit.

## 🔮 Future Improvements

- **Collaborative Filtering**: Incorporate user viewing history for personalized recommendations.
- **Advanced NLP**: Use BERT or sentence embeddings for richer text feature extraction.
- **Interactive Dashboard**: Build a Plotly Dash app for real-time recommendation exploration.
- **Expanded Dataset**: Include user ratings or additional metadata for enhanced accuracy.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Netflix titles data from Kaggle (as of 2021).
- **Tools**: Built with `pandas`, `scikit-learn`, `plotly`, and other open-source Python libraries.
- **Inspiration**: Thanks to the data science community and Aman Kharwal for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in recommendation systems and data visualization. Last updated: August 15, 2025.*
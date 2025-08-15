# T20 World Cup 2022 Analysis

![Project Banner](https://via.placeholder.com/1200x200.png?text=T20+World+Cup+2022+Analysis)  
*Uncovering team and player performance insights through interactive data visualization*

## 📖 Project Overview

This project analyzes match data from the T20 World Cup 2022 to uncover patterns in team performance, player contributions, and strategic decisions. Using a dataset of 33 matches, it employs advanced data preprocessing, exploratory data analysis (EDA), and interactive Plotly visualizations to provide actionable insights for cricket analysts, teams, and fans. The analysis is implemented in a Jupyter notebook, designed for clarity and professional presentation, making it a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Analyze Team Performance**: Identify top-performing teams and their winning strategies.
- **Evaluate Player Contributions**: Highlight key players through awards and scoring metrics.
- **Assess Strategic Factors**: Examine the impact of toss decisions and venue on match outcomes.
- **Deliver Visual Insights**: Create interactive visualizations for engaging presentations.

## 📊 Dataset Description

The dataset (`t20-world-cup-22.csv`) contains 33 matches from the T20 World Cup 2022, with 14 features capturing match details:

- **Key Features**:
  - `venue`: Match location (e.g., SCG, MCG).
  - `team1`, `team2`: Competing teams.
  - `stage`: Tournament stage (e.g., Super 12, Semi-final, Final).
  - `toss winner`, `toss decision`: Toss outcome and decision (Bat/Field).
  - `first innings score`, `first innings wickets`: Batting team’s performance.
  - `second innings score`, `second innings wickets`: Chasing team’s performance.
  - `winner`, `won by`: Match winner and margin (Runs/Wickets).
  - `player of the match`, `top scorer`, `highest score`, `best bowler`, `best bowling figure`: Individual performance metrics.
- **Insights**:
  - Size: 33 rows, 14 columns.
  - Missing Data: Handled for cancelled matches (e.g., imputed as ‘No Result’).
  - Notable Teams: England (final winner), India, Pakistan, New Zealand.

## 🛠 Methodology

The analysis is implemented in `T20_World_Cup_Analysis.ipynb` with the following pipeline:

1. **Data Preprocessing**:
   - Loaded dataset and handled missing values (e.g., ‘No Result’ for cancelled matches).
   - Standardized numerical columns (`first innings score`, `highest score`) using `pd.to_numeric`.
   - Ensured data consistency for robust analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized match wins by team to identify top performers.
   - Analyzed toss decision impact (Bat vs. Field) on outcomes.
   - Highlighted top players through Player of the Match awards.
   - Examined venue trends across tournament stages.
   - Plotted innings performance (scores and wickets) distributions.

3. **Visualization**:
   - Used Plotly for interactive bar charts, histograms, and subplots.
   - Saved visualizations as HTML files for presentation flexibility.

4. **Outputs**:
   - Saved processed dataset as `processed_t20_world_cup.csv`.
   - Generated interactive HTML plots (e.g., `match_wins.html`, `toss_decision.html`).

## 📈 Key Results

- **Team Performance**:
  - England and India emerged as top winners, with England clinching the final.
  - Teams choosing to field won more matches, suggesting a strategic advantage.
- **Player Contributions**:
  - Players like Virat Kohli, Sam Curran, and Suryakumar Yadav frequently earned Player of the Match awards.
  - Top scorers (e.g., Virat Kohli with 82) and bowlers (e.g., Sam Curran with 5-10) drove match outcomes.
- **Venue Trends**:
  - SCG and MCG hosted critical matches, including Super 12 and semi-finals.
  - Venues influenced scoring patterns, with higher scores at SCG.
- **Visualizations**:
  - Interactive bar chart of match wins by team.
  - Grouped histogram showing toss decision impact.
  - Bar plot of top 10 Players of the Match.
  - Stacked histogram of matches by venue and stage.
  - Subplots for first innings scores and wickets distributions.
- **Insights**:
  - Fielding first is a strategic advantage in T20 matches.
  - All-rounders like Sam Curran are critical for success.
  - Teams should tailor strategies based on venue scoring trends.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- Dataset: `t20-world-cup-22.csv` (included in the repository)

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
   ```

3. Ensure `t20-world-cup-22.csv` is in the project directory.

### Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run `T20_World_Cup_Analysis.ipynb` to execute the analysis and generate interactive visualizations.
3. Open HTML files (e.g., `match_wins.html`) in a browser for interactive exploration.

## 📋 Usage

- **For Stakeholders**: Present the notebook’s visualizations and “Key Insights” section to highlight team strategies and player performance for cricket analysts or teams.
- **For Data Scientists**: Extend the analysis with predictive models (e.g., match outcome prediction) or additional metrics (e.g., run rates).
- **For Developers**: Integrate visualizations into a Plotly Dash dashboard for real-time analysis.

## 🔮 Future Improvements

- **Predictive Modeling**: Develop models to predict match outcomes based on toss, venue, and team form.
- **Advanced Metrics**: Calculate run rates, economy rates, or player consistency across matches.
- **Interactive Dashboard**: Build a Plotly Dash app for dynamic exploration of match data.
- **Expanded Dataset**: Include historical T20 data for longitudinal analysis.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Dataset**: T20 World Cup 2022 match data from `t20-world-cup-22.csv`.
- **Tools**: Built with `pandas`, `plotly`, and other open-source Python libraries.
- **Inspiration**: Thanks to the cricket and data science communities for resources and insights.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing skills in data analysis and visualization. Last updated: August 15, 2025.*
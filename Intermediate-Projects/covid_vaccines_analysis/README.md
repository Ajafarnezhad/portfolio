# COVID-19 Vaccines Analysis

![Project Banner](https://via.placeholder.com/1200x200.png?text=COVID-19+Vaccines+Analysis)  
*Exploring Global Vaccine Usage with Data Science and Interactive Visualizations*

## 📖 Project Overview

The COVID-19 pandemic highlighted vaccines as a critical tool for global health and economic recovery. This project delivers a comprehensive Python-based pipeline to analyze COVID-19 vaccine usage across countries, leveraging data from Our World in Data. It includes data preprocessing, exploratory data analysis (EDA) with Plotly, and an interactive Streamlit app for real-time insights into vaccine combinations and vaccination progress. Designed for public health analysts, policymakers, and data scientists, this project is a valuable addition to an intermediate-level data science portfolio.

### Objectives
- **Analyze Vaccine Usage**: Identify which vaccines and combinations are used by each country.
- **Track Vaccination Progress**: Examine vaccination trends over time for key countries.
- **Provide Interactive Interface**: Deploy a Streamlit app for dynamic data exploration.
- **Support Public Health Decisions**: Offer insights for vaccine distribution and policy planning.

## 📊 Dataset Description

The dataset, sourced from Our World in Data ([download here](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv)), contains vaccination data for numerous countries:

- **Key Features**:
  - `location`: Country or region name.
  - `vaccines`: Vaccine combination used (e.g., "Pfizer/BioNTech, Moderna").
  - `date`: Date of vaccination record.
  - `total_vaccinations`: Cumulative number of vaccine doses administered.
  - `people_vaccinated`: Number of people who received at least one dose.
  - `people_fully_vaccinated`: Number of people fully vaccinated.
- **Insights**:
  - Size: ~10,000+ records covering multiple countries and dates.
  - Data Issues: Redundant UK regions (England, Scotland, Wales, Northern Ireland) removed during preprocessing.
  - Preprocessing: Handles missing values and standardizes date formats.

## 🛠 Methodology

The project is implemented in a single Python script (`covid_vaccines_analysis.py`):

1. **Data Acquisition and Preprocessing**:
   - Fetches data from the Our World in Data URL.
   - Removes redundant UK regions and fills missing vaccination metrics with zeros.
   - Converts dates to a standardized format.

2. **Exploratory Data Analysis (EDA)**:
   - Generates interactive Plotly visualizations:
     - Bar chart of top vaccine combinations.
     - Sunburst chart of vaccine usage by country.
     - Line chart of vaccination progress for top countries.
   - Saves visualizations and processed data to the `outputs` directory.

3. **Vaccine-Country Analysis**:
   - Maps vaccine combinations to countries using a structured DataFrame.
   - Saves the mapping for further analysis.

4. **Deployment**:
   - Deploys a Streamlit app for interactive exploration.
   - Supports filtering by country or vaccine combination.
   - Provides downloadable data and visualizations.

## 📈 Key Results

- **Vaccine Usage**:
  - Common combinations include "Moderna, Oxford/AstraZeneca, Pfizer/BioNTech" and "Pfizer/BioNTech" alone.
  - Countries like the United States and Canada use multiple vaccine types, while others rely on single vaccines like Sinovac.
- **Visualizations**:
  - Interactive Plotly dashboards for vaccine combinations, country-wise usage, and temporal trends.
  - Sunburst charts reveal hierarchical vaccine adoption patterns.
- **Practical Insights**:
  - Enables policymakers to optimize vaccine distribution strategies.
  - Supports public health campaigns by identifying vaccination gaps.
  - Facilitates global comparisons of vaccine adoption and progress.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `plotly`, `streamlit`
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
   plotly==5.15.0
   streamlit==1.22.0
   ```

### Running the Project
1. **Run the Pipeline**:
   ```bash
   python covid_vaccines_analysis.py
   ```
   This:
   - Fetches and preprocesses the dataset (`outputs/processed_vaccines_data.csv`).
   - Generates visualizations (`outputs/vaccine_usage.html`, `outputs/vaccine_by_country.html`, `outputs/vaccine_progress.html`).
   - Saves vaccine-country mappings (`outputs/vaccine_countries.csv`).
   - Launches the Streamlit app.

2. **Access the Streamlit App**:
   - Navigate to `http://localhost:8501` to filter data and explore visualizations interactively.

3. **View Visualizations**:
   - Open HTML files in the `outputs` directory in a browser for interactive visualizations.

## 📋 Usage

- **For Public Health Analysts**: Use the Streamlit app to explore vaccine usage and prioritize distribution efforts.
- **For Data Scientists**: Extend the analysis with additional metrics (e.g., vaccination rates per capita).
- **For Policymakers**: Leverage insights to inform vaccine procurement and public health strategies.
- **For Developers**: Deploy the Streamlit app on cloud platforms (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Additional Metrics**: Incorporate vaccination rates, efficacy data, or side-effect reports.
- **Geospatial Analysis**: Add maps to visualize vaccine distribution geographically.
- **Real-Time Updates**: Integrate live data feeds for ongoing vaccination monitoring.
- **Comparative Analysis**: Compare vaccine effectiveness across countries and combinations.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Data Source**: Our World in Data COVID-19 vaccinations dataset.
- **Tools**: Built with `pandas`, `plotly`, and `streamlit`.
- **Inspiration**: Thanks to Aman Kharwal and the public health data science community for foundational insights.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in data analysis, visualization, and public health analytics. Last updated: August 15, 2025.*
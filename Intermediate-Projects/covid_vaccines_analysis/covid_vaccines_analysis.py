
# covid_vaccines_analysis.py
# Integrated script for analyzing COVID-19 vaccine usage by country
# Author: Ajafarnezhad (aiamirjd@gmail.com)
# Last Updated: August 15, 2025

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os
from datetime import datetime

# Configuration
DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
OUTPUT_DIR = "outputs"
DATA_PATH = os.path.join(OUTPUT_DIR, "processed_vaccines_data.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "vaccine_usage.html")
VACCINE_COUNTRY_PLOT = os.path.join(OUTPUT_DIR, "vaccine_by_country.html")
VACCINE_PROGRESS_PLOT = os.path.join(OUTPUT_DIR, "vaccine_progress.html")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Fetch and preprocess the COVID-19 vaccinations dataset."""
    try:
        df = pd.read_csv(DATA_URL)
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        # Remove redundant UK regions
        df = df[~df['location'].isin(["England", "Scotland", "Wales", "Northern Ireland"])]
        # Handle missing values
        df['total_vaccinations'] = df['total_vaccinations'].fillna(0)
        df['people_vaccinated'] = df['people_vaccinated'].fillna(0)
        df['people_fully_vaccinated'] = df['people_fully_vaccinated'].fillna(0)
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis and save visualizations."""
    # Vaccine combinations count
    vaccine_counts = df['vaccines'].value_counts().reset_index()
    vaccine_counts.columns = ['Vaccines', 'Count']
    fig_vaccines = px.bar(
        vaccine_counts.head(10),
        x='Count',
        y='Vaccines',
        title="Top 10 Vaccine Combinations Used Globally",
        labels={'Count': 'Number of Records', 'Vaccines': 'Vaccine Combination'},
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig_vaccines.update_layout(yaxis={'tickmode': 'linear'}, showlegend=False)

    # Vaccine usage by country (sunburst chart)
    vaccine_country = df.groupby(['location', 'vaccines']).size().reset_index(name='count')
    fig_country = px.sunburst(
        vaccine_country,
        path=['location', 'vaccines'],
        values='count',
        title="Vaccine Usage by Country",
        color='count',
        color_continuous_scale='Blues'
    )

    # Vaccination progress over time (example: top 5 countries by total vaccinations)
    top_countries = df.groupby('location')['total_vaccinations'].max().sort_values(ascending=False).head(5).index
    progress_df = df[df['location'].isin(top_countries)][['location', 'date', 'total_vaccinations']]
    fig_progress = px.line(
        progress_df,
        x='date',
        y='total_vaccinations',
        color='location',
        title="Vaccination Progress Over Time (Top 5 Countries)",
        labels={'total_vaccinations': 'Total Vaccinations', 'date': 'Date'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Save visualizations
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Top Vaccine Combinations", "Vaccine Usage by Country", "Vaccination Progress"),
        specs=[[{"type": "bar"}], [{"type": "sunburst"}], [{"type": "xy"}]]
    )
    for trace in fig_vaccines.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig_country.data:
        fig.add_trace(trace, row=2, col=1)
    for trace in fig_progress.data:
        fig.add_trace(trace, row=3, col=1)
    fig.update_layout(height=1200, showlegend=True, title_text="COVID-19 Vaccines Analysis")
    fig.write_html(PLOT_PATH)

    # Save individual plots
    fig_vaccines.write_html(VACCINE_COUNTRY_PLOT)
    fig_progress.write_html(VACCINE_PROGRESS_PLOT)

    return vaccine_counts, vaccine_country, progress_df

def analyze_vaccine_countries(df):
    """Analyze which countries use each vaccine combination."""
    vaccine_dict = {}
    for vaccine in df['vaccines'].unique():
        countries = df[df['vaccines'] == vaccine]['location'].unique().tolist()
        vaccine_dict[vaccine] = countries
    vaccine_df = pd.DataFrame([(k, v) for k, v_list in vaccine_dict.items() for v in v_list], columns=['Vaccine', 'Country'])
    vaccine_df.to_csv(os.path.join(OUTPUT_DIR, "vaccine_countries.csv"), index=False)
    return vaccine_df

def streamlit_app():
    """Run the Streamlit app for interactive vaccine analysis."""
    st.set_page_config(page_title="COVID-19 Vaccines Analysis", layout="wide")
    st.title("💉 COVID-19 Vaccines Analysis Dashboard")
    st.markdown("""
    Explore global COVID-19 vaccine usage by country, vaccine combination, and vaccination progress over time.
    Select a country or vaccine combination to view detailed insights.
    """)

    # Load data
    df = load_data()
    if df is None:
        return

    # Perform EDA
    vaccine_counts, vaccine_country, progress_df = perform_eda(df)

    # Sidebar for user input
    st.sidebar.header("Analysis Filters")
    selected_country = st.sidebar.selectbox("Select Country", ['All'] + sorted(df['location'].unique()))
    selected_vaccine = st.sidebar.selectbox("Select Vaccine Combination", ['All'] + sorted(df['vaccines'].unique()))

    # Filter data
    filtered_df = df
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['location'] == selected_country]
    if selected_vaccine != 'All':
        filtered_df = filtered_df[filtered_df['vaccines'] == selected_vaccine]

    # Display filtered data
    st.subheader("Filtered Vaccine Data")
    st.dataframe(filtered_df[['location', 'vaccines', 'date', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']])

    # Display visualizations
    st.header("Exploratory Data Analysis")
    st.markdown("View interactive visualizations of vaccine usage and progress:")
    if os.path.exists(PLOT_PATH):
        with open(PLOT_PATH, 'r') as f:
            st.components.v1.html(f.read(), height=1200)

    # Country-wise vaccine analysis
    st.subheader("Vaccine Usage by Country")
    vaccine_df = analyze_vaccine_countries(df)
    st.dataframe(vaccine_df)
    if selected_vaccine != 'All':
        st.write(f"Countries using {selected_vaccine}:")
        st.write(vaccine_df[vaccine_df['Vaccine'] == selected_vaccine]['Country'].tolist())

    # Download data
    st.sidebar.header("Download Results")
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'rb') as f:
            st.sidebar.download_button(
                label="Download Processed Data",
                data=f,
                file_name=f"vaccines_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    if os.path.exists(os.path.join(OUTPUT_DIR, "vaccine_countries.csv")):
        with open(os.path.join(OUTPUT_DIR, "vaccine_countries.csv"), 'rb') as f:
            st.sidebar.download_button(
                label="Download Vaccine-Country Mapping",
                data=f,
                file_name=f"vaccine_countries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Main function to execute the pipeline."""
    df = load_data()
    if df is not None:
        vaccine_counts, vaccine_country, progress_df = perform_eda(df)
        vaccine_df = analyze_vaccine_countries(df)
        print("COVID-19 Vaccines analysis completed. Results saved in outputs/")
        print("Processed data saved:", DATA_PATH)
        print("Vaccine-country mapping saved:", os.path.join(OUTPUT_DIR, "vaccine_countries.csv"))
    
    # Run Streamlit app
    streamlit_app()

if __name__ == "__main__":
    main()

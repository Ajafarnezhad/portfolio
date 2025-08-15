# International Air Traffic Analysis

## Overview
This project provides a comprehensive analysis of international air traffic to and from Australia, leveraging a dataset of passenger, freight, and mail traffic between city pairs. It includes three main components: Exploratory Data Analysis (EDA), time-series forecasting, and anomaly detection/clustering. Using Python libraries like Pandas, Matplotlib, Seaborn, Prophet, Scikit-learn, and IsolationForest, the project uncovers trends, forecasts future traffic, and identifies anomalies in air traffic patterns. The analysis is designed for scalability, reproducibility, and visualization, making it a robust addition to a data science portfolio.

## Features
- **Exploratory Data Analysis (EDA)**:
  - Analyzes traffic distribution by country, region, and port.
  - Visualizes trends over time for passengers, freight, and mail.
  - Identifies top foreign ports and countries by traffic volume.
- **Time-Series Forecasting**:
  - Uses Prophet to forecast passenger and freight traffic for 12 months beyond August 2025.
  - Visualizes trends, seasonality, and forecast predictions with confidence intervals.
- **Anomaly Detection and Clustering**:
  - Applies IsolationForest to detect anomalies in passenger and freight data.
  - Uses KMeans clustering to group foreign ports by traffic behavior.
  - Visualizes anomalies and clusters for insights into traffic patterns.
- **Data Preprocessing**:
  - Feature engineering (e.g., Passenger_Diff, Freight_Ratio, normalization).
  - Handles datetime conversions and aggregations for time-series analysis.
- **Visualization**:
  - Generates high-resolution plots (bar charts, line plots, scatter plots, pie charts) for traffic trends, anomalies, and clusters.
- **Scalability**:
  - Processes a large dataset (+89k rows) efficiently with Pandas and Scikit-learn.
- **Artifacts**:
  - Saves processed datasets, forecasts, and cluster assignments as CSV files.
  - Produces publication-quality visualizations for analysis.

## Requirements
- **Python**: 3.8+
- **Libraries**: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `prophet`, `numpy`
- **Kaggle API**: Configure with `kaggle.json` for dataset download (place in `~/.kaggle/` and run `chmod 600 ~/.kaggle/kaggle.json`).

Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn prophet numpy kaggle
```

## Dataset
- **Source**: Kaggle dataset "International Airlines Traffic by City Pairs" by imtkaggleteam.
- **Structure**: Contains +89k rows with columns for Australian and foreign ports, country, passengers (in/out), freight (in/out), mail (in/out), and derived totals.
- **Columns**:
  - `AustralianPort`: Port in Australia.
  - `ForeignPort`: Corresponding foreign port.
  - `Country`: Country of the foreign port.
  - `Passengers_In`, `Passengers_Out`, `Passengers_Total`: Passenger counts.
  - `Freight_In_(tonnes)`, `Freight_Out_(tonnes)`, `Freight_Total_(tonnes)`: Freight in tonnes.
  - `Mail_In_(tonnes)`, `Mail_Out_(tonnes)`, `Mail_Total_(tonnes)`: Mail in tonnes.
  - `Year`, `Month`, `Month_num`: Temporal data.
- **Access**: Automatically downloaded via Kaggle API during script execution.

## How to Run
1. **Setup Kaggle API**:
   - Place `kaggle.json` in `~/.kaggle/` and set permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - Download the dataset:
     ```bash
     kaggle datasets download -d imtkaggleteam/international-airlines-traffic-by-city-pairs -p data/ --unzip
     ```

2. **Run the Notebooks**:
   - **EDA**:
     ```bash
     jupyter notebook 1-EDA_Ajafarnezhad.ipynb
     ```
     - Performs data exploration, visualizes traffic distributions, and saves processed data as `processed_city_pairs_eda.csv`.
   - **Traffic Forecasting**:
     ```bash
     jupyter notebook 2-Traffic_Forecasting_Ajafarnezhad.ipynb
     ```
     - Forecasts passenger and freight traffic, saves results as `passenger_forecast.csv` and `freight_forecast.csv`.
   - **Anomaly Detection and Clustering**:
     ```bash
     jupyter notebook 3-AnomalyDetection_Ajafarnezhad.ipynb
     ```
     - Detects anomalies, clusters ports, and saves results as `final_processed_city_pairs.csv` and `port_clusters.csv`.

3. **Output**:
   - **EDA**:
     - Visualizations: Bar charts (top countries/ports), pie charts (region distribution), line plots (yearly trends).
     - Saved file: `processed_city_pairs_eda.csv`.
   - **Forecasting**:
     - Visualizations: Forecast plots with trends and seasonality.
     - Saved files: `passenger_forecast.csv`, `freight_forecast.csv`.
   - \- **Anomaly Detection**:
     - Visualizations: Scatter plots for anomalies and clusters.
     - Saved files: `final_processed_city_pairs.csv`, `port_clusters.csv`.

## Example Output
**EDA**:
```
Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 89320 entries, 0 to 89319
Data columns (total 15 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   Month                   89320 non-null  object
 1   AustralianPort          89320 non-null  object
 2   ForeignPort             89320 non-null  object
 ...
```

**Forecasting**:
```
2025-08-15 19:00:00 - INFO - Chain [1] start processing
2025-08-15 19:00:01 - INFO - Chain [1] done processing
[Plots of passenger and freight forecasts with confidence intervals]
```

**Anomaly Detection**:
```
Detected Anomalies:
       Date  ForeignPort  Country  Passengers_Total  Freight_Total_(tonnes)  Passenger_Diff
...
[Scatter plot of anomalies in passenger traffic]
```

## Artifacts
- **Data Files**:
  - `processed_city_pairs_eda.csv`: Processed dataset from EDA.
  - `passenger_forecast.csv`, `freight_forecast.csv`: Forecast results.
  - `final_processed_city_pairs.csv`, `port_clusters.csv`: Anomaly and cluster results.
- **Plots**:
  - Bar charts, pie charts, line plots, and scatter plots saved within notebooks and displayed during execution.

## Improvements and Future Work
- **Advanced Forecasting**: Incorporate multivariate time-series models (e.g., ARIMA, LSTM) for improved accuracy.
- **Anomaly Detection**: Add DBSCAN or autoencoders for more robust anomaly detection.
- **Interactive Visualizations**: Use Plotly or Bokeh for interactive dashboards.
- **Feature Engineering**: Include external factors (e.g., economic indicators, holidays) to enhance forecasting.
- **Cloud Deployment**: Deploy analysis pipeline on cloud platforms like AWS or Google Cloud for scalability.

## License
MIT License
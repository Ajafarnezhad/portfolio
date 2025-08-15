# Cryptocurrency Price Prediction

![Project Banner](https://via.placeholder.com/1200x200.png?text=Cryptocurrency+Price+Prediction)  
*Forecasting Bitcoin prices with advanced machine learning and real-time analysis*

## 📖 Project Overview

This project delivers an end-to-end machine learning pipeline for forecasting Bitcoin prices using data from the CoinGecko API. It integrates Prophet and LSTM models for accurate predictions, interactive Plotly visualizations for price trends and error analysis, and a Streamlit app for real-time forecasting. Designed for financial stakeholders and investors, this project is a standout addition to an intermediate-level data science portfolio.

### Objectives
- **Forecast Bitcoin Prices**: Predict future prices using time-series and sequence-based models.
- **Analyze Price Trends**: Provide interactive visualizations for historical and forecasted prices.
- **Deploy Real-Time Interface**: Build a Streamlit app for dynamic forecasting.
- **Enable Financial Insights**: Offer actionable recommendations for trading and investment.

## 📊 Dataset Description

The dataset is sourced from the CoinGecko API, containing daily Bitcoin prices (USD):

- **Features**:
  - `Date`: Date of observation.
  - `Close`: Closing price of Bitcoin.
- **Insights**:
  - Size: ~365 records (1-year historical data).
  - Context: Captures daily price fluctuations for Bitcoin.
  - Preprocessing: Standardized for time-series and sequence modeling.

## 🛠 Methodology

The pipeline is implemented across two scripts:

1. **train.py** (Data Acquisition and Forecasting):
   - Fetched Bitcoin price data from CoinGecko API.
   - Trained Prophet for time-series forecasting and LSTM for sequence-based forecasting.
   - Evaluated models using RMSE and MAE.
   - Visualized price forecasts and error distributions with Plotly.
   - Saved processed data, model, and visualizations.

2. **app.py** (Deployment):
   - Deployed a Streamlit app for real-time Bitcoin price forecasting.
   - Allowed user inputs for forecast horizon.
   - Visualized historical and forecasted prices.

## 📈 Key Results

- **Forecast Performance**:
  - Prophet captures seasonal trends; LSTM models complex price patterns.
  - Low RMSE and MAE indicate reliable forecasts.
- **Visualizations**:
  - Interactive line chart of historical vs. forecasted prices.
  - Histogram of LSTM forecast error distribution.
- **Financial Insights**:
  - Accurate forecasts inform trading strategies and investment decisions.
  - Real-time analysis supports dynamic market monitoring.
  - Applicable to cryptocurrency trading and portfolio management.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `pycoingecko`, `prophet`, `tensorflow`, `plotly`, `streamlit`, `matplotlib`, `seaborn`
- Internet access for CoinGecko API

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
   pycoingecko==3.0.0
   prophet==1.1.4
   tensorflow==2.12.0
   plotly==5.15.0
   streamlit==1.22.0
   matplotlib==3.7.1
   seaborn==0.12.2
   ```

### Running the Project
1. **Train the Model**:
   ```bash
   python train.py
   ```
   This generates the processed data (`processed_bitcoin_data.csv`), model (`lstm_bitcoin_model.keras`), and visualizations (`price_forecast.html`, `error_distribution.html`).

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```
   Access the app at `http://localhost:8501` to forecast Bitcoin prices.

3. **View Visualizations**:
   - Open HTML files (e.g., `price_forecast.html`) in a browser for standalone visualizations.

## 📋 Usage

- **For Financial Stakeholders**: Use the Streamlit app to forecast Bitcoin prices and analyze trends, presenting visualizations for trading strategies.
- **For Data Scientists**: Extend the pipeline with additional models (e.g., ARIMA) or features (e.g., trading volume).
- **For Developers**: Deploy the Streamlit app on a cloud platform (e.g., Streamlit Cloud) for broader access.

## 🔮 Future Improvements

- **Enhanced Features**: Incorporate market sentiment or trading volume data.
- **Multi-Cryptocurrency Support**: Extend to other cryptocurrencies (e.g., Ethereum).
- **Real-Time Data**: Stream live price data via WebSocket APIs.
- **Advanced Visualizations**: Add volatility or trend analysis dashboards.

## 📧 Contact

For questions, feedback, or collaboration opportunities, reach out via:
- **GitHub**: [Ajafarnezhad](https://github.com/Ajafarnezhad)
- **Email**: aiamirjd@gmail.com

## 🙏 Acknowledgments

- **Data Source**: Bitcoin price data from CoinGecko API.
- **Tools**: Built with `pycoingecko`, `prophet`, `tensorflow`, `plotly`, `streamlit`, and other open-source libraries.
- **Inspiration**: Thanks to Aman Kharwal and the data science community for foundational ideas.

---

*This project is part of the [Intermediate Projects](https://github.com/Ajafarnezhad/portfolio/tree/main/Intermediate-Projects) portfolio by Ajafarnezhad, showcasing expertise in time-series forecasting and financial analytics. Last updated: August 15, 2025.*
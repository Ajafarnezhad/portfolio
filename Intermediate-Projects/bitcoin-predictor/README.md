\# Bitcoin Predictor



\## Overview

This intermediate Python project predicts Bitcoin prices using an LSTM neural network on historical data. It includes advanced data preprocessing (cleaning, feature engineering like MA and RSI), comprehensive exploration with visualizations, model training with callbacks and evaluation metrics (MAE, RMSE, R2), and future price forecasting. The project features a modular design, CLI interface, model/scaler persistence, and robust error handling, making it a strong portfolio piece for time series forecasting in finance.



\## Features

\- \*\*Data Loading \& Preprocessing\*\*: Cleans volume data, handles missing values, adds features (moving averages, RSI), and scales with MinMaxScaler.

\- \*\*Exploration\*\*: Generates price trends, correlation heatmaps, and RSI plots.

\- \*\*Model Training\*\*: Builds LSTM with dropout, trains with EarlyStopping and ModelCheckpoint, evaluates on test set.

\- \*\*Prediction\*\*: Forecasts future prices for specified steps using the trained model.

\- \*\*CLI Interface\*\*: Supports modes for training, prediction, and exploration with configurable parameters.

\- \*\*Model Persistence\*\*: Saves/loads model (H5) and scaler (joblib).

\- \*\*Error Handling \& Logging\*\*: Comprehensive checks and detailed logs for debugging.



\## Requirements

\- Python 3.8+

\- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow`, `joblib`



Install dependencies:

```bash

pip install pandas numpy matplotlib seaborn tensorflow joblib





Dataset



The dataset (bitcoin.csv) contains Bitcoin historical data:



Columns: Date, Open, High, Low, Close, Vol.





From 2013 to 2023. Cleaned and engineered in the code.



How to Run



Explore data:

bashpython bitcoin\_predictor.py --mode explore



Train model:

bashpython bitcoin\_predictor.py --mode train --epochs 100 --batch\_size 64 --test\_size 0.1



Predict future prices:

bashpython bitcoin\_predictor.py --mode predict --steps 60





Custom options:



--data\_path: Path to dataset.

--model\_path: Path to save/load model.

--scaler\_path: Path to save/load scaler.



Example Output



Training:

textINFO: LSTM model built and compiled.

INFO: Model training completed.

Test MAE: 500.23, RMSE: 700.45, R2: 0.95



Prediction: Predicted Prices: \[28000.12 28500.34 ...]



Plots saved in plots/ folder: price\_trend.png, correlation\_heatmap.png, rsi\_plot.png, predictions.png.

Improvements and Future Work



Add more features (e.g., sentiment from news, on-chain metrics).

Implement hybrid models (e.g., LSTM + XGBoost).

Deploy as a web app with Streamlit for interactive forecasting.

Use Prophet or ARIMA for comparison.

Add unit tests with pytest for data processing and predictions.



License

MIT License




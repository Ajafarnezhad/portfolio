# Bitcoin Predictor: Forecast Crypto Prices with Confidence 📈💰

Welcome to the **Bitcoin Predictor**, an intermediate Python project that harnesses the power of LSTM neural networks to forecast Bitcoin prices using historical data. Designed for finance enthusiasts and data scientists, this tool offers advanced data preprocessing, insightful visualizations, and accurate price predictions. With a modular design, intuitive CLI interface, and robust error handling, it’s a standout addition to your portfolio for time series forecasting in the exciting world of cryptocurrency.

---

## 🌟 Project Highlights
This project combines cutting-edge deep learning with financial time series analysis, featuring a clean CLI, model persistence, and comprehensive data exploration. It’s perfect for showcasing your skills in machine learning, financial modeling, and software engineering.

---

## 🚀 Features
- **Data Loading & Preprocessing**: Cleans volume data, handles missing values, and enriches with features like Moving Averages (MA) and Relative Strength Index (RSI), scaled using MinMaxScaler.
- **Data Exploration**: Visualize price trends, correlation heatmaps, and RSI plots for deeper insights into Bitcoin’s behavior.
- **Model Training**: Train a high-performance LSTM model with dropout, EarlyStopping, and ModelCheckpoint for optimal results.
- **Price Forecasting**: Predict future Bitcoin prices for a specified number of steps using the trained model.
- **CLI Interface**: Easily switch between training, prediction, and exploration modes with customizable parameters.
- **Model Persistence**: Save and load models (H5 format) and scalers (joblib) for seamless reuse.
- **Error Handling & Logging**: Robust checks and detailed logs ensure smooth operation and easy debugging.

---

## 🛠️ Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`
  - `joblib`

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn tensorflow joblib
```

---

## 📂 Dataset
- **Bitcoin Historical Data**: Source your dataset (e.g., from Kaggle, CoinGecko, or Yahoo Finance) in CSV format with columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- **Setup**: Place the dataset in a folder (e.g., `data/bitcoin.csv`) or specify a custom path via CLI.

---

## 🎮 How to Run

### 1. Explore the Data
Visualize trends and correlations in Bitcoin data:
```bash
python bitcoin_predictor.py --mode explore
```

### 2. Train the Model
Build and train an LSTM model with customizable parameters:
```bash
python bitcoin_predictor.py --mode train --epochs 50 --batch_size 32 --future_steps 7
```

### 3. Predict Future Prices
Forecast Bitcoin prices for a specified number of days:
```bash
python bitcoin_predictor.py --mode predict --future_steps 7
```

### 4. Customize Your Workflow
- `--dataset_path`: Path to the Bitcoin dataset (e.g., `data/bitcoin.csv`).
- `--model_path`: Save/load the trained model (e.g., `models/bitcoin_model.h5`).
- `--scaler_path`: Save/load the scaler (e.g., `models/scaler.joblib`).
- `--future_steps`: Number of days to forecast (e.g., `7` for a week).

---

## 📈 Example Output
- **Exploration**:
  ```
  INFO: Generating visualizations...
  Plots saved: price_trend.png, correlation_heatmap.png, rsi_plot.png
  ```
- **Training**:
  ```
  Epoch 1/50: Loss: 0.045, MAE: 0.032
  ...
  INFO: Best model saved at models/bitcoin_model.h5
  INFO: Test set metrics - MAE: 250.32, RMSE: 320.45, R2: 0.89
  ```
- **Prediction**:
  ```
  Predicted Bitcoin Prices (next 7 days): [45231.50, 45312.75, ...]
  ```

---

## 🔮 Future Enhancements
Take this project to the next level with these exciting ideas:
- **Additional Features**: Incorporate more technical indicators (e.g., MACD, Bollinger Bands) for richer predictions.
- **Alternative Models**: Experiment with Transformer-based models or Prophet for time series forecasting.
- **Web App Deployment**: Build an interactive dashboard with Flask or Streamlit for real-time predictions.
- **Live Data Integration**: Fetch real-time Bitcoin data from APIs like CoinGecko.
- **Unit Testing**: Add `pytest` for robust validation of preprocessing and model evaluation.

---

## 📜 License
This project is licensed under the **MIT License**—use, modify, and share it freely!

Dive into the crypto world with the **Bitcoin Predictor** and forecast the future of finance! 🚀
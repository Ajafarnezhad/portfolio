# 📈 AI-Powered Crypto Trading Bot: Sentiment-Driven Trading with Real-Time Insights

## 🌟 Project Vision
Step into the world of algorithmic trading with the **AI-Powered Crypto Trading Bot**, a sophisticated Python application that leverages sentiment analysis and technical indicators to generate trading signals for cryptocurrencies like Bitcoin (BTC). By streaming real-time comments from Reddit and Twitter (via X API), analyzing sentiment with TextBlob, and integrating the CNN Fear and Greed Index, this bot provides actionable buy/sell signals based on market sentiment and RSI trends. With a modular design, robust error handling, and interactive visualizations, this project is a standout for fintech enthusiasts and data scientists looking to explore sentiment-driven trading.

## ✨ Core Features
- **Real-Time Sentiment Analysis** 🧠: Streams comments from Reddit and Twitter (X) to compute sentiment scores using TextBlob, capturing market mood.
- **Technical Indicators** 📊: Utilizes RSI and SMA from the `ta` library to identify overbought/oversold conditions for precise trading signals.
- **Fear and Greed Index Integration** ⚖️: Scrapes CNN’s Fear and Greed Index to enhance trading decisions with market psychology insights.
- **Asynchronous API Streaming** ⚡: Employs `asyncio` for efficient, non-blocking streaming of Reddit and Twitter (X) data.
- **Binance Integration** 💸: Connects to Binance API for real-time price data and potential trade execution (simulated or live with configuration).
- **Interactive Visualizations** 📈: Generates Plotly charts for sentiment and price trends, saved as HTML files.
- **Robust Error Handling & Logging** 🛡️: Ensures reliability with comprehensive logging and graceful error recovery.
- **Scalable CLI Interface** 🖥️: Offers intuitive commands for running the bot, analyzing data, or visualizing trends.

## 🛠️ Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher
- **API Credentials**:
  - **Reddit**: Obtain `client_id`, `client_secret`, `username`, `password`, and `user_agent` from [Reddit Apps](https://www.reddit.com/prefs/apps).
  - **Binance**: Get `api_key` and `api_secret` from [Binance API Management](https://www.binance.com/en/my/settings/api-management).
  - **Twitter (X)**: Acquire `bearer_token` from [Twitter Developer Portal](https://developer.twitter.com).
- **Dependencies**: A curated suite of libraries:
  - `pandas`
  - `numpy`
  - `praw`
  - `requests`
  - `textblob`
  - `python-binance`
  - `ta`
  - `beautifulsoup4`
  - `plotly`
  - `langdetect`
  - `tweepy` (for Twitter/X streaming)

Install them with:
```bash
pip install pandas numpy praw requests textblob python-binance ta beautifulsoup4 plotly langdetect tweepy
```

### Configuration
1. Create a `config.py` file in the project root with:
```python
# config.py
REDDIT_ID = "your_reddit_client_id"
REDDIT_SEC = "your_reddit_client_secret"
REDDIT_USER = "your_reddit_username"
REDDIT_PASS = "your_reddit_password"
BINANCE_KEY = "your_binance_api_key"
BINANCE_SEC = "your_binance_api_secret"
BEARER_TOKEN = "your_twitter_bearer_token"
```

2. Ensure secure storage of `config.py` (e.g., add to `.gitignore`).

## 🎉 How to Use

### 1. Run the Trading Bot
Start the bot to stream sentiment data and generate trading signals:
```bash
python main.py --mode trade
```

### 2. Analyze Sentiment Data
Perform exploratory data analysis on collected sentiments:
```bash
python main.py --mode analyze
```

### 3. Visualize Trends
Generate interactive Plotly charts for sentiment and price trends:
```bash
python main.py --mode visualize
```

### CLI Options
- `--mode`: Choose `trade` (run bot), `analyze` (EDA), or `visualize` (plot generation) (default: `trade`).
- `--symbol`: Trading pair (e.g., `BTCUSDT`, default: `BTCUSDT`).
- `--output_dir`: Directory for saving visualizations (default: `./plots`).
- `--sentiment_count`: Number of sentiments to average (default: 300).

## 📊 Sample Output

### Trading Output
```
2025-08-23 20:38:12,345 - INFO - Reddit Sentiment: 0.2456 (300 comments)
2025-08-23 20:38:12,567 - INFO - Twitter Sentiment: 0.2103 (300 tweets)
2025-08-23 20:38:12,789 - INFO - RSI: 28.45, Fear and Greed: 35
2025-08-23 20:38:12,901 - INFO - Signal: BUY (BTCUSDT)
```

### Analysis Output
```
🌟 Sentiment Analysis Summary:
Reddit Sentiment (last 300): 0.2456
Twitter Sentiment (last 300): 0.2103
Price Trend (BTCUSDT): $60,450.32 (1-min candles)
RSI: 28.45 (Oversold)
Fear and Greed Index: 35 (Fear)
```

### Visualizations
Find interactive plots in the `plots/` folder:
- `sentiment_trend.html`: Line chart of Reddit and Twitter sentiment over time.
- `price_rsi_trend.html`: Combined chart of BTCUSDT price and RSI.

## 🌈 Future Enhancements
- **Advanced Sentiment Models** 🚀: Integrate BERT or VADER for more accurate sentiment analysis.
- **Live Trading Execution** 💰: Enable actual trade placement via Binance API with user confirmation.
- **Web Dashboard** 🌐: Develop a Streamlit or Flask app for real-time signal visualization.
- **Additional Data Sources** 📡: Incorporate news APIs or other social platforms for broader sentiment analysis.
- **Backtesting Framework** 📜: Add support for historical data backtesting to evaluate strategy performance.
- **Unit Testing** 🛠️: Implement `pytest` for robust validation of trading logic.

## 📜 License
Proudly licensed under the **MIT License**, fostering open collaboration and innovation in algorithmic trading.

---

🌟 **AI-Powered Crypto Trading Bot**: Where sentiment meets strategy for smarter trading! 🌟
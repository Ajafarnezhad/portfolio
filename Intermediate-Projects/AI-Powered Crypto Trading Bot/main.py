import asyncio
import logging
import pandas as pd
import numpy as np
import praw
import tweepy
import requests
from textblob import TextBlob
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1MINUTE
from ta.momentum import RSIIndicator
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import re
import plotly.graph_objs as go
import argparse
from typing import List, Tuple
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentTradingBot:
    """A sentiment-driven crypto trading bot using Reddit, Twitter, and Binance data."""
    
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_ID,
            client_secret=config.REDDIT_SEC,
            password=config.REDDIT_PASS,
            user_agent="CryptoSentimentBot/1.0",
            username=config.REDDIT_USER,
        )
        self.binance = Client(config.BINANCE_KEY, config.BINANCE_SEC)
        self.twitter_auth = tweepy.OAuth2BearerHandler(config.BEARER_TOKEN)
        self.twitter_api = tweepy.API(self.twitter_auth)
        self.sentiment_count = 300
        self.reddit_sentiments = []
        self.twitter_sentiments = []
        self.prices = []
        self.trade_symbol = 'BTCUSDT'
        self.upper_band = 70
        self.lower_band = 30
        self.output_dir = './plots'

    async def fear_and_greed(self) -> int:
        """Scrape CNN Fear and Greed Index."""
        try:
            url = "https://money.cnn.com/data/fear-and-greed/"
            req = Request(url, headers={'user-agent': 'CryptoBot/1.0'})
            response = urlopen(req)
            html = BeautifulSoup(response, 'html.parser')
            needle = html.find(id='needleChart')
            data_rows = needle.findAll('li')
            index_str = re.findall(r'[0-9]+', str(data_rows[0]))[0]
            logging.info(f"Fear and Greed Index: {index_str}")
            return int(index_str)
        except Exception as e:
            logging.error(f"Error fetching Fear and Greed Index: {str(e)}")
            return 50  # Default value on error

    def average_sentiment(self, sentiments: List[float]) -> float:
        """Calculate average sentiment for the last N entries."""
        if len(sentiments) == 0:
            return 0.0
        return round(sum(sentiments[-self.sentiment_count:]) / min(len(sentiments), self.sentiment_count), 4)

    async def reddit_stream(self):
        """Stream Reddit comments and compute sentiment."""
        try:
            async for comment in self.reddit.subreddit("bitcoin").stream.comments(skip_existing=True):
                blob = TextBlob(comment.body)
                polarity = blob.sentiment.polarity
                if polarity != 0.0:
                    self.reddit_sentiments.append(polarity)
                    avg = self.average_sentiment(self.reddit_sentiments)
                    logging.info(f"Reddit Sentiment: {avg} ({len(self.reddit_sentiments)} comments)")
                    await self.evaluate_trading_signal()
        except Exception as e:
            logging.error(f"Reddit stream error: {str(e)}")

    async def twitter_stream(self):
        """Stream Twitter (X) tweets and compute sentiment."""
        class StreamListener(tweepy.Stream):
            def __init__(self, bot, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.bot = bot

            def on_status(self, status):
                tweet = status.text
                tweet = re.sub(r'http\S+|[^a-zA-Z0-9\s]', '', tweet)
                blob = TextBlob(tweet)
                polarity = blob.sentiment.polarity
                if polarity != 0.0:
                    self.bot.twitter_sentiments.append(polarity)
                    avg = self.bot.average_sentiment(self.bot.twitter_sentiments)
                    logging.info(f"Twitter Sentiment: {avg} ({len(self.bot.twitter_sentiments)} tweets)")
                    asyncio.create_task(self.bot.evaluate_trading_signal())

            def on_error(self, status_code):
                logging.error(f"Twitter stream error: {status_code}")
                return True

        try:
            stream = StreamListener(self, auth=self.twitter_auth)
            stream.filter(track=['bitcoin'], languages=['en'], threaded=True)
        except Exception as e:
            logging.error(f"Twitter stream error: {str(e)}")

    async def fetch_prices(self):
        """Fetch recent price data from Binance."""
        try:
            candles = self.binance.get_historical_klines(
                self.trade_symbol, KLINE_INTERVAL_1MINUTE, "1 Minutes ago UTC"
            )
            price = float(candles[-1][4])  # Closing price
            if not self.prices or self.prices[-1] != price:
                self.prices.append(price)
            logging.info(f"Current {self.trade_symbol} price: ${price:,.2f}")
        except Exception as e:
            logging.error(f"Error fetching prices: {str(e)}")

    async def evaluate_trading_signal(self):
        """Evaluate trading conditions and generate signals."""
        await self.fetch_prices()
        if len(self.prices) < 14:  # Minimum for RSI
            return

        rsi = RSIIndicator(pd.Series(self.prices)).rsi().iloc[-1]
        fear_greed = await self.fear_and_greed()
        reddit_avg = self.average_sentiment(self.reddit_sentiments)
        twitter_avg = self.average_sentiment(self.twitter_sentiments)

        if (rsi < self.lower_band and reddit_avg > 0.2 and twitter_avg > 0.2 and
                len(self.reddit_sentiments) >= self.sentiment_count and
                len(self.twitter_sentiments) >= self.sentiment_count and fear_greed < 40):
            logging.info(f"Signal: BUY ({self.trade_symbol}) - RSI: {rsi:.2f}, Fear & Greed: {fear_greed}")
        elif (rsi > self.upper_band and reddit_avg < -0.2 and twitter_avg < -0.2 and
              len(self.reddit_sentiments) >= self.sentiment_count and
              len(self.twitter_sentiments) >= self.sentiment_count and fear_greed > 60):
            logging.info(f"Signal: SELL ({self.trade_symbol}) - RSI: {rsi:.2f}, Fear & Greed: {fear_greed}")

    def generate_visualizations(self):
        """Generate Plotly visualizations for sentiment and price trends."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Sentiment trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.reddit_sentiments, name='Reddit Sentiment', mode='lines'))
        fig.add_trace(go.Scatter(y=self.twitter_sentiments, name='Twitter Sentiment', mode='lines'))
        fig.update_layout(title='Sentiment Trends', xaxis_title='Sample', yaxis_title='Sentiment Polarity')
        fig.write_html(f"{self.output_dir}/sentiment_trend.html")

        # Price and RSI
        rsi = RSIIndicator(pd.Series(self.prices)).rsi()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.prices, name='Price', mode='lines'))
        fig.add_trace(go.Scatter(y=rsi, name='RSI', mode='lines', yaxis='y2'))
        fig.update_layout(
            title=f'{self.trade_symbol} Price and RSI',
            xaxis_title='Time',
            yaxis_title='Price (USDT)',
            yaxis2=dict(title='RSI', overlaying='y', side='right')
        )
        fig.write_html(f"{self.output_dir}/price_rsi_trend.html")
        logging.info(f"Visualizations saved to {self.output_dir}")

    def analyze_data(self):
        """Perform exploratory data analysis."""
        reddit_avg = self.average_sentiment(self.reddit_sentiments)
        twitter_avg = self.average_sentiment(self.twitter_sentiments)
        rsi = RSIIndicator(pd.Series(self.prices)).rsi().iloc[-1] if len(self.prices) >= 14 else None
        fear_greed = asyncio.run(self.fear_and_greed())
        logging.info(
            f"Sentiment Analysis Summary:\n"
            f"Reddit Sentiment (last {self.sentiment_count}): {reddit_avg}\n"
            f"Twitter Sentiment (last {self.sentiment_count}): {twitter_avg}\n"
            f"Price Trend ({self.trade_symbol}): ${self.prices[-1]:,.2f}\n"
            f"RSI: {rsi:.2f if rsi else 'N/A'}\n"
            f"Fear and Greed Index: {fear_greed}"
        )

async def main():
    parser = argparse.ArgumentParser(description="AI-Powered Crypto Trading Bot")
    parser.add_argument('--mode', choices=['trade', 'analyze', 'visualize'], default='trade')
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--output_dir', default='./plots')
    parser.add_argument('--sentiment_count', type=int, default=300)
    args = parser.parse_args()

    bot = SentimentTradingBot()
    bot.trade_symbol = args.symbol
    bot.output_dir = args.output_dir
    bot.sentiment_count = args.sentiment_count

    if args.mode == 'trade':
        await asyncio.gather(bot.reddit_stream(), bot.twitter_stream())
    elif args.mode == 'analyze':
        bot.analyze_data()
    elif args.mode == 'visualize':
        bot.generate_visualizations()

if __name__ == "__main__":
    asyncio.run(main())
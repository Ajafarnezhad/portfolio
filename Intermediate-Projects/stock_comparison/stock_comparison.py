import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pio.templates.default = "plotly_white"

def fetch_stock_data(tickers, start_date, end_date, benchmark_ticker='^GSPC'):
    """
    Fetch historical stock data for given tickers and benchmark.
    
    Args:
        tickers (list): List of stock tickers.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        benchmark_ticker (str): Ticker for market benchmark (default: ^GSPC).
    
    Returns:
        dict: Dictionary of DataFrames for each ticker and benchmark.
    """
    try:
        data = {}
        for ticker in tickers + [benchmark_ticker]:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty:
                raise ValueError(f"No data fetched for {ticker}")
            df['Daily_Return'] = df['Adj Close'].pct_change()
            data[ticker] = df
        logging.info(f"Fetched data for {', '.join(tickers + [benchmark_ticker])} from {start_date} to {end_date}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise

def calculate_metrics(data, tickers, benchmark_ticker):
    """
    Calculate performance metrics (volatility, beta, cumulative returns).
    
    Args:
        data (dict): Dictionary of DataFrames for each ticker and benchmark.
        tickers (list): List of stock tickers.
        benchmark_ticker (str): Ticker for market benchmark.
    
    Returns:
        dict: Metrics including volatility, beta, and cumulative returns.
    """
    try:
        metrics = {}
        for ticker in tickers:
            df = data[ticker]
            metrics[ticker] = {
                'volatility': df['Daily_Return'].std(),
                'cumulative_return': (1 + df['Daily_Return']).cumprod() - 1
            }
            cov = df['Daily_Return'].cov(data[benchmark_ticker]['Daily_Return'])
            var_market = data[benchmark_ticker]['Daily_Return'].var()
            metrics[ticker]['beta'] = cov / var_market if var_market != 0 else np.nan
        logging.info("Metrics calculated successfully")
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        raise

def visualize_data(data, metrics, tickers, output_dir):
    """
    Generate interactive visualizations for stock comparison.
    
    Args:
        data (dict): Dictionary of DataFrames for each ticker.
        metrics (dict): Calculated metrics for each ticker.
        tickers (list): List of stock tickers.
        output_dir (str): Directory to save plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Daily Returns
        fig = go.Figure()
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, ticker in enumerate(tickers):
            fig.add_trace(go.Scatter(x=data[ticker].index, y=data[ticker]['Daily_Return'],
                                     mode='lines', name=ticker, line=dict(color=colors[i % len(colors)])))
        fig.update_layout(title='Daily Returns Comparison',
                          xaxis_title='Date', yaxis_title='Daily Return',
                          legend=dict(x=0.02, y=0.95))
        fig.write(os.path.join(output_dir, 'daily_returns.png'))
        logging.info("Saved daily returns plot")

        # Cumulative Returns
        fig = go.Figure()
        for i, ticker in enumerate(tickers):
            fig.add_trace(go.Scatter(x=metrics[ticker]['cumulative_return'].index,
                                     y=metrics[ticker]['cumulative_return'],
                                     mode='lines', name=ticker, line=dict(color=colors[i % len(colors)])))
        fig.update_layout(title='Cumulative Returns Comparison',
                          xaxis_title='Date', yaxis_title='Cumulative Return',
                          legend=dict(x=0.02, y=0.95))
        fig.write(os.path.join(output_dir, 'cumulative_returns.png'))
        logging.info("Saved cumulative returns plot")

        # Volatility Comparison
        fig = go.Figure()
        fig.add_bar(x=tickers, y=[metrics[ticker]['volatility'] for ticker in tickers],
                    text=[f"{metrics[ticker]['volatility']:.4f}" for ticker in tickers],
                    textposition='auto', marker=dict(color=colors[:len(tickers)]))
        fig.update_layout(title='Volatility Comparison',
                          xaxis_title='Stock', yaxis_title='Volatility (Standard Deviation)',
                          bargap=0.5)
        fig.write(os.path.join(output_dir, 'volatility_comparison.png'))
        logging.info("Saved volatility comparison plot")

    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for stock comparison analysis.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Validate inputs
        tickers = args.tickers.split(',')
        if not tickers:
            raise ValueError("At least one ticker must be provided")
        
        # Set default dates if not provided
        if not args.end_date:
            args.end_date = datetime.now().strftime('%Y-%m-%d')
        if not args.start_date:
            args.start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        # Fetch data
        data = fetch_stock_data(tickers, args.start_date, args.end_date, args.benchmark)

        # Calculate metrics
        metrics = calculate_metrics(data, tickers, args.benchmark)

        if args.mode == 'analyze':
            # Print metrics
            print("🌟 Analysis Results:")
            for ticker in tickers:
                print(f"- {ticker} Volatility: {metrics[ticker]['volatility']:.4f} | Beta: {metrics[ticker]['beta']:.4f}")
            max_beta_ticker = max(tickers, key=lambda x: metrics[x]['beta'])
            print(f"✅ Conclusion: {max_beta_ticker} is more volatile (higher Beta) compared to others")

        if args.mode == 'visualize':
            # Generate visualizations
            visualize_data(data, metrics, tickers, args.output_dir)

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stock Market Comparison Analysis: Compare stock performance with Python")
    parser.add_argument('--mode', choices=['analyze', 'visualize'], default='analyze', help="Mode: analyze or visualize")
    parser.add_argument('--tickers', default='AAPL,GOOGL', help="Comma-separated list of stock tickers")
    parser.add_argument('--start_date', default=None, help="Start date (YYYY-MM-DD, default: 90 days ago)")
    parser.add_argument('--end_date', default=None, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument('--benchmark', default='^GSPC', help="Benchmark ticker (default: S&P 500)")
    parser.add_argument('--output_dir', default='./plots', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)
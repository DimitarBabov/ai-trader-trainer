import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_data(ticker, start_date, interval):
    """
    Fetch historical data for a given ticker, date range, and interval.
    """
    return yf.download(ticker, start=start_date, interval=interval)

def calculate_moving_average(data, period):
    """
    Calculate the simple moving average (SMA).
    """
    return data.rolling(window=period).mean()

def calculate_derivative(ma):
    """
    Calculate the rate of change (derivative) of a moving average.
    """
    return ma.diff()

def backtest_strategy(data, threshold):
    """
    Backtest the buy/sell strategy.
    """
    # Compute moving averages
    data['MA_weekly'] = calculate_moving_average(data['weekly'], 5)
    data['MA_daily'] = calculate_moving_average(data['daily'], 5)
    data['MA_hourly'] = calculate_moving_average(data['hourly'], 5)
    
    # Compute derivatives
    data['dMA_weekly'] = calculate_derivative(data['MA_weekly'])
    data['dMA_daily'] = calculate_derivative(data['MA_daily'])
    data['dMA_hourly'] = calculate_derivative(data['MA_hourly'])
    
    # Buy/Sell Signals
    data['Buy'] = (
        (data['dMA_weekly'] > -threshold) &
        (data['dMA_daily'] < -threshold) &
        (data['dMA_hourly'] < -threshold)
    )
    data['Sell'] = (
        (data['dMA_daily'] > threshold) &
        (data['dMA_hourly'] > threshold)
    )
    
    # Track holdings
    data['Position'] = 0
    position = 0
    for i in range(len(data)):
        if data['Buy'].iloc[i]:
            position = 1  # Buy
        elif data['Sell'].iloc[i]:
            position = 0  # Sell
        data['Position'].iloc[i] = position
    
    return data

def plot_strategy(data):
    """
    Plot the strategy results.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot close price
    plt.plot(data.index, data['daily'], label='Daily Price', color='blue', alpha=0.5)
    
    # Plot moving averages
    plt.plot(data.index, data['MA_weekly'], label='Weekly MA (5)', color='orange', linestyle='--')
    plt.plot(data.index, data['MA_daily'], label='Daily MA (5)', color='green', linestyle='--')
    plt.plot(data.index, data['MA_hourly'], label='Hourly MA (5)', color='red', linestyle='--')
    
    # Plot buy signals
    buy_signals = data[data['Buy']]
    plt.scatter(buy_signals.index, buy_signals['daily'], marker='^', color='green', label='Buy Signal', alpha=1, zorder=5)
    
    # Plot sell signals
    sell_signals = data[data['Sell']]
    plt.scatter(sell_signals.index, sell_signals['daily'], marker='v', color='red', label='Sell Signal', alpha=1, zorder=5)
    
    # Add labels and legend
    plt.title('Trading Strategy: Moving Averages and Signals', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

# Main Function
if __name__ == "__main__":
    ticker = "SLV"  # Example stock ticker
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    threshold = 0.01  # Define your threshold for slopes
    
    # Fetch data for weekly, daily, and hourly intervals
    data_weekly = fetch_data(ticker, start_date, "1wk")
    data_daily = fetch_data(ticker, start_date, "1d")
    data_hourly = fetch_data(ticker, start_date, "1h")
    
    # Combine data into a single DataFrame
    data_combined = pd.DataFrame({
        'weekly': data_weekly['Close'],
        'daily': data_daily['Close'].reindex(data_weekly.index),
        'hourly': data_hourly['Close'].reindex(data_weekly.index)
    }).dropna()
    
    # Backtest the strategy
    results = backtest_strategy(data_combined, threshold)
    
    # Plot the strategy results
    plot_strategy(results)

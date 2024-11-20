import yfinance as yf
import pandas as pd
import os
# Function to fetch and save data for specific intervals
def fetch_yahoo_finance_data(ticker):
    # Dictionary to store the results
    intervals = {
        '1d': {'interval': '1d', 'period': 'max'},   # Daily data
        '5d': {'interval': '5d', 'period': 'max'},   # 5-day data
        '1wk': {'interval': '1wk', 'period': 'max'}, # Weekly data
        '1mo': {'interval': '1mo', 'period': 'max'}, # Monthly data
        '3mo': {'interval': '3mo', 'period': 'max'}, # Quarterly data
    }

    # Create a directory to store the data if not exists
    output_dir = f"data_csv/{ticker}/"
    os.makedirs(output_dir, exist_ok=True)
    
    for timeframe, settings in intervals.items():
        try:
            # Download the data
            print(f"Fetching {timeframe} data for {ticker}...")
            data = yf.download(ticker, interval=settings['interval'], period=settings['period'], progress=False)
            
            # Check if data is returned
            if not data.empty:
                # Save data to CSV
                csv_filename = f"{output_dir}{ticker}_{timeframe}_data.csv"
                data.to_csv(csv_filename)
                print(f"{timeframe} data saved to {csv_filename}.")
            else:
                print(f"No {timeframe} data available for {ticker}.")
        
        except Exception as e:
            print(f"Error fetching {timeframe} data for {ticker}: {e}")

# Main function to fetch data for a given ticker
def main():
    ticker = input("Enter the ticker symbol (e.g., AAPL, MSFT, BTC-USD): ").upper()
    
    # Fetch and save the data
    fetch_yahoo_finance_data(ticker)

if __name__ == '__main__':
    main()

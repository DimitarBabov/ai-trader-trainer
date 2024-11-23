import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_max_1h_data(ticker):
    """
    Download the maximum available 1-hour interval data for a given ticker from Yahoo Finance.

    Parameters:
        ticker (str): Stock/crypto/forex ticker symbol (e.g., "AAPL", "BTC-USD").

    Returns:
        pandas.DataFrame: A DataFrame with the 1-hour interval data.
    """
    try:
        # Calculate the maximum allowable start date (730 days from today)
        end_date = datetime.now().strftime('%Y-%m-%d')  # Today's date
        start_date = (datetime.now() - timedelta(days=729)).strftime('%Y-%m-%d')  # 730 days ago

        # Download 1-hour data
        print(f"Downloading 1-hour data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, interval="1h", start=start_date, end=end_date)

        # Check if data is empty
        if data.empty:
            print(f"No data found for {ticker}.")
        else:
            print("Sample Data:")
            print(data.head())

        return data

    except Exception as e:
        print(f"Error while downloading data for {ticker}: {e}")
        return None

# Specify the ticker
ticker = "SLV"  # Example: Apple stock

# Download the maximum available data
data = download_max_1h_data(ticker)

# Save to a CSV file
if data is not None and not data.empty:
    output_file = f"{ticker}_1h_data.csv"
    data.to_csv(output_file)
    print(f"1-hour data saved to '{output_file}'.")

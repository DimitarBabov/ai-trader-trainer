import yfinance as yf
import pandas as pd

def download_1m_data(ticker, start_date, end_date):
    """
    Download 1-minute interval data for a given ticker from Yahoo Finance.

    Parameters:
        ticker (str): Stock/crypto/forex ticker symbol (e.g., "AAPL", "BTC-USD").
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame with the 1-minute interval data.
    """
    try:
        # Download 1-minute data
        print(f"Downloading 1-minute data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, interval="1m", start=start_date, end=end_date)

        # Show the first few rows
        print("Sample Data:")
        print(data.head())

        return data

    except Exception as e:
        print(f"Error while downloading data: {e}")
        return None

# Specify parameters
ticker = "SLV"  # Example: Apple stock
start_date = "2024-10-25"  # Start date (must be within 7 days for 1m data)
end_date = "2024-11-01"    # End date (must be within 7 days of today)

# Download data
data = download_1m_data(ticker, start_date, end_date)

# Save to a CSV file
if data is not None:
    output_file = f"{ticker}_1m_data.csv"
    data.to_csv(output_file)
    print(f"1-minute data saved to '{output_file}'.")

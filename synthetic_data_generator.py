import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Load the data from the specified directory
file_path = "data_csv/SLV/SLV_1d_data.csv"  # Update path if needed
data = pd.read_csv(file_path)

# Display the first few rows to inspect the structure
print(data.head())

# Assume columns: ['Date', 'Open', 'High', 'Low', 'Close'] exist
# Adjust column names if necessary based on your dataset

# 1. Calculate distributions
high_wick_diff = data['High'] - np.maximum(data['Open'], data['Close'])
low_wick_diff = np.minimum(data['Open'], data['Close']) - data['Low']
close_open_diff = data['Close'] - data['Open']
starting_open_sample = data['Open']

# 2. Fit KDE distributions
high_wick_kde = gaussian_kde(high_wick_diff[high_wick_diff > 0])  # Remove anomalies
low_wick_kde = gaussian_kde(low_wick_diff[low_wick_diff > 0])    # Remove anomalies
close_open_kde = gaussian_kde(close_open_diff)

# 3. Visualize distributions
plt.figure(figsize=(12, 6))
plt.hist(high_wick_diff, bins=50, density=True, alpha=0.5, label='High Wick Diff')
plt.hist(low_wick_diff, bins=50, density=True, alpha=0.5, label='Low Wick Diff')
plt.title("Distributions of Wick Sizes (High-Wick and Low-Wick)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(close_open_diff, bins=50, density=True, alpha=0.5, label='Close-Open Body Height')
plt.title("Distribution of Candle Body Height (Close - Open)")
plt.legend()
plt.show()

# 4. Generate synthetic candlesticks
def generate_candlesticks_within_bounds(n_candles, slope, distance_pct, high_wick_kde, low_wick_kde, close_open_kde, starting_open_sample):
    """
    Generate synthetic candlestick (HOLC) data with all candles staying within trend lines.
    
    Parameters:
    - n_candles: Number of candlesticks to generate.
    - slope: Slope of the trend lines (price units per candlestick).
    - distance_pct: Vertical distance between trend lines as a percentage of initial price.
    - high_wick_kde, low_wick_kde: KDE objects for high wick and low wick differences.
    - close_open_kde: KDE object for body height (Close - Open).
    - starting_open_sample: Series of sampled starting Open prices.
    
    Returns:
    - candlesticks: List of dictionaries with High, Open, Low, Close prices.
    - trend_lines: Tuple of upper and lower trend lines (arrays).
    """
    # Randomly pick a starting Open price from the CSV data
    initial_price = np.random.choice(starting_open_sample)
    
    # Define trend lines
    x = np.arange(n_candles)
    centerline = initial_price + slope * x
    distance = (distance_pct / 100) * initial_price
    upper_line = centerline + distance / 2
    lower_line = centerline - distance / 2
    
    candlesticks = []
    prev_close = initial_price  # Initialize with the random starting Open price
    
    for i in range(n_candles):
        # Open (current candle) is equal to Close (previous candle)
        open_price = prev_close
        
        # Generate body height (Close - Open)
        body_height = close_open_kde.resample(1)[0]
        close_price = open_price + body_height
        
        # Ensure Open and Close stay within trend lines
        open_price = min(max(open_price, lower_line[i]), upper_line[i])
        close_price = min(max(close_price, lower_line[i]), upper_line[i])
        
        # Generate wicks
        high_wick = abs(high_wick_kde.resample(1)[0])
        low_wick = abs(low_wick_kde.resample(1)[0])
        high_price = max(open_price, close_price) + high_wick
        low_price = min(open_price, close_price) - low_wick
        
        # Constrain High and Low within trend lines
        high_price = min(high_price, upper_line[i])
        low_price = max(low_price, lower_line[i])
        
        # Append candlestick
        candlesticks.append({
            'High': high_price,
            'Open': open_price,
            'Low': low_price,
            'Close': close_price
        })
        
        # Update prev_close for the next iteration
        prev_close = close_price
    
    return candlesticks, (upper_line, lower_line)

# Parameters for synthetic generation
n_candles = 16
slope = 0.5  # Price units per candlestick
distance_pct = 5  # Distance between trend lines as a percentage of initial price

# Generate synthetic data
candlesticks, (upper_line, lower_line) = generate_candlesticks_within_bounds(
    n_candles=n_candles,
    slope=slope,
    distance_pct=distance_pct,
    high_wick_kde=high_wick_kde,
    low_wick_kde=low_wick_kde,
    close_open_kde=close_open_kde,
    starting_open_sample=starting_open_sample
)

# Extract values for plotting
highs = [c['High'] for c in candlesticks]
opens = [c['Open'] for c in candlesticks]
lows = [c['Low'] for c in candlesticks]
closes = [c['Close'] for c in candlesticks]

# Plot the generated candlestick chart
plt.figure(figsize=(12, 6))
plt.plot(upper_line, label="Upper Trend Line", color="green", linestyle="--")
plt.plot(lower_line, label="Lower Trend Line", color="red", linestyle="--")

for i in range(len(candlesticks)):
    color = 'green' if closes[i] > opens[i] else 'red'
    plt.plot([i, i], [lows[i], highs[i]], color='black')  # High-Low line
    plt.plot([i, i], [opens[i], closes[i]], color=color, linewidth=4)  # Open-Close bar

plt.title("Synthetic Candlestick Data (All Candles Within Trend Lines)")
plt.xlabel("Candlestick Index")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

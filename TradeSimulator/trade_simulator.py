import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

# Prompt user for ticker and timeframe
ticker = input("Enter the ticker symbol: ")
timeframe = input("Enter the timeframe (e.g., '1h'): ")

# Specify directories and files
labels_json_path = os.path.join('data_processed_imgs', ticker, timeframe, 'labels', f'{ticker}_{timeframe}_labels.json')
csv_data_path = os.path.join('data_csv', ticker, f'{ticker}_{timeframe}_data.csv')

# Load JSON labels
def load_json_file(filename, default_value):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:  # File exists and is not empty
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {filename} is corrupted. Reinitializing.")
    return default_value

labels_data = load_json_file(labels_json_path, {})

# Load CSV data
if not os.path.exists(csv_data_path):
    print(f"Error: CSV file not found at {csv_data_path}")
    exit()

csv_data = pd.read_csv(csv_data_path, parse_dates=['Datetime'])
csv_data.set_index('Datetime', inplace=True)


# Variables to track holdings and trades
holdings = 0
buy_price = 0.0
cash = 1000.0  # Start with $1000

# Lists to store trading actions for plotting
dates = []
prices = []
actions = []
balances = []

# Process labeled images and execute trades
for image_name, label in labels_data.items():
    # Extract datetime from the image file name
    date_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", image_name)
    if not date_match:
        print(f"Warning: Could not extract datetime from {image_name}")
        continue

    image_datetime = pd.to_datetime(date_match.group(), format="%Y-%m-%d %H-%M-%S", utc=True)
    
    
  
    # Check if the datetime exists in the CSV data
    # Debug loop to check each csv_data.index
    
    
    if image_datetime  not in csv_data.index:
        print(f"Warning: No data for datetime {image_datetime} in CSV for image {image_name}")
        continue
    # Get the next row for trading
    try:
        current_idx = csv_data.index.get_loc(image_datetime)
        if current_idx + 1 >= len(csv_data):
            print(f"Warning: No next row available for datetime {image_datetime}")
            continue
        next_row = csv_data.iloc[current_idx + 1]
    except KeyError:
        print(f"Warning: Datetime {image_datetime} not found in CSV for image {image_name}")
        continue

    next_open_price = next_row['Open']

    # Execute trades based on label
    if label < 0 and holdings == 0 and (not actions or actions[-1] == 'Sell'):  # Buy signal, ensure no consecutive buys  # Buy signal
            holdings = 10
            buy_price = next_open_price
            cash -= buy_price * 10
            equity = cash + (holdings * next_open_price)
            print(f"Bought 10 shares at {buy_price:.2f} on {next_row.name}. Cash balance: ${cash:.2f} Equity: ${equity:.2f}")
            dates.append(next_row.name)
            prices.append(buy_price)
            actions.append('Buy')
            balances.append(cash)
    elif label > 0 and holdings > 0 and (not actions or actions[-1] == 'Buy'):  # Sell signal, ensure no consecutive sells  # Sell signal
            shares_to_sell = 10  # Sell a maximum of 10 shares
            sell_price = next_open_price
            cash += sell_price * shares_to_sell
            holdings -= shares_to_sell
            equity = cash + (holdings * next_open_price)
            print(f"Sold {shares_to_sell} shares at {sell_price:.2f} on {next_row.name}. Cash balance: ${cash:.2f} Equity: ${equity:.2f}")
            dates.append(next_row.name)
            prices.append(sell_price)
            actions.append('Sell')
            balances.append(cash)

# Summary of final portfolio
print(f"\nFinal Holdings: {holdings} shares")
print(f"Final Cash Balance: ${cash:.2f}")

# Plot the trades on the price chart
plt.figure(figsize=(12, 6))
plt.plot(csv_data.index, csv_data['Open'], label='Open Price', color='blue', alpha=0.5)

# Add buy and sell markers
# Add buy and sell markers without duplicate entries
buy_dates = [date for date, action in zip(dates, actions) if action == 'Buy']
buy_prices = [price for price, action in zip(prices, actions) if action == 'Buy']
sell_dates = [date for date, action in zip(dates, actions) if action == 'Sell']
sell_prices = [price for price, action in zip(prices, actions) if action == 'Sell']

# Plot buy and sell markers
for date, price in zip(buy_dates, buy_prices):
    plt.scatter(date, price, color='green', marker='^', label='Buy', zorder=5)
for date, price in zip(sell_dates, sell_prices):
    plt.scatter(date, price, color='red', marker='v', label='Sell', zorder=5)

# Ensure unique legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title(f'Trade Actions for {ticker.upper()} ({timeframe})')
plt.grid(True)
plt.show()

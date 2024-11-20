import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Prompt user for ticker and timeframe
ticker = input("Enter the ticker symbol: ")
timeframe = input("Enter the timeframe (e.g., '1d'): ")

# Specify directories and files
labels_json = os.path.join('data_processed_imgs', ticker, timeframe, 'labels', f'{ticker}_{timeframe}_labels.json')
csv_data_path = os.path.join('data_csv', ticker, f'{ticker}_{timeframe}_data.csv')

# Load existing labels or initialize an empty dictionary
def load_json_file(filename, default_value):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:  # Check if file exists and is not empty
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {filename} is corrupted. Reinitializing.")
    return default_value

labels_data = load_json_file(labels_json, {})

# Load CSV data
if not os.path.exists(csv_data_path):
    print(f"Error: CSV file not found at {csv_data_path}")
    exit()

csv_data = pd.read_csv(csv_data_path, parse_dates=['Date'])
csv_data.set_index('Date', inplace=True)

# Variables to track holdings
holdings = 0
buy_price = 0.0
cash = 1000.0  # Start with $1000

# Lists to store trading actions for plotting
dates = []
prices = []
actions = []
balances = []

# Iterate through labeled images and execute trades based on signals
for image_name, label in labels_data.items():
    # Extract date from image name
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", image_name)
    if not date_match:
        continue

    image_date = pd.to_datetime(date_match.group(), format="%Y-%m-%d")

    # Find the next available row in the CSV
    if image_date not in csv_data.index:
        print(f"Warning: No data for date ({image_date.date()}) for image {image_name}")
        continue

    try:
        next_row_index = csv_data.index.get_loc(image_date) + 1
        if next_row_index >= len(csv_data):
            print(f"Warning: No next day data available for image {image_name}")
            continue
    except KeyError:
        print(f"Warning: Date {image_date.date()} not found in CSV for image {image_name}")
        continue

    next_day_open_price = csv_data.iloc[next_row_index]['Open']

    # Execute trades based on label
    if label == 1:  # Buy signal
        if holdings == 0:
            holdings = 10
            buy_price = next_day_open_price
            cash -= buy_price * 10
            equity = cash + (holdings * next_day_open_price)
            print(f"Bought 10 shares at {buy_price} on {csv_data.index[next_row_index].date()}. Cash balance: ${cash:.2f} Equity: ${equity:.2f}")
            dates.append(csv_data.index[next_row_index])
            prices.append(buy_price)
            actions.append('Buy')
            balances.append(cash)
    elif label == -1:  # Sell signal
        if holdings > 0:
            shares_to_sell = min(holdings, 10)  # Sell a maximum of 10 shares
            sell_price = next_day_open_price
            cash += sell_price * shares_to_sell
            holdings -= shares_to_sell
            equity = cash + (holdings * next_day_open_price)
            print(f"Sold {shares_to_sell} shares at {sell_price} on {csv_data.index[next_row_index].date()}. Cash balance: ${cash:.2f} Equity: ${equity:.2f}")
            dates.append(csv_data.index[next_row_index])
            prices.append(sell_price)
            actions.append('Sell')
            balances.append(cash)

# Summary
print(f"Final holdings: {holdings} shares")
print(f"Cash balance: ${cash:.2f}")

# Plot the trades
plt.figure(figsize=(10, 6))
plt.plot(csv_data.index, csv_data['Open'], label='Open Price', color='blue')

# Plot buy and sell actions
for date, price, action, balance in zip(dates, prices, actions, balances):
    if action == 'Buy':
        plt.scatter(date, price, color='green', marker='^', label='Buy', zorder=5)
    elif action == 'Sell':
        plt.scatter(date, price, color='red', marker='v', label='Sell', zorder=5)

# Ensure that the legend does not have duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Trade Actions for {ticker} ({timeframe})')
plt.grid(True)
plt.show()

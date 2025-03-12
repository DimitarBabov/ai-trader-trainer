import json
import pandas as pd
import os

# Read the regression data
regression_file = "data_processed_imgs/slv/1d/regression_data/slv_1d_regression_data_normalized.json"
if not os.path.exists(regression_file):
    print(f"Error: Could not find regression data file at {regression_file}")
    exit(1)

with open(regression_file, "r") as f:
    regression_data = json.load(f)

print("\nFirst few entries in regression data:")
for filename in list(regression_data.keys())[:5]:
    print(f"{filename}: {regression_data[filename]}")

# Read the price data
price_file = "data_csv/SLV/SLV_1d_data.csv"
if not os.path.exists(price_file):
    print(f"Error: Could not find price data file at {price_file}")
    exit(1)

# Read CSV with datetime parsing
price_data = pd.read_csv(price_file, parse_dates=['Datetime'])

print("\nFirst few rows of price data:")
print(price_data.head())

# Create market data dictionary
market_data = {}

# Process each entry in regression data
for filename, data in regression_data.items():
    try:
        # Extract date from filename (format: slv_1d_16c_YYYY-MM-DD 00-00-00.png)
        date_parts = filename.split('_')
        if len(date_parts) < 4:
            print(f"Warning: Unexpected filename format: {filename}")
            continue
            
        date_str = date_parts[3].split(' ')[0]  # Get YYYY-MM-DD
        print(f"\nProcessing date: {date_str}")
        
        # Find matching price data
        matching_price = price_data[price_data['Datetime'].dt.strftime('%Y-%m-%d') == date_str]
        print(f"Found {len(matching_price)} matching price entries")
        
        if not matching_price.empty:
            market_data[date_str] = {
                'price': float(matching_price['Close'].iloc[0]),
                'trend': float(data['trend_strength'])
            }
            print(f"Added entry: {date_str}: {market_data[date_str]}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        continue

# Save to JSON file
output_path = "LSM_experimets/market_data.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
with open(output_path, "w") as f:
    json.dump(market_data, f, indent=4)

print(f"\nCreated market_data.json with {len(market_data)} entries")

# Print first few entries as example
print("\nFirst few entries in market_data.json:")
for date, values in list(market_data.items())[:5]:
    print(f"{date}: {values}") 
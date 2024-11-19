import os
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

# Load financial data from CSV
def load_data(csv_file):
    """Load financial data from a CSV file."""
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
    df.set_index('Date', inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def price_to_pixel(value, min_price, max_price, height):
    """Convert a price to a pixel position based on image height and price range."""
    scale = (max_price - min_price) / (height - 1)
    return height - int((value - min_price) / scale) - 1

def create_candlestick_with_regression_image(data, height=224, candlestick_width=3, spacing=1, blur=False):
    """Create a candlestick image with all candles rendered in white and draw regression lines below."""
    num_candlesticks = len(data)
    min_price = data[['Low']].min().min()
    max_price = data[['High']].max().max()
    total_width = (candlestick_width + spacing) * num_candlesticks

    # Render the combined image in grayscale mode ('L') with black background
    combined_height = height * 2  # Double the height to fit both candlesticks and regression lines
    combined_image = Image.new('L', (total_width, combined_height), 0)  # Black background
    draw = ImageDraw.Draw(combined_image)

    # Draw the candlesticks in the upper half
    for i, (index, row) in enumerate(data.iterrows()):
        open_price, high_price, low_price, close_price = row['Open'], row['High'], row['Low'], row['Close']

        # Convert prices to pixel positions for the upper half of the image
        open_pixel = price_to_pixel(open_price, min_price, max_price, height)
        close_pixel = price_to_pixel(close_price, min_price, max_price, height)
        high_pixel = price_to_pixel(high_price, min_price, max_price, height)
        low_pixel = price_to_pixel(low_price, min_price, max_price, height)

        # Candle color: white (255 in grayscale)
        color = 255

        # Calculate x position
        x_start = i * (candlestick_width + spacing)

        # Draw wick
        wick_x = x_start + candlestick_width // 2
        draw.line((wick_x, high_pixel, wick_x, low_pixel), fill=color)

        # Draw body
        draw.rectangle([x_start, min(open_pixel, close_pixel), x_start + candlestick_width - 1, max(open_pixel, close_pixel)], fill=color)

    # Prepare data for regression calculation by treating each (Open, High, Low, Close) as separate data points
    x_values = []
    y_values = []

    for i in range(num_candlesticks):
        # Four data points per candlestick
        x_values.extend([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])
        y_values.extend([data['Open'].iloc[i], data['High'].iloc[i], data['Low'].iloc[i], data['Close'].iloc[i]])

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # First half regression
    x_values_first = x_values[:len(x_values) // 2]
    y_values_first = y_values[:len(y_values) // 2]
    coefficients_first = np.polyfit(x_values_first, y_values_first, 1)
    regression_line_first = np.poly1d(coefficients_first)
    slope_first = coefficients_first[0]  # Slope of the first regression line

    # Second half regression
    x_values_second = x_values[len(x_values) // 2:]
    y_values_second = y_values[len(y_values) // 2:]
    coefficients_second = np.polyfit(x_values_second, y_values_second, 1)
    regression_line_second = np.poly1d(coefficients_second)
    slope_second = coefficients_second[0]  # Slope of the second regression line

    # Last 4 candles regression
    num_last_candles = 4
    x_values_last = x_values[-num_last_candles * 4:]
    y_values_last = y_values[-num_last_candles * 4:]
    coefficients_last = np.polyfit(x_values_last, y_values_last, 1)
    regression_line_last = np.poly1d(coefficients_last)
    slope_last = coefficients_last[0]  # Slope of the third regression line

    # Calculate the standard deviation for all candlesticks
    std_dev = np.std(y_values)

    # Normalize y values to fit into the lower half of the image
    min_y = min(y_values_first.min(), y_values_second.min(), y_values_last.min())
    max_y = max(y_values_first.max(), y_values_second.max(), y_values_last.max())

    def normalize_y(value, min_value, max_value, height):
        """Normalize y values to fit the lower half of the image."""
        scale = (max_value - min_value) / (height - 1)
        return height - int((value - min_value) / scale) - 1

    # Draw first regression line
    for x in x_values_first:
        y = regression_line_first(x)
        y_pixel = normalize_y(y, min_y, max_y, height)
        x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
        if 0 <= y_pixel < height:
            draw.point((x_pixel, height + y_pixel), fill=128)  # Draw in gray (128) for visibility

    # Draw second regression line
    for x in x_values_second:
        y = regression_line_second(x)
        y_pixel = normalize_y(y, min_y, max_y, height)
        x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
        if 0 <= y_pixel < height:
            draw.point((x_pixel, height + y_pixel), fill=128)  # Draw in gray (128) for visibility

    # Draw last 4 candles regression line
    for x in x_values_last:
        y = regression_line_last(x)
        y_pixel = normalize_y(y, min_y, max_y, height)
        x_pixel = (x // 4) * (candlestick_width + spacing) + (x % 4) * (candlestick_width + spacing) // 4
        if 0 <= y_pixel < height:
            draw.point((x_pixel, height + y_pixel), fill=200)  # Draw in light gray (200) for visibility

    if blur:
        # Apply Gaussian blur
        combined_image = combined_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return combined_image, slope_first, slope_second, slope_last, std_dev

def save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder):
    """Save the candlestick image with a filename based on the ticker, timeframe, window size, and end date."""
    filename = f"{ticker}_{timeframe}_{window_size}c_{end_date}.png"
    filepath = os.path.join(output_folder, filename)
    image.save(filepath)
    print(f"Saved: {filepath}")
    return filename

def process_data_into_images(csv_file, ticker, timeframe, window_size=56, height=224, output_folder='output_images', overlap=23, blur=False):
    """Process all data in the CSV file into candlestick images with specified window size and overlap."""
    data = load_data(csv_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate the step size for the sliding window to create specified overlap
    step_size = window_size - overlap

    # Dictionary to store regression slopes for each image
    regression_data = {}

    # Slide through the dataset with specified overlap
    for i in range(0, len(data) - window_size + 1, step_size):
        window_data = data.iloc[i:i + window_size]
        end_date = window_data.index[-1].strftime('%Y-%m-%d')
        
        image, slope_first, slope_second, slope_third, std = create_candlestick_with_regression_image(window_data, height=height, candlestick_width=3, spacing=1, blur=blur)
        filename = save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder)

        # Save the regression slopes for this image
        regression_data[filename] = {
            "slope_first": slope_first,
            "slope_second": slope_second,
            "slope_third": slope_third,
            "std_dev":std
        }

    # Save the regression data to a JSON file
    regression_file = os.path.join(output_folder, f"{ticker}_{timeframe}_regression_data.json")
    with open(regression_file, 'w') as json_file:
        json.dump(regression_data, json_file, indent=4)
    print(f"Regression data saved to '{regression_file}'")

if __name__ == "__main__":
    # Prompt user for ticker and timeframe in the same line
    user_input = input("Enter the ticker symbol and timeframe (e.g., 'SLV 1d'): ")
    ticker, timeframe = user_input.split()
    ticker_path = os.path.join('data_csv', ticker)
    
    if not os.path.exists(ticker_path):
        print(f"No data available for ticker {ticker}.")
        exit()
    
    # Find the CSV file for the specified timeframe
    available_files = [f for f in os.listdir(ticker_path) if f.endswith('.csv') and timeframe in f]
    if not available_files:
        print(f"No CSV files available for ticker {ticker} and timeframe {timeframe}.")
        exit()
    
    # Get the CSV file
    csv_file = os.path.join(ticker_path, available_files[0])
    
    # Parameters for processing
    output_folder = os.path.join('data_processed_imgs', ticker, timeframe)
    window_size = 32             # Number of candlesticks per image
    height = 128                 # Image height in pixels (candlesticks take half, regression lines take the other half)
    overlap = 31                 # Number of overlapping candlesticks between consecutive windows    
    blur = False                 # Apply blur for natural mammalian vision effect
    blur_radius = 1

    # Process the data and generate images
    process_data_into_images(csv_file, ticker, timeframe, window_size, height, output_folder, overlap, blur)

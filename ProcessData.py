import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import os
from datetime import datetime

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

def create_candlestick_image(data, height=224, candlestick_width=3, spacing=1, min_alpha=64, blur=False):
    """Create a candlestick image with 3-pixel wide candlesticks and 1-pixel spacing between each."""
    num_candlesticks = len(data)
    min_price = data[['Low']].min().min()
    max_price = data[['High']].max().max()
    min_volume = data['Volume'].min()
    max_volume = data['Volume'].max()
    
    # Calculate total width: (candlestick width + spacing) * num_candlesticks
    total_width = (candlestick_width + spacing) * num_candlesticks
    background_color = 0  # Black background for grayscale
    combined_image = Image.new('L', (total_width, height), background_color)  # Grayscale image
    draw = ImageDraw.Draw(combined_image)

    for i, (_, row) in enumerate(data.iterrows()):
        open_price, high_price, low_price, close_price, volume = row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
        
        # Set color to white with transparency based on volume
        alpha = int(min_alpha + (volume - min_volume) / (max_volume - min_volume) * (255 - min_alpha))
        color = alpha  # Grayscale value

        # Convert prices to pixel positions
        open_pixel = price_to_pixel(open_price, min_price, max_price, height)
        close_pixel = price_to_pixel(close_price, min_price, max_price, height)
        high_pixel = price_to_pixel(high_price, min_price, max_price, height)
        low_pixel = price_to_pixel(low_price, min_price, max_price, height)
        
        # Calculate the x position for this candlestick with spacing
        x_start = i * (candlestick_width + spacing)

        # Draw the high-low wick on the central pixel of the candlestick width
        wick_x = x_start + 1  # Center of 3-pixel candlestick
        draw.line((wick_x, high_pixel, wick_x, low_pixel), fill=color)
        
        # Draw the candlestick body (open to close) across the 3-pixel width
        draw.rectangle([x_start, min(open_pixel, close_pixel), x_start + candlestick_width - 1, max(open_pixel, close_pixel)], fill=color)

    if blur:
        # Apply Gaussian blur to simulate natural mammalian vision blur
        combined_image = combined_image.filter(ImageFilter.GaussianBlur(radius=0.5))

    return combined_image

def save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder):
    """Save the candlestick image with a filename based on the ticker, timeframe, window size, and end date."""
    filename = f"{ticker}_{timeframe}_{window_size}c_{end_date}.png"
    filepath = os.path.join(output_folder, filename)
    image.save(filepath)
    print(f"Saved: {filepath}")

def process_data_into_images(csv_file, ticker, timeframe, window_size=56, height=224, output_folder='output_images', overlap=23, min_alpha=64, blur=False):
    """Process all data in the CSV file into candlestick images with specified window size and overlap."""
    data = load_data(csv_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Calculate the step size for the sliding window to create 23-candlestick overlap
    step_size = window_size - overlap

    # Slide through the dataset with specified overlap
    for i in range(0, len(data) - window_size + 1, step_size):
        window_data = data.iloc[i:i + window_size]
        end_date = window_data.index[-1].strftime('%Y-%m-%d')
        
        image = create_candlestick_image(window_data, height=height, candlestick_width=3, spacing=1, min_alpha=min_alpha, blur=blur)
        save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder)

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
    window_size = 56             # Number of candlesticks per image
    height = 224                 # Image height in pixels
    overlap = 20                 # Number of overlapping candlesticks between consecutive windows
    min_alpha = 64               # Minimum alpha value for lowest volume candlestick
    blur = False                  # Apply blur for natural mammalian vision effect

    # Process the data and generate images
    process_data_into_images(csv_file, ticker, timeframe, window_size, height, output_folder, overlap, min_alpha, blur)

# File: main.py
import os
import json

import pandas as pd
from data_loader import load_data
from image_utils import create_candlestick_with_regression_image
from save_utils import save_candlestick_image
from json_utils import normalize_json

def process_data_into_images(csv_file, ticker, timeframe, window_size=56, height=224, 
                             output_folder='data_processed_imgs',
                             regression_folder='data_processed_imgs', 
                             overlap=23, blur=False, blur_radius=0, 
                             draw_regression_lines=True,
                             color_candles=True,
                             create_regression_labels=True):
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
        # Adjust format for hourly data
        end_date = window_data.index[-1].strftime('%Y-%m-%d %H-%M-%S')
        (image, 
         slope_first, slope_second, slope_third, slope_whole, 
         price_change, 
         max_dev_scaled,
         colored_pixels_ratio) = (
        create_candlestick_with_regression_image(window_data, 
                                                 height=height, 
                                                 candlestick_width=3, 
                                                 spacing=1, 
                                                 blur=blur, 
                                                 blur_radius=blur_radius,
                                                 draw_regression_lines=draw_regression_lines, 
                                                 color_candles=color_candles))
        
        filename = save_candlestick_image(image, ticker, timeframe, window_size, end_date, output_folder)

        # Save the regression slopes and additional data for this image
        regression_data[filename] = {
            "slope_first": slope_first,
            "slope_second": slope_second,
            "slope_third": slope_third,
            "slope_whole": slope_whole,
            "max_dev": max_dev_scaled,
            "price_change": price_change,
            "colored_pixels_ratio":colored_pixels_ratio
        }
    #regression lables are only allowed if blur is false 
    #this is do because later we feed the resnet train model with blured images but 
    #want to use proper lables
    if(create_regression_labels and not blur):
        # Save the regression data to a JSON file
        if not os.path.exists(regression_folder):
            os.makedirs(regression_folder)

        regression_file = os.path.join(regression_folder, f"{ticker}_{timeframe}_regression_data.json")
        with open(regression_file, 'w') as json_file:
            json.dump(regression_data, json_file, indent=4)
        print(f"Regression data saved to '{regression_file}'")
        normalized_json = normalize_json(regression_file)
        print(f"Normalized regression data saved to '{normalized_json}'")



if __name__ == "__main__":   

    ticker = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (e.g., '1d'): ")
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
    output_folder = os.path.join('data_processed_imgs', ticker, timeframe, "images")
    regression_folder = os.path.join('data_processed_imgs', ticker, timeframe, 'regression_data')
    window_size = 16           # Number of candlesticks per image
    height = 64                # Image height in pixels 
    overlap = 15               # Number of overlapping candlesticks between consecutive windows    
    blur = True                # Apply blur for natural mammalian vision effect
    blur_radius = 1.25
    draw_regression_lines = False
    color_candles = True
    create_regression_labels = False
    # Process the data and generate images
    process_data_into_images(csv_file, ticker, timeframe, window_size, height, output_folder, regression_folder, overlap, blur, blur_radius, draw_regression_lines, color_candles=color_candles,create_regression_labels=create_regression_labels)

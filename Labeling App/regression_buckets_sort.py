# Script to Sort Images Based on Slopes and max_dev, and Organize into Buckets

import json
import os
import shutil

# Step 1: Load the JSON Data
def load_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None

# Step 2: Create a Function to Get the Bucket Name
def get_bucket_name(attributes):
    if not isinstance(attributes, dict):
        return None

    slope_keys = ['slope_first', 'slope_second', 'slope_third', 'slope_whole']
    slope_conditions = [attributes.get(key, 0) > 0 for key in slope_keys]
    max_dev = attributes.get('max_dev', 0)
    if max_dev < 15:
        max_dev_part = "maxdev_lt15"
    elif 15 <= max_dev <= 25:
        max_dev_part = "maxdev_15to25"
    else:
        max_dev_part = "maxdev_gt25"
    
    # Create a bucket identifier string, e.g., "pos_pos_neg_pos_maxdev_gt25"
    slope_part = "_".join(["pos" if cond else "neg" for cond in slope_conditions])
    
    return f"{slope_part}_{max_dev_part}"

# Step 3: Sort Images into Buckets
def sort_images_into_buckets(base_dir, data):
    if data is None:
        print("No data to process.")
        return
    
    buckets = {}
    for image_name, attributes in data.items():
        bucket_name = get_bucket_name(attributes)
        if bucket_name is None:
            continue
        
        if bucket_name not in buckets:
            buckets[bucket_name] = []
        buckets[bucket_name].append(image_name)
    
    # Create folders for each bucket and copy images
    buckets_dir = os.path.join(base_dir, 'sorted_buckets')
    if not os.path.exists(buckets_dir):
        os.makedirs(buckets_dir)
    
    for bucket_name, images in buckets.items():
        bucket_folder = os.path.join(buckets_dir, bucket_name)
        if not os.path.exists(bucket_folder):
            os.makedirs(bucket_folder)
        
        print(f"Bucket {bucket_name} has {len(images)} images.")
        for img in images:
            # Calculate trending value and add it before the extension
            attributes = data.get(img, {})
            price_change = attributes.get('price_change', 0)
            colored_pixels_ratio = attributes.get('colored_pixels_ratio', 1)  # Avoid division by zero
            trending_value = price_change / colored_pixels_ratio
            img_name, img_ext = os.path.splitext(img)
            updated_image_name = f"{img_name}_trending_{trending_value:.2f}{img_ext}"

            src_image_path = os.path.join(base_dir, 'images', img)
            dst_image_path = os.path.join(bucket_folder, updated_image_name)
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image {img} not found in the images directory.")

# Prompt user for the ticker and timeframe
ticker = input("Enter the ticker symbol: ")
timeframe = input("Enter the timeframe (e.g., '1d'): ")

# Specify the file path
base_dir = os.path.join('data_processed_imgs', ticker, timeframe)
regression_json = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_regression_data.json')

# Load the JSON data
data = load_json_data(regression_json)

# Sort images into buckets based on slopes and max_dev
sort_images_into_buckets(base_dir, data)

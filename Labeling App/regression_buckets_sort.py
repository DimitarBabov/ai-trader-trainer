# Script to Sort Images Based on Slopes and max_dev, and Organize into Buckets

import json
import os
import shutil

import numpy as np

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
        max_dev_part = "dev_lt15"
    elif 15 <= max_dev <= 25:
        max_dev_part = "dev_15to25"
    else:
        max_dev_part = "dev_gt25"
    
    # Create a bucket identifier string, e.g., "pos_pos_neg_pos_maxdev_gt25"
    slope_part = "_".join(["p" if cond else "n" for cond in slope_conditions])
    
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
            trending_value = attributes.get('trending', 0)
            img_name, img_ext = os.path.splitext(img)
            updated_image_name = f"{img_name}_{trending_value:.2f}{img_ext}"

            src_image_path = os.path.join(base_dir, 'images', img)
            dst_image_path = os.path.join(bucket_folder, updated_image_name)
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image {img} not found in the images directory.")


def normalize_json(input_json_path):
    """
    Normalizes max_dev and colored_pixels_ratio in the input JSON file and saves the output
    to a new JSON file with '_normalized' appended to the original file name.
    
    Args:
        input_json_path (str): Path to the input JSON file.
    
    Returns:
        None
    """
    # Load the JSON data
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    # Extract filename and directory
    base, ext = os.path.splitext(input_json_path)
    output_json_path = f"{base}_normalized{ext}"

    # Parameters
    epsilon = 1e-6  # Small constant to avoid log(0)

    # Extract max_dev and colored_pixels_ratio values for calculations
    max_dev_values = [item["max_dev"] for item in json_data.values() if isinstance(item, dict)]
    colored_pixels_ratios = [item["colored_pixels_ratio"] for item in json_data.values() if isinstance(item, dict)]
    max_dev_mean = np.mean(max_dev_values)
    max_dev_std = np.std(max_dev_values)
    colored_pixels_mean = np.mean(colored_pixels_ratios)
    colored_pixels_std = np.std(colored_pixels_ratios)

    # Process the data
    modified_json = {
        "_comments": [
            "shape: Represents the sequence of slopes. 'n' = negative slope, 'p' = positive slope.",
            "max_dev_log: Logarithmic representation of the normalized signed max_dev distance from the mean.",
            "colored_pixels_ratio_scaled: Adjusted colored pixel ratio, scaled by the mean and spread."
        ]
    }

    for key, value in json_data.items():
        if not isinstance(value, dict):  # Skip metadata or comments
            modified_json[key] = value
            continue

        # Normalize max_dev by its standard deviation
        max_dev = value["max_dev"]
        max_dev_signed_distance = (max_dev - max_dev_mean) / (max_dev_std + epsilon)  # Normalized by std_dev
        max_dev_log = np.sign(max_dev_signed_distance) * np.log(abs(max_dev_signed_distance) + epsilon)  # Log-transformed with sign

        # Adjust colored_pixels_ratio
        colored_pixels_ratio = value["colored_pixels_ratio"]
        adjusted_difference = (colored_pixels_ratio - colored_pixels_mean) / (colored_pixels_std + epsilon)  # Adjusted by std_dev
        colored_pixels_ratio_scaled = 1 + adjusted_difference  # Shift to 1.0

        # Determine "shape" sequence
        slopes = [value["slope_first"], value["slope_second"], value["slope_third"], value["slope_whole"]]
        shape = "".join("p" if s > 0 else "n" for s in slopes)

        # Create modified entry
        modified_json[key] = {
            "shape": shape,
            "max_dev_log": max_dev_log,
            "colored_pixels_ratio_scaled": colored_pixels_ratio_scaled
        }

    # Save the modified JSON to the output file
    with open(output_json_path, 'w') as file:
        json.dump(modified_json, file, indent=4)

    print(f"Normalized JSON saved to: {output_json_path}")

# Prompt user for the ticker and timeframe
ticker = input("Enter the ticker symbol: ")
timeframe = input("Enter the timeframe (e.g., '1d'): ")

# Specify the file path
base_dir = os.path.join('data_processed_imgs', ticker, timeframe)
regression_json = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_regression_data.json')


# Load the JSON data
data = load_json_data(regression_json)


normalize_json(regression_json)

# Sort images into buckets based on slopes and max_dev
# TO DO.....
#sort_images_into_buckets(base_dir, data)

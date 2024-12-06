# Script to Bin the Data for Labeling and Display Sample Images

import json
import numpy as np
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

# Step 2: Extract max_dev and trending Values
def extract_features(data):
    features = []
    for attributes in data.values():
        if 'max_dev' in attributes and 'trending' in attributes:
            features.append([np.abs(attributes['max_dev']), np.abs(attributes['trending'])])
    return np.array(features)

# Step 3: Bin Data for Labeling
def bin_data(data):
    max_dev_values = [np.abs(attributes['max_dev']) for attributes in data.values() if 'max_dev' in attributes]
    trending_values = [np.abs(attributes['trending']) for attributes in data.values() if 'trending' in attributes]

    # Define bin edges for max_dev and trending
    max_dev_bins = np.percentile(max_dev_values, [0, 25, 50, 75, 100]).astype(int)
    trending_bins = np.percentile(trending_values, [0, 40, 70, 90, 100]).astype(int)

    # Assign bin labels to data
    for key, attributes in data.items():
        if 'max_dev' in attributes and 'trending' in attributes:
            max_dev = np.abs(attributes['max_dev'])
            trending = np.abs(attributes['trending'])
            max_dev_label = np.digitize(max_dev, max_dev_bins) - 1
            trending_label = np.digitize(trending, trending_bins) - 1
            data[key]['max_dev_bin'] = int(max_dev_label)
            data[key]['trending_bin'] = int(trending_label)
    return data, max_dev_bins, trending_bins

# Step 4: Display Sample Images from Each Bin
def save_sample_images(data, max_dev_bins, trending_bins, base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bins = {}
    for key, attributes in data.items():
        max_dev_bin_index = attributes['max_dev_bin']
        trending_bin_index = attributes['trending_bin']

        # Define labels for max_dev and trending bins
        if max_dev_bin_index == 0:
            max_dev_bin_label = f"lt_{int(max_dev_bins[1])}"
        elif max_dev_bin_index == len(max_dev_bins) - 1:
            max_dev_bin_label = f"gt_{int(max_dev_bins[-2])}"
        else:
            max_dev_bin_label = f"{int(max_dev_bins[max_dev_bin_index])}_{int(max_dev_bins[max_dev_bin_index + 1])}"

        if trending_bin_index == 0:
            trending_bin_label = f"lt_{int(trending_bins[1])}"
        elif trending_bin_index == len(trending_bins) - 1:
            trending_bin_label = f"gt_{int(trending_bins[-2])}"
        else:
            trending_bin_label = f"{int(trending_bins[trending_bin_index])}_{int(trending_bins[trending_bin_index + 1])}"

        bin_name = f"max_dev_{max_dev_bin_label}_trending_{trending_bin_label}"
        if bin_name not in bins:
            bins[bin_name] = []
        bins[bin_name].append(key)

    for bin_name, images in bins.items():
        bin_folder = os.path.join(output_dir, bin_name)
        if not os.path.exists(bin_folder):
            os.makedirs(bin_folder)
        
        # Copy all images to the bin folder
        for img in images:
            src_path = os.path.join(base_dir, 'images', img)
            dst_path = os.path.join(bin_folder, img)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            else:
                print(f"Image not found: {src_path}")

# Example usage
if __name__ == "__main__":
    # Prompt user for the ticker and timeframe
    ticker = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (e.g., '1d'): ")

    # Specify the file path
    base_dir = os.path.join('data_processed_imgs', ticker, timeframe)
    regression_json = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_regression_data.json')
    output_json = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_binned_data.json')
    output_dir = os.path.join(base_dir, 'binned_samples')

    # Load the JSON data
    data = load_json_data(regression_json)

    if data is not None:
        # Extract features
        features = extract_features(data)

        # Bin the data for labeling
        binned_data, max_dev_bins, trending_bins = bin_data(data)

        # Save the binned data to a new JSON file
        with open(output_json, 'w') as f:
            json.dump(binned_data, f, indent=4)
        print(f"Binned data saved to {output_json}")

        # Save sample images from each bin
        save_sample_images(binned_data, max_dev_bins, trending_bins, base_dir, output_dir)
        print(f"Sample images saved to {output_dir}")

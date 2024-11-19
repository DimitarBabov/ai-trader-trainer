import os
import json
import numpy as np
from PIL import Image

# Specify directories and files (adjust these paths as needed)
ticker = 'slv'  # Example ticker
timeframe = '1d'  # Example timeframe
labels_json = os.path.join('data_processed_imgs', ticker, timeframe, 'labels', f'{ticker}_{timeframe}_labels.json')
image_dir = os.path.join('data_processed_imgs', ticker, timeframe)

# Load labels JSON file
def load_labels(labels_json):
    if os.path.exists(labels_json) and os.path.getsize(labels_json) > 0:
        try:
            with open(labels_json, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error: {labels_json} is corrupted.")
    return {}

# Overlay all images labeled as "buy" by accumulating pixel values
def overlay_buy_images(labels_json, image_dir, output_file='overlayed_buy_images.png'):
    labels_data = load_labels(labels_json)
    buy_images = [name for name, label in labels_data.items() if label == -1]

    if not buy_images:
        print("No images labeled as 'buy' found.")
        return

    # Open the first image and convert to 'L' mode (greyscale)
    base_image_path = os.path.join(image_dir, buy_images[0])
    base_image = Image.open(base_image_path).convert("L")
    base_array = np.zeros_like(np.array(base_image, dtype=np.float32))

    # Accumulate pixel values for all images labeled as "buy"
    for image_name in buy_images:
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist. Skipping.")
            continue

        image = Image.open(image_path).convert("L")
        image_array = np.array(image, dtype=np.float32)

        # Accumulate pixel values (treat non-black pixels as white)
        base_array += (image_array > 0).astype(np.float32)

    # Find the maximum number of overlaps (i.e., maximum "hits" for any pixel)
    max_overlaps = np.max(base_array)
    if max_overlaps == 0:
        print("No overlapping pixels found.")
        return

    # Normalize the accumulated array to fit in the range [0, 255]
    # Scale the pixel values so that the maximum overlap corresponds to intensity 255
    normalized_array = (base_array / max_overlaps) * 255
    normalized_array = np.clip(normalized_array, 0, 255)

    # Convert the array back to an image
    accumulated_image = Image.fromarray(normalized_array.astype(np.uint8))

    # Save the resulting overlayed image
    accumulated_image.save(output_file)
    print(f"Overlayed image saved as '{output_file}'")

# Example usage
overlay_buy_images(labels_json, image_dir)

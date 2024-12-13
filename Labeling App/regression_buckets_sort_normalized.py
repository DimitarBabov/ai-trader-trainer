import os
import shutil
import json
import numpy as np
from sklearn.cluster import KMeans

def normalize_json(input_json_path):
    epsilon = 1e-6
    with open(input_json_path, 'r') as file:
        json_data = json.load(file)

    base, ext = os.path.splitext(input_json_path)
    output_json_path = f"{base}_normalized{ext}"

    # Extract parameters
    max_dev_values = [item["max_dev"] for item in json_data.values() if isinstance(item, dict)]
    colored_pixels_ratios = [item["colored_pixels_ratio"] for item in json_data.values() if isinstance(item, dict)]
    max_dev_std = np.std(max_dev_values)
    colored_pixels_mean = np.mean(colored_pixels_ratios)
    colored_pixels_std = np.std(colored_pixels_ratios)

    modified_json = {
        "_comments": [
            "shape: sequence of slopes ('n' = negative, 'p' = positive).",
            "trend_strength = (price_change / max_dev_log) * log(max_dev_std)",
            "colored_pixels_ratio_scaled: adjusted colored pixel ratio."
        ]
    }

    for key, value in json_data.items():
        if not isinstance(value, dict):
            modified_json[key] = value
            continue

        max_dev = value["max_dev"]
        price_change = value["price_change"]
        colored_pixels_ratio = value["colored_pixels_ratio"]

        # Compute max_dev_distance without mean subtraction
        max_dev_distance = max_dev / (max_dev_std + epsilon)
        max_dev_log = np.sign(max_dev_distance) * np.log(abs(max_dev_distance) + epsilon)

        trend_strength = (price_change / (max_dev_log + epsilon)) * np.log(max_dev_std + epsilon)

        adjusted_difference = (colored_pixels_ratio - colored_pixels_mean) / (colored_pixels_std + epsilon)
        colored_pixels_ratio_scaled = 1 + adjusted_difference

        slopes = [value["slope_first"], value["slope_second"], value["slope_third"], value["slope_whole"]]
        shape = "".join("p" if s > 0 else "n" for s in slopes)

        modified_json[key] = {
            "shape": shape,
            "trend_strength": trend_strength,
            "colored_pixels_ratio_scaled": colored_pixels_ratio_scaled
        }

    with open(output_json_path, 'w') as file:
        json.dump(modified_json, file, indent=4)

    print(f"Normalized JSON saved to: {output_json_path}")
    return modified_json, output_json_path

def sort_images_by_shape(base_dir, json_data):
    sorted_buckets_dir = os.path.join(base_dir, 'sorted_buckets')
    os.makedirs(sorted_buckets_dir, exist_ok=True)

    images_dir = os.path.join(base_dir, 'images')

    for image_name, attributes in json_data.items():
        if not isinstance(attributes, dict):
            continue

        shape_folder = attributes["shape"]
        bucket_folder = os.path.join(sorted_buckets_dir, shape_folder)
        os.makedirs(bucket_folder, exist_ok=True)

        trend_strength = attributes["trend_strength"]
        colored_pixels_ratio_scaled = attributes["colored_pixels_ratio_scaled"]
        img_name, img_ext = os.path.splitext(image_name)
        updated_image_name = f"{img_name}_trend_{trend_strength:.3f}_col_{colored_pixels_ratio_scaled:.3f}{img_ext}"

        src_image_path = os.path.join(images_dir, image_name)
        dst_image_path = os.path.join(bucket_folder, updated_image_name)
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)
        else:
            print(f"Warning: Image {image_name} not found in the images directory.")

    print(f"Images sorted by shape in: {sorted_buckets_dir}")

def cluster_by_kmeans(base_dir, json_data, n_clusters=4):
    """
    For each shape folder:
    - Extract features (trend_strength, colored_pixels_ratio_scaled) from the JSON data.
    - Apply K-Means clustering with n_clusters=4.
    - Move images into cluster_0, cluster_1, cluster_2, cluster_3 subfolders under 'clusters_kmeans'.
    """
    sorted_buckets_dir = os.path.join(base_dir, 'sorted_buckets')
    if not os.path.exists(sorted_buckets_dir):
        print("sorted_buckets directory not found. Please run sort_images_by_shape first.")
        return

    # Get unique shapes from the JSON data
    shapes = {attr["shape"] for attr in json_data.values() if isinstance(attr, dict)}

    for shape in shapes:
        shape_dir = os.path.join(sorted_buckets_dir, shape)
        if not os.path.exists(shape_dir):
            continue

        # Collect images and features for K-Means
        image_files = []
        X = []
        
        for fname in os.listdir(shape_dir):
            fpath = os.path.join(shape_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith('.'):
                # Extract original image name from fname if needed
                # Actually, we can just match by removing the appended parts:
                # fname looks like: "originalname_trend_XXX_col_YYY.ext"
                # We'll try to find the original key in json_data by removing the appended parts.
                # However, we have the original json_data keyed by the original image name.
                # Let's just search json_data by matching prefix and extension:
                # safer to just find the underscore index:
                parts = fname.split("_trend_")
                if len(parts) < 2:
                    continue
                original_prefix = parts[0]
                ext = os.path.splitext(fname)[1]
                original_name = original_prefix + ext

                if original_name in json_data:
                    attributes = json_data[original_name]
                    image_files.append(fname)
                    X.append([attributes["trend_strength"], attributes["colored_pixels_ratio_scaled"]])

        if not X:
            print(f"No images found for shape {shape} to cluster.")
            continue

        X = np.array(X)

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        # Create clusters_kmeans directory
        clusters_dir = os.path.join(shape_dir, 'clusters_kmeans')
        os.makedirs(clusters_dir, exist_ok=True)

        # Create cluster directories
        for c in range(n_clusters):
            os.makedirs(os.path.join(clusters_dir, f"cluster_{c}"), exist_ok=True)

        # Move images to cluster folders
        for img_file, label in zip(image_files, labels):
            src = os.path.join(shape_dir, img_file)
            dst = os.path.join(clusters_dir, f"cluster_{label}", img_file)
            shutil.move(src, dst)

        print(f"K-Means clustering completed for shape: {shape}. Clusters saved in {clusters_dir}")

if __name__ == "__main__":
    ticker = input("Enter the ticker symbol: ")
    timeframe = input("Enter the timeframe (e.g., '1d'): ")

    base_dir = os.path.join('data_processed_imgs', ticker, timeframe)
    regression_json = os.path.join(base_dir, 'regression_data', f'{ticker}_{timeframe}_regression_data.json')

    # Normalize JSON
    normalized_data, normalized_json_path = normalize_json(regression_json)

    # Sort images by shape
    sort_images_by_shape(base_dir, normalized_data)

    # Cluster images by K-Means (4 clusters) in each shape folder
    cluster_by_kmeans(base_dir, normalized_data, n_clusters=4)

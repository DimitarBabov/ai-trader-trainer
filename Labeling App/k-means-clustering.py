# Script to Read JSON File, Perform K-means Clustering, and Organize Images into Folders

import json
import os
import shutil
import numpy as np
from sklearn.cluster import KMeans

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

# Step 2: Prepare Data for Clustering
def prepare_features(data):
    features = []
    for key, value in data.items():
        if isinstance(value, dict):
            try:
                feature_vector = [
                    value['slope_first'],
                    value['slope_second'],
                    value['slope_third'],
                    value['slope_whole'],
                    value['std_dev'],
                    value['max_dev'],
                    value['colored_pixels_ratio'],
                    value['price_change']
                ]
                features.append((key, feature_vector))
            except KeyError as e:
                print(f"Missing key {e} in image: {key}, skipping.")
    return features

# Step 3: Perform K-means Clustering
def perform_kmeans_clustering(features, n_clusters=8):
    if len(features) == 0:
        print("No valid features found. Clustering cannot be performed.")
        return None, None
    
    feature_vectors = [f[1] for f in features]
    keys = [f[0] for f in features]
    feature_vectors = np.array(feature_vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(feature_vectors)
    return kmeans.labels_, keys

# Step 4: Print Clustering Results and Organize Images
def organize_images_into_clusters(base_dir, labels, keys, n_clusters):
    if labels is None:
        return
    
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(keys[i])
    
    # Create folders for each cluster and copy images
    kmeans_cluster_dir = os.path.join(base_dir, 'kmeans_clusters')
    if not os.path.exists(kmeans_cluster_dir):
        os.makedirs(kmeans_cluster_dir)
    
    for cluster_id, images in clusters.items():
        cluster_folder = os.path.join(kmeans_cluster_dir, f'cluster_{cluster_id}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        
        print(f"Cluster {cluster_id} has {len(images)} images.")
        top_images = images  # Display top 10 images from each cluster
        for img in top_images:
            print(f"  - {img}")
            # Copy all images to the corresponding cluster folder
        for img in images:
            src_image_path = os.path.join(base_dir, 'images', img)
            dst_image_path = os.path.join(cluster_folder, img)
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

# Prepare features and perform clustering
features = prepare_features(data)
labels, keys = perform_kmeans_clustering(features, n_clusters=8)

# Organize images into cluster folders and print clustering results
organize_images_into_clusters(base_dir, labels, keys, n_clusters=8)

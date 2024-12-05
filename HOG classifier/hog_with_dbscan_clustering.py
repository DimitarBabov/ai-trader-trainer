import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

# Path to your images folder
image_folder = os.path.join('data_processed_imgs', 'slv', '1d',"images")  # Replace with the folder path



image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

def compute_hog_features(image_path):
    image = imread(image_path)  # Read the image
    gray_image = rgb2gray(image)  # Convert to grayscale
    # Compute HOG features
    features = hog(
        gray_image,
        orientations=9,  # Number of gradient bins
        pixels_per_cell=(8, 8),  # Size of a cell
        cells_per_block=(2, 2),  # Number of cells per block
        block_norm="L2-Hys",  # Block normalization
        visualize=False,  # Set to False since we don't need the visualization
        feature_vector=True,  # Return features as a vector
    )
    return features


# Compute HOG features for all images
hog_features = []
for path in image_paths:
    hog_features.append(compute_hog_features(path))

hog_features = np.array(hog_features)  # Convert to a NumPy array
print(f"HOG features shape: {hog_features.shape}")  # (Number of images, feature vector size)


# Apply DBSCAN
dbscan = DBSCAN(eps=4.9, min_samples=5)  # Adjust eps and min_samples for your dataset
labels = dbscan.fit_predict(hog_features)

# Display cluster distribution
cluster_counts = Counter(labels)
print("Cluster distribution:", cluster_counts)

# Number of clusters (excluding noise, labeled as -1)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters: {num_clusters}")

def plot_cluster_representatives(cluster_id, labels, image_paths, num_images=20):
    """
    Plots representative images for a given cluster.

    Parameters:
        cluster_id (int): Cluster ID to visualize.
        labels (array): DBSCAN cluster labels for each image.
        image_paths (list): List of image paths.
        num_images (int): Number of images to display for the cluster.
    """
    cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == cluster_id]
    cluster_images = cluster_images[:num_images]

    if len(cluster_images) == 0:
        print(f"No images found for Cluster {cluster_id}")
        return

    # Determine grid layout
    num_columns = 7
    num_rows = -(-len(cluster_images) // num_columns)  # Ceiling division
    plt.figure(figsize=(10, 2 * num_rows))

    for i, img_path in enumerate(cluster_images):
        img = Image.open(img_path)
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.suptitle(f"Cluster {cluster_id} (Top {num_images} Images)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize clusters (excluding noise, labeled as -1)
plot_cluster_representatives(-1, labels, image_paths, num_images=20)
for cluster_id in range(num_clusters):
    plot_cluster_representatives(cluster_id, labels, image_paths, num_images=20)

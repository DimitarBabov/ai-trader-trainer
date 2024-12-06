import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np

# Path to your images folder
image_folder = os.path.join('data_processed_imgs', 'slv', '1d', "images")  # Replace with your actual folder path

# Load all image paths (ensure they align with the order of features in `features.npy`)
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Load features
features = np.load("RESNET clasifier/features.npy")

# Perform K-means clustering
kmeans = KMeans(n_clusters=30, random_state=42)  # Set n_clusters based on expected patterns
labels = kmeans.fit_predict(features)

# Display cluster labels for each image
print("Cluster labels:", labels)

# Verify that the number of images matches the number of features
if len(image_paths) != len(labels):
    raise ValueError(f"Number of image paths ({len(image_paths)}) does not match number of labels ({len(labels)}).")

# Function to plot representative images for a cluster
def plot_cluster_representatives(cluster_id, labels, image_paths, num_images=20):
    """
    Plots representative images for a given cluster.

    Parameters:
        cluster_id (int): The cluster ID to visualize.
        labels (array): Array of cluster labels for all images.
        image_paths (list): List of image file paths.
        num_images (int): Number of images to display for the cluster.
    """
    # Get images belonging to the current cluster
    cluster_images = [image_paths[i] for i in range(len(labels)) if labels[i] == cluster_id]
    
    if len(cluster_images) == 0:
        print(f"No images found for Cluster {cluster_id}")
        return
    
    # Limit the number of images to display
    cluster_images = cluster_images[:num_images]

    # Determine the number of rows and columns for the grid
    num_columns = 5  # Number of images per row
    num_rows = -(-len(cluster_images) // num_columns)  # Ceiling division for rows

    # Plot the images
    plt.figure(figsize=(10, 3 * num_rows/2))  # Adjust height based on rows
    for i, img_path in enumerate(cluster_images):
        img = Image.open(img_path)
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id} (Top {num_images} Images)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Display cluster distribution
cluster_counts = Counter(labels)
print("Cluster distribution:", cluster_counts)

# Number of images to display per cluster
num_images = 20  # Change this to display more or fewer images per cluster

# Visualize images for each cluster
for cluster_id in range(50):  # Adjust range if you have more or fewer clusters
    plot_cluster_representatives(cluster_id, labels, image_paths, num_images=num_images)

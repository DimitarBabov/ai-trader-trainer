import os
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

# Path to your images folder
image_folder = os.path.join('data_processed_imgs', 'slv', '1d',"images")  # Replace with the folder path



image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Function to compute HOG features
def compute_hog_features(image_path):
    image = imread(image_path)  # Read the image
    gray_image = rgb2gray(image)  # Convert to grayscale
    # Compute HOG features
    features, _ = hog(
        gray_image,
        orientations=9,  # Number of gradient bins
        pixels_per_cell=(8, 8),  # Size of a cell
        cells_per_block=(2, 2),  # Number of cells per block
        block_norm="L2-Hys",  # Block normalization
        visualize=True,  # Set to True if you want the HOG image
        feature_vector=True,  # Return features as a vector
    )
    return features

# Compute HOG features for all images
hog_features = []
for path in image_paths:
    hog_features.append(compute_hog_features(path))

hog_features = np.array(hog_features)  # Convert to a NumPy array
print(f"HOG features shape: {hog_features.shape}")  # (Number of images, feature vector size)

# Step 2: Perform K-means clustering
n_clusters = 10  # Set the desired number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(hog_features)

# Display cluster labels for each image
print("Cluster labels:", labels)

# Display cluster distribution
cluster_counts = Counter(labels)
print("Cluster distribution:", cluster_counts)

# Step 3: Visualize representative images for each cluster
def plot_cluster_representatives(cluster_id, labels, image_paths, num_images=10):
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
    plt.figure(figsize=(15, 3 * num_rows))  # Adjust height based on rows
    for i, img_path in enumerate(cluster_images):
        img = Image.open(img_path)
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Cluster {cluster_id} (Top {num_images} Images)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize images for each cluster
for cluster_id in range(n_clusters):
    plot_cluster_representatives(cluster_id, labels, image_paths, num_images=10)

import os
import shutil
from PIL import Image

def calculate_nonblack_ratio(image_path):
    """
    Calculate the ratio of non-black pixels to total pixels in a 64x64 image.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure image is in RGB mode
        pixels = img.load()
        width, height = img.size

        if width != 64 or height != 64:
            raise ValueError(f"Image {image_path} is not 64x64 in size.")

        total_pixels = width * height
        non_black_pixels = sum(
            1 for x in range(width) for y in range(height) if pixels[x, y] != (0, 0, 0)
        )
        return non_black_pixels / total_pixels

def copy_and_rename_images(src_folder, dest_folder):
    """
    Copies images from the source folder to the destination folder, renames them 
    to include the non-black pixel ratio, and saves them in the destination folder.
    """
    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    for file_name in os.listdir(src_folder):
        file_path = os.path.join(src_folder, file_name)

        if not os.path.isfile(file_path):
            continue  # Skip non-file entries

        try:
            ratio = calculate_nonblack_ratio(file_path)
            ratio_str = f"{ratio:.4f}"
            file_base, file_ext = os.path.splitext(file_name)
            new_file_name = f"{file_base}_{ratio_str}{file_ext}"
            dest_path = os.path.join(dest_folder, new_file_name)

            shutil.copy2(file_path, dest_path)  # Copy the file to the destination
            print(f"Copied and renamed: {file_name} -> {dest_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Path to the source folder (e.g., images to process)
    src_folder = r"data_processed_imgs/slv/1d/sorted_buckets/nnnn/clusters_kmeans/cluster_0"

    # Path to the destination folder
    dest_folder = r"TEST/output_color_ratio_images"

    if os.path.exists(src_folder):
        copy_and_rename_images(src_folder, dest_folder)
    else:
        print(f"Source folder not found: {src_folder}")

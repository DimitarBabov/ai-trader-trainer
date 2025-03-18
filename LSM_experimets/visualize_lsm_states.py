"""
LSM State Visualization Script
=============================

This script provides a simple way to visualize the state images generated
by the LSM network simulation using OpenCV.
"""

import cv2
import os
import glob
import time

def load_state_images(image_dir):
    """Load all state images and sort them by date"""
    image_files = glob.glob(os.path.join(image_dir, "state_*.png"))
    # Sort by date in filename
    image_files.sort(key=lambda x: os.path.basename(x).replace("state_", "").replace(".png", ""))
    return image_files

def display_states(image_dir="LSM_experimets/state_images", delay=200):  # delay in milliseconds
    """Display state images in sequence"""
    # Load image files
    image_files = load_state_images(image_dir)
    if not image_files:
        print(f"No state images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} state images")
    
    # Create window
    cv2.namedWindow("LSM States", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LSM States", 500, 500)  # Set window size
    
    try:
        for img_file in image_files:
            # Read image
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Extract date from filename for display
            date = os.path.basename(img_file).replace("state_", "").replace(".png", "")
            
            # Add date text to image
            img_with_text = cv2.putText(img.copy(), date, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                      (255, 255, 255), 2)
            
            # Show image
            cv2.imshow("LSM States", img_with_text)
            
            # Wait for delay and check for exit key
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                raise KeyboardInterrupt
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting LSM state visualization...")
    print("Press 'q' to exit")
    display_states()
    print("Visualization complete.")
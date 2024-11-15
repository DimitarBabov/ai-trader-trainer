import os
import random
import json
import re
from datetime import datetime
from tkinter import Tk, Button, Label, Frame, simpledialog
from PIL import Image, ImageTk

# Set up the main window
root = Tk()
root.withdraw()  # Hide the main window while getting user input

# Prompt user for ticker and timeframe
ticker = simpledialog.askstring("Input", "Enter the ticker symbol:")
timeframe = simpledialog.askstring("Input", "Enter the timeframe (e.g., '1d'):")

root.destroy()  # Destroy the prompt window after getting user input

# Specify directories and files
image_dir = os.path.join('data_processed_imgs', ticker, timeframe)  # Directory with images
labels_json = os.path.join('data_processed_imgs', ticker, timeframe, 'labels', f'{ticker}_{timeframe}_labels.json')
used_images_json = os.path.join('data_processed_imgs', ticker, timeframe, 'labels', f'{ticker}_{timeframe}_used.json')

# Function to safely load JSON data
def load_json_file(filename, default_value):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:  # Check if file exists and is not empty
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {filename} is corrupted. Reinitializing.")
    return default_value

# Helper function to extract date from filename and sort by date
def sort_by_date(filenames):
    # Regular expression to extract date from filename
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    
    # Sort filenames by the date portion extracted
    return sorted(filenames, key=lambda x: datetime.strptime(date_pattern.search(x).group(), "%Y-%m-%d"))

# Load and sort existing labels or initialize an empty dictionary
if not os.path.exists(os.path.dirname(labels_json)):
    os.makedirs(os.path.dirname(labels_json))
labels_data = load_json_file(labels_json, {})

# Sort the dictionary by the date in filenames
labels_data = dict(sorted(labels_data.items(), key=lambda x: datetime.strptime(re.search(r"\d{4}-\d{2}-\d{2}", x[0]).group(), "%Y-%m-%d")))

# Load used images or initialize an empty set
if not os.path.exists(os.path.dirname(used_images_json)):
    os.makedirs(os.path.dirname(used_images_json))
used_images = set(load_json_file(used_images_json, []))  # Ensure used_images is a set

# Get a list of image files excluding used images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and f not in used_images]
image_files = sort_by_date(image_files)  # Sort image_files by date

# Calculate total images and images left to label
total_images = len(image_files) + len(used_images)
images_left = len(image_files)

# Set up the main window
gui_root = Tk()
gui_root.title("Image Labeling Tool")
gui_root.configure(bg='grey')

# Function to update the counter display
def update_counter_display():
    counter_label.config(text=f"{images_left} out of {total_images}")

# Function to display a random image
def display_next_image():
    global img_label, current_image_path, current_image_name

    # Check if there are any images left to label
    if not image_files:
        img_label.config(text="All images labeled!", image='')
        return

    # Randomly select an image
    current_image_name = random.choice(image_files)
    current_image_path = os.path.join(image_dir, current_image_name)

    # Load, resize, and display the image
    img = Image.open(current_image_path)
    img = img.resize((img.width * 2, img.height * 2))  # Scale image for display
    img_tk = ImageTk.PhotoImage(img)

    # Update the label with the image
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function to save the label for the current image
def save_labeled_image(label):
    global current_image_name, images_left

    # Define numeric label mapping
    label_map = {
        "sell": -1,
        "hold": 0,
        "buy": 1
    }
    numeric_label = label_map[label]

    # Update the JSON data with the label
    labels_data[current_image_name] = numeric_label
    used_images.add(current_image_name)  # Mark image as used

    # Save the updated labels to JSON file, sorted by date
    sorted_labels = dict(sorted(labels_data.items(), key=lambda x: datetime.strptime(re.search(r"\d{4}-\d{2}-\d{2}", x[0]).group(), "%Y-%m-%d")))
    with open(labels_json, 'w') as file:
        json.dump(sorted_labels, file, indent=4)

    # Save the updated used images to JSON file, sorted by date
    sorted_used_images = sort_by_date(list(used_images))  # Convert used_images to a sorted list
    with open(used_images_json, 'w') as file:
        json.dump(sorted_used_images, file, indent=4)

    # Remove the labeled image from the list, update counters, and display the next image
    image_files.remove(current_image_name)
    images_left = len(image_files)
    update_counter_display()

    if image_files:
        display_next_image()
    else:
        img_label.config(text="All images labeled!", image='')

# Set up the GUI layout
# Display a single line counter for total images and images left to label
counter_label = Label(gui_root, text="", bg='grey', font=("Helvetica", 10))
counter_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

# Label to display images
img_label = Label(gui_root, bg='grey')
img_label.grid(row=1, column=0, padx=10, pady=10)

# Frame for labeling buttons below the image
buttons_frame = Frame(gui_root, bg='grey')
buttons_frame.grid(row=2, column=0, padx=10, pady=10)

# Button configuration
button_width = 10
button_height = 2

# Label buttons
sell_button = Button(buttons_frame, text="Sell", command=lambda: save_labeled_image("sell"),
                     width=button_width, height=button_height)
sell_button.pack(side="left", padx=5, pady=5)

hold_button = Button(buttons_frame, text="Hold", command=lambda: save_labeled_image("hold"),
                     width=button_width, height=button_height)
hold_button.pack(side="left", padx=5, pady=5)

buy_button = Button(buttons_frame, text="Buy", command=lambda: save_labeled_image("buy"),
                    width=button_width, height=button_height)
buy_button.pack(side="left", padx=5, pady=5)

# Display the first random image and initialize counter display
display_next_image()
update_counter_display()

# Run the application
gui_root.mainloop()

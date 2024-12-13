import os
import json
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

class ImageViewerApp:
    def __init__(self, master, json_file, images_dir):
        self.master = master
        self.master.title("Image Viewer - Trend Strength")
        
        # Load JSON data
        with open(json_file, 'r') as file:
            self.image_data = [
                (image_name, attributes["trend_strength"])
                for image_name, attributes in json.load(file).items()
                if isinstance(attributes, dict)
            ]
        
        # Sort images by trend_strength in descending order
        self.image_data.sort(key=lambda x: x[1], reverse=True)
        
        self.images_dir = images_dir
        self.current_index = 0

        # Create UI elements
        self.label = Label(self.master, text="", font=("Arial", 14))
        self.label.pack(pady=10)

        self.image_label = Label(self.master)
        self.image_label.pack()

        self.prev_button = Button(self.master, text="Prev", command=self.prev_image, state="disabled")
        self.prev_button.pack(side="left", padx=20)

        self.next_button = Button(self.master, text="Next", command=self.next_image)
        self.next_button.pack(side="right", padx=20)

        # Display the first image
        self.display_image()

    def display_image(self):
        if 0 <= self.current_index < len(self.image_data):
            image_name, trend_strength = self.image_data[self.current_index]
            image_path = os.path.join(self.images_dir, image_name)

            if os.path.exists(image_path):
                # Display image and its details
                img = Image.open(image_path)
                img.thumbnail((800, 600))  # Resize for display
                self.img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.img_tk)
                self.label.config(text=f"{image_name}\nTrend Strength: {trend_strength:.3f}")
            else:
                # If the image file is missing
                self.label.config(text=f"Image not found: {image_name}")
                self.image_label.config(image="")

            # Update button states
            self.prev_button.config(state="normal" if self.current_index > 0 else "disabled")
            self.next_button.config(state="normal" if self.current_index < len(self.image_data) - 1 else "disabled")
        else:
            self.label.config(text="No images to display.")
            self.image_label.config(image="")
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def next_image(self):
        if self.current_index < len(self.image_data) - 1:
            self.current_index += 1
            self.display_image()

# Run the app
if __name__ == "__main__":
    # Ask for JSON file and images directory
    root = Tk()
    root.withdraw()
    json_file = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON Files", "*.json")])
    images_dir = filedialog.askdirectory(title="Select Images Directory")
    root.deiconify()

    if json_file and images_dir:
        app = ImageViewerApp(root, json_file, images_dir)
        root.mainloop()
    else:
        print("JSON file or images directory not selected.")

import os
import random
from PIL import Image, ImageTk
from utils import load_json_file, sort_by_date

class ImageManager:
    def __init__(self):
        self.ticker = "SLV"
        self.timeframe = "1d"
        self.image_dir = os.path.join('data_processed_imgs', self.ticker, self.timeframe)
        self.labels_json = os.path.join('data_processed_imgs', self.ticker, self.timeframe, 'labels', f'{self.ticker}_{self.timeframe}_labels.json')
        self.used_images_json = os.path.join('data_processed_imgs', self.ticker, self.timeframe, 'labels', f'{self.ticker}_{self.timeframe}_used.json')
        self.regression_json = os.path.join('data_processed_imgs', self.ticker, self.timeframe, f'{self.ticker}_{self.timeframe}_regression_data.json')
        
        self.labels_data = load_json_file(self.labels_json, {})
        self.used_images = set(load_json_file(self.used_images_json, []))
        self.regression_data = load_json_file(self.regression_json, {})
        
        self.filtered_images = self.apply_filtering_conditions()
        self.image_files = sort_by_date([f for f in self.filtered_images if f not in self.used_images and f in os.listdir(self.image_dir)])
        self.total_images = len(self.image_files) + len(self.used_images)
        self.images_left = len(self.image_files)
        self.current_image_name = None

    def apply_filtering_conditions(self):
        return [
            image for image, slopes in self.regression_data.items()
            if (
                (slopes["slope_first"] > 0) and
                (slopes["slope_second"] > 0) and
                (slopes["slope_second"] > slopes["slope_first"]) and
                (slopes["slope_third"] >= slopes["slope_second"])
            ) or (
                (slopes["slope_first"] < 0) and
                (slopes["slope_second"] < 0) and
                (slopes["slope_second"] < slopes["slope_first"]) and
                (slopes["slope_third"] <= slopes["slope_second"])
            )
        ]

    def display_next_image(self):
        if not self.image_files:
            return '', "All images labeled!"
        
        self.current_image_name = random.choice(self.image_files)
        current_image_path = os.path.join(self.image_dir, self.current_image_name)

        img = Image.open(current_image_path)
        img = img.resize((img.width * 2, img.height * 2))  # Scale image for display
        img_tk = ImageTk.PhotoImage(img)

        std_dev_value = self.regression_data.get(self.current_image_name, {}).get("std_dev", "N/A")
        if isinstance(std_dev_value, (int, float)):
            std_dev_text = f"Standard Deviation: {std_dev_value:.4f}"
        else:
            std_dev_text = "Standard Deviation: N/A"

        return img_tk, std_dev_text

    def remove_current_image(self):
        self.image_files.remove(self.current_image_name)
        self.images_left = len(self.image_files)

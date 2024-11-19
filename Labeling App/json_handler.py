from datetime import datetime
import json
from utils import load_json_file, sort_by_date
from image_manager import ImageManager
import re

class JSONHandler:
    def __init__(self, image_manager):
        self.image_manager = image_manager

    def save_labeled_image(self, label):
        label_map = {
            "sell": -1,
            "hold": 0,
            "buy": 1
        }
        numeric_label = label_map[label]
        self.image_manager.labels_data[self.image_manager.current_image_name] = numeric_label
        self.image_manager.used_images.add(self.image_manager.current_image_name)

        sorted_labels = dict(sorted(self.image_manager.labels_data.items(), key=lambda x: datetime.strptime(re.search(r"\d{4}-\d{2}-\d{2}", x[0]).group(), "%Y-%m-%d")))
        with open(self.image_manager.labels_json, 'w') as file:
            json.dump(sorted_labels, file, indent=4)

        sorted_used_images = sort_by_date(list(self.image_manager.used_images))
        with open(self.image_manager.used_images_json, 'w') as file:
            json.dump(sorted_used_images, file, indent=4)


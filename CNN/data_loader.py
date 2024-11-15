# data_loader.py
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config  # Import configuration

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels = self.load_labels(labels_file)
        self.transform = transform

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        # Filter labels to only include sell (-1), hold (0), and buy (1)
        return {k: v for k, v in labels.items() if v in {-1, 0, 1}}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image filename and label
        image_name = list(self.labels.keys())[idx]
        label = self.labels[image_name]

        # Load grayscale image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("L")  # "L" mode for grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to create a data loader
def get_data_loader(batch_size=config.BATCH_SIZE, shuffle=True):
    # Use global config variables for ticker and timeframe
    image_dir = os.path.join("data_processed_imgs", config.TICKER, config.TIMEFRAME)
    labels_file = os.path.join(image_dir, "labels", f"{config.TICKER.lower()}_{config.TIMEFRAME}_labels.json")

    # Define transformations for grayscale images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Single mean and std for grayscale
    ])
    
    dataset = ImageDataset(image_dir=image_dir, labels_file=labels_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler

# Combined Dataset Class
class TrendStrengthDataset(Dataset):
    def __init__(self, json_files, image_dirs, transform=None, use_grayscale=False, augment=False):
        self.data = []
        self.image_dirs = image_dirs
        self.transform = transform
        self.use_grayscale = use_grayscale
        self.augment = augment

        # Combine data from multiple JSON files
        for json_file, image_dir in zip(json_files, image_dirs):
            with open(json_file, 'r') as file:
                dataset = json.load(file)
                self.data.extend([
                    (os.path.join(image_dir, image_name), attributes["trend_strength"])
                    for image_name, attributes in dataset.items()
                    if isinstance(attributes, dict)  # Filter out comments
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, trend_strength = self.data[idx]

        if self.use_grayscale:
            image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.augment and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            trend_strength = -trend_strength

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(trend_strength, dtype=torch.float32)


# Transform for images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match training dimensions
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Add slight color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Paths to 1d and 1h datasets
json_files = [
    "data_processed_imgs/slv/1d/regression_data/slv_1d_regression_data_normalized.json",
    "data_processed_imgs/slv/1h/regression_data/slv_1h_regression_data_normalized.json",
    "data_processed_imgs/slv/1wk/regression_data/slv_1wk_regression_data_normalized.json"
]
image_dirs = [
    "data_processed_imgs/slv/1d/images",
    "data_processed_imgs/slv/1h/images",
    "data_processed_imgs/slv/1wk/images"
]

# Create combined dataset
combined_dataset = TrendStrengthDataset(
    json_files=json_files,
    image_dirs=image_dirs,
    transform=transform,
    use_grayscale=False,
    augment=True
)

# Split into train and validation sets
train_data, val_data = train_test_split(combined_dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define the model (ResNet18 for regression)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features

# Adjust the first convolution layer for grayscale images if needed
use_grayscale = False
if use_grayscale:
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Add Dropout before the final fully connected layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1)
)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop
criterion = nn.MSELoss()
num_epochs = 25
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_resnet18_trend_strength.pth")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

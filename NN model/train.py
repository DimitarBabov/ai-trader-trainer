# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loader
from model import ImageClassifierCNN
from utils import map_label
import config  # Import configuration
import os

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize data loader using config parameters
train_loader = get_data_loader(batch_size=config.BATCH_SIZE)

# Initialize model, loss function, and optimizer
model = ImageClassifierCNN().to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Training loop
for epoch in range(config.NUM_EPOCHS):
    total_loss = 0
    model.train()  # Set model to training mode

    for images, labels in train_loader:
        # Move data to the GPU if available
        images = images.to(device)
        labels = torch.tensor([map_label(label.item()) for label in labels]).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# Save the model using the ticker and timeframe from config
model_save_path = os.path.join("CNN", "saved_models", f"{config.TICKER}_{config.TIMEFRAME}_classifier.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved as {model_save_path}")

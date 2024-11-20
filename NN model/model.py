# model.py
import torch
import torch.nn as nn

class ImageClassifierCNN(nn.Module):
    def __init__(self):
        super(ImageClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 1 channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 3)  # 3 output classes for sell, hold, buy

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits for 3 classes
        return x

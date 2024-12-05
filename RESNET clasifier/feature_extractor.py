import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from tqdm import tqdm  # For progress bar

# Step 1: Define the folder path and output file
input_folder =os.path.join('data_processed_imgs', 'slv', '1d',"images")
output_file = os.path.join('RESNET clasifier','features.npy')  # File to save extracted features

# Step 2: Load the pre-trained ResNet18 model
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
model.eval()  # Set to evaluation mode

# Step 3: Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# Step 4: Function to process a single image
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(input_tensor).flatten().numpy()  # Extract and flatten features
    return features

# Step 5: Process all images in the folder
all_features = []
image_paths = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_paths)} images. Extracting features...")

for img_path in tqdm(image_paths, desc="Extracting Features"):
    features = process_image(img_path)
    all_features.append(features)

# Step 6: Save the features to a .npy file
all_features = np.array(all_features)
np.save(output_file, all_features)

print(f"Feature extraction complete. Features saved to {output_file}.")

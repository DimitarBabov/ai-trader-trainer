# inference.py
import os
import torch
from PIL import Image
from torchvision import transforms
from model import ImageClassifierCNN
from utils import inverse_map_label  # For mapping output classes to labels
import config

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to load the trained model on the specified device
def load_model(model_path):
    model = ImageClassifierCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load directly on the device
    model.eval()  # Set model to evaluation mode
    return model

# Function to preprocess a single image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # For grayscale normalization
    ])
    image = Image.open(image_path).convert("L")  # Convert image to grayscale
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Function to run inference on images in a folder
def run_inference(model, image_folder):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate through each image and predict
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        input_tensor = preprocess_image(image_path)  # Preprocess and move the image to GPU

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Map the predicted class index to the label (-1, 0, or 1)
        label = inverse_map_label(predicted_class)
        label_str = { -1: "Sell", 0: "Hold", 1: "Buy" }.get(label, "Unknown")
        
        print(f"Image: {image_file}, Prediction: {label_str} ({label})")

# Main function
if __name__ == "__main__":
    # Define paths
    model_path = os.path.join("CNN", "saved_models", f"{config.TICKER}_{config.TIMEFRAME}_classifier.pth")
    image_folder = os.path.join("data_processed_imgs", config.TICKER, config.TIMEFRAME)  # Folder to run inference on

    # Load the model
    model = load_model(model_path)

    # Run inference on the specified folder
    run_inference(model, image_folder)

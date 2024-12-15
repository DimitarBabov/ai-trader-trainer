import os
import json
import torch
import shutil
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # Progress bar for inference

# Custom Dataset for Inference
class InferenceDataset:
    def __init__(self, json_file, images_dir, transform=None, use_grayscale=False):
        """
        Dataset for inference to predict trend_strength.
        Args:
            json_file: Path to JSON file with trend_strength values.
            images_dir: Directory where images are stored.
            transform: Torchvision transforms to apply to the images.
            use_grayscale: If True, convert images to grayscale.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)

        self.data = [
            (image_name, attributes["trend_strength"])
            for image_name, attributes in data.items()
            if isinstance(attributes, dict)  # Filter out comments
        ]
        self.images_dir = images_dir
        self.transform = transform
        self.use_grayscale = use_grayscale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, trend_strength = self.data[idx]
        image_path = os.path.join(self.images_dir, image_name)

        # Conditionally convert to grayscale or RGB
        if self.use_grayscale:
            image = Image.open(image_path).convert("L")  # Grayscale
        else:
            image = Image.open(image_path).convert("RGB")  # RGB

        if self.transform:
            image = self.transform(image)

        return image, trend_strength, image_name


def perform_inference(model_path, json_file, images_dir, output_dir, use_grayscale=False):
    """
    Perform inference and rename images based on predictions.
    Args:
        model_path: Path to the trained model (.pth file).
        json_file: Path to JSON file with true labels.
        images_dir: Directory containing images for inference.
        output_dir: Directory to save renamed images.
        use_grayscale: If True, use grayscale images; otherwise, use RGB.
    """
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to match training dimensions
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Load the dataset
    dataset = InferenceDataset(json_file, images_dir, transform=transform, use_grayscale=use_grayscale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load the model
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features

    # Adjust for grayscale input if needed
    if use_grayscale:
        model.conv1 = torch.nn.Conv2d(
            in_channels=1,  # Single channel for grayscale
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

    # Replace the final layer
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 1)  # Regression output
    )

    # Load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Performing inference and renaming images...")
    for images, true_labels, image_names in tqdm(dataloader, desc="Processing Images"):
        images = images.to(device)
        
        # Ensure true_labels is iterable
        true_labels = true_labels.numpy().flatten()  # Convert to 1D array

        with torch.no_grad():
            predictions = model(images).squeeze().cpu().numpy()

        # Ensure predictions is iterable
        predictions = predictions.reshape(-1)  # Convert to 1D array if scalar

        # Iterate over the batch
        for true_label, prediction, image_name in zip(true_labels, predictions, image_names):
            true_label = float(true_label)  # Ensure scalar for formatting
            prediction = float(prediction)  # Ensure scalar for formatting
            new_name = f"{os.path.splitext(image_name)[0]}_[{true_label:.2f}]_[{prediction:.2f}].png"
            src_path = os.path.join(images_dir, image_name)
            dst_path = os.path.join(output_dir, new_name)
            shutil.copy(src_path, dst_path)

    print(f"All images processed and saved in '{output_dir}'.")



# Main Script
if __name__ == "__main__":
    # Parameters
    model_path = "best_resnet18_trend_strength.pth"  # Path to trained model
    timeframe = "1d"  # Change this to the appropriate timeframe
    base_dir = "data_processed_imgs/slv"
    json_file = f"{base_dir}/{timeframe}/regression_data/slv_{timeframe}_regression_data_normalized.json"
    images_dir = f"{base_dir}/{timeframe}/images"
    output_dir = f"{base_dir}/{timeframe}/predicted_images"

    # Set to True if the model was trained on grayscale images
    use_grayscale = False

    # Perform inference
    perform_inference(model_path, json_file, images_dir, output_dir, use_grayscale=use_grayscale)

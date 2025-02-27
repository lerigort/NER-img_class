# inference_image_classifier.py (Adapted for older model file)

import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

def predict_image(model_path, image_path, image_size, model_name="resnet18", num_classes=10): #Added arguments
    """Predicts the class of an image using a trained model (without metadata)."""

    # --- 1. Load Model (No Checkpoint Metadata) ---
    # We have to *assume* the architecture and number of classes.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Model Loading (Hardcoded Architecture and Classes) ---
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False)  # Don't need pretrained here
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)  # Use loaded num_classes
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    # Load the state dict directly (no checkpoint dictionary)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
         print(f"Error: Model file not found at {model_path}")
         return None
    except Exception as e:
        print(f"Error: loading the weights: {e}")
        return None

    model = model.to(device)
    model.eval()

    # --- 3. Image Preprocessing (Same as before) ---

    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    # --- 4. Prediction (Same as before) ---

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()

    # --- 5. Get Class Label (Hardcoded Path to Training Data) ---
    # We *assume* the training data is still in 'data/train'
    train_dir = 'data/train'  # Hardcoded path!
    if not os.path.exists(train_dir):
        print("Error: Could not find training data directory.  Cannot determine class names.")
        return None

    class_names = sorted(os.listdir(train_dir))
    if predicted_class_index >= len(class_names):
        print(f"Error predicted index {predicted_class_index} is larger than classes in train dir")
        return None
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of an image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to.')
    # New arguments to specify the model and classes, required for this "older" model.
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model architecture (e.g., resnet18).')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the model.')

    args = parser.parse_args()

    predicted_class = predict_image(args.model_path, args.image_path, args.image_size, args.model_name, args.num_classes)

    if predicted_class is not None:
        print(f"Predicted class: {predicted_class}")
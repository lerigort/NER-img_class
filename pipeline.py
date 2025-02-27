# pipeline.py (Corrected Image Selection)

import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from inference_ner_general import predict_ner_general  # Import general NER
import os
import random
import warnings
from bs4 import MarkupResemblesLocatorWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging


def predict_image_class(model_path, image_path, image_size, model_name):
    """Predicts image class, handles English names (older model), uses correct key."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 10)
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 10)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True) # Add weights_only=True
        # Load the state dict correctly, handling both older and newer save formats:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading image classification model: {e}")
        return None

    model.to(device)
    model.eval()

    # Preprocess image (same)
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()

    # English Class Name Mapping (Handles both older and newer models)
    try:
        # Try loading class_to_idx (newer model)
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        original_labels = [idx_to_class[i] for i in range(len(idx_to_class))]
        english_class_names = ['dog', 'cat', 'horse', 'spider', 'butterfly',
                               'chicken', 'sheep', 'cow', 'squirrel', 'elephant']
        idx_to_class_english = {i: english_class_names[i] for i in range(len(english_class_names))}
        predicted_class_name = idx_to_class_english[predicted_class_index]
    except KeyError:
        # Fallback to older model method (hardcoded English names)
        english_class_names = ['dog', 'cat', 'horse', 'spider', 'butterfly',
                               'chicken', 'sheep', 'cow', 'squirrel', 'elephant']
        idx_to_class_english = {i: english_class_names[i] for i in range(len(english_class_names))}
        predicted_class_name = idx_to_class_english[predicted_class_index]

    return predicted_class_name

def main(image_dir, text, image_model_path, ner_model_path, image_size, image_model_name, animal_class):
    # --- Corrected Image Selection ---
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    if not image_files:
        print(f"Error: No image files found in directory: {image_dir}")
        return

    # Randomly select an image (no filtering by filename)
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)
    print(f"Selected Image: {image_path}")

    predicted_class = predict_image_class(image_model_path, image_path, image_size, image_model_name)
    if predicted_class is None: return False
    print(f"Image Classification Result: {predicted_class}")

    extracted_animal = predict_ner_general(ner_model_path, text)
    if extracted_animal is None: return False
    print(f"NER Result: {extracted_animal}")

    if predicted_class.lower() in extracted_animal.lower() or extracted_animal.lower() in predicted_class.lower():
        print("Result: Matched")
        return True
    else:
        print("Result: Not Matched")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification and NER Pipeline.')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--image_model_path', type=str, required=True)
    parser.add_argument('--ner_model_path', type=str, required=True,
                        help='Path to the *directory* containing the saved general NER model.')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--image_model_name', type=str, default='resnet18')
    parser.add_argument('--animal_class', type=str, default=None) # We are not using it anymore
    args = parser.parse_args()
    main(args.image_dir, args.text, args.image_model_path, args.ner_model_path, args.image_size, args.image_model_name, args.animal_class)
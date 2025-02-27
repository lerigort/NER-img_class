# pipeline.py (Dynamic Image Selection)

import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from inference_ner_bio import predict_ner_bio  # Import our BioBERT inference function
import os
import random  # Import the 'random' module


def predict_image_class(model_path, image_path, image_size, model_name):
    """Predicts the class of an image (image classification)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the image classification model
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
        # Add more EfficientNet variants as needed...
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading image classification model: {e}")
        return None

    model = model.to(device)
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        return None

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
    # Get class name
    class_to_idx = checkpoint.get('class_to_idx')

    if class_to_idx is None: #Using old model
        print("Warning, using old model, that hasn't class_to_idx. Hardcoding idx_to_class")
        train_dir = 'data/train'
        class_names = sorted(os.listdir(train_dir))
        predicted_class_name = class_names[predicted_class_index]

    else:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        predicted_class_name = idx_to_class[predicted_class_index]

    return predicted_class_name

def main(image_dir, text, image_model_path, ner_model_name, image_size, image_model_name, animal_class):
    # --- 1. Dynamic Image Selection ---
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    if not image_files:
        print(f"Error: No image files found in directory: {image_dir}")
        return

    # Randomly select an image
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)
    print(f"Selected image: {image_path}")  # Print the selected image path

    # --- 2. Image Classification ---
    predicted_class = predict_image_class(image_model_path, image_path, image_size, image_model_name)

    if predicted_class is None:
        print("Image classification failed.")
        return
    print(f"Image Classification: Predicted animal: {predicted_class}")

    # --- 3. NER (using BioBERT) ---
    extracted_animal = predict_ner_bio(ner_model_name, text)

    if extracted_animal is None:
        print("NER extraction failed.")
        return

    print(f"NER Extraction: Animal name: {extracted_animal}")

    # --- 4. Verification ---
    if predicted_class.lower() in extracted_animal.lower() or extracted_animal.lower() in predicted_class.lower():
        print("Verification: NER output matches image classification output.")
        result = True
    else:
        print("Verification: NER output DOES NOT MATCH image classification output.")
        result = False
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification and NER Pipeline.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to the directory containing images.')
    parser.add_argument('--text', type=str, required=True,
                        help='Input text describing the image.')
    parser.add_argument('--image_model_path', type=str, required=True,
                        help='Path to the trained image classification model.')
    parser.add_argument('--ner_model_name', type=str, default='dmis-lab/biobert-base-cased-v1.2',
                        help='Name of the pre-trained BioBERT model.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for classification.')
    parser.add_argument('--image_model_name', type=str, default='resnet18',
                        help='Name of the image model architecture.')
    parser.add_argument('--animal_class', type=str, default = 'dog',
                        help='Class of the animal (for directory selection).') # Added for dynamic selection

    args = parser.parse_args()

    main(args.image_dir, args.text, args.image_model_path, args.ner_model_name, args.image_size, args.image_model_name, args.animal_class)
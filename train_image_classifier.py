# train_image_classifier.py (Modified for Fine-Tuning)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

def train_model(dataset_path, model_name, epochs, batch_size, learning_rate, output_dir, image_size):
    """Trains an image classification model with layer freezing for fine-tuning."""

    # --- 1. Data Loading and Augmentation (Same as before) ---

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')

    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # --- 2. Model Loading and Layer Freezing ---

    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=True)
        elif model_name == "resnet50":
             model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")

        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False  # This is the key for freezing!

        # Unfreeze the final fully connected layer (and optionally, some layers before it)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)  # New, untrained layer
        for param in model.fc.parameters():
            param.requires_grad = True  # Unfreeze the FC layer

        # *Optional*: Unfreeze the last few convolutional layers (for more fine-tuning)
        #  Uncomment the following lines to unfreeze, for example, layer4 of ResNet:
        # for param in model.layer4.parameters():
        #     param.requires_grad = True


    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(pretrained=True)
        # Add more EfficientNet variants as needed...
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the final classifier layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)  # New, untrained layer
        for param in model.classifier[1].parameters():
             param.requires_grad = True

        # *Optional*: Unfreeze some of the later convolutional blocks (EfficientNet calls them 'blocks')
        #  Example: Unfreeze the last few blocks:
        # for block in model.features[7:]:  # Unfreeze blocks 7 and onwards (adjust as needed)
        #     for param in block.parameters():
        #         param.requires_grad = True


    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # --- 3. Loss Function, Optimizer, and Scheduler (Same as before) ---

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that are set to be trainable (requires_grad=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 4. Training Loop (Same as before) ---

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()

        # Training Phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct_predictions / total_samples

        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / len(val_dataset)
        epoch_val_acc = correct_predictions / total_samples

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"  Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f} - "
              f"Duration: {epoch_duration:.2f} seconds")

        scheduler.step()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Best model saved to {os.path.join(output_dir, 'best_model.pth')}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an image classification model.')
    parser.add_argument('--dataset_path', type=str, default='data', help='Path to the dataset root directory.')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Name of the model (resnet18, resnet34, efficientnet_b0, etc.)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='models/image_model', help='Directory to save the trained model.')
    parser.add_argument('--image_size', type=int, default=224, help='Size to resize images to.')

    args = parser.parse_args()

    train_model(args.dataset_path, args.model_name, args.epochs, args.batch_size,
                args.learning_rate, args.output_dir, args.image_size)
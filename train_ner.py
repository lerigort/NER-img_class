# train_ner.py (Modified for Reduced Overfitting)

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm

class NERDataset(Dataset):
    """Custom Dataset class for NER."""
    def __init__(self, data_file, tokenizer, label_to_id):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

        with open(data_file, 'r') as f:
            sentence = []
            label = []
            for line in f:
                line = line.strip()
                if not line:  # Empty line indicates end of sentence
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(label)
                    sentence = []
                    label = []
                else:
                    word, tag = line.split()
                    sentence.append(word)
                    label.append(tag)

            if sentence:  # Append last sentence
                self.sentences.append(sentence)
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoded = self.tokenizer(
            sentence,
            is_split_into_words=True,  # Important
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Convert labels to IDs
        label_ids = [self.label_to_id[label] for label in labels]
        # Pad label IDs to match the length of the input IDs
        padding_length = encoded['input_ids'].size(1) - len(label_ids)
        label_ids += [-100] * padding_length  # Use -100 as ignore index

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def train_ner(model_name, train_data, validation_data, epochs, batch_size, learning_rate, output_dir):
    """Trains a BERT-based NER model."""

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create label mappings
    unique_labels = set()
    with open(train_data, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                _, tag = line.split()
                unique_labels.add(tag)

    label_to_id = {label: i for i, label in enumerate(sorted(unique_labels))}
    id_to_label = {i: label for label, i in label_to_id.items()}
    num_labels = len(label_to_id)
    print(f"Labels: {label_to_id}")

    train_dataset = NERDataset(train_data, tokenizer, label_to_id)
    val_dataset = NERDataset(validation_data, tokenizer, label_to_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 2. Model Loading ---
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # --- 3. Optimizer and Scheduler ---
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,  # Add weight decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8) # Added eps

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # --- Add Dropout (for Regularization) ---
    dropout_prob = 0.3  # You can adjust this value
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = dropout_prob

    # --- 4. Training Loop ---
    best_val_f1 = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            # Clip gradients (to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"  Average training loss: {avg_train_loss:.4f}")

        # --- 5. Validation ---
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=2)

                # Convert to lists and remove padding
                for i in range(labels.size(0)):
                    true_label_seq = []
                    pred_label_seq = []
                    for j in range(labels.size(1)):
                        if labels[i, j] != -100:  # Ignore padding
                            true_label_seq.append(id_to_label[labels[i, j].item()])
                            pred_label_seq.append(id_to_label[predicted_labels[i, j].item()])
                    true_labels.append(true_label_seq)
                    predictions.append(pred_label_seq)

        # --- 6. Evaluation (using seqeval) ---
        f1 = f1_score(true_labels, predictions)
        report = classification_report(true_labels, predictions)
        print(f"  Validation F1 Score: {f1:.4f}")
        print(report)

        # --- 7. Save Model ---
        if f1 > best_val_f1:
            best_val_f1 = f1
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            # Save label mapping
            torch.save({
                'label_to_id': label_to_id,
                'id_to_label': id_to_label
            }, os.path.join(output_dir, 'label_mapping.pt'))

            print(f"Best model saved to {output_dir}")

    print("Training complete.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a BERT-based NER model.')
    parser.add_argument('--model_name', type=str, default='bert-base-cased',
                        help='Name of the pre-trained BERT model.')
    parser.add_argument('--train_data', type=str, default='data/ner_data/train.txt',
                        help='Path to the training data file.')
    parser.add_argument('--validation_data', type=str, default='data/ner_data/validation.txt',
                        help='Path to the validation data file.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='models/ner_model',
                        help='Directory to save the trained model.')

    args = parser.parse_args()

    train_ner(args.model_name, args.train_data, args.validation_data, args.epochs,
              args.batch_size, args.learning_rate, args.output_dir)
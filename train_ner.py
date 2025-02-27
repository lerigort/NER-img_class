# train_ner.py (Complete - Uses Pre-trained Model's Labels, Augmentation, Weighted Loss)

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class NERDataset(Dataset):
    def __init__(self, data_file, tokenizer, label_map, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = label_map  # Use the provided label_map
        self.sentences, self.labels = self.load_data(data_file)

    def load_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sentences = []
        labels = []
        sentence = []
        label = []

        for line in lines:
            line = line.strip()
            if not line:  # Empty line indicates end of sentence
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Invalid line format: {line}")
                    continue
                word, tag = parts
                sentence.append(word)
                label.append(tag)
        if sentence:
            sentences.append(sentence)
            labels.append(label)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoded = self.tokenizer(
            sentence,
            is_split_into_words=True,  # Important!
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Convert labels to IDs using the provided label_map
        label_ids = [self.label_map[lbl] for lbl in labels]
        # Pad label_ids to max_len, using -100 to ignore during loss calculation
        padding_length = self.max_len - len(label_ids)
        label_ids = label_ids + [-100] * padding_length
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': label_ids
        }

def augment_data(data_file, animal_classes):
    """Augments the NER dataset by replacing animal names."""
    augmented_sentences = []
    augmented_labels = []

    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentence = []
    label = []
    for line in lines:
        line = line.strip()
        if not line:
            if sentence:
                # Check if the sentence contains an animal entity
                if "B-ANIMAL" in label:
                    # Augment the sentence by replacing with other animals
                    for animal in animal_classes:
                        new_sentence = []
                        new_label = []
                        for i, word in enumerate(sentence):
                            if label[i] == "B-ANIMAL":
                                new_sentence.append(animal)  # Replace the animal name
                                new_label.append("B-ANIMAL")  # Keep B-ANIMAL tag
                            elif label[i] == 'I-ANIMAL':
                                new_sentence.append(word) #keep I-tag token
                                new_label.append('I-ANIMAL') #Keep I-ANIMAL tag

                            else:
                                new_sentence.append(word)
                                new_label.append(label[i])
                        augmented_sentences.append(new_sentence)
                        augmented_labels.append(new_label)

                # Add original sentence and labels
                augmented_sentences.append(sentence)
                augmented_labels.append(label)
                sentence = []
                label = []
        else:
            parts = line.split()
            if len(parts) != 2:
                print(f"Warning: Invalid line format in augmentation: {line}")
                continue
            word, tag = parts
            sentence.append(word)
            label.append(tag)

    # Handle last sentence
    if sentence:
      if "B-ANIMAL" in label:
        for animal in animal_classes:
          new_sentence = []
          new_label = []
          for i, word in enumerate(sentence):
            if label[i] == 'B-ANIMAL':
              new_sentence.append(animal)
              new_label.append('B-ANIMAL')
            elif label[i] == 'I-ANIMAL':
              new_sentence.append(word)
              new_label.append('I-ANIMAL')
            else:
              new_sentence.append(word)
              new_label.append(label[i])
          augmented_sentences.append(new_sentence)
          augmented_labels.append(new_label)
      augmented_sentences.append(sentence)
      augmented_labels.append(label)
    return augmented_sentences, augmented_labels

def train(model, train_dataloader, dev_dataloader, optimizer, scheduler, num_epochs, device, label_list, output_dir, loss_fn):
    """Trains the NER model."""

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Reshape logits and labels for loss calculation
            active_loss = labels.view(-1) != -100  # Only consider non-padded tokens
            active_logits = logits.view(-1, len(label_list))[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = loss_fn(active_logits, active_labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluate on the development set
        results = evaluate(model, dev_dataloader, device, label_list)
        f1 = results['f1']

        # Save the best model
        if f1 > best_f1:
            best_f1 = f1
            print(f"New best F1: {best_f1:.4f}")
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir) #Save tokenizer


def evaluate(model, dataloader, device, label_list):
    """Evaluates the model."""

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=2)

            # Convert to lists and remove padding
            preds = preds.detach().cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()

            for i in range(len(labels)):
                temp_1 = []
                temp_2 = []
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:  # Remove padding
                        temp_1.append(label_list[labels[i][j]])
                        temp_2.append(label_list[preds[i][j]])
                true_labels.append(temp_1)
                predictions.append(temp_2)

    report = classification_report(true_labels, predictions, digits=4)
    print("\n"+ report)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)

    return {'f1': f1, 'precision': precision, 'recall': recall}

def main():
    parser = argparse.ArgumentParser(description='Train a NER model.')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the training data file.')
    parser.add_argument('--dev_file', type=str, required=True,
                        help='Path to the development data file.')
    parser.add_argument('--model_name', type=str, default='dslim/bert-base-NER',
                        help='Name of the pre-trained model.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model.')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate.')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
     # Add animals
    parser.add_argument('--animal_classes', type=str, nargs='+',
                        default=['dog', 'cat', 'horse', 'spider', 'butterfly',
                                 'chicken', 'sheep', 'cow', 'squirrel', 'elephant'],
                        help='List of animal classes to recognize.')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get initial label list from the pre-trained model
    label_list = list(model.config.id2label.values())
    # IMPORTANT: We DO NOT create the label_map yet.

    # --- Data Augmentation ---
    augmented_sentences, augmented_labels = augment_data(args.train_file, args.animal_classes)

    # --- Update Label List and Label Map AFTER Augmentation ---
    if "B-ANIMAL" not in label_list:
        label_list.append("B-ANIMAL")
    if "I-ANIMAL" not in label_list:
        label_list.append("I-ANIMAL")

    # *Now* create/update the label_map:
    label_map = {label: i for i, label in enumerate(label_list)}
    print(f"DEBUG: Label List: {label_list}")
    print(f"DEBUG: Label Map: {label_map}")

    # --- Resize Model Embeddings ---
    num_new_labels = len(label_list)
    model.resize_token_embeddings(len(tokenizer))  # Usually, you don't need to resize token embeddings for NER
    model.config.id2label = {i: label for i, label in enumerate(label_list)}
    model.config.label2id = label_map
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_new_labels).to(device) #Resizing classifier
    model.config.num_labels = num_new_labels


    #Create augmented file
    augmented_train_file = os.path.join(os.path.dirname(args.train_file), "train_augmented.txt")
    with open(augmented_train_file, 'w', encoding='utf-8') as f:
      for sent, labels in zip(augmented_sentences, augmented_labels):
        for word, label in zip(sent, labels):
          f.write(f"{word} {label}\n")
        f.write("\n")
    # --- Prepare Datasets and Dataloaders ---
    train_dataset = NERDataset(augmented_train_file, tokenizer, label_map, max_len=args.max_len)
    dev_dataset = NERDataset(args.dev_file, tokenizer, label_map, max_len=args.max_len)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=args.batch_size
    )

    # --- Optimizer and Scheduler ---
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Learning rate scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # --- Weighted Loss ---
    # Count occurrences of each label
    label_counts = {}
    for labels in train_dataset.labels:
        for label in labels:
          if label != '-100':
            label_id = label_map[label] # Now label_map is updated.
            label = label_list[label_id]
            label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate weights
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / (count + 1e-8) for label, count in label_counts.items()}

    # Normalize weights
    sum_weights = sum(class_weights.values())
    class_weights = {label: (weight / sum_weights) * len(label_list) for label, weight in class_weights.items()}

    # Convert to tensor
    weights_tensor = torch.tensor([class_weights[label] for label in label_list], dtype=torch.float).to(device)
    print(f"DEBUG: Class Weights: {weights_tensor}")
    loss_fn = CrossEntropyLoss(weight=weights_tensor, ignore_index=-100)

    # --- Training Loop ---
    train(model, train_dataloader, dev_dataloader, optimizer, scheduler, args.num_epochs, device, label_list, args.output_dir, loss_fn)

if __name__ == '__main__':
    main()
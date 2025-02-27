# inference_ner.py (Corrected)

import argparse
import os
import torch
from transformers import BertTokenizer, BertForTokenClassification

def predict_ner(model_path, text):
    """Predicts the named entities in a given text using a trained NER model."""

    # --- 1. Load Model and Tokenizer ---
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

     # --- 2. Load label mapping ---
    try:
        label_mapping = torch.load(os.path.join(model_path, 'label_mapping.pt'))
        id_to_label = label_mapping['id_to_label']
    except FileNotFoundError:
        print("Error: label_mapping.pt file not found.")
        return None
    except Exception as e:
        print("Error loading label_mapping.pt")
        return None


    # --- 3. Preprocess Input Text ---

    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # --- 4. Prediction ---

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=2)

    # --- 5. Convert to Labels and Extract Animal ---

    predicted_label_seq = []
    for i in range(predicted_labels.size(1)):
         if encoded['attention_mask'][0,i] == 1: # Only consider non-padded tokens
            predicted_label_seq.append(id_to_label[predicted_labels[0, i].item()])


    # Extract the animal name (if any)
    animal_name = None
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    extracted_tokens = []

    for token, label in zip(tokens, predicted_label_seq):
        if label == 'B-ANIMAL':
            if animal_name is None:
                animal_name = token
            else:
                animal_name += " " + token # In case if we will have two words in a row
        elif label == 'I-ANIMAL':
            if animal_name is not None:
             animal_name += " " + token.replace("##", "")
        else:
            if animal_name is not None: # If not animal
                break # Stop searching

    # Post-process to remove special tokens and handle sub-word pieces
    if animal_name is not None:
        animal_name = animal_name.replace("[CLS]", "").replace("[SEP]", "").strip()
        animal_name = animal_name.replace(" ##", "")

    return animal_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform NER on a given text.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained NER model directory.')
    parser.add_argument('--text', type=str, required=True,
                        help='Input text for NER.')
    args = parser.parse_args()

    animal = predict_ner(args.model_path, args.text)

    if animal:
        print(f"Extracted animal: {animal}")
    else:
        print("No animal found in the text.")
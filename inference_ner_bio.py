# inference_ner_bio.py (Adapted for Pre-trained BioBERT)

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def predict_ner_bio(model_name, text):
    """Predicts named entities using a pre-trained BioBERT model."""

    # --- 1. Load Model and Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- 2. Preprocess Input Text ---
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # --- 3. Prediction ---
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=2)

    # --- 4. Convert to Labels and Extract Entities ---
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_tags = [model.config.id2label[label_id.item()] for label_id in predicted_labels[0]]

    # --- 5. Filter and Extract Animal Names (Heuristic) ---
    animal_name = None
    current_entity = ""
    for token, tag in zip(tokens, predicted_tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:  # Skip special tokens
            continue

        # Heuristic filtering:  Look for B- and I- tags (beginning and inside of entities)
        if tag.startswith("B-") or tag.startswith("I-"):
            # Add more specific types here if needed, based on the BioBERT model's label set.
            # You might need to inspect model.config.id2label to see the full list.
            if any(label_type in tag for label_type in ["ORGANISM", "TAXON", "SPECIES", "ANIMAL"]):  # Add/remove types as needed
                if tag.startswith("B-"):
                    if current_entity: #Start new entity
                        animal_name = current_entity.strip()
                        break #If finds first animal
                    current_entity = token
                else: #tag startswith "I-"
                    current_entity += " " + token.replace("##", "") # Remove "##"

    #Remove special characters
    if current_entity:
      animal_name = current_entity.strip()
    return animal_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform NER using a pre-trained BioBERT model.')
    parser.add_argument('--model_name', type=str, default='dmis-lab/biobert-base-cased-v1.2',
                        help='Name of the pre-trained BioBERT model.')
    parser.add_argument('--text', type=str, required=True,
                        help='Input text for NER.')
    args = parser.parse_args()

    animal = predict_ner_bio(args.model_name, args.text)

    if animal:
        print(f"Extracted animal: {animal}")
    else:
        print("No animal found in the text.")
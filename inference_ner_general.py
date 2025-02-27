# inference_ner_general.py (Debugging Version)

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def predict_ner_general(model_path, text):
    """Predicts named entities (with debugging prints)."""

    print(f"DEBUG: Loading model from: {model_path}")

    # --- 1. Load Model and Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")
    model.to(device)
    model.eval()

    # --- 2. Preprocess Input Text ---
    print(f"DEBUG: Input text: {text}")
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"DEBUG: Encoded input_ids: {input_ids}")
    print(f"DEBUG: Encoded attention_mask: {attention_mask}")

    # --- 3. Prediction ---
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=2)

    print(f"DEBUG: Predicted labels (indices): {predicted_labels}")

    # --- 4. Convert to Labels and Extract Entities ---
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    predicted_tags = [model.config.id2label[label_id.item()] for label_id in predicted_labels[0]]

    print(f"DEBUG: Tokens: {tokens}")
    print(f"DEBUG: Predicted tags: {predicted_tags}")
    print(f"DEBUG: All Labels: {model.config.id2label}") #Print all labels

    # --- 5. Filter and Extract Animal Names (Heuristic) ---
    animal_name = None
    current_entity = ""
    for token, tag in zip(tokens, predicted_tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:  # Skip special tokens
            continue

        # Relaxed filtering:  Accept *ANY* B- or I- tag initially
        if tag.startswith("B-") or tag.startswith("I-"):
            print(f"DEBUG: Found potential entity: Token={token}, Tag={tag}") # Check Extracted Tokens
            if tag.startswith("B-"):
                if current_entity:
                    animal_name = current_entity.strip()
                    break
                current_entity = token
            else:
                current_entity += " " + token.replace("##", "") #Remove ##
    if current_entity:
        animal_name =  current_entity.strip()
    return animal_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform NER using a general-purpose NER model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained NER model directory.')
    parser.add_argument('--text', type=str, required=True,
                        help='Input text for NER.')
    args = parser.parse_args()

    animal = predict_ner_general(args.model_path, args.text)

    if animal:
        print(f"Extracted animal: {animal}")
    else:
        print("No animal found in the text.")
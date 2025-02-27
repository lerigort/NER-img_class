# download_models.py (Downloads ONLY the NER Model)

import argparse
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification

def download_models(ner_model_name, output_dir):
    """Downloads the NER model."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- NER Model (dslim/bert-base-NER) ---
    tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_model_dir = os.path.join(output_dir, "ner_model_pretrained")  # Save to a consistent directory
    tokenizer.save_pretrained(ner_model_dir)
    model.save_pretrained(ner_model_dir)
    print(f"Downloaded NER model and tokenizer to: {ner_model_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download pre-trained NER model.')
    parser.add_argument('--ner_model_name', type=str, default='dslim/bert-base-NER',
                        help='Name of the NER model.')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Base directory to save the model.')
    args = parser.parse_args()

    download_models(args.ner_model_name, args.output_dir)
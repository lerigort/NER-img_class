# process_inaturalist_data.py (Explicit NLTK Data Path)
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import os
import argparse
import warnings
from bs4 import MarkupResemblesLocatorWarning

# --- Add this section to set the NLTK data path ---
nltk.data.path.append("/content/nltk_data")  # Add a custom path
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir='/content/nltk_data', quiet=True) # Download to custom path
# -----------------------------------------------------

def clean_text(text):
    # ... (rest of your clean_text function remains the same) ...
    if pd.isna(text):  # Handle missing values
        return ""

    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)

    # Remove non-alphanumeric characters (except sentence punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def create_bio_tags(sentence, target_words):
    # ... (rest of your create_bio_tags function remains the same) ...
    words = sentence.split()
    labels = []
    inside = False

    for word in words:
        # Use regular expressions for more flexible matching
        is_target = False
        for target in target_words:
            if re.search(r'\b' + re.escape(target) + r'\b', word): # Match whole words
                is_target = True
                break
            #Also check if target word is the beginning of the word
            elif re.search(r'\b' + re.escape(target), word):
                is_target = True
                break

        if is_target:
            if not inside:
                labels.append('B-ANIMAL')
                inside = True
            else:
                labels.append('I-ANIMAL')
        else:
            labels.append('O')
            inside = False  # Reset inside flag

    return words, labels

def process_inaturalist_data(input_csv, output_dir, target_words, train_ratio=0.7, val_ratio=0.15):
    # ... (rest of your process_inaturalist_data function remains the same) ...
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {input_csv}")
        return

    # Drop rows with missing descriptions
    df.dropna(subset=['description'], inplace=True)

    # Keep only necessary columns
    df = df[['id', 'description']]

    # Clean the descriptions
    df['cleaned_description'] = df['description'].apply(clean_text)

    # Split into sentences
    sentences = []
    for desc in df['cleaned_description']:
        sentences.extend(sent_tokenize(desc))

    # Create BIO tags
    all_words = []
    all_labels = []
    for sentence in sentences:
        words, labels = create_bio_tags(sentence, target_words)
        if 'B-ANIMAL' in labels:  # Only keep sentences with at least one animal mention.
            all_words.append(words)
            all_labels.append(labels)

    # Split into train/val/test
    num_sentences = len(all_words)
    num_train = int(train_ratio * num_sentences)
    num_val = int(val_ratio * num_sentences)

    train_words = all_words[:num_train]
    train_labels = all_labels[:num_train]
    val_words = all_words[num_train:num_train + num_val]
    val_labels = all_labels[num_train:num_train + num_val]
    test_words = all_words[num_train + num_val:]
    test_labels = all_labels[num_train + num_val:]
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to files (CoNLL format)
    def write_to_file(words, labels, filepath):
        with open(filepath, 'w') as f:
            for sentence_words, sentence_labels in zip(words, labels):
                for word, label in zip(sentence_words, sentence_labels):
                    f.write(f"{word} {label}\n")
                f.write("\n")  # Separate sentences

    write_to_file(train_words, train_labels, os.path.join(output_dir, 'train.txt'))
    write_to_file(val_words, val_labels, os.path.join(output_dir, 'validation.txt'))
    write_to_file(test_words, test_labels, os.path.join(output_dir, 'test.txt'))

    print(f"Processed iNaturalist data saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process iNaturalist data for NER.')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input iNaturalist CSV file.')
    parser.add_argument('--output_dir', type=str, default='data/ner_data_inaturalist',
                        help='Directory to save the processed NER data.')
    parser.add_argument('--target_words', type=str, nargs='+',
                        default=['dog', 'dogs', 'puppy', 'pup'],
                        help='List of target words (animal names and variations).')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio for the training data split')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio for the validation data split')
    args = parser.parse_args()

    # Suppress BeautifulSoup warning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

    process_inaturalist_data(args.input_csv, args.output_dir, args.target_words, args.train_ratio, args.val_ratio)
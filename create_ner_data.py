# create_ner_data.py (Modified for Data Variety)

import os
import random
import argparse
import re #Import regular expressions

def create_ner_dataset(animal_classes, output_dir, num_sentences=10000, seed=42):
    """Generates a more diverse synthetic NER dataset."""

    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Expanded Templates (Much More Variety!) ---
    templates = [
        "There is a {animal} in the picture.",
        "I see a {animal}.",
        "Is that a {animal}?",
        "The {animal} is cute.",
        "Look at the {animal}!",
        "A {animal} is running.",
        "I think it's a {animal}.",
        "That's definitely a {animal}.",
        "It might be a {animal}.",
        "Could that be a {animal}?",
        "I'm not sure if it's a {animal}.",
        "The {animal} looks like it's sleeping.",
        "A wild {animal} appeared!",
        "Is the {animal} friendly?",
        "{animal} are often found in zoos.",
        "I've never seen a {animal} before.",
        "The {animal} has beautiful fur/feathers/scales.",  # Adapt to animal type
        "That {animal} is very large/small.",
        "What kind of {animal} is that?",
        "I'm afraid of {animal_plural}!",  # Plural form
        "There are several {animal_plural} in the field.",
        "A group of {animal_plural} is called a herd/flock/pack.", # Adapt to animal
        "The {animal} is eating grass/leaves/insects.", # Diet
        "I saw a {animal} yesterday.",
        "Do you see the {animal} over there?",
        "That looks like a {animal}, doesn't it?",
        "I'm pretty sure that's a {animal}.",
        "It's hard to tell, but it could be a {animal}.",
        "I can't believe how big that {animal} is!",
        "The {animal} is making a strange noise.",
        "We spotted a {animal} during our hike.",
        "The {animal} is camouflaged in the trees.",
        "Be careful, that {animal} might be dangerous.",
        "I read a book about {animal_plural}.",
        "The documentary featured a {animal}.",
        "My favorite animal is the {animal}."
    ]

    sentences = []
    labels = []

    for _ in range(num_sentences):
        animal = random.choice(animal_classes)
        template = random.choice(templates)

        # --- Handle Plurals and Other Variations ---
        animal_plural = animal + "s"  # Simple pluralization (works for most)
        if animal.endswith("y"):
            animal_plural = animal[:-1] + "ies"  # butterfly -> butterflies
        if animal == "sheep" or animal == "deer": # Handle irregular plurals (add more as needed)
            animal_plural = animal

        sentence = template.format(animal=animal, animal_plural=animal_plural)

         # --- Create BIO labels (Improved Logic) ---
        label = []
        words = sentence.split()
        inside = False  # Keep track of whether we're inside an animal name
        for i, word in enumerate(words):
             cleaned_word = re.sub(r'[^\w\s]', '', word).lower()
             if cleaned_word == animal.lower() or cleaned_word == animal_plural.lower() :
                 if not inside:
                     label.append('B-ANIMAL')
                     inside = True
                 else:
                     label.append('I-ANIMAL')
             else:
                 label.append('O')
                 inside = False  # Reset inside flag when we're not in an animal


        sentences.append(sentence)
        labels.append(label)

    # Split into train, validation, and test sets
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    num_train = int(train_ratio * num_sentences)
    num_val = int(val_ratio * num_sentences)

    train_sentences = sentences[:num_train]
    train_labels = labels[:num_train]
    val_sentences = sentences[num_train:num_train + num_val]
    val_labels = labels[num_train:num_train + num_val]
    test_sentences = sentences[num_train + num_val:]
    test_labels = labels[num_train + num_val:]

    # Save to files
    def write_to_file(sentences, labels, filepath):
        with open(filepath, 'w') as f:
            for sentence, label_seq in zip(sentences, labels):
                for word, tag in zip(sentence.split(), label_seq):
                    f.write(f"{word} {tag}\n")
                f.write("\n")

    write_to_file(train_sentences, train_labels, os.path.join(output_dir, 'train.txt'))
    write_to_file(val_sentences, val_labels, os.path.join(output_dir, 'validation.txt'))
    write_to_file(test_sentences, test_labels, os.path.join(output_dir, 'test.txt'))

    print(f"NER dataset created and saved to {output_dir}")
if __name__ == '__main__':
 parser = argparse.ArgumentParser(description='Generate synthetic NER data.')
 parser.add_argument('--animal_classes', type=str, nargs='+',
                     default=['dog', 'cat', 'horse', 'spider', 'butterfly',
                              'chicken', 'sheep', 'cow', 'squirrel', 'elephant'],
                     help='List of animal classes.')
 parser.add_argument('--output_dir', type=str, default='data/ner_data',
                     help='Directory to save the generated data.')
 parser.add_argument('--num_sentences', type=int, default=10000,
                     help='Number of sentences to generate.')
 parser.add_argument('--seed', type=int, default=42,
                     help='Number of sentences to generate.')
 args = parser.parse_args()
 create_ner_dataset(args.animal_classes, args.output_dir, args.num_sentences, args.seed)
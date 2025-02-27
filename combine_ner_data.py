# combine_ner_data.py
import os
import random
import argparse
import glob

def combine_ner_data(input_dirs, output_dir, seed=42):
    """Combines NER data from multiple directories into single train/val/test files."""

    random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def read_data(filepath):
        """Reads data from a single CoNLL file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return lines

    def write_data(filepath, data):
        """Writes data to a CoNLL file."""
        with open(filepath, 'w') as f:
            f.writelines(data)

    # Gather all train, validation, and test files
    train_files = []
    val_files = []
    test_files = []

    for input_dir in input_dirs:
        train_files.extend(glob.glob(os.path.join(input_dir, 'train.txt')))
        val_files.extend(glob.glob(os.path.join(input_dir, 'validation.txt')))
        test_files.extend(glob.glob(os.path.join(input_dir, 'test.txt')))

    # Read and combine data, keeping track of file type
    combined_train_data = []
    for train_file in train_files:
      combined_train_data.extend(read_data(train_file))
    combined_val_data = []
    for val_file in val_files:
      combined_val_data.extend(read_data(val_file))
    combined_test_data = []
    for test_file in test_files:
      combined_test_data.extend(read_data(test_file))


    # Shuffle the combined data
    random.shuffle(combined_train_data)
    random.shuffle(combined_val_data)
    random.shuffle(combined_test_data)

    # Write combined data to output files
    write_data(os.path.join(output_dir, 'train.txt'), combined_train_data)
    write_data(os.path.join(output_dir, 'validation.txt'), combined_val_data)
    write_data(os.path.join(output_dir, 'test.txt'), combined_test_data)

    print(f"Combined NER data saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine NER data from multiple directories.')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                        help='List of input directories (one for each animal).')
    parser.add_argument('--output_dir', type=str, default='data/ner_data_combined',
                        help='Directory to save the combined NER data.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling.')
    args = parser.parse_args()

    combine_ner_data(args.input_dirs, args.output_dir, args.seed)
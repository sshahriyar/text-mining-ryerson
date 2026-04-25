import json

def load_item_text(path):
    with open(path) as f:
        item2text = json.load(f)
    return item2text


def load_sequences(path):
    """
    Load user interaction sequences from a text file.

    Each line represents a sequence of item IDs separated by spaces.

    Args:
        path (str): Path to the sequence file.

    Returns:
        list[list[str]]: List of user interaction sequences.
    """
    sequences = []
    with open(path, "r") as f:
        for line in f:
            sequences.append(line.strip().split())
    return sequences


def load_val_sequences(path):
    sequences = []
    with open(path, "r") as f:
        for line in f:
            sequences.append(line.strip().split())
    return sequences
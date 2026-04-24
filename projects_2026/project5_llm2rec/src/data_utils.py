def read_sequences(file_path, max_users=7000, min_len=3, max_seq_len=50):
    """
    Read user interaction sequences from a file.

    Each line represents a user sequence of item IDs.

    Args:
        file_path (str): Path to sequence file
        max_users (int): Maximum number of users to load
        min_len (int): Minimum sequence length to keep
        max_seq_len (int): Maximum sequence length per user

    Returns:
        list: List of sequences (each sequence is a list of item IDs)
    """
    sequences = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_users:
                break

            items = line.strip().split()

            if len(items) >= min_len:
                sequences.append(items[-max_seq_len:])

    return sequences


def build_sequence_dicts(train_sequences_raw, valid_sequences_raw, test_sequences_raw):
    """
    Convert raw sequence lists into dictionary format.

    For validation and test:
        - history = sequence[:-1]
        - target = sequence[-1]

    Args:
        train_sequences_raw (list)
        valid_sequences_raw (list)
        test_sequences_raw (list)

    Returns:
        tuple:
            train_sequences (dict)
            val_sequences (dict)
            test_sequences (dict)
    """
    train_sequences = {}
    val_sequences = {}
    test_sequences = {}

    for i, seq in enumerate(train_sequences_raw):
        user_id = f"user_{i}"
        train_sequences[user_id] = seq

    for i, seq in enumerate(valid_sequences_raw):
        user_id = f"user_{i}"
        val_sequences[user_id] = (seq[:-1], seq[-1])

    for i, seq in enumerate(test_sequences_raw):
        user_id = f"user_{i}"
        test_sequences[user_id] = (seq[:-1], seq[-1])

    return train_sequences, val_sequences, test_sequences


def build_item_title_map(info_file):
    """
    Build mapping from item_id to item_title.

    File format:
        title \\t item_id

    Args:
        info_file (str): Path to item info file

    Returns:
        dict: {item_id: item_title}
    """
    item_title_map = {}

    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                title, item_id = parts
                item_title_map[item_id] = title

    return item_title_map


def create_csft_samples(train_sequences, max_seq_len=10):
    """
    Create CSFT training samples from sequences.

    Each sample:
        (history, target)

    Args:
        train_sequences (dict): {user_id: [item_ids]}
        max_seq_len (int): Maximum history length

    Returns:
        list: List of (history, target) tuples
    """
    samples = []

    for seq in train_sequences.values():
        for i in range(1, len(seq), 2):
            start = max(0, i - max_seq_len)
            history = seq[start:i]
            target = seq[i]
            samples.append((history, target))

    return samples


def load_data(train_file, valid_file, test_file, info_file, max_users=7000):
    """
    Load and preprocess all data for training.

    Steps:
        1. Read sequences from files
        2. Convert to dictionary format
        3. Build item-title mapping
        4. Generate CSFT samples

    Args:
        train_file (str)
        valid_file (str)
        test_file (str)
        info_file (str)
        max_users (int)

    Returns:
        tuple:
            train_sequences (dict)
            val_sequences (dict)
            test_sequences (dict)
            item_title_map (dict)
            csft_samples (list)
    """
    train_sequences_raw = read_sequences(train_file, max_users=max_users)
    valid_sequences_raw = read_sequences(valid_file, max_users=max_users)
    test_sequences_raw = read_sequences(test_file, max_users=max_users)

    train_sequences, val_sequences, test_sequences = build_sequence_dicts(
        train_sequences_raw, valid_sequences_raw, test_sequences_raw
    )

    item_title_map = build_item_title_map(info_file)
    csft_samples = create_csft_samples(train_sequences)

    return train_sequences, val_sequences, test_sequences, item_title_map, csft_samples
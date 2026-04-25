import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial


def create_csft_samples(sequences, max_seq_len=10):
    """
    Create CSFT (Causal Sequential Fine-Tuning) samples.

    Args:
        sequences (dict): User sequences {user_id: [item1, item2, ...]}
        max_seq_len (int): Maximum length of history to consider

    Returns:
        list: List of (history, target) tuples
    """
    samples = []

    for seq in sequences.values():
        for i in range(1, len(seq), 2):
            start = max(0, i - max_seq_len)
            history = seq[start:i]
            target = seq[i]
            samples.append((history, target))

    return samples


class CSFTDataset(Dataset):
    """
    PyTorch Dataset for CSFT training.

    Converts user interaction sequences into tokenized inputs suitable
    for training a language model for recommendation tasks.

    Each sample consists of:
        - input_ids: tokenized sequence (history + target)
        - attention_mask: mask for valid tokens
        - labels: masked labels (only target contributes to loss)
    """

    def __init__(
        self,
        train_sequences,
        item_title_map,
        tokenizer,
        max_seq_len=10,
        max_token_len=128,
    ):
        """
        Initialize dataset.

        Args:
            train_sequences (dict): {user_id: [item_ids]}
            item_title_map (dict): {item_id: item_title}
            tokenizer: HuggingFace tokenizer
            max_seq_len (int): Max history length
            max_token_len (int): Max token length per sample
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.max_seq_len = max_seq_len

        # cache tokenized texts for speed
        self.token_cache = {}

        def tokenize_cached(text):
            """
            Tokenize text with caching to avoid repeated computation.
            """
            if text not in self.token_cache:
                self.token_cache[text] = tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_token_len,
                    return_tensors="pt"
                )
            return self.token_cache[text]

        for user_id, items in train_sequences.items():
            if len(items) < 2:
                continue

            for end in range(1, len(items), 2):
                start = max(0, end - self.max_seq_len)
                history = items[start:end]
                target = items[end]

                if len(history) == 0:
                    continue

                if target not in item_title_map:
                    continue

                # Convert history item IDs to titles
                history_text = [
                    item_title_map[x]
                    for x in history
                    if x in item_title_map
                ]

                if len(history_text) == 0:
                    continue

                target_text = item_title_map[target]

                # Create input text
                input_text = " ; ".join(history_text) + " ; "
                full_text = input_text + target_text

                # Tokenize
                full_enc = tokenize_cached(full_text)
                input_enc = tokenize_cached(input_text)

                input_ids = full_enc["input_ids"].squeeze(0)
                attention_mask = full_enc["attention_mask"].squeeze(0)

                if len(input_ids) < 2:
                    continue

                input_len = input_enc["input_ids"].shape[1]
                labels = input_ids.clone()

                # Mask history tokens (ignore in loss)
                input_len = min(input_len, len(labels) - 1)
                labels[:input_len] = -100

                # Ensure at least one valid target token
                if (labels != -100).sum().item() < 1:
                    continue

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

        print("Total CSFT samples:", len(self.samples))

    def __len__(self):
        """
        Return dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get one sample by index.
        """
        return self.samples[idx]


def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader.

    Pads variable-length sequences in a batch.

    Args:
        batch (list): List of samples
        tokenizer: Tokenizer (for pad token)

    Returns:
        dict: Batched tensors (input_ids, attention_mask, labels)
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    return {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0
        ),
        "labels": pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
    }


def build_csft_loader(dataset, tokenizer, batch_size=4, shuffle=True):
    """
    Build DataLoader for CSFT training.

    Args:
        dataset (CSFTDataset): Dataset object
        tokenizer: Tokenizer
        batch_size (int): Batch size
        shuffle (bool): Shuffle data or not

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )


def train_csft(model, loader, optimizer, device, epochs=1, accum_steps=2):
    """
    Training loop for CSFT model.

    Uses gradient accumulation to simulate larger batch sizes.

    Args:
        model: Language model
        loader: DataLoader
        optimizer: Optimizer
        device: CPU/GPU
        epochs (int): Number of epochs
        accum_steps (int): Gradient accumulation steps

    Returns:
        model: Trained model
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        valid_steps = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Skip if no valid labels
            if (labels != -100).sum() == 0:
                continue

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Skip invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}")
                continue

            loss = loss / accum_steps
            loss.backward()

            # Update weights after accumulation
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            valid_steps += 1

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, loss = {loss.item() * accum_steps:.4f}")

        # Final step if not aligned with accum_steps
        if (batch_idx + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch done")
        print("Valid steps:", valid_steps)

        if valid_steps > 0:
            print("Avg loss:", total_loss / valid_steps)

    return model
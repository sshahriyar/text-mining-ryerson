import torch
from torch.utils.data import Dataset

PAD_IDX = 0

class SeqRecDataset(Dataset):
    """
    Dataset for sequential recommendation.

    Converts raw sequences into:
    - Padded input sequences
    - Target next item
    - Padding mask

    Used for training the SASRec model.
    """
    def __init__(self, sequences, item_id_to_idx, max_len=10):
        self.sequences = sequences
        self.item_id_to_idx = item_id_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Convert item IDs to indices, filtering out items not in our embedding matrix
        indices = [self.item_id_to_idx[item] for item in seq if item in self.item_id_to_idx]

        if len(indices) < 2:  # Need at least one history item and one target
            return self.__getitem__((idx + 1) % len(self.sequences))

        # Input is everything except the last item
        # Target is the last item
        input_ids = indices[:-1][-self.max_len:]
        target_id = indices[-1]

        # Padding
        pad_len = self.max_len - len(input_ids)
        padding_mask = [True] * pad_len + [False] * len(input_ids)
        input_ids = [PAD_IDX] * pad_len + input_ids

        return {
            "item_seq": torch.tensor(input_ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
            "target": torch.tensor(target_id, dtype=torch.long)
        }
# sasrec.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SASRecDataset(Dataset):
    """
    Dataset for SASRec model.

    Converts user sequences into:
        - padded item sequences
        - padding masks
        - target item
    """
    def __init__(
        self,
        sequences_dict,
        item_to_idx,
        max_seq_len,
        pad_idx=0
    ):
        """
        Args:
            sequences_dict (dict): {user_id: (history_seq, target)}
            item_to_idx (dict): {item_id: index}
            max_seq_len (int): Maximum sequence length
            pad_idx (int): Padding index
        """
        self.item_to_idx = item_to_idx
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.samples = []

        for history_seq, target in sequences_dict.values():

            # keep only known history items
            mapped_history = [
                self.item_to_idx[str(x)]
                for x in history_seq
                if str(x) in self.item_to_idx
            ]

            # skip if target unseen
            if str(target) not in self.item_to_idx:
                continue

            mapped_target = self.item_to_idx[str(target)]

            # skip empty histories
            if len(mapped_history) == 0:
                continue

            self.samples.append(
                (mapped_history, mapped_target)
            )

        print(f"Valid SASRec samples: {len(self.samples)}")

    def __len__(self):
        """
        Return dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one sample:
            - item_seq (padded sequence)
            - padding_mask
            - target
        """
        history_seq, target = self.samples[idx]

        history_seq = history_seq[-self.max_seq_len:]

        padded_seq = (
            [self.pad_idx] *
            (self.max_seq_len - len(history_seq))
            + history_seq
        )

        padding_mask = [
            x == self.pad_idx
            for x in padded_seq
        ]

        return {
            "item_seq": torch.tensor(
                padded_seq,
                dtype=torch.long
            ),
            "padding_mask": torch.tensor(
                padding_mask,
                dtype=torch.bool
            ),
            "target": torch.tensor(
                target,
                dtype=torch.long
            )
        }


class SASRec(nn.Module):
    """
    SASRec model (Self-Attentive Sequential Recommendation).

    Uses:
        - Pretrained item embeddings
        - Transformer encoder
        - Positional embeddings
        - Dot-product scoring for next-item prediction
    """
    def __init__(
        self,
        embeddings_np,
        hidden_dim=128,
        num_heads=2,
        num_layers=2,
        max_len=10,
        dropout=0.2,
        pad_idx=0
    ):
        """
        Args:
            embeddings_np (np.array): Pretrained item embeddings
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
            pad_idx (int): Padding index
        """
        super().__init__()

        self.pad_idx = pad_idx
        self.max_len = max_len

        emb_tensor = torch.tensor(
            embeddings_np,
            dtype=torch.float32
        )

        # Frozen pretrained embeddings
        self.item_embeddings = nn.Embedding.from_pretrained(
            emb_tensor,
            freeze=True,
            padding_idx=pad_idx
        )

        # Adapter to match hidden dimension
        self.adapter = nn.Linear(
            embeddings_np.shape[1],
            hidden_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            max_len,
            hidden_dim
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, item_seq, padding_mask):
        """
        Forward pass.

        Steps:
            1. Embed items
            2. Add positional embeddings
            3. Apply transformer encoder
            4. Extract last valid hidden state
            5. Compute scores against all items

        Args:
            item_seq (tensor): Input sequences
            padding_mask (tensor): Mask for padding

        Returns:
            tensor: Scores for all items
        """
        seq_emb = self.item_embeddings(item_seq)
        seq_emb = self.adapter(seq_emb)

        positions = torch.arange(
            item_seq.size(1),
            device=item_seq.device
        ).unsqueeze(0)

        pos_emb = self.position_embedding(positions)

        x = seq_emb + pos_emb
        x = self.dropout(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # Get last valid position
        seq_lengths = (~padding_mask).sum(dim=1) - 1
        seq_lengths = torch.clamp(seq_lengths, min=0)

        batch_idx = torch.arange(
            x.size(0),
            device=x.device
        )

        last_hidden = x[batch_idx, seq_lengths]
        last_hidden = self.output(last_hidden)

        # Compute scores with all items
        all_item_embs = self.item_embeddings.weight
        all_item_embs = self.adapter(all_item_embs)
        all_item_embs = self.output(all_item_embs)

        scores = last_hidden @ all_item_embs.T

        # Mask padding index
        scores[:, self.pad_idx] = -1e9

        return scores
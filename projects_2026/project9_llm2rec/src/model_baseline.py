import torch
import torch.nn as nn

class BERTSASRec(nn.Module):
    """
    BERT + SASRec sequential recommendation model.

    Combines:
    - Frozen BERT embeddings
    - Linear adapter layer (768 → hidden_dim)
    - Transformer encoder (sequence modeling)

    Predicts next item based on user interaction history.
    """
    def __init__(self, item_embeddings, hidden_dim=128, num_heads=2, num_layers=2, max_len=10):
        super().__init__()

        # Load the frozen BERT embeddings
        self.item_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(item_embeddings, dtype=torch.float32),
            freeze=True
        )

        # PROJECTION LAYER: Crucial for Choice B
        # Maps BERT 768 -> SASRec 128
        self.adapter = nn.Linear(768, hidden_dim)

        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_seq, padding_mask):
        # Get BERT embeddings and project them
        seq_emb = self.adapter(self.item_embeddings(item_seq))

        # Add Positional Encoding
        pos = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)

        # Transformer Layers
        x = self.transformer(seq_emb + pos_emb, src_key_padding_mask=padding_mask)

        # Final Hidden State (last item in sequence)
        user_vector = self.output_layer(x[:, -1, :])

        # Score against all projected item embeddings
        all_items_projected = self.output_layer(self.adapter(self.item_embeddings.weight))
        scores = torch.matmul(user_vector, all_items_projected.T)

        return scores
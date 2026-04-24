import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np


def enable_bidirectional_attention(model):
    """
    Enable bidirectional attention in a causal model.

    Modifies:
        - Sets is_causal = False in all modules
        - Disables cache
        - Converts model from decoder to encoder-style behavior

    Args:
        model: Transformer model

    Returns:
        model: Modified model with bidirectional attention
    """
    changed = 0
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            changed += 1

    if hasattr(model.config, "is_causal"):
        model.config.is_causal = False
    
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.config.is_decoder = False
    print("Bidirectional attention enabled")
    print("Patched modules:", changed)
    
    return model


class ItemDataset(Dataset):
    """
    Dataset of item titles for MNTP and contrastive learning.

    Each sample contains:
        - item_id
        - tokenized title
    """
    def __init__(self, item_title_map, tokenizer, max_len=64):
        """
        Args:
            item_title_map (dict): {item_id: title}
            tokenizer: Tokenizer
            max_len (int): Max token length
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for item_id, title in item_title_map.items():
            if title is not None and str(title).strip() != "":
                self.samples.append((item_id, str(title)))

        print("Total items:", len(self.samples))

    def __len__(self):
        """
        Return number of items.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one tokenized item sample.
        """
        item_id, title = self.samples[idx]
        enc = self.tokenizer(
            title,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "item_id": item_id,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def item_collate_fn(batch, tokenizer):
    """
    Collate function for item dataset.

    Pads sequences to same length.

    Args:
        batch (list): List of samples
        tokenizer: Tokenizer

    Returns:
        dict: Batched tensors
    """
    item_ids = [x["item_id"] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    return {
        "item_ids": item_ids,
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0)
    }


def mask_tokens(input_ids, attention_mask, tokenizer, mask_prob=0.2):
    """
    Apply random masking for MNTP (Masked Next Token Prediction).

    Args:
        input_ids (tensor)
        attention_mask (tensor)
        tokenizer
        mask_prob (float): Probability of masking

    Returns:
        tuple:
            masked_input_ids
            labels (only masked positions contribute to loss)
    """
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    maskable = attention_mask.bool()
    random_matrix = torch.rand(input_ids.shape, device=input_ids.device)
    mask_positions = (random_matrix < mask_prob) & maskable

    labels[mask_positions] = input_ids[mask_positions]
    masked_input_ids[mask_positions] = tokenizer.mask_token_id
    return masked_input_ids, labels


from functools import partial


def build_item_loader(dataset, tokenizer, batch_size=16, shuffle=True):
    """
    Build DataLoader for item dataset.

    Args:
        dataset
        tokenizer
        batch_size (int)
        shuffle (bool)

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(item_collate_fn, tokenizer=tokenizer)
    )


def train_mntp(
    model,
    loader,
    optimizer,
    device,
    tokenizer,
    epochs=1,
    mask_prob=0.2,
    max_steps=None
):
    """
    Train model using MNTP (Masked Next Token Prediction).

    Args:
        model
        loader
        optimizer
        device
        tokenizer
        epochs (int)
        mask_prob (float)
        max_steps (int or None)

    Returns:
        model: Trained model
    """
    model.train()

    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.pad_token
    if tokenizer.mask_token_id is None:
        tokenizer.mask_token_id = tokenizer.pad_token_id

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        valid_steps = 0

        for batch_idx, batch in enumerate(loader):
            if max_steps is not None and global_step >= max_steps:
                print("Reached max_steps, stopping MNTP.")
                return model

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_input_ids, labels = mask_tokens(
                input_ids,
                attention_mask,
                tokenizer,
                mask_prob=mask_prob
            )

            if (labels != -100).sum() == 0:
                continue

            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            valid_steps += 1
            global_step += 1

            if batch_idx % 50 == 0:
                print(f"MNTP epoch {epoch+1}, batch {batch_idx}, loss = {loss.item():.4f}")

        print(f"MNTP epoch {epoch+1} done")
        print("Valid steps:", valid_steps)
        if valid_steps > 0:
            print("Avg loss:", running_loss / valid_steps)

    return model


class EmbeddingModel(nn.Module):
    """
    Model wrapper for generating normalized embeddings for contrastive learning.

    Uses:
        - Mean pooling
        - Dropout (for augmentation)
        - L2 normalization
    """
    def __init__(self, base_model, dropout_p=0.2):
        super().__init__()
        self.base_model = base_model
        self.view_dropout = nn.Dropout(dropout_p)

    def get_embedding(self, input_ids, attention_mask):
        """
        Generate a single embedding for input sequence.

        Args:
            input_ids
            attention_mask

        Returns:
            tensor: Normalized embedding
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

        if torch.isnan(last_hidden).any():
            last_hidden = torch.nan_to_num(last_hidden, nan=0.0)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        pooled = summed / counts
        pooled = self.view_dropout(pooled)
        pooled = F.normalize(pooled, p=2, dim=-1)

        if torch.isnan(pooled).any():
            pooled = torch.nan_to_num(pooled, nan=0.0)
        return pooled

    def forward(self, input_ids, attention_mask):
        """
        Generate two augmented views for contrastive learning.
        """
        z1 = self.get_embedding(input_ids, attention_mask)
        z2 = self.get_embedding(input_ids, attention_mask)
        return z1, z2


def nt_xent_loss(z1, z2, temperature=0.2):
    """
    Compute NT-Xent contrastive loss.

    Args:
        z1, z2: Embedding views
        temperature (float)

    Returns:
        loss (tensor)
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])

    return F.cross_entropy(sim, labels)


def train_contrastive(
    emb_model,
    loader,
    optimizer,
    device,
    epochs=1,
    max_steps=None
):
    """
    Train embedding model using contrastive learning.

    Args:
        emb_model
        loader
        optimizer
        device
        epochs (int)
        max_steps (int or None)

    Returns:
        emb_model: Trained embedding model
    """
    emb_model.train()
    global_step = 0

    for epoch in range(epochs):
        emb_model.train()
        running_loss = 0.0
        valid_steps = 0

        for batch_idx, batch in enumerate(loader):
            if max_steps is not None and global_step >= max_steps:
                print("Reached max_steps, stopping contrastive training.")
                return emb_model

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            z1, z2 = emb_model(input_ids, attention_mask)
            loss = nt_xent_loss(z1, z2)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            valid_steps += 1
            global_step += 1

            if batch_idx % 50 == 0:
                print(f"Contrastive epoch {epoch+1}, batch {batch_idx}, loss = {loss.item():.4f}")

        print(f"Contrastive epoch {epoch+1} done")
        print("Valid steps:", valid_steps)
        if valid_steps > 0:
            print("Avg loss:", running_loss / valid_steps)

    return emb_model


class OrderedItemDataset(Dataset):
    """
    Dataset for ordered item processing during embedding extraction.

    Ensures consistent order for building embedding matrix.
    """
    def __init__(self, ordered_items, tokenizer, max_len=64):
        self.ordered_items = ordered_items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Return dataset size.
        """
        return len(self.ordered_items)

    def __getitem__(self, idx):
        """
        Return tokenized item for embedding extraction.
        """
        item_id, title = self.ordered_items[idx]
        enc = self.tokenizer(
            str(title),
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "item_id": item_id,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def extraction_collate_fn(batch, tokenizer):
    """
    Collate function for embedding extraction.

    Pads sequences and groups item IDs.

    Args:
        batch
        tokenizer

    Returns:
        dict
    """
    item_ids = [x["item_id"] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    return {
        "item_ids": item_ids,
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0)
    }


@torch.no_grad()
def run_embedding_extraction(
    emb_model,
    item_title_map,
    tokenizer,
    device,
    batch_size=32,
    max_len=64
):
    """
    Extract embeddings for all items.

    Steps:
        1. Build ordered dataset
        2. Run model to get embeddings
        3. Stack into matrix

    Args:
        emb_model
        item_title_map (dict)
        tokenizer
        device
        batch_size (int)
        max_len (int)

    Returns:
        tuple:
            item_embeddings_matrix (numpy array)
            all_item_ids (list)
    """
    emb_model.eval()
    
    ordered_items = [
        (item_id, title) for item_id, title in item_title_map.items() 
        if title is not None and str(title).strip() != ""
    ]
    
    dataset = OrderedItemDataset(
        ordered_items,
        tokenizer,
        max_len=max_len
    )

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: extraction_collate_fn(x, tokenizer)
    )

    all_embeddings = []
    all_item_ids = []

    print(f"Extracting embeddings for {len(ordered_items)} items...")

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        emb = emb_model.get_embedding(input_ids, attention_mask)

        all_embeddings.append(emb.cpu().numpy())
        all_item_ids.extend(batch["item_ids"])

    item_embeddings_matrix = np.vstack(all_embeddings)
    
    return item_embeddings_matrix, all_item_ids
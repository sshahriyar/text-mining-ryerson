import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

BERT_MODEL_NAME = "bert-base-uncased"


def load_bert(device):
    print(f"Loading Baseline Model: {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_model.eval()
    return tokenizer, bert_model


def extract_bert_embeddings(model, tokenizer, titles, device, batch_size=64):
    """
    Generate BERT embeddings for item titles.

    Uses the [CLS] token representation as the embedding.

    Args:
        model: Pretrained BERT model.
        tokenizer: BERT tokenizer.
        titles (list): List of item titles.
        device: CPU or GPU.
        batch_size (int): Batch size for processing.

    Returns:
        np.ndarray: Matrix of item embeddings.
    """
    
    all_embeddings = []

    # Index 0 is reserved for PADDING
    all_embeddings.append(np.zeros(768))

    for i in tqdm(range(0, len(titles), batch_size), desc="BERT Embedding Extraction"):
        batch_titles = titles[i: i + batch_size]
        inputs = tokenizer(
            batch_titles,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token (index 0) as the representative embedding for the title
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)
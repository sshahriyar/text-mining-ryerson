# evaluation.py

import math
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, device, k_list=[10, 20]):
    """
    Evaluate recommendation model using Recall@K and NDCG@K.

    For each sample:
        - Predict scores for all items
        - Compute rank of the true target item
        - Update Recall and NDCG metrics

    Args:
        model: Trained recommendation model
        loader: DataLoader for evaluation data
        device: CPU/GPU device
        k_list (list): List of K values for evaluation (e.g., [10, 20])

    Returns:
        dict: Evaluation metrics (Recall@K, NDCG@K)
    """
    model.eval()

    # Initialize metric storage
    metrics = {
        f"Recall@{k}": 0.0 for k in k_list
    }

    metrics.update({
        f"NDCG@{k}": 0.0 for k in k_list
    })

    total = 0

    for batch in tqdm(loader, leave=False):
        """
        Loop over evaluation batches.
        Each batch contains:
            - item_seq: input sequences
            - padding_mask: mask for padded positions
            - target: true next item
        """
        item_seq = batch["item_seq"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["target"].to(device)

        # Get prediction scores for all items
        scores = model(item_seq, padding_mask)

        batch_size = scores.size(0)
        total += batch_size

        for i in range(batch_size):
            """
            Evaluate each sample in the batch:
                - Get score of true item
                - Compute its rank among all items
            """
            target_score = scores[i, targets[i]].item()

            # Rank = number of items with higher score + 1
            rank = (
                scores[i] > target_score
            ).sum().item() + 1

            for k in k_list:
                """
                Update metrics:
                    - Recall@K: whether target is in top-K
                    - NDCG@K: ranking quality
                """
                if rank <= k:
                    metrics[f"Recall@{k}"] += 1
                    metrics[f"NDCG@{k}"] += (
                        1 / math.log2(rank + 1)
                    )

    # Normalize metrics by total samples
    for k in k_list:
        metrics[f"Recall@{k}"] /= total
        metrics[f"NDCG@{k}"] /= total

    return metrics
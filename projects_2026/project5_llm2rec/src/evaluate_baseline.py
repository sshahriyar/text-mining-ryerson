import torch
import numpy as np
from tqdm import tqdm

def evaluate(model, data_loader, device, k_list=[10, 20]):
    """
    Evaluate model performance.

    Metrics:
    - Recall@K
    - NDCG@K

    Args:
        model: Trained model.
        data_loader: Validation data loader.
        device: CPU or GPU.
        k_list (list): List of K values.

    Returns:
        dict: Evaluation metrics.
    """
    model.eval()
    results = {f"Recall@{k}": 0 for k in k_list}
    results.update({f"NDCG@{k}": 0 for k in k_list})
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            item_seq = batch["item_seq"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            targets = batch["target"].to(device)

            # Get scores for ALL items
            scores = model(item_seq, padding_mask)

            for i in range(len(targets)):
                target_item = targets[i].item()
                user_scores = scores[i]

                # Get top K indices
                _, top_indices = torch.topk(user_scores, max(k_list))
                top_indices = top_indices.cpu().numpy()

                for k in k_list:
                    if target_item in top_indices[:k]:
                        results[f"Recall@{k}"] += 1
                        rank = np.where(top_indices[:k] == target_item)[0][0]
                        results[f"NDCG@{k}"] += 1 / np.log2(rank + 2)

                total_samples += 1

    # Average the results
    for key in results:
        results[key] /= total_samples

    return results
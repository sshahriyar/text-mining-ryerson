import torch

@torch.no_grad()
def predict_next_items_baseline(
    model,
    history_titles,
    find_item_id_by_title_fn,
    title_to_item_ids,
    item_id_to_idx,
    idx_to_item,
    item_id_to_title,
    device,
    top_k=10,
    max_len=10
):
    """
    Predict next items given user history.

    Steps:
    - Convert titles to item IDs
    - Map to indices
    - Apply model
    - Return top-K predictions

    Args:
        model: Trained model.
        history_titles (list): User input titles.
        top_k (int): Number of recommendations.

    Returns:
        list: Predicted items with titles.
    """
    
    model.eval()

    found_item_ids = []
    not_found_titles = []

    # convert input titles -> item ids
    for title in history_titles:
        item_id = find_item_id_by_title_fn(title, title_to_item_ids)
        if item_id is None:
            not_found_titles.append(title)
        else:
            found_item_ids.append(item_id)

    if len(found_item_ids) == 0:
        print("No valid movie titles found in dataset.")
        return []

    # convert item ids -> indices mapped to BERT matrix
    history_indices = [item_id_to_idx[x] for x in found_item_ids if x in item_id_to_idx]

    if len(history_indices) == 0:
        print("Titles found, but not present in the current item_id_to_idx map.")
        return []

    # Match the max_len used in BERT training
    history_indices = history_indices[-max_len:]

    # Left pad with PAD_IDX (0)
    padded = [0] * (max_len - len(history_indices)) + history_indices
    padding_mask = [x == 0 for x in padded]

    item_seq = torch.tensor([padded], dtype=torch.long).to(device)
    padding_mask = torch.tensor([padding_mask], dtype=torch.bool).to(device)

    # get scores from model
    scores = model(item_seq, padding_mask)
    scores = scores[0]

    # Mask already seen items
    for idx in history_indices:
        scores[idx] = -1e9

    top_indices = torch.topk(scores, k=top_k).indices.tolist()

    predictions = []
    for idx in top_indices:
        if idx == 0:
            continue  # Skip padding index
        item_id = idx_to_item.get(idx)
        title = item_id_to_title.get(item_id, "Unknown Title")
        predictions.append((item_id, title))

    if not_found_titles:
        print("Note: These titles were not found in the BERT index:", not_found_titles)

    return predictions
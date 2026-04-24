# inference.py

import torch


def build_title_maps(item_title_map, item_to_idx):
    """
    Build helper mappings for inference.

    Creates:
        - idx_to_item: maps index → item_id
        - title_to_item_ids: maps normalized title → list of item_ids

    Args:
        item_title_map (dict): {item_id: title}
        item_to_idx (dict): {item_id: index}

    Returns:
        tuple:
            idx_to_item (dict)
            title_to_item_ids (dict)
    """
    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    title_to_item_ids = {}

    for item_id, title in item_title_map.items():
        if title is None:
            continue

        title_lower = title.strip().lower()

        if title_lower not in title_to_item_ids:
            title_to_item_ids[title_lower] = []

        title_to_item_ids[title_lower].append(item_id)

    return idx_to_item, title_to_item_ids


def find_item_id_by_title(query_title, title_to_item_ids):
    """
    Find item_id using a given title (exact or partial match).

    Args:
        query_title (str): Input movie title
        title_to_item_ids (dict): title → list of item_ids

    Returns:
        str or None: Matching item_id if found, else None
    """
    query = query_title.strip().lower()

    # Exact match
    if query in title_to_item_ids:
        return title_to_item_ids[query][0]

    # Partial match
    for title_lower, item_ids in title_to_item_ids.items():
        if query in title_lower:
            return item_ids[0]

    return None


@torch.no_grad()
def predict_next_items(
    model,
    history_titles,
    item_to_idx,
    idx_to_item,
    item_title_map,
    title_to_item_ids,
    device=None,
    max_seq_len=10,
    pad_idx=0,
    top_k=10
):
    """
    Predict next recommended items based on user history.

    Steps:
        1. Convert input titles → item_ids
        2. Convert item_ids → indices
        3. Pad sequence to fixed length
        4. Run model to get scores
        5. Remove already seen items
        6. Return top-K recommendations

    Args:
        model: Trained recommendation model
        history_titles (list): List of input movie titles
        item_to_idx (dict): {item_id: index}
        idx_to_item (dict): {index: item_id}
        item_title_map (dict): {item_id: title}
        title_to_item_ids (dict): title → item_ids
        device: Optional device (CPU/GPU)
        max_seq_len (int): Max sequence length
        pad_idx (int): Padding index
        top_k (int): Number of recommendations

    Returns:
        list: List of (item_id, title) predictions
    """
    model.eval()
    model_device = next(model.parameters()).device

    found_item_ids = []
    not_found_titles = []

    # Convert titles to item_ids
    for title in history_titles:
        item_id = find_item_id_by_title(title, title_to_item_ids)

        if item_id is None:
            not_found_titles.append(title)
        else:
            found_item_ids.append(item_id)

    if len(found_item_ids) == 0:
        print("No valid movie titles found.")
        return []

    # Convert item_ids to indices
    history_indices = [
        item_to_idx[x]
        for x in found_item_ids
        if x in item_to_idx
    ]

    if len(history_indices) == 0:
        return []

    # Keep only recent history
    history_indices = history_indices[-max_seq_len:]

    # Pad sequence
    padded = [pad_idx] * (max_seq_len - len(history_indices)) + history_indices
    padding_mask = [x == pad_idx for x in padded]

    # Convert to tensors
    item_seq = torch.tensor([padded], dtype=torch.long, device=model_device)
    padding_mask = torch.tensor([padding_mask], dtype=torch.bool, device=model_device)

    # Get prediction scores
    scores = model(item_seq, padding_mask)[0]

    # Remove already seen items
    for idx in history_indices:
        scores[idx] = -1e9

    # Get top-K predictions
    top_indices = torch.topk(scores, k=top_k).indices.tolist()

    predictions = []
    for idx in top_indices:
        if idx == pad_idx:
            continue

        item_id = idx_to_item.get(idx)
        title = item_title_map.get(item_id, "Unknown Title")
        predictions.append((item_id, title))

    return predictions


def search_titles(keyword, item_title_map, max_results=10):
    """
    Search items by keyword in titles.

    Args:
        keyword (str): Search keyword
        item_title_map (dict): {item_id: title}
        max_results (int): Max number of results

    Returns:
        list: List of (item_id, title) matches
    """
    keyword = keyword.strip().lower()

    matches = []

    for item_id, title in item_title_map.items():
        if title is None:
            continue

        if keyword in title.lower():
            matches.append((item_id, title))

    return matches[:max_results]
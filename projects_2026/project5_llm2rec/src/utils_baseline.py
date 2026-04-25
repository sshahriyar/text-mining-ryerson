def build_mappings(item2text):
    """
    Build mapping dictionaries for item IDs and titles.

    Creates:
    - item_id_to_idx
    - idx_to_item
    - title_to_item_ids

    Args:
        item2text (dict): Item ID to title mapping.

    Returns:
        tuple: Mapping dictionaries.
    """
    item_ids = list(item2text.keys())

    item_id_to_idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
    idx_to_item = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    item_id_to_title = item2text

    title_to_item_ids = {}
    for item_id, title in item_id_to_title.items():
        title_lower = title.strip().lower()
        if title_lower not in title_to_item_ids:
            title_to_item_ids[title_lower] = []
        title_to_item_ids[title_lower].append(item_id)

    return item_ids, item_id_to_idx, idx_to_item, item_id_to_title, title_to_item_ids


def find_item_id_by_title(query_title, title_to_item_ids):
    query = query_title.strip().lower()

    # exact match
    if query in title_to_item_ids:
        return title_to_item_ids[query][0]

    # partial match fallback
    for title_lower, item_ids in title_to_item_ids.items():
        if query in title_lower:
            return item_ids[0]

    return None
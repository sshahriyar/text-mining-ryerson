from dataclasses import dataclass

def dt0(feature_row):
    node1 = int(feature_row['Round'] > 6)
    node2 = int(feature_row['Official'] == 1)
    node3 = int(feature_row['Judge'] in [0, 1, 2, 3])
    node4 = int(feature_row['Round'] <= 3)
    node5 = int(feature_row['Judge'] in [4, 5, 6])
    node6 = int(feature_row['Round'] > 9)
    emb = [node1, node2, node3, node4, node5, node6]

    if node1 == 1 and node2 == 1 and node3 == 1:
        return 1, emb
    elif node1 == 0 and node4 == 1 and node5 == 1:
        return 0, emb
    elif node1 == 1 and node6 == 1:
        return 0, emb
    else:
        return 1, emb
    
def dt1(row):
    # Precompute binary vector embeddings
    node1 = int(row['Judge'] == 0)
    node2 = int(row['Judge'] == 2)
    node3 = int(row['Official'] == 1)
    node4 = int(row['Round'] <= 6)
    node5 = int(row['Round'] <= 3)

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1:
        prediction = 0
    elif node2:
        prediction = 1
    elif node3 and node4:
        if node5:
            prediction = 0
        else:
            prediction = 1
    elif node4:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb

def dt2(feature_row):
    # Precompute binary vector embeddings
    node1 = int(feature_row['Judge'] in {0, 1, 2, 3, 4})
    node2 = int(feature_row['Official'] == 1)
    node3 = int(feature_row['Round'] < 5)
    node4 = int(feature_row['Round'] < 8)
    node5 = int(feature_row['Round'] < 11)
    node6 = int(feature_row['Judge'] in {5, 6, 7, 8, 9, 10})
    emb = [node1, node2, node3, node4, node5, node6]

    # Apply decision tree logic
    if node1 and node2 and node3:
        prediction = 0
    elif node1 and node2 and node4 and node5:
        prediction = 1
    elif node1 and not node2 and node3:
        prediction = 1
    elif node1 and not node2 and not node3 and node4:
        prediction = 0
    elif node6 and node3:
        prediction = 1
    elif node6 and not node3 and node4:
        prediction = 0
    else:
        prediction = 0

    return prediction, emb


def dt3(feature_row):
    node1 = int(feature_row['Judge'] in [0, 1, 2, 3])
    node2 = int(feature_row['Judge'] in [4, 5, 6, 7, 8])
    node3 = int(feature_row['Judge'] in [9, 10])
    node4 = int(feature_row['Official'] == 1)
    node5 = int(feature_row['Round'] <= 6)
    node6 = int(feature_row['Round'] > 6)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1:
        return 0, emb
    elif node2 and node4 and node5:
        return 1, emb
    elif node2 and node4 and node6:
        return 0, emb
    elif node3 and node5:
        return 1, emb
    else:
        return 0, emb


def dt4(row):
    # Precompute binary vector embeddings
    node1 = int(row['Round'] <= 6)
    node2 = int(row['Judge'] in {0, 1, 2, 3, 4})
    node3 = int(row['Judge'] in {5, 6, 7, 8, 9, 10})
    node4 = int(row['Round'] > 3 and row['Round'] <= 9)
    node5 = int(row['Official'] == 1)

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1 and node2:
        prediction = 0  # Trinidad
    elif node1 and node3:
        prediction = 1  # de la Hoya
    elif node4 and node5:
        prediction = 0  # Trinidad
    else:
        prediction = 1  # de la Hoya

    return prediction, emb

@dataclass
class MistralBoxing2Embedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
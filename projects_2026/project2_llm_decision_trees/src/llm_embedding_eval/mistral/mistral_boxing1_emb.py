from dataclasses import dataclass

def dt0(row):
    node1 = int(row['Judge'] == 0 or row['Judge'] == 3)
    node2 = int(row['Judge'] == 5)
    node3 = int(row['Judge'] == 8)
    node4 = int(row['Judge'] == 9)
    node5 = int(row['Official'] == 1)
    node6 = int(row['Round'] <= 6)
    node7 = int(row['Round'] > 6)
    emb = [node1, node2, node3, node4, node5, node6, node7]

    if node1:
        return 1, emb
    elif node2 and node6:
        return 0, emb
    elif node2 and node7:
        return 1, emb
    elif node3 and node5:
        return 1, emb
    elif node3 and node6:
        return 0, emb
    elif node4 and node7:
        return 1, emb
    else:
        return 0, emb

def dt1(feature_row):
    # Precompute binary vector embeddings
    node1 = int(feature_row['Judge'] in [0, 1, 2, 5, 9])
    node2 = int(feature_row['Judge'] in [3, 4, 6, 7, 8])
    node3 = int(feature_row['Round'] <= 6)
    node4 = int(feature_row['Round'] > 6)
    node5 = int(feature_row['Official'] == 1)

    emb = [node1, node2, node3, node4, node5]

    # Apply decision tree logic
    if node1 and node3 and node5:
        return 1, emb
    elif node1 and node3 and not node5:
        return 0, emb
    elif node1 and not node3 and node5:
        return 1, emb
    elif node1 and not node3 and not node5:
        return 0, emb
    elif node2 and node4 and node5:
        return 0, emb
    elif node2 and node4 and not node5:
        return 1, emb
    elif node2 and not node4 and node5:
        return 0, emb
    else:
        return 1, emb
 
def dt2(row):
    node1 = int(row['Judge'] == 0)
    node2 = int(row['Judge'] == 1)
    node3 = int(row['Judge'] == 2)
    node4 = int(row['Judge'] == 3)
    node5 = int(row['Judge'] == 4)
    node6 = int(row['Judge'] == 5)
    node7 = int(row['Judge'] == 6)
    node8 = int(row['Judge'] == 7)
    node9 = int(row['Judge'] == 8)
    node10 = int(row['Judge'] == 9)
    node11 = int(row['Official'] == 1)
    node12 = int(row['Official'] == 0)
    node13 = int(row['Round'] <= 6)
    node14 = int(row['Round'] > 6)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14]

    if node1:
        prediction = 1
    elif node2:
        prediction = 0
    elif node3:
        prediction = 1
    elif node4:
        prediction = 0
    elif node5:
        prediction = 1
    elif node6:
        prediction = 0
    elif node7:
        prediction = 1
    elif node8:
        prediction = 0
    elif node9:
        prediction = 1
    elif node10:
        prediction = 0
    elif node11:
        if node13:
            prediction = 1
        else:
            prediction = 0
    elif node12:
        if node14:
            prediction = 1
        else:
            prediction = 0
    else:
        prediction = 0

    return prediction, emb

def dt3(row):
    # Precompute binary embeddings
    node1 = int(row['Judge'] in [0, 1, 2])  # E. Williams, L. O'Connell, S. Christodoulu
    node2 = int(row['Official'] == 1)  # Official judge
    node3 = int(row['Round'] <= 6)  # First half of the rounds
    node4 = int(row['Round'] <= 3)  # First third of the rounds
    node5 = int(row['Judge'] in [5, 6, 7])  # Boxing Times, Sportsline, Associated Press
    node6 = int(row['Round'] % 2 == 0)  # Even-numbered rounds

    emb = [node1, node2, node3, node4, node5, node6]

    # Decision tree logic
    if node1 and node4:
        prediction = 1  # Lewis likely to win in early rounds with certain judges
    elif node2 and node3:
        prediction = 0  # Holyfield likely to win in first half with official judges
    elif node5 and node6:
        prediction = 0  # Holyfield likely to win in even rounds with specific judges
    elif node1 and not node2:
        prediction = 1  # Lewis likely to win with non-official judges
    else:
        prediction = 0  # Default to Holyfield

    return prediction, emb


def dt4(feature_row):
    node1 = int(feature_row['Judge'] in [1, 2, 3, 4, 5])
    node2 = int(feature_row['Official'] == 1)
    node3 = int(feature_row['Round'] <= 6)

    emb = [node1, node2, node3]

    if node1:
        if node2:
            prediction = 1
        else:
            if node3:
                prediction = 0
            else:
                prediction = 1
    else:
        prediction = 0

    return prediction, emb

@dataclass
class MistralBoxing1Embedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
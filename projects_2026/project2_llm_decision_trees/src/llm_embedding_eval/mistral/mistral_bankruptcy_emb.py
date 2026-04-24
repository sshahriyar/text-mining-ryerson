from dataclasses import dataclass


def dt_func_0(row):
    # Precompute binary vector embeddings
    node1 = int(row['WC/TA'] < 0)
    node2 = int(row['RE/TA'] < 0)
    node3 = int(row['EBIT/TA'] < 0)
    node4 = int(row['S/TA'] < 0)
    node5 = int(row['BVE/BVL'] < 0.5)

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1 and node2:
        prediction = 1  # Bankrupt
    elif node3 and node4:
        prediction = 1  # Bankrupt
    elif node5:
        prediction = 1  # Bankrupt
    else:
        prediction = 0  # Not bankrupt

    return prediction, emb

def dt_func_1(feature_row):
    # Precompute the binary vector embeddings
    node1 = int(feature_row['WC/TA'] < 0)
    node2 = int(feature_row['RE/TA'] < 0)
    node3 = int(feature_row['EBIT/TA'] < 0)
    node4 = int(feature_row['S/TA'] < 0)
    node5 = int(feature_row['BVE/BVL'] < 0.5)
    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1:
        if node2:
            prediction = 1
        else:
            prediction = 0
    elif node3:
        if node4:
            prediction = 1
        else:
            prediction = 0
    elif node5:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb

def dt_func_2(feature_row):
    node1 = int(feature_row['WC/TA'] < 20)
    node2 = int(feature_row['RE/TA'] < 15)
    node3 = int(feature_row['EBIT/TA'] < 10)
    node4 = int(feature_row['S/TA'] < 30)
    node5 = int(feature_row['BVE/BVL'] < 0.5)

    emb = [node1, node2, node3, node4, node5]

    if node1 == 1 and node2 == 1 and node3 == 1:
        return 1, emb
    elif node1 == 0 and node4 == 0 and node5 == 0:
        return 0, emb
    elif node1 == 1 and node4 == 0 and node5 == 1:
        return 1, emb
    else:
        return 0, emb
    

def dt_func_3(row):
    node1 = int(row['WC/TA'] < 10)
    node2 = int(row['RE/TA'] < 5)
    node3 = int(row['EBIT/TA'] < 7)
    node4 = int(row['S/TA'] < 15)
    node5 = int(row['BVE/BVL'] < 0.5)

    emb = [node1, node2, node3, node4, node5]

    if node1:
        if node2:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3:
            if node4:
                return 1, emb
            else:
                return 0, emb
        else:
            if node5:
                return 1, emb
            else:
                return 0, emb
            
            

def dt_func_4(feature_row):
    node1 = int(feature_row['WC/TA'] > 15)
    node2 = int(feature_row['RE/TA'] < 5)
    node3 = int(feature_row['EBIT/TA'] < 10)
    node4 = int(feature_row['S/TA'] > 50)
    node5 = int(feature_row['BVE/BVL'] < 0.5)

    emb = [node1, node2, node3, node4, node5]

    if node1 == 1 and node2 == 1 and node3 == 1 and node4 == 1:
        prediction = 1
    elif node1 == 0 and node2 == 0 and node3 == 0 and node4 == 0:
        prediction = 0
    elif node1 == 1 and node2 == 0 and node3 == 0 and node4 == 1:
        prediction = 1
    elif node1 == 0 and node2 == 1 and node3 == 1 and node4 == 0:
        prediction = 0
    else:
        prediction = 1

    return prediction, emb

@dataclass
class MistralBankruptcyEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
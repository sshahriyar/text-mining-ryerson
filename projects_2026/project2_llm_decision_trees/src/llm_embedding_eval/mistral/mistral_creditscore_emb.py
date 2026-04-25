from dataclasses import dataclass

def dt0(feature_row):
    # Precompute the binary vector embeddings
    node1 = int(feature_row['Age'] >= 30)
    node2 = int(feature_row['Income.per.dependent'] >= 5)
    node3 = int(feature_row['Monthly.credit.card.exp'] <= 500)
    node4 = int(feature_row['Own.home'] == 1)
    node5 = int(feature_row['Self.employed'] == 1)
    node6 = int(feature_row['Derogatory.reports'] == 0)

    emb = [node1, node2, node3, node4, node5, node6]

    # Decision Tree Logic
    if feature_row['Age'] >= 30 and feature_row['Income.per.dependent'] >= 5:
        if feature_row['Monthly.credit.card.exp'] <= 500:
            if feature_row['Own.home'] == 1 or feature_row['Self.employed'] == 1:
                if feature_row['Derogatory.reports'] == 0:
                    prediction = 1  # Credit application accepted
                else:
                    prediction = 0  # Credit application not accepted
            else:
                prediction = 0  # Credit application not accepted
        else:
            prediction = 0  # Credit application not accepted
    else:
        prediction = 0  # Credit application not accepted

    return prediction, emb

def dt1(feature_row):
    node1 = int(feature_row['Age'] >= 30)
    node2 = int(feature_row['Income.per.dependent'] >= 3)
    node3 = int(feature_row['Monthly.credit.card.exp'] <= 500)
    node4 = int(feature_row['Own.home'] == 1)
    node5 = int(feature_row['Self.employed'] == 1)
    node6 = int(feature_row['Derogatory.reports'] == 0)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1 and node2 and node3 and node4 and node5 and node6:
        prediction = 1
    elif node1 and node2 and node3 and node4 and node5 and not node6:
        prediction = 0
    elif node1 and node2 and node3 and node4 and not node5 and node6:
        prediction = 1
    elif node1 and node2 and node3 and not node4 and node5 and node6:
        prediction = 0
    elif node1 and node2 and not node3 and node4 and node5 and node6:
        prediction = 1
    elif node1 and not node2 and node3 and node4 and node5 and node6:
        prediction = 0
    elif not node1 and node2 and node3 and node4 and node5 and node6:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb

def dt2(feature_row):
    node1 = int(feature_row['Derogatory.reports'] == 0)
    node2 = int(feature_row['Own.home'] == 1)
    node3 = int(feature_row['Self.employed'] == 1)
    node4 = int(feature_row['Income.per.dependent'] >= 5)
    node5 = int(feature_row['Monthly.credit.card.exp'] <= 500)
    node6 = int(feature_row['Age'] >= 30)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1:
        if node2:
            if node3:
                if node4:
                    prediction = 1
                else:
                    prediction = 0
            else:
                prediction = 1
        else:
            prediction = 0
    else:
        if node5:
            if node6:
                prediction = 1
            else:
                prediction = 0
        else:
            prediction = 0

    return prediction, emb

def dt3(feature_row):
    # Precompute the binary vector embeddings
    node1 = int(feature_row['Age'] >= 30)
    node2 = int(feature_row['Income.per.dependent'] > 5)
    node3 = int(feature_row['Monthly.credit.card.exp'] < 1000)
    node4 = int(feature_row['Own.home'] == 1)
    node5 = int(feature_row['Self.employed'] == 1)
    node6 = int(feature_row['Derogatory.reports'] == 0)

    emb = [node1, node2, node3, node4, node5, node6]

    # Apply the decision tree logic
    if node1 and node4 and node6:
        prediction = 1
    elif node2 and node3 and node5:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb

def dt4(feature_row):
    # Precompute binary vector embeddings
    node1 = int(feature_row['Age'] > 30)
    node2 = int(feature_row['Income.per.dependent'] > 5)
    node3 = int(feature_row['Monthly.credit.card.exp'] < 500)
    node4 = int(feature_row['Own.home'] == 1)
    node5 = int(feature_row['Self.employed'] == 1)
    node6 = int(feature_row['Derogatory.reports'] == 0)

    emb = [node1, node2, node3, node4, node5, node6]

    # Decision tree logic
    if feature_row['Derogatory.reports'] == 0:
        if feature_row['Own.home'] == 1:
            if feature_row['Age'] > 30:
                prediction = 1
            else:
                prediction = 0
        else:
            if feature_row['Income.per.dependent'] > 5:
                prediction = 1
            else:
                prediction = 0
    else:
        if feature_row['Monthly.credit.card.exp'] < 500:
            if feature_row['Self.employed'] == 1:
                prediction = 1
            else:
                prediction = 0
        else:
            prediction = 0

    return prediction, emb


@dataclass
class MistralCreditscoreEmbedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
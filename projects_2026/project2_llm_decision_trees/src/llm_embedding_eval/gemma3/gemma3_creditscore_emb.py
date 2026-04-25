from dataclasses import dataclass


def dt_func_0(row):
    """
    Decision tree classifier for credit application approval.

    Args:
        row (dict): A dictionary containing the feature values.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """
    node1 = int(row['Age'] > 30)
    node2 = int(row['Income.per.dependent'] > 4)
    node3 = int(row['Monthly.credit.card.exp'] > 1000)
    node4 = int(row['Own.home'] == 1)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Derogatory.reports'] > 1)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1 == 0:
        if node2 == 0:
            if node3 == 0:
                if node4 == 1:
                    return 1, emb
                else:
                    return 0, emb
            else:
                if node5 == 1:
                    return 1, emb
                else:
                    return 0, emb
        else:
            if node6 == 0:
                return 1, emb
            else:
                return 0, emb
    else:
        if node3 == 0:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb


def dt_func_1(row):
    """
    Predicts whether a credit application is accepted based on the provided features.

    Args:
        row (dict): A dictionary containing the feature values for a single application.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """

    node1 = int(row['Age'] > 30)
    node2 = int(row['Income.per.dependent'] > 5)
    node3 = int(row['Monthly.credit.card.exp'] > 1000)
    node4 = int(row['Own.home'] == 1)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Derogatory.reports'] > 1)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1 == 0:
        if node2 == 1:
            if node3 == 0:
                return 1, emb
            else:
                return 0, emb
        else:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
    else:
        if node5 == 1:
            if node6 == 0:
                return 1, emb
            else:
                return 0, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
            
            

def dt_func_2(row):
    """
    Predicts whether a credit application is accepted based on the given features.

    Args:
        row (dict): A dictionary containing the feature values for a single application.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """

    node1 = int(row['Age'] > 35)
    node2 = int(row['Income.per.dependent'] > 5)
    node3 = int(row['Monthly.credit.card.exp'] > 1000)
    node4 = int(row['Own.home'] == 1)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Derogatory.reports'] > 1)


    emb = [node1, node2, node3, node4, node5, node6]

    if node1 == 0:
        if node2 == 1:
            if node3 == 0:
                return 1, emb
            else:
                return 0, emb
        else:
            if node4 == 1:
                return 1, emb
            else:
                if node5 == 1:
                    return 1, emb
                else:
                    if node6 == 1:
                        return 0, emb
                    else:
                        return 0, emb
    else:
        if node3 == 0:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb


def dt_func_3(row):
    """
    Classifies whether a credit application is accepted based on the provided features.

    Args:
        row (dict): A dictionary containing the feature values for a single application.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """

    # Precompute binary vector embeddings
    node1 = int(row['Age'] > 30)
    node2 = int(row['Income.per.dependent'] > 5)
    node3 = int(row['Monthly.credit.card.exp'] > 1000)
    node4 = int(row['Own.home'] == 1)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Derogatory.reports'] > 1)
    emb = [node1, node2, node3, node4, node5, node6]

    # Decision Tree Logic
    if row['Age'] > 40:
        return 1, emb
    elif row['Income.per.dependent'] > 8:
        return 1, emb
    elif row['Derogatory.reports'] == 0:
        if row['Own.home'] == 1:
            return 1, emb
        else:
            return 0, emb
    elif row['Self.employed'] == 1:
        if row['Monthly.credit.card.exp'] < 1500:
            return 1, emb
        else:
            return 0, emb
    else:
        return 0, emb


def dt_func_4(row):
    """
    Classifies whether a credit application is accepted based on the provided features.

    Args:
        row (dict): A dictionary containing the feature values for a single application.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary node embeddings.
    """

    node1 = int(row['Age'] > 30)
    node2 = int(row['Income.per.dependent'] > 5)
    node3 = int(row['Monthly.credit.card.exp'] > 500)
    node4 = int(row['Own.home'] == 1)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Derogatory.reports'] > 1)

    emb = [node1, node2, node3, node4, node5, node6]

    if node1 == 0:
        if node2 == 0:
            if node3 == 0:
                return 0, emb
            else:
                return 0, emb
        else:
            if node4 == 0:
                if node5 == 0:
                    return 0, emb
                else:
                    return 1, emb
            else:
                return 1, emb
    else:
        if node6 == 0:
            if node3 == 0:
                return 1, emb
            else:
                return 1, emb
        else:
            return 0, emb

@dataclass
class Gemma3CreditscoreEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
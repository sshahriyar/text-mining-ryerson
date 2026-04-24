from dataclasses import dataclass

def dt_func_0(row):
    """
    Predicts the winner of a boxing round (Trinidad/de la Hoya) based on Judge, Official, and Round.

    Args:
        row (dict): A dictionary containing the values for 'Judge', 'Official', and 'Round'.

    Returns:
        tuple: A tuple containing the prediction (0 for Trinidad, 1 for de la Hoya) and a list of binary node embeddings.
    """

    node1 = int(row['Judge'] <= 2)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)

    emb = [node1, node2, node3]

    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
        else:
            return 0, emb
    else:
        if node2 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 1, emb
        else:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb


def dt_func_1(row):
    """
    Predicts the winner of a boxing round (Trinidad vs. de la Hoya) 
    based on the judge, official status, and round number.

    Args:
        row (dict): A dictionary containing the feature values.

    Returns:
        tuple: A tuple containing the predicted winner (0 or 1) 
               and a list representing the truth values of the inner nodes.
    """

    # Precompute binary vector embeddings
    node1 = int(row['Judge'] <= 3)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)

    emb = [node1, node2, node3]

    # Decision Tree Logic
    if node1 == 1:
        if node3 == 1:
            return 0, emb  # Trinidad
        else:
            return 1, emb  # de la Hoya
    else:
        if node2 == 1:
            return 1, emb  # de la Hoya
        else:
            if node3 == 1:
                return 0, emb  # Trinidad
            else:
                return 1, emb  # de la Hoya


def dt_func_2(row):
    """
    Predicts the winner of a boxing round based on Judge, Official, and Round.

    Args:
        row (dict): A dictionary containing the values for Judge, Official, and Round.

    Returns:
        tuple: A tuple containing the predicted winner (0 = Trinidad, 1 = de la Hoya) and a list of binary node embeddings.
    """

    node1 = int(row['Judge'] < 3)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)

    emb = [node1, node2, node3]

    if node1 == 1:
        if node3 == 1:
            return 0, emb
        else:
            return 1, emb
    else:
        if node2 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
            

def dt_func_3(row):
    """
    Predicts the winner of a boxing round (Trinidad/de la Hoya) based on Judge, Official, and Round.

    Args:
        row (dict): A dictionary containing the values for Judge, Official, and Round.

    Returns:
        tuple: A tuple containing the prediction (0 for Trinidad, 1 for de la Hoya) 
               and a list representing the truth values of the inner nodes.
    """
    node1 = int(row['Judge'] <= 4)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]

    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
        else:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
    else:
        if node2 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb


def dt_func_4(row):
    """
    Predicts the winner of a boxing round based on Judge, Official, and Round.

    Args:
        row (dict): A dictionary containing the values for Judge, Official, and Round.

    Returns:
        tuple: A tuple containing the predicted winner (0 = Trinidad, 1 = de la Hoya)
               and a list of binary node embeddings.
    """

    node1 = int(row['Judge'] <= 2)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]

    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
        else:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
    else:
        if node2 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb

@dataclass
class Gemma3Boxing2Embedding:
    runner  = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
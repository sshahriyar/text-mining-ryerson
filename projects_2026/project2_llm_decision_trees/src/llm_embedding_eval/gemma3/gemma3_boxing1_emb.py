from dataclasses import dataclass

def dt_func_0(row):
    """
    Predicts the winner of a boxing round (Lewis or Holyfield) based on Judge, Official judge, and Round.

    Args:
        row (dict): A dictionary containing the feature values for a single round.
            Keys are feature names ('Judge', 'Official', 'Round').

    Returns:
        tuple: A tuple containing the prediction (1 for Lewis, 0 for Holyfield)
            and a list of binary embeddings representing the truth values of the inner nodes.
    """

    # Precompute binary vector embeddings
    node1 = int(row['Judge'] == 0)  # E. Williams
    node2 = int(row['Judge'] == 1)  # L. O'Connell
    node3 = int(row['Official'] == 1) # Official Judge
    node4 = int(row['Round'] <= 6)

    emb = [node1, node2, node3, node4]

    # Decision Tree Logic
    if node1:  # Judge is E. Williams
        if node3: # Official Judge
            return 1, emb  # Lewis
        else:
            return 0, emb  # Holyfield
    elif node2:  # Judge is L. O'Connell
        if node4:  # Round <= 6
            return 0, emb # Holyfield
        else:
            return 1, emb #Lewis
    elif node3: # Official judge
        if node4:
            return 1, emb #Lewis
        else:
            return 0, emb #Holyfield
    else: #Other Judges
        if node4:
            return 0, emb  # Holyfield
        else:
            return 1, emb  # Lewis
        
def dt_func_1(row):
    """
    Predicts the winner of a boxing round (Lewis or Holyfield) based on Judge, Official judge, and Round.

    Args:
        row (dict): A dictionary containing the values for Judge, Official, and Round.

    Returns:
        tuple: A tuple containing the predicted winner (1 for Lewis, 0 for Holyfield) and a list of binary node embeddings.
    """

    # Precompute binary node embeddings
    node1 = int(row['Judge'] < 3)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)


    emb = [node1, node2, node3]

    # Decision tree logic
    if node1 == 1:
        if node3 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node2 == 1:
            return 1, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 0, emb
            
def dt_func_2(row):
    """
    Predicts the winner of a boxing round (Lewis vs. Holyfield) based on Judge, Official, and Round.

    Args:
        row (dict): A dictionary containing the values for Judge, Official, and Round.

    Returns:
        tuple: A tuple containing the prediction (1 for Lewis, 0 for Holyfield) and a list of binary node embeddings.
    """

    # Precompute binary node embeddings
    node1 = int(row['Judge'] <= 2)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]

    # Decision tree logic
    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                return 1, emb  # Lewis
            else:
                return 0, emb  # Holyfield
        else:
            return 0, emb  # Holyfield
    else:
        if node2 == 1:
            if node3 == 1:
                return 0, emb  # Holyfield
            else:
                return 1, emb  # Lewis
        else:
            if node3 == 1:
                return 1, emb  # Lewis
            else:
                return 0, emb  # Holyfield
            
def dt_func_3(row):
    """
    Predicts the winner of a boxing round (Lewis vs. Holyfield) based on judge,
    official judge, and round number.

    Args:
        row (dict): A dictionary containing the values for 'Judge', 'Official', and 'Round'.

    Returns:
        tuple: A tuple containing the predicted winner (1 = Lewis, 0 = Holyfield) and
               a list representing the binary vector embeddings of the inner nodes.
    """

    node1 = int(row['Judge'] < 3)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]

    if node1 == 1:
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
            if row['Round'] <= 4:
                return 0, emb
            else:
                return 1, emb

def dt_func_4(row):
    """
    Predicts the winner of a boxing round (Lewis or Holyfield) based on Judge, Official judge, and Round.

    Args:
        row (dict): A dictionary containing the values for 'Judge', 'Official', and 'Round'.

    Returns:
        tuple: A tuple containing the prediction (0 = Holyfield, 1 = Lewis) and a list representing the truth values of the inner nodes.
    """
    node1 = int(row['Judge'] < 3)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]

    if node1 == 1:
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
            if row['Round'] <= 4:
                return 0, emb
            else:
                return 1, emb

@dataclass
class Gemma3Boxing1Embedding:
    runner =  [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
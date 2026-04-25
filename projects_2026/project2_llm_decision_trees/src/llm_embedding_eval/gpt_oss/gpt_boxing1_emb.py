from dataclasses import dataclass


def dt_func_0(row):
    # Precompute binary vector embeddings
    node1 = int(row['Official'] == 1)                # Official judge
    node2 = int(row['Judge'] == 7 or row['Judge'] == 8)  # Judge is Associated Press or ESPN
    node3 = int(row['Round'] <= 6)                   # Early rounds

    emb = [node1, node2, node3]

    # Decision tree logic
    if node1:  # Official judge
        if node2:  # AP or ESPN
            pred = 1  # Lewis
        else:
            pred = 0  # Holyfield
    else:  # Not official judge
        if node3:  # Early round
            pred = 1  # Lewis
        else:
            pred = 0  # Holyfield

    return pred, emb


def dt_func_1(row):
    # Binary node evaluations
    node1 = int(row['Judge'] in ['Associated Press', 'ESPN', 'Boxing Monthly-Leach'])
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] >= 10)

    emb = [node1, node2, node3]

    # Decision tree logic
    if node1 == 1:
        return 1, emb          # Lewis wins
    elif node2 == 1:
        return 1, emb          # Lewis wins
    elif node3 == 1:
        return 0, emb          # Holyfield wins
    else:
        return 1, emb          # Lewis wins


def dt_func_2(row):
    # Precompute binary embeddings for the inner nodes
    node1 = int(row['Official'] == 1)           # Is the judge official?
    node2 = int(row['Judge'] == 0)             # Judge is E. Williams
    node3 = int(row['Round'] <= 6)             # Early rounds (1‑6)
    
    emb = [node1, node2, node3]
    
    # Decision tree logic
    if node1:
        # Official judges tend to favor Lewis
        return 1, emb
    if node2:
        # E. Williams often awards the round to Lewis
        return 1, emb
    if node3:
        # Early rounds may favor Lewis
        return 1, emb
    # Remaining cases favor Holyfield
    return 0, emb


def dt_func_3(row):
    """
    Decision tree classifier for boxing round winner.

    Parameters
    ----------
    row : dict
        Dictionary containing feature values with keys:
            - 'Judge'          : str  (e.g., 'ESPN', 'Associated Press', ...)
            - 'Official'       : int  (1 = Yes, 0 = No)
            - 'Round'          : int  (1 to 12)

    Returns
    -------
    tuple
        (prediction, embedding_vector)
        prediction : int (1 = Lewis, 0 = Holyfield)
        embedding_vector : list of ints (0 or 1) representing inner node evaluations
    """
    # Inner node evaluations
    node1 = int(row['Official'] == 1)                                   # Official judge?
    node2 = int(row['Round'] <= 6)                                      # Early rounds (<=6)?
    node3 = int(row['Judge'] in ['ESPN', 'Associated Press', 'Sportsline'])  # Judge favoring Lewis?

    emb = [node1, node2, node3]

    # Decision tree logic
    if node1 == 1:          # Official judge present
        if node2 == 1:      # Early round
            prediction = 0  # Holyfield
        else:              # Late round
            prediction = 1  # Lewis
    else:                  # No official judge
        if node3 == 1:      # Judge from Lewis-favoring outlets
            prediction = 1  # Lewis
        else:              # Other judges
            prediction = 0  # Holyfield

    return prediction, emb


def dt_func_4(row):
    """
    Decision tree classifier for boxing round winner.
    Features: Judge (string), Official (int, 1=yes 0=no), Round (int 1-12)
    Target: 1 = Lewis wins, 0 = Holyfield wins
    """
    # Precompute binary node evaluations
    node1 = int(row['Official'] == 1)                         # Is the judge official?
    node2 = int(row['Round'] <= 6)                            # Is the round in the first half?
    node3 = int(row['Judge'] in ['ESPN', 'Associated Press'])  # Judge is ESPN or AP?
    node4 = int(row['Judge'] == 'ESPN')                       # Judge is ESPN?

    emb = [node1, node2, node3, node4]

    # Decision tree logic
    if node1 == 1:
        pred = 1  # Official judges tend to favor Lewis
    else:
        if node2 == 1:
            # Early rounds: ESPN/Associated Press favor Lewis
            pred = 1 if node3 == 1 else 0
        else:
            # Late rounds: only ESPN favors Lewis
            pred = 1 if node4 == 1 else 0

    return pred, emb

@dataclass
class GptBoxing1Embedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]

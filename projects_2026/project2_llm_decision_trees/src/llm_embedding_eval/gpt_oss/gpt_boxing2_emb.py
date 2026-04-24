from dataclasses import dataclass



def dt_func_0(row):
    # Compute binary embeddings for the inner nodes
    node1 = int(row['Official'] == 1)                     # Official judge
    node2 = int(row['Round'] <= 6)                        # Early rounds
    node3 = int(row['Judge'] in ['J. Roth', 'USA Today', 'Associated Press'])
    emb = [node1, node2, node3]

    # Decision tree logic
    if node1:  # Official judge
        if node3:
            return 1, emb  # de la Hoya wins
        else:
            return 0, emb  # Trinidad wins
    else:      # Non‑official judge
        if node2:
            return 0, emb  # Trinidad wins
        else:
            return 1, emb  # de la Hoya wins
        

def dt_func_1(row):
    # Precompute binary embeddings for the inner nodes
    node1 = int(row['Judge'] == 'Associated Press')          # Judge is Associated Press ?
    node2 = int(row['Round'] <= 6)                          # Round number <= 6 ?
    node3 = int(row['Official'] == 1)                       # Official judge == Yes ?
    
    emb = [node1, node2, node3]
    
    # Decision tree logic
    if node1:                     # If Judge is Associated Press
        if node2:                 # Round <= 6
            prediction = 1       # de la Hoya wins
        else:                     # Round > 6
            prediction = 0       # Trinidad wins
    else:                         # Judge is not Associated Press
        if node3:                 # Official judge is Yes
            prediction = 1       # de la Hoya wins
        else:                     # Official judge is No
            prediction = 0       # Trinidad wins
    
    return prediction, emb


def dt_func_2(row):
    # Binary feature nodes
    node1 = int(row['Official'] == 1)          # Official judge is yes
    node2 = int(row['Round'] <= 6)                  # Early rounds
    node3 = int(row['Judge'] in ['G. Hamada', 'B. Logist', 'J. Roth'])
    
    # Embedding vector
    emb = [node1, node2, node3]
    
    # Decision tree logic
    if node1:           # Official judge
        if node2:       # Early round
            prediction = 1   # de la Hoya wins
        else:           # Late round
            prediction = 0   # Trinidad wins
    else:               # Not an official judge
        if node3:       # Judge is one of the three named
            prediction = 0   # Trinidad wins
        else:           # Other judge
            prediction = 1   # de la Hoya wins
    
    return prediction, emb

def dt_func_3(row):
    # Precompute binary embeddings for inner nodes
    node1 = int(row['Official'] == 1)                        # Official judge? 1=yes, 0=no
    node2 = int(row['Judge'] in [0, 2, 5, 9])                # Judge in preferred list
    node3 = int(row['Round'] > 6)                            # Round > 6 ?
    emb = [node1, node2, node3]

    # Decision tree logic
    if node1:   # Official judge
        if node2:
            pred = 1  # de la Hoya
        else:
            pred = 0  # Trinidad
    else:       # Not an official judge
        if node3:
            pred = 1  # de la Hoya
        else:
            pred = 0  # Trinidad

    return pred, emb


def dt_func_4(row):
    # Precompute binary node conditions
    node0 = int(row['Official'] == 1)  # Official judge?
    node1 = int(row['Judge'] in ['USA Today', 'Associated Press'])
    node2 = int(row['Judge'] in ['G. Hamada', 'B. Logist', 'J. Roth'])
    node3 = int(row['Judge'] in ['HBO-Lederman', 'Sportsticker', 'Los Angeles Times-Kawakami'])
    node4 = int(row['Round'] <= 6)
    node5 = int(row['Judge'] in ['Los Angeles Times-Springer', 'Las Vegas Review-Journal', 'van de Wiele'])
    
    emb = [node0, node1, node2, node3, node4, node5]
    
    # Decision tree logic
    if node0:  # Official judge
        if node1:           # USA Today or Associated Press
            pred = 0
        elif node2:         # G. Hamada, B. Logist, J. Roth
            pred = 1
        elif node3:         # HBO-Lederman, Sportsticker, Los Angeles Times-Kawakami
            pred = 1
        else:
            pred = 0
    else:  # Non‑official judge
        if node4:  # Round 1-6
            if node5:  # Los Angeles Times-Springer, Las Vegas Review‑Journal, van de Wiele
                pred = 0
            else:
                pred = 1
        else:  # Round 7-12
            pred = 1
    
    return pred, emb


@dataclass
class GptBoxing2Embedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
from dataclasses import dataclass


def dt_func_0(row):
    # Binary evaluation of inner nodes
    node1 = int(row['rectal_temperature'] > 39.5)      # high rectal temperature
    node2 = int(row['pulse'] > 120)                   # tachycardia
    node3 = int(row['capillary_refill_time'] == 0)    # > 3 sec refill time
    node4 = int(row['abdominal_distension'] == 3)     # severe distension
    
    emb = [node1, node2, node3, node4]
    
    # Decision tree logic
    if node1:                          # High temperature
        if node2:                      # Tachycardia
            return 1, emb              # Surgery likely
        else:                          # Normal pulse
            if node3:                  # Slow refill
                return 1, emb          # Surgery likely
            else:
                return 0, emb          # No surgery
    else:                              # Normal temperature
        if node4:                      # Severe distension
            return 1, emb              # Surgery likely
        else:
            return 0, emb              # No surgery

def dt_func_1(row):
    """
    Predict whether a horse requires surgical intervention based on key clinical signs.

    Parameters
    ----------
    row : dict
        Dictionary containing the horse's clinical measurements. Expected keys are:
        - 'rectal_temperature' (float, in °C)
        - 'pulse' (int, beats per minute)
        - 'capillary_refill_time' (int, encoded as 0=>3 sec, 1=<3 sec, 2=missing, 3=unknown)
        - 'peripheral_pulse' (int, 1=strong, 2=moderate, 3=weak, 4=reduced)
        - 'temp_extremities' (int, 1=cold, 2=normal, 3=warm, 4=hot)

    Returns
    -------
    pred : int
        1 if surgery is recommended, 0 otherwise.
    emb : list of int
        Binary embedding of internal node activations (in order of computation).
    """
    # Internal node evaluations
    node1 = int(row['rectal_temperature'] > 39.0)          # High fever
    node2 = int(row['pulse'] > 120)                       # Tachycardia
    node3 = int(row['capillary_refill_time'] == 0)        # Prolonged refill (>3 sec)
    node4 = int(row['peripheral_pulse'] == 4)             # Reduced peripheral pulse
    node5 = int(row['temp_extremities'] == 4)             # Warm extremities

    emb = [node1, node2, node3, node4, node5]

    # Decision logic
    if node1:  # High rectal temperature
        if node2:  # Tachycardia
            if node3:  # Prolonged capillary refill
                pred = 1  # Surgery recommended
            else:
                pred = 0  # No surgery
        else:
            pred = 0  # No surgery
    else:  # Normal temperature
        if node4 or node5:  # Shock indicators
            pred = 1  # Surgery recommended
        else:
            pred = 0  # No surgery

    return pred, emb


def dt_func_2(row):
    # pre‑compute binary node conditions
    node1 = int(row['rectal_temperature'] > 38.5)                     # high temperature
    node2 = int(row['pulse'] > 140)                                  # tachycardia
    node3 = int(row['peripheral_pulse'] <= 1 or row['peripheral_pulse'] >= 4)  # absent or reduced pulse
    node4 = int(row['mucous_membranes'] >= 5)                         # bright red or pale cyanotic membranes
    node5 = int(row['capillary_refill_time'] == 0)                    # refill >3 s
    node6 = int(row['pain'] >= 5)                                     # severe pain
    node7 = int(row['peristalsis'] <= 1 or row['peristalsis'] >= 3)   # absent or hypomotile peristalsis

    emb = [node1, node2, node3, node4, node5, node6, node7]

    # decision tree logic
    if node1:                        # rectal_temperature > 38.5
        if node2:                    # pulse > 140
            pred = 1                 # surgery
        else:
            if node3:                # abnormal peripheral pulse
                pred = 1             # surgery
            else:
                if node4:            # severe mucous membranes
                    pred = 1         # surgery
                else:
                    pred = 0         # no surgery
    else:                            # rectal_temperature ≤ 38.5
        if node5:                    # capillary refill > 3 s
            pred = 1                 # surgery
        else:
            if node6:                # severe pain
                pred = 1             # surgery
            else:
                if node7:            # abnormal peristalsis
                    pred = 1         # surgery
                else:
                    pred = 0         # no surgery

    return pred, emb


def dt_func_3(row):
    """
    Predict whether a horse colic lesion is surgical (1) or not (0).
    
    Parameters
    ----------
    row : dict
        Dictionary containing feature values. Keys must match the feature names:
        'rectal_temperature', 'pulse', 'peristalsis', 'abdomen'.
    
    Returns
    -------
    tuple
        (prediction, embedding)
        prediction : int (0 or 1)
        embedding : list of ints (1 if the corresponding node condition is true, 0 otherwise)
    """
    # Precompute binary node embeddings
    node1 = int(row['rectal_temperature'] > 39.5)          # High rectal temperature
    node2 = int(row['pulse'] > 140)                       # Tachycardia
    node3 = int(row['peristalsis'] == 1)                  # Absent peristalsis
    node4 = int(row['abdomen'] == 1)                      # Distended large abdomen
    
    emb = [node1, node2, node3, node4]
    
    # Decision tree logic
    if node1:
        pred = 1  # Surgery likely with high rectal temperature
    elif node2:
        pred = 1  # Surgery likely with tachycardia
    elif node3:
        pred = 1  # Surgery likely with absent peristalsis
    elif node4:
        pred = 1  # Surgery likely with distended large abdomen
    else:
        pred = 0  # Not surgery
    
    return pred, emb


def dt_func_4(row):
    # Precompute binary decisions for each internal node
    node1 = int(row['rectal_temperature'] > 39.5)           # high temperature
    node2 = int(row['pulse'] > 80)                          # high pulse
    node3 = int(row['peripheral_pulse'] == 2)               # increased peripheral pulse
    node4 = int(row['capillary_refill_time'] == 0)          # refill time > 3 s
    node5 = int(row['pain'] >= 3)                           # depressed or more severe pain

    emb = [node1, node2, node3, node4, node5]

    # Decision tree logic
    if node1:
        if node2:
            pred = 1
        else:
            if node3:
                pred = 1
            else:
                pred = 0
    else:
        if node4:
            pred = 0
        else:
            if node5:
                pred = 1
            else:
                pred = 0

    return pred, emb

@dataclass
class GptColicEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
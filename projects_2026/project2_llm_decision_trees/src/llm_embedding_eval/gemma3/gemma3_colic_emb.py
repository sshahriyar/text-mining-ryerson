from dataclasses import dataclass

def dt_func_0(row):
    """
    Predicts whether a horse colic lesion requires surgery based on the provided features.

    Args:
        row (dict): A dictionary containing the feature values for a single horse.

    Returns:
        tuple: A tuple containing the predicted class (0 or 1) and a list of binary embeddings representing the decision path.
    """

    # Binary Embeddings
    node1 = int(row['Age'] == 'young')
    node2 = int(row['pulse'] > 40)
    node3 = int(row['respiratory_rate'] > 20)
    node4 = int(row['temp_extremities'] == 'cold')
    node5 = int(row['peripheral_pulse'] == 'absent')
    node6 = int(row['capillary_refill_time'] > 1)
    node7 = int(row['pain'] > 2)
    node8 = int(row['peristalsis'] == 'absent')
    node9 = int(row['abdominal_distension'] == 'severe')
    node10 = int(row['nasogastric_tube'] == 'significant')
    node11 = int(row['nasogastric_reflux'] == 'none')
    node12 = int(row['rectal_examination'] == 'decreased')
    node13 = int(row['abdomen'] == 'distended large')
    node14 = int(row['packed_cell_volume'] > 30)
    node15 = int(row['abdominocentesis_appearance'] == 'cloudy')
    node16 = int(row['abdomcentesis_total_protein'] > 2)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14, node15, node16]

    if node1 == 0:  # Age is adult
        if node2 == 0:  # pulse <= 40
            if node3 == 0:  # respiratory_rate <= 20
                if node4 == 0:  # extremities not cold
                    return 0, emb
                else:
                    return 1, emb
            else:
                if node5 == 0:  # peripheral_pulse not absent
                    return 0, emb
                else:
                    return 1, emb
        else:
            if node6 == 0: #capillary refill time <= 1
                if node7 == 0: # pain <= 2
                    return 0, emb
                else:
                   return 1, emb
            else:
                return 1, emb
    else: # Age is young
        if node8 == 1: #peristalsis absent
            if node9 == 1: #abdominal distension severe
                return 1, emb
            else:
                if node10 == 1: #nasogastric tube is significant
                    return 1, emb
                else:
                    return 0, emb

        else:
            if node11 == 1: #nasogastric reflux is none
                if node12 == 1: #rectal examination decreased
                    if node13 == 1: #abdomen is distended large
                        return 1, emb
                    else:
                        return 0, emb
                else:
                    return 0, emb
            else:
                if node14 == 1: #packed cell volume > 30
                    if node15 == 1: #abdominocentesis appearance is cloudy
                        return 1, emb
                    else:
                        return 0, emb
                else:
                    if node16 == 1: #abdominocentesis total protein > 2
                        return 1, emb
                    else:
                        return 0, emb


def dt_func_1(row):
    node1 = int(row['rectal_temperature'] > 38.5)
    node2 = int(row['pulse'] > 60)
    node3 = int(row['respiratory_rate'] > 30)
    node4 = int(row['peripheral_pulse'] > 3)
    node5 = int(row['capillary_refill_time'] > 1)
    node6 = int(row['pain'] > 2)
    node7 = int(row['peristalsis'] > 2)
    node8 = int(row['abdominal_distension'] > 1)
    node9 = int(row['nasogastric_tube'] > 1)
    node10 = int(row['packed_cell_volume'] > 30)
    node11 = int(row['abdominocentesis_appearance'] > 1)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11]
    
    if node1 == 0 and node2 == 0:
        return 0, emb
    elif node3 == 1 and node4 == 0:
        return 1, emb
    elif node5 == 1 and node6 == 1:
        return 1, emb
    elif node7 == 0 and node8 == 1:
        return 1, emb
    elif node9 == 1 and node10 == 0:
        return 1, emb
    elif node11 == 1:
        return 1, emb
    else:
        return 0, emb


def dt_func_2(row):
    node1 = int(row['Age'] == 0)
    node2 = int(row['rectal_temperature'] > 38.5)
    node3 = int(row['pulse'] > 60)
    node4 = int(row['respiratory_rate'] > 30)
    node5 = int(row['peripheral_pulse'] > 3)
    node6 = int(row['capillary_refill_time'] > 1)
    node7 = int(row['pain'] > 2)
    node8 = int(row['peristalsis'] == 2)
    node9 = int(row['abdominal_distension'] > 2)
    node10 = int(row['nasogastric_tube'] > 1)
    node11 = int(row['nasogastric_reflux_PH'] < 6)
    node12 = int(row['packed_cell_volume'] > 35)
    node13 = int(row['abdominocentesis_appearance'] == 2)
    node14 = int(row['abdomcentesis_total_protein'] > 3)
    
    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14]
    
    if node1 == 0:
        if node2 == 0:
            if node3 == 0:
                if node4 == 0:
                    if node5 > 0:
                        return 0, emb
                    else:
                        return 1, emb
                else:
                    return 1, emb
            else:
                return 1, emb
        else:
            if node6 > 0:
                return 1, emb
            else:
                return 0, emb
    else:
        if node7 > 0:
            if node8 > 0:
                return 1, emb
            else:
                if node9 > 0:
                    return 1, emb
                else:
                    if node10 > 0:
                        return 1, emb
                    else:
                        if node11 > 0:
                            return 1, emb
                        else:
                            if node12 > 0:
                                return 1, emb
                            else:
                                if node13 > 0:
                                    return 1, emb
                                else:
                                    if node14 > 0:
                                        return 1, emb
                                    else:
                                        return 0, emb
        else:
            return 0, emb
        

def dt_func_3(row):
    """
    Predicts whether a horse colic lesion is surgical or not based on a decision tree.

    Args:
        row (dict): A dictionary containing the feature values for a single horse.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and a list of binary embeddings representing the inner node evaluations.
    """

    # Binary Embeddings
    node1 = int(row['Age'] == 1)
    node2 = int(row['pulse'] > 40)
    node3 = int(row['respiratory_rate'] > 30)
    node4 = int(row['temp_extremities'] == 1)
    node5 = int(row['peripheral_pulse'] == 1)
    node6 = int(row['capillary_refill_time'] > 0)
    node7 = int(row['pain'] > 2)
    node8 = int(row['peristalsis'] == 1)
    node9 = int(row['abdominal_distension'] == 3)
    node10 = int(row['nasogastric_tube'] == 2)
    node11 = int(row['nasogastric_reflux'] == 0)
    node12 = int(row['rectal_examination'] == 1)
    node13 = int(row['abdomen'] == 1)
    node14 = int(row['packed_cell_volume'] > 30)
    node15 = int(row['abdominocentesis_appearance'] == 2)
    

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14, node15]

    if node1 == 0:  # Age is adult
        if node2 == 1:  # Pulse > 40
            if node3 == 1:  # Respiratory Rate > 30
                prediction = 1
            else:
                prediction = 0
        else:
            if node4 == 1: # Extremities are cold
                prediction = 1
            else:
                prediction = 0
    else: # Age is young
        if node5 == 1:  # Peripheral Pulse is absent
            prediction = 1
        else:
            if node6 == 1: # Capillary Refill time > 0
                if node7 == 1: # pain is severe
                   prediction = 1
                else:
                   prediction = 0
            else:
                if node8 == 1: # Peristalsis is absent
                    prediction = 1
                else:
                    if node9 == 1: # Abdominal Distension is severe
                        prediction = 1
                    else:
                        if node10 == 1: # Nasogastric tube is significant
                            if node11 == 1: # Nasogastric reflux is none
                                prediction = 1
                            else:
                                prediction = 0
                        else:
                            if node12 == 1:  # Rectal Examination is absent
                                prediction = 1
                            else:
                                if node13 == 1: # Abdomen is distended
                                    prediction = 1
                                else:
                                    if node14 == 1: # Packed cell volume > 30
                                        prediction = 1
                                    else:
                                        prediction = 0

    return prediction, emb


def dt_func_4(row):
    node1 = int(row['Age'] == 0)
    node2 = int(row['rectal_temperature'] > 38.5)
    node3 = int(row['pulse'] > 60)
    node4 = int(row['respiratory_rate'] > 30)
    node5 = int(row['peripheral_pulse'] == 1)
    node6 = int(row['capillary_refill_time'] > 0)
    node7 = int(row['pain'] > 2)
    node8 = int(row['peristalsis'] == 1)
    node9 = int(row['abdominal_distension'] == 3)
    node10 = int(row['nasogastric_reflux'] == 0)
    node11 = int(row['rectal_examination'] == 1)
    node12 = int(row['abdomen'] == 1)
    node13 = int(row['packed_cell_volume'] > 30)
    node14 = int(row['abdominocentesis_appearance'] == 2)
    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14]

    if node1 == 0:
        if node2 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node4 == 1:
                if node5 == 1:
                    return 1, emb
                else:
                    return 0, emb
            else:
                if node6 > 0:
                    if node7 == 1:
                        return 1, emb
                    else:
                        return 0, emb
                else:
                    if node8 == 1:
                        return 1, emb
                    else:
                        return 0, emb
    else:
        if node9 == 1:
            if node10 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node11 == 1:
                if node12 == 1:
                    return 1, emb
                else:
                    return 0, emb
            else:
                if node13 == 1:
                    if node14 == 1:
                        return 1, emb
                    else:
                        return 0, emb
                else:
                    return 0, emb
@dataclass
class Gemma3ColicEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
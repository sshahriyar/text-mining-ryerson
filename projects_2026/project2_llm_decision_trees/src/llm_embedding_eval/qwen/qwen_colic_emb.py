from dataclasses import dataclass

def dt0(row):
    node1 = int(row['rectal_temperature'] <= 38.5)
    node2 = int(row['pulse'] <= 60)
    node3 = int(row['respiratory_rate'] <= 20)
    node4 = int(row['packed_cell_volume'] <= 40)
    node5 = int(row['total_protein'] <= 6.5)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
        else:
            if node4 == 1:
                return 0, emb
            else:
                return 1, emb
    else:
        if node5 == 1:
            return 1, emb
        else:
            return 0, emb
        

def dt1(row):
    """
    Predicts whether a horse requires surgery for colic.
    
    Parameters:
    - row: A dictionary-like object containing feature values.
    
    Returns:
    - 1 if surgery is likely; 0 if not.
    """
    node1 = int(row['abdominal_distension'] == 3)
    node2 = int(row['peristalsis'] == 1)
    node3 = int(row['abdominocentesis_appearance'] in [2, 3])
    node4 = int(row['rectal_temperature'] < 37)
    node5 = int(row['pulse'] > 60)
    emb = [node1, node2, node3, node4, node5]
    # First, check if the horse has severe abdominal distension
    if row['abdominal_distension'] == 3:
        return 1, emb   # Severe distension is a strong indicator for surgery
    
    # Next, check for absence of peristalsis
    elif row['peristalsis'] == 1:
        return 1, emb  # Absence of peristalsis is indicative of intestinal obstruction
    
    # Then, check for abnormal results from abdominocentesis
    elif row['abdominocentesis_appearance'] in [2, 3]:
        return 1, emb  # Cloudy or serosanguinous fluid may indicate peritonitis
    
    # Check for abnormal rectal temperature
    elif row['rectal_temperature'] < 37:
        return 1, emb  # Hypothermia is a sign of severe colic or peritonitis
    
    # Check for high pulse rate
    elif row['pulse'] > 60:
        return 1, emb  # Tachycardia can be associated with severe illness
    
    # If none of the above, predict 'no surgery'
    else:
        return 0, emb


def dt2(row):
    # Precompute binary embeddings
    node1 = int(row['capillary_refill_time'] == 0)  # Capillary Refill Time > 3 seconds?
    node2 = int(row['peripheral_pulse'] == 1)  # Peripheral Pulse is absent?
    node3 = int(row['mucous_membranes'] in [3, 5])  # Mucous Membranes is dark or pale cyanotic?
    node4 = int(row['packed_cell_volume'] < 25)  # Packed Cell Volume < 25%?
    node5 = int(row['abdominal_distension'] == 3)  # Abdominal Distension is severe?
    node6 = int(row['pulse'] > 60)  # Pulse > 60 bpm?
    node7 = int(row['respiratory_rate'] > 20)  # Respiratory Rate > 20?
    node8 = int(row['rectal_temperature'] < 37)  # Rectal Temperature < 37°C?
    emb = [node1, node2, node3, node4, node5, node6, node7, node8]
    
    # Decision logic
    if node1:
        if node2:
            prediction = 1
        else:
            if node3:
                prediction = 1
            else:
                if node4:
                    prediction = 1
                else:
                    if node5:
                        prediction = 1
                    else:
                        prediction = 0
    else:
        if node6:
            if node7:
                if node8:
                    prediction = 0
                else:
                    prediction = 1 if node5 else 0
            else:
                prediction = 0
        else:
            prediction = 0
    return prediction, emb

def dt3(row):
    node1 = int(row['rectal_temperature'] <= 37.5)
    node2 = int(row['pulse'] > 60)
    node3 = int(row['respiratory_rate'] > 20)
    node4 = int(row['mucous_membranes'] in [3,5])
    node5 = int(row['capillary_refill_time'] >= 3)
    node6 = int(row['abdominal_distension'] == 3)
    node7 = int(row['peristalsis'] == 1)
    node8 = int(row['packed_cell_volume'] < 30)
    node9 = int(row['total_protein'] < 5.5)
    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9]
    
    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                if node4 == 1:
                    return 1, emb
                else:
                    if node5 == 1:
                        return 1, emb
                    else:
                        return 0, emb
            else:
                return 0, emb
        else:
            return 0, emb
    else:
        if node6 == 1:
            return 1, emb
        else:
            if node7 == 1:
                return 1, emb
            else:
                if node8 == 1:
                    return 1, emb
                else:
                    if node9 == 1:
                        return 1, emb
                    else:
                        return 0, emb

def dt4(row):
    node1 = int(row['capillary_refill_time'] >= 3)
    node2 = int(row['rectal_temperature'] <= 37.5)
    node3 = int(row['pulse'] > 60)
    node4 = int(row['abdominal_distension'] == 3)
    node5 = int(row['pain'] == 2)
    node6 = int(row['peristalsis'] == 1)
    node7 = int(row['mucous_membranes'] == 3)
    node8 = int(row['packed_cell_volume'] < 25)
    emb = [node1, node2, node3, node4, node5, node6, node7, node8]
    
    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
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
            if node6 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node7 == 1:
                return 1, emb
            else:
                if node8 == 1:
                    return 1, emb
                else:
                    return 0, emb

@dataclass
class QwenColicEmbedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
from dataclasses import dataclass

def dt0(feature_row):
    node1 = int(feature_row['surgery'] == 1)
    node2 = int(feature_row['pulse'] > 60)
    node3 = int(feature_row['respiratory_rate'] > 30)
    node4 = int(feature_row['temp_extremities'] < 3)
    node5 = int(feature_row['peripheral_pulse'] != 3)
    node6 = int(feature_row['capillary_refill_time'] != 1)
    node7 = int(feature_row['pain'] > 1)
    node8 = int(feature_row['peristalsis'] != 4)
    node9 = int(feature_row['abdominal_distension'] > 2)
    node10 = int(feature_row['nasogastric_tube'] != 1)
    node11 = int(feature_row['nasogastric_reflux'] != 3)
    node12 = int(feature_row['rectal_examination'] != 4)
    node13 = int(feature_row['abdomen'] != 4)
    node14 = int(feature_row['packed_cell_volume'] < 30)
    node15 = int(feature_row['total_protein'] < 6)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14, node15]

    if node1:
        prediction = 0
    elif node2 and node3 and node4 and node5 and node6 and node7 and node8 and node9 and node10 and node11 and node12 and node13 and node14 and node15:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb


def dt1(feature_row):
    node1 = int(feature_row['surgery'] == 2)
    node2 = int(feature_row['Age'] == 0)
    node3 = int(feature_row['rectal_temperature'] <= 37.5)
    node4 = int(feature_row['pulse'] > 60)
    node5 = int(feature_row['respiratory_rate'] > 40)
    node6 = int(feature_row['temp_extremities'] == 3)
    node7 = int(feature_row['peripheral_pulse'] == 3)
    node8 = int(feature_row['mucous_membranes'] == 4)
    node9 = int(feature_row['capillary_refill_time'] == 1)
    node10 = int(feature_row['pain'] <= 2)
    node11 = int(feature_row['peristalsis'] == 4)
    node12 = int(feature_row['abdominal_distension'] == 2)
    node13 = int(feature_row['nasogastric_tube'] == 1)
    node14 = int(feature_row['nasogastric_reflux'] == 3)
    node15 = int(feature_row['nasogastric_reflux_PH'] < 7.0)
    node16 = int(feature_row['rectal_examination'] == 4)
    node17 = int(feature_row['abdomen'] == 4)
    node18 = int(feature_row['packed_cell_volume'] >= 30)
    node19 = int(feature_row['total_protein'] >= 6.0)
    node20 = int(feature_row['abdominocentesis_appearance'] == 1)
    node21 = int(feature_row['abdomcentesis_total_protein'] < 2.5)
    node22 = int(feature_row['outcome'] == 3)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14, node15, node16, node17, node18, node19, node20, node21, node22]

    if node1 and node2 and node3 and node4 and node5 and node6 and node7 and node8 and node9 and node10 and node11 and node12 and node13 and node14 and node15 and node16 and node17 and node18 and node19 and node20 and node21 and node22:
        return 0, emb
    elif node1 and not node2 and node3 and not node4 and node5 and node6 and not node7 and node8 and node9 and node10 and not node11 and node12 and node13 and not node14 and node15 and node16 and not node17 and node18 and node19 and node20 and node21 and node22:
        return 1, emb
    else:
        return 0, emb


def dt2(feature_row):
    node1 = int(feature_row['Age'] == 1)  # Young
    node2 = int(feature_row['respiratory_rate'] > 30)
    node3 = int(feature_row['rectal_temperature'] < 37.5)
    node4 = int(feature_row['mucous_membranes'] in [3, 5])  # dark cyanotic, pale cyanotic
    node5 = int(feature_row['peripheral_pulse'] in [1, 4])  # absent, reduced
    node6 = int(feature_row['capillary_refill_time'] == 0)  # more than 3 seconds
    node7 = int(feature_row['pain'] in [2, 5])  # continuous severe pain, intermittent severe pain
    node8 = int(feature_row['abdominal_distension'] in [3, 1])  # severe, moderate
    node9 = int(feature_row['nasogastric_reflux'] == 0)  # more then 1 liter
    node10 = int(feature_row['abdomen'] in [1, 2])  # distended large, distended small
    node11 = int(feature_row['total_protein'] < 5.0)
    node12 = int(feature_row['abdominocentesis_appearance'] in [2, 3])  # cloudy, serosanguinous

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12]

    if node1 and (node2 or node3 or node4 or node5 or node6 or node7):
        prediction = 1  # Surgery
    elif node8 and (node9 or node10 or node11 or node12):
        prediction = 1  # Surgery
    else:
        prediction = 0  # No surgery

    return prediction, emb


def dt3(feature_row):
    # Precompute the binary vector embeddings
    node1 = int(feature_row['surgery'] == 2)
    node2 = int(feature_row['Age'] == 1)
    node3 = int(feature_row['rectal_temperature'] > 38.5)
    node4 = int(feature_row['pulse'] > 60)
    node5 = int(feature_row['respiratory_rate'] > 40)
    node6 = int(feature_row['temp_extremities'] == 3 or feature_row['temp_extremities'] == 4)
    node7 = int(feature_row['peripheral_pulse'] == 3 or feature_row['peripheral_pulse'] == 4)
    node8 = int(feature_row['mucous_membranes'] == 4)
    node9 = int(feature_row['capillary_refill_time'] == 1)
    node10 = int(feature_row['pain'] == 1 or feature_row['pain'] == 4)
    node11 = int(feature_row['peristalsis'] == 4)
    node12 = int(feature_row['abdominal_distension'] == 2 or feature_row['abdominal_distension'] == 4)
    node13 = int(feature_row['nasogastric_tube'] == 1)
    node14 = int(feature_row['nasogastric_reflux'] == 3)
    node15 = int(feature_row['nasogastric_reflux_PH'] < 7)
    node16 = int(feature_row['rectal_examination'] == 4)
    node17 = int(feature_row['abdomen'] == 4)
    node18 = int(feature_row['packed_cell_volume'] > 30)
    node19 = int(feature_row['total_protein'] > 6)
    node20 = int(feature_row['abdominocentesis_appearance'] == 1)
    node21 = int(feature_row['abdomcentesis_total_protein'] < 2.5)
    node22 = int(feature_row['outcome'] == 3)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10, node11, node12, node13, node14, node15, node16, node17, node18, node19, node20, node21, node22]

    # Apply the decision tree logic
    if node1 and node2 and node3 and node4 and node5 and node6 and node7 and node8 and node9 and node10 and node11 and node12 and node13 and node14 and node15 and node16 and node17 and node18 and node19 and node20 and node21 and node22:
        prediction = 0
    else:
        prediction = 1

    return prediction, emb


def dt4(feature_row):
    # Precompute binary vector embeddings
    node1 = int(feature_row['surgery'] == 2)
    node2 = int(feature_row['Age'] == 1)
    node3 = int(feature_row['rectal_temperature'] < 38)
    node4 = int(feature_row['pulse'] > 70)
    node5 = int(feature_row['respiratory_rate'] > 30)
    node6 = int(feature_row['temp_extremities'] < 3)
    node7 = int(feature_row['peripheral_pulse'] == 1)
    node8 = int(feature_row['mucous_membranes'] == 1)
    node9 = int(feature_row['capillary_refill_time'] == 0)
    node10 = int(feature_row['pain'] == 2)
    node11 = int(feature_row['peristalsis'] == 1)
    node12 = int(feature_row['abdominal_distension'] == 3)
    node13 = int(feature_row['nasogastric_tube'] == 2)
    node14 = int(feature_row['nasogastric_reflux'] == 0)
    node15 = int(feature_row['nasogastric_reflux_PH'] < 5)
    node16 = int(feature_row['rectal_examination'] == 1)
    node17 = int(feature_row['abdomen'] == 1)
    node18 = int(feature_row['packed_cell_volume'] < 30)
    node19 = int(feature_row['total_protein'] < 5)
    node20 = int(feature_row['abdominocentesis_appearance'] == 2)
    node21 = int(feature_row['abdomcentesis_total_protein'] > 3)
    node22 = int(feature_row['outcome'] == 1)

    emb = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,
           node11, node12, node13, node14, node15, node16, node17, node18, node19,
           node20, node21, node22]

    # Decision tree logic
    if node1 and node2:
        prediction = 1
    elif node3 and node4:
        prediction = 1
    elif node5 and node6:
        prediction = 1
    elif node7 and node8:
        prediction = 1
    elif node9 and node10:
        prediction = 1
    elif node11 and node12:
        prediction = 1
    elif node13 and node14:
        prediction = 1
    elif node15 and node16:
        prediction = 1
    elif node17 and node18:
        prediction = 1
    elif node19 and node20:
        prediction = 1
    elif node21 and node22:
        prediction = 1
    else:
        prediction = 0

    return prediction, emb

@dataclass
class MistralColicEmbedding:
    runner = [dt0, dt1, dt2, dt3, dt4]

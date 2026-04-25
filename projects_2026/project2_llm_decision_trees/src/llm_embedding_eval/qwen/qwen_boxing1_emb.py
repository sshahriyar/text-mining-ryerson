from dataclasses import dataclass

def dt0(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Round'] <= 6)
    node3 = int(row['Judge'] == 'ESPN')
    node4 = int(row['Judge'] == 'Boxing Monthly-Leach')
    node5 = int(row['Round'] > 6)
    emb = [node1, node2, node3, node4, node5]
    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            return 1, emb
        elif node4 == 1:
            return 0, emb
        else:
            if node5 == 1:
                return 0, emb
            else:
                return 1, emb
def dt1(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Round'] <= 6)
    emb = [node1, node2]
    if node1 == 1:
        if node2 == 1:
            prediction = 1
        else:
            prediction = 0
    else:
        if node2 == 1:
            prediction = 0
        else:
            prediction = 1
    return prediction, emb

def dt2(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Round'] <= 6)
    node3 = int(row['Judge'] in [3, 4])
    emb = [node1, node2, node3]
    if node1 == 1:
        if node2 == 1:
            return 1, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            return 1, emb
        else:
            return 0, emb
        
def dt3(row):
    node1 = int(row['Judge'] == 'E. Williams')
    node2 = int(row['Round'] <= 6)
    node3 = int(row['Judge'] == 'L. O\'Connell')
    node4 = int(row['Official'] == 1)
    node5 = int(row['Judge'] == 'HBO-Lederman')
    node6 = int(row['Round'] <= 7)
    node7 = int(row['Round'] <= 5)
    emb = [node1, node2, node3, node4, node5, node6, node7]
    
    if node1 == 1:
        if node2 == 1:
            prediction = 1
        else:
            prediction = 0
    elif node3 == 1:
        if node4 == 1:
            prediction = 1
        else:
            prediction = 0
    elif node5 == 1:
        if node6 == 1:
            prediction = 1
        else:
            prediction = 0
    else:
        if node7 == 1:
            prediction = 1
        else:
            prediction = 0
    
    return prediction, emb

def dt4(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Judge'] == 8)  # ESPN
    node3 = int(row['Round'] <= 6)
    node4 = int(row['Round'] <= 4)
    emb = [node1, node2, node3, node4]
    
    if node1 == 1:
        if node2 == 1:
            prediction = 1
        else:
            if node3 == 1:
                prediction = 1
            else:
                prediction = 0
    else:
        if node4 == 1:
            prediction = 0
        else:
            prediction = 1
    return prediction, emb

@dataclass
class QwenBoxing1Embedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
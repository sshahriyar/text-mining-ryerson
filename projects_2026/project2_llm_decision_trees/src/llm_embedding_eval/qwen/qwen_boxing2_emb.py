from dataclasses import dataclass

def dt0(row):
    node1 = int(row['Judge'] <= 5)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Round'] <= 6)
    emb = [node1, node2, node3]
    if node1 == 1:
        if node2 == 1:
            return 0, emb
        else:
            if node3 == 1:
                return 0, emb
            else:
                return 1, emb
    else:
        if node2 == 1:
            return 1, emb
        else:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
def dt1(row):
    node1 = int(row['Round'] <= 6)
    node2 = int(row['Official'] == 1)
    node3 = int(row['Judge'] in ['USA Today', 'Associated Press'])
    emb = [node1, node2, node3]
    if node1 == 1:
        prediction = 0
    else:
        if node2 == 1:
            prediction = 1
        else:
            if node3 == 1:
                prediction = 1
            else:
                prediction = 0
    return prediction, emb

def dt2(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Round'] <= 6)
    node3 = int(row['Judge'] == 'USA Today')
    emb = [node1, node2, node3]
    if node1 == 1:
        if node2 == 1:
            return 0, emb
        else:
            return 1, emb
    else:
        if node3 == 1:
            return 0, emb
        else:
            return 1, emb
        
def dt3(row):
    node1 = int(row['Official'] == 1)
    node2 = int(row['Round'] <= 6)
    node3 = int(row['Round'] <= 4)
    emb = [node1, node2, node3]
    if node1 == 1:
        if node2 == 1:
            return 0, emb
        else:
            return 1, emb
    else:
        if node3 == 1:
            return 0, emb
        else:
            return 1, emb
def dt4(row):
    node1 = int(row['Judge'] == 0)
    node2 = int(row['Judge'] == 1)
    node3 = int(row['Official'] == 1)
    node4 = int(row['Round'] <= 6)
    emb = [node1, node2, node3, node4]
    if node1:
        if node3:
            return 0, emb
        else:
            return 1, emb
    elif node2:
        if node4:
            return 1, emb
        else:
            return 0, emb
    else:
        if row['Round'] > 6:
            return 1, emb
        else:
            return 0, emb

@dataclass
class QwenBoxing2Embedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
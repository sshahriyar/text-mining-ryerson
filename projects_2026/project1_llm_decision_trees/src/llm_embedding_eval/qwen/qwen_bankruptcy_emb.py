from dataclasses import dataclass

def dt0(row):
    node1 = int(row['WC/TA'] <= 10)
    node2 = int(row['EBIT/TA'] <= 5)
    node3 = int(row['S/TA'] <= 20)
    node4 = int(row['BVE/BVL'] <= 0.5)
    node5 = int(row['RE/TA'] <= 3)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        return 1, emb
    elif node2 == 1:
        return 1, emb
    elif node3 == 1:
        return 1, emb
    elif node4 == 1:
        return 1, emb
    elif node5 == 1:
        return 1, emb
    else:
        return 0, emb
    
def dt1(row):
    node1 = int(row['WC/TA'] <= 0.10)
    node2 = int(row['RE/TA'] <= 0.05)
    node3 = int(row['EBIT/TA'] <= 0.05)
    node4 = int(row['S/TA'] <= 0.20)
    node5 = int(row['BVE/BVL'] <= 1.0)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        prediction = 1
    else:
        if node2 == 1:
            prediction = 1
        else:
            if node3 == 1:
                prediction = 1
            else:
                if node4 == 1:
                    prediction = 1
                else:
                    if node5 == 1:
                        prediction = 1
                    else:
                        prediction = 0
    return prediction, emb

def dt2(row):
    node1 = int(row['BVE/BVL'] <= 1.0)
    node2 = int(row['WC/TA'] <= 10.0)
    node3 = int(row['EBIT/TA'] <= 5.0)
    node4 = int(row['S/TA'] <= 20.0)
    node5 = int(row['RE/TA'] <= 5.0)
    emb = [node1, node2, node3, node4, node5]
    if node1 == 1:
        if node2 == 1:
            if node3 == 1:
                if node4 == 1:
                    if node5 == 1:
                        return 1, emb
                    else:
                        return 0, emb
                else:
                    return 0, emb
            else:
                return 0, emb
        else:
            return 0, emb
    else:
        return 0, emb
    
def dt3(row):
    node1 = int(row['WC/TA'] <= 0.10)
    node2 = int(row['RE/TA'] <= 0.05)
    node3 = int(row['EBIT/TA'] <= 0.05)
    node4 = int(row['S/TA'] <= 0.20)
    node5 = int(row['BVE/BVL'] <= 0.50)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        if node2 == 1:
            prediction = 1
        else:
            prediction = 0
    else:
        if node3 == 1:
            if node4 == 1:
                prediction = 1
            else:
                prediction = 0
        else:
            if node5 == 1:
                prediction = 1
            else:
                prediction = 0
    return prediction, emb

def dt4(row):
    node1 = int(row['BVE/BVL'] <= 0.5)
    node2 = int(row['EBIT/TA'] <= 5)
    node3 = int(row['WC/TA'] <= 5)
    node4 = int(row['S/TA'] <= 10)
    emb = [node1, node2, node3, node4]
    
    if node1 == 1:
        return 1, emb
    else:
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

@dataclass
class QwenBankruptcyEmbedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
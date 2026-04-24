from dataclasses import dataclass

def dt0(row):
    node1 = int(row['Derogatory.reports'] > 0)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Income.per.dependent'] > 5)
    node4 = int(row['Monthly.credit.card.exp'] < 500)
    node5 = int(row['Self.employed'] == 1)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        return 0, emb
    if node2 == 1:
        if node3 == 1:
            return 1, emb
        else:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
    else:
        if node5 == 1:
            if node3 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
def dt1(row):
    node1 = int(row['Derogatory.reports'] > 2)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Income.per.dependent'] > 5)
    node4 = int(row['Monthly.credit.card.exp'] <= 500)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Age'] > 35)
    emb = [node1, node2, node3, node4, node5, node6]
    
    if node1 == 1:
        return 0, emb
    elif node2 == 1:
        if node5 == 1:
            if node6 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 1, emb
    else:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb
        
def dt2(row):
    node1 = int(row['Derogatory.reports'] > 2)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Income.per.dependent'] > 5)
    node4 = int(row['Monthly.credit.card.exp'] > 500)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Monthly.credit.card.exp'] < 300)
    emb = [node1, node2, node3, node4, node5, node6]
    
    if node1 == 1:
        prediction = 0
    elif node2 == 1:
        if node3 == 1:
            if node4 == 1:
                prediction = 1
            else:
                prediction = 0
        else:
            prediction = 0
    else:
        if node5 == 1:
            if node6 == 1:
                prediction = 1
            else:
                prediction = 0
        else:
            prediction = 0
    return prediction, emb

def dt3(row):
    node1 = int(row['Derogatory.reports'] > 2)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Income.per.dependent'] > 5)
    node4 = int(row['Monthly.credit.card.exp'] < 1000)
    node5 = int(row['Self.employed'] == 1)
    emb = [node1, node2, node3, node4, node5]
    
    if node1 == 1:
        return 0, emb
    elif node2 == 1:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb
    else:
        if node3 == 1:
            if node5 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb
        
def dt4(row):
    node1 = int(row['Derogatory.reports'] <= 2)
    node2 = int(row['Own.home'] == 1)
    node3 = int(row['Income.per.dependent'] > 5)
    node4 = int(row['Monthly.credit.card.exp'] < 3000)
    node5 = int(row['Self.employed'] == 1)
    node6 = int(row['Income.per.dependent'] > 6)
    node7 = int(row['Monthly.credit.card.exp'] < 2000)
    emb = [node1, node2, node3, node4, node5, node6, node7]
    if node1 == 0:
        return 0, emb
    if node2 == 1:
        if node3 == 1:
            if node4 == 1:
                return 1, emb
            else:
                if node5 == 1:
                    return 0, emb
                else:
                    return 1, emb
        else:
            return 0, emb
    else:
        if node6 == 1:
            if node7 == 1:
                return 1, emb
            else:
                return 0, emb
        else:
            return 0, emb

@dataclass
class QwenCreditscoreEmbedding:
    runner = [dt0, dt1, dt2, dt3, dt4]
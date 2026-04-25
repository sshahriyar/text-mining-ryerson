from dataclasses import dataclass

def dt_func_0(row):
    # Precompute binary embeddings for each decision node
    node1 = int(row['BVE/BVL'] <= 0.5)          # High leverage → risk
    node2 = int(row['EBIT/TA'] <= 0)            # Negative earnings → risk
    node3 = int(row['WC/TA'] <= 0)              # Negative working capital → risk
    node4 = int(row['RE/TA'] <= 0)              # No retained earnings → risk
    node5 = int(row['S/TA'] <= 0.3)             # Low asset turnover → risk
    
    emb = [node1, node2, node3, node4, node5]
    
    # Decision tree logic (depth = 5)
    if node1:
        pred = 1          # Bankrupt
    elif node2:
        pred = 1
    elif node3:
        pred = 1
    elif node4:
        pred = 1
    elif node5:
        pred = 1
    else:
        pred = 0          # Not bankrupt
    
    return pred, emb

def dt_func_1(row):
    # Precompute binary node embeddings
    node0 = int(row['BVE/BVL'] <= 0.6)   # BVE/BVL <= 0.6
    node1 = int(row['WC/TA'] <= 20)     # WC/TA <= 20
    node2 = int(row['EBIT/TA'] <= 5)    # EBIT/TA <= 5
    node3 = int(row['RE/TA'] <= 8)      # RE/TA <= 8
    emb = [node0, node1, node2, node3]

    # Decision tree logic
    if node0:
        if node1:
            pred = 1   # bankrupt
        else:
            if node3:
                pred = 1   # bankrupt
            else:
                pred = 0   # not bankrupt
    else:
        if node2:
            pred = 1   # bankrupt
        else:
            pred = 0   # not bankrupt

    return pred, emb


def dt_func_2(row):
    """
    Predict bankruptcy (1 = bankrupt, 0 = not bankrupt) and return the binary
    truth values of the inner decision nodes.
    
    Parameters
    ----------
    row : dict or pandas.Series
        Must contain the keys:
        - 'BVE/BVL'
        - 'WC/TA'
        - 'RE/TA'
        - 'EBIT/TA'
        - 'S/TA'
    
    Returns
    -------
    prediction : int
        1 if the company is predicted to bankrupt, 0 otherwise.
    emb : list of int
        Binary vector representing the truth values of the inner nodes:
        [node1, node2, node3, node4]
    """
    # Precompute binary node embeddings
    node1 = int(row['BVE/BVL'] <= 0.3)   # Node 1: low leverage
    node2 = int(row['WC/TA']   <= 0.10)  # Node 2: low liquidity
    node3 = int(row['RE/TA']   <= 0.0)   # Node 3: negative retained earnings
    node4 = int(row['EBIT/TA'] <= 0.02)  # Node 4: low asset profitability
    emb = [node1, node2, node3, node4]
    
    # Decision tree logic
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
                    prediction = 0
    
    return prediction, emb



def dt_func_3(row):
    # compute binary node conditions
    node1 = int(row['BVE/BVL'] <= 0.5)          # BVE/BVL very low → high risk
    node2 = int(row['BVE/BVL'] <= 1.0)          # moderate BVE/BVL
    node3 = int(row['EBIT/TA'] <= 1.0)          # low profitability
    node4 = int(row['WC/TA'] <= 0.1)            # poor liquidity
    node5 = int(row['RE/TA'] <= 0.05)           # negligible retained earnings

    emb = [node1, node2, node3, node4, node5]

    # decision logic
    if node1 == 1:
        prediction = 1  # bankrupt
    else:
        if node2 == 1:  # BVE/BVL <= 1.0
            if node3 == 1:
                prediction = 1  # bankrupt
            else:
                prediction = 0  # not bankrupt
        else:  # BVE/BVL > 1.0
            if node4 == 1:
                prediction = 1  # bankrupt
            else:
                if node5 == 1:
                    prediction = 1  # bankrupt
                else:
                    prediction = 0  # not bankrupt

    return prediction, emb


def dt_func_4(row):
    # Precompute binary node embeddings
    node1 = int(row['BVE/BVL'] <= 0.6)          # BVE/BVL <= 0.6 ?
    node2 = int(row['EBIT/TA'] <= 0.02)         # EBIT/TA <= 0.02 ?
    node3 = int(row['WC/TA'] <= 0.05)           # WC/TA <= 0.05 ?
    emb = [node1, node2, node3]

    # Decision tree logic
    if node1 == 1:
        pred = 0            # Not bankrupt
    else:
        if node2 == 1:
            pred = 1        # Bankrupt
        else:
            if node3 == 1:
                pred = 1    # Bankrupt
            else:
                pred = 0    # Not bankrupt
    return pred, emb

@dataclass
class GptBankruptcyEmbedding:
    runner = [dt_func_0, dt_func_1, dt_func_2, dt_func_3, dt_func_4]
